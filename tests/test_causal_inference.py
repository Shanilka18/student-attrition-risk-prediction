"""Unit tests for Phase 7 Applied Causal Inference Framework."""

import json
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import pytest

from src.causal.causal_forest import can_run_causal_forest
from src.causal.dag import (
    EXPOSURES,
    OUTCOME,
    build_causal_dag,
    get_adjustment_set,
)
from src.causal.diagnostics import (
    compute_e_value,
    run_refutation_tests,
)
from src.causal.estimators import (
    estimate_doubly_robust_aipw,
    estimate_g_computation,
    estimate_matching_sensitivity,
    estimate_stabilized_ipw,
    estimate_unadjusted_rd,
)
from src.causal.narrative import MANDATORY_CAUSAL_HEADER, generate_causal_narrative


class TestCausalDAGIntegrity:
    def test_dag_acyclic_and_connected(self):
        G = build_causal_dag()
        assert nx.is_directed_acyclic_graph(G), "Causal DAG must not contain cycles"
        assert nx.is_weakly_connected(G), "Causal DAG must be a single connected graph"

    def test_adjustment_sets_exclude_exposure_and_outcome(self):
        for exp in EXPOSURES:
            adj_set = get_adjustment_set(exp)
            assert (
                exp not in adj_set
            ), f"Exposure '{exp}' cannot be in its own adjustment set"
            assert (
                OUTCOME not in adj_set
            ), f"Outcome '{OUTCOME}' cannot be in adjustment set"


class TestEstimatorsAndConfoundingAdjustment:
    @pytest.fixture
    def synthetic_confounded_data(self):
        np.random.seed(42)
        n = 500
        # Confounder W affects both A and Y
        W = np.random.normal(0, 1, size=n)
        # Treatment assignment biased by W
        p_A = 1 / (1 + np.exp(-(0.8 * W)))
        A = np.random.binomial(1, p_A)
        # Outcome has true treatment effect of +0.10, confounded by W (+0.25)
        latent_logit = -1.5 + 0.6 * A + 0.9 * W
        p_Y = 1 / (1 + np.exp(-latent_logit))
        Y = np.random.binomial(1, p_Y)
        return pd.DataFrame({"confounder_w": W, "exposure_a": A, "attrition": Y})

    def test_confounding_bias_removal(self, synthetic_confounded_data):
        df = synthetic_confounded_data
        unadj = estimate_unadjusted_rd(df, "exposure_a")
        sipw = estimate_stabilized_ipw(df, "exposure_a", ["confounder_w"])
        gcomp = estimate_g_computation(
            df, "exposure_a", ["confounder_w"], n_bootstraps=30
        )

        # Unadjusted RD must be systematically higher due to positive confounding
        assert unadj["risk_difference"] > sipw["risk_difference"]
        # Both adjusted estimators should yield consistent risk differences
        assert abs(sipw["risk_difference"] - gcomp["risk_difference"]) < 0.05

    def test_doubly_robust_aipw_and_matching(self, synthetic_confounded_data):
        df = synthetic_confounded_data
        aipw = estimate_doubly_robust_aipw(df, "exposure_a", ["confounder_w"])
        match = estimate_matching_sensitivity(df, "exposure_a", ["confounder_w"])

        assert not np.isnan(aipw["risk_difference"])
        assert match["n_matched_pairs"] > 50


class TestDiagnosticsAndRefutations:
    def test_e_value_properties(self):
        # E-value with positive RD
        res = compute_e_value(risk_difference=0.10, baseline_risk=0.17)
        assert res["e_value_point"] > 1.5
        # E-value with zero RD should equal 1.0 (no confounding required to explain away null)
        res_null = compute_e_value(risk_difference=0.0, baseline_risk=0.17)
        assert res_null["e_value_point"] == 1.0

    def test_placebo_refutation_detects_null(self):
        # On larger sample size (n=1000), a purely random placebo permutation converges to near zero (|RD| < 0.035)
        np.random.seed(42)
        n = 1000
        W = np.random.randn(n)
        A = np.random.binomial(1, 0.5, size=n)
        Y = np.random.binomial(1, 0.2, size=n)
        df = pd.DataFrame({"w": W, "exposure": A, "attrition": Y})

        ref_res = run_refutation_tests(df, "exposure", ["w"], observed_rd=0.0)
        assert ref_res["placebo_treatment"]["passed"] is True


class TestCausalForestGating:
    def test_causal_forest_gate_pass_and_fail(self):
        # Pass conditions
        passed, reason = can_run_causal_forest(
            n=500, n_events=80, min_exposure_group_size=50, overlap_satisfied=True
        )
        assert passed is True
        assert reason is None

        # Fail: insufficient sample size
        passed, reason = can_run_causal_forest(
            n=250, n_events=80, min_exposure_group_size=50, overlap_satisfied=True
        )
        assert passed is False
        assert "Sample size" in reason

        # Fail: insufficient events
        passed, reason = can_run_causal_forest(
            n=500, n_events=40, min_exposure_group_size=50, overlap_satisfied=True
        )
        assert passed is False
        assert "events" in reason

        # Fail: poor overlap
        passed, reason = can_run_causal_forest(
            n=500, n_events=80, min_exposure_group_size=50, overlap_satisfied=False
        )
        assert passed is False
        assert "overlap" in reason


class TestPayloadSeparationAndGuardrails:
    def test_narrative_label_and_anti_shap_isolation(self):
        narrative_res = generate_causal_narrative(
            exposure_name="long_commute",
            unadjusted_rd=0.095,
            adjusted_rd=0.062,
            ci_lower=0.02,
            ci_upper=0.10,
            e_value=1.85,
            refutations_passed=True,
            planted_ate=0.06,
        )

        # Assert mandatory disclaimer header is strictly present
        assert narrative_res["header"] == MANDATORY_CAUSAL_HEADER
        assert narrative_res["type"] == "causal_inference"
        # Assert SHAP terminology is strictly absent
        assert "shap" not in narrative_res["narrative"].lower()
        assert "attribution" not in narrative_res["narrative"].lower()

    def test_strict_payload_type_separation_from_shap(self):
        """Assert that predictive SHAP and causal inference payloads maintain distinct type identifiers."""
        causal_summary_path = Path("artifacts/frontend_data/causal_summary.json")
        shap_summary_path = Path("artifacts/frontend_data/global_importance.json")

        if causal_summary_path.exists():
            with open(causal_summary_path, "r", encoding="utf-8") as f:
                causal_data = json.load(f)
            assert causal_data.get("type") == "causal_inference"

        if shap_summary_path.exists():
            with open(shap_summary_path, "r", encoding="utf-8") as f:
                shap_data = json.load(f)
            # Must not be labelled causal_inference
            if isinstance(shap_data, dict):
                assert shap_data.get("type") != "causal_inference"
            elif isinstance(shap_data, list):
                for item in shap_data:
                    if isinstance(item, dict):
                        assert item.get("type") != "causal_inference"
