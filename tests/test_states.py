"""Unit and integration tests for student engagement states, patterns, and unsupervised sequence modeling."""

import numpy as np
import pandas as pd
import pytest

from mei.features.transformers import CANONICAL_INDICATORS
from mei.states.clustering import (
    evaluate_clusterings,
    extract_composite_trajectory_matrix,
    profile_cluster,
)
from mei.states.evaluation import (
    evaluate_ground_truth_recovery,
    run_state_ablation_cv,
)
from mei.states.hmm import (
    decode_hmm_sequences,
    fit_gaussian_hmm_with_restarts,
    prepare_sequence_data,
    select_optimal_hmm,
)
from mei.states.patterns import (
    classify_all_trajectories,
    classify_trajectory_pattern,
)
from mei.states.transformers import StateFeatureTransformer


@pytest.fixture
def sample_survey_dataframe() -> pd.DataFrame:
    """Generate reproducible 60-student survey dataframe with 28 canonical columns."""
    rng = np.random.RandomState(42)
    n_students = 60
    data = {}

    for ind in CANONICAL_INDICATORS:
        for p in [1, 2, 3, 4]:
            # Discrete ordinal values 1..5
            data[f"{ind}_p{p}"] = rng.choice([1, 2, 3, 4, 5], size=n_students)

    df = pd.DataFrame(data)
    df["student_id"] = [f"STU_{i:04d}" for i in range(n_students)]
    df["at_risk"] = rng.choice([0, 1], size=n_students, p=[0.75, 0.25])
    return df


class TestPatternRules:
    """Tests for deterministic pattern and engagement level classification."""

    def test_consistent_patterns(self):
        # Exactly constant
        res = classify_trajectory_pattern([3, 3, 3, 3], k=3)
        assert res.pattern == "Consistent"
        assert res.level == "High"

        res_low = classify_trajectory_pattern([1, 1, 1, 1], k=3)
        assert res_low.pattern == "Consistent"
        assert res_low.level == "Low"

        # Monotonic step of size 1 without return
        res_step_up = classify_trajectory_pattern([2, 2, 3, 3], k=3)
        assert res_step_up.pattern == "Consistent"

        res_step_down = classify_trajectory_pattern([3, 2, 2, 2], k=3)
        assert res_step_down.pattern == "Consistent"

    def test_declining_patterns(self):
        # Monotonic decline
        res = classify_trajectory_pattern([3, 3, 2, 1], k=3)
        assert res.pattern == "Declining"
        assert res.net_change <= -1

        # Steep decline
        res2 = classify_trajectory_pattern([3, 2, 1, 1], k=3)
        assert res2.pattern == "Declining"

    def test_improving_patterns(self):
        # Monotonic improvement
        res = classify_trajectory_pattern([1, 1, 2, 3], k=3)
        assert res.pattern == "Improving"
        assert res.net_change >= 1

    def test_sporadic_patterns(self):
        # Alternating shocks: up, down, up (>= 2 reversals)
        res = classify_trajectory_pattern([1, 3, 1, 3], k=3)
        assert res.pattern == "Sporadic"
        assert res.n_reversals >= 2

        # Return to base (not monotonic): [2, 3, 2, 2]
        res2 = classify_trajectory_pattern([2, 3, 2, 2], k=3)
        assert res2.pattern == "Sporadic"

    def test_invalid_path_raises_error(self):
        with pytest.raises(ValueError, match="Expected 4-period state path"):
            classify_trajectory_pattern([1, 2, 3])

    def test_classify_all_trajectories(self):
        paths = np.array(
            [
                [3, 3, 3, 3],
                [3, 3, 2, 1],
                [1, 1, 2, 3],
                [1, 3, 1, 3],
            ]
        )
        patterns, levels, summary_df = classify_all_trajectories(paths, k=3)
        assert patterns == ["Consistent", "Declining", "Improving", "Sporadic"]
        assert len(levels) == 4
        assert len(summary_df) == 4


class TestHMMSequenceModeling:
    """Tests for GaussianHMM model selection, fitting, ordering, and decoding."""

    def test_toy_sequence_recovery(self):
        """Verify HMM recovers known state transitions from structured synthetic sequence."""
        rng = np.random.RandomState(42)
        n_samples = 40
        # State 0: Low values (mean 1.0)
        # State 1: High values (mean 5.0)
        X = np.zeros((n_samples * 4, 7))
        for i in range(n_samples):
            if i < 20:
                # Persistent low state
                X[i * 4 : (i + 1) * 4] = 1.0 + rng.normal(0, 0.1, size=(4, 7))
            else:
                # Persistent high state
                X[i * 4 : (i + 1) * 4] = 5.0 + rng.normal(0, 0.1, size=(4, 7))

        lengths = [4] * n_samples
        model, ll, converged = fit_gaussian_hmm_with_restarts(
            X, lengths, k=2, n_restarts=3, random_seed=42
        )
        assert converged
        assert ll > -np.inf

        # State 0 must be lower than State 1
        assert model.means_[0].mean() < model.means_[1].mean()

        # Predictions on low students should decode as state 1 (1-indexed)
        paths, posteriors, scores = decode_hmm_sequences(model, X, lengths)
        assert np.all(paths[:20] == 1)
        assert np.all(paths[20:] == 2)
        assert posteriors.shape == (n_samples, 4, 2)
        assert len(scores) == n_samples

    def test_state_ordering_is_deterministic(self, sample_survey_dataframe):
        """Assert state ordering places lowest composite engagement at State 0."""
        X_seq_scaled, lengths, _, _ = prepare_sequence_data(
            sample_survey_dataframe, fit_scaler=True
        )
        model, _, _ = fit_gaussian_hmm_with_restarts(
            X_seq_scaled, lengths, k=3, n_restarts=3, random_seed=42
        )

        composite_means = model.means_.mean(axis=1)
        # Monotonically non-decreasing
        assert composite_means[0] <= composite_means[1] <= composite_means[2]

    def test_select_optimal_hmm(self, sample_survey_dataframe):
        train_df = sample_survey_dataframe.iloc[:40]
        val_df = sample_survey_dataframe.iloc[40:]

        res = select_optimal_hmm(
            train_df=train_df,
            val_df=val_df,
            k_range=[2, 3],
            n_restarts=2,
            random_seed=42,
        )
        assert res.best_k in [2, 3]
        assert not res.model_table.empty
        assert len(res.expected_dwell_times) == res.best_k
        assert res.transition_matrix.shape == (res.best_k, res.best_k)


class TestClustering:
    """Tests for longitudinal clustering model selection and profiling."""

    def test_extract_composite_trajectory_matrix(self, sample_survey_dataframe):
        matrix, feat_names = extract_composite_trajectory_matrix(
            sample_survey_dataframe
        )
        assert matrix.shape == (len(sample_survey_dataframe), 7)
        assert feat_names == [
            "composite_p1",
            "composite_p2",
            "composite_p3",
            "composite_p4",
            "composite_slope",
            "composite_volatility",
            "composite_level",
        ]

    def test_cluster_profiling(self):
        # High Consistent
        c_high = np.array([80, 80, 80, 80, 0.0, 2.0, 80.0])
        names = [
            "composite_p1",
            "composite_p2",
            "composite_p3",
            "composite_p4",
            "composite_slope",
            "composite_volatility",
            "composite_level",
        ]
        profile = profile_cluster(c_high, names)
        assert profile == "High Consistent"

        # Moderate Declining
        c_decl = np.array([60, 55, 45, 35, -7.0, 8.0, 48.0])
        assert profile_cluster(c_decl, names) == "Moderate Declining"

        # Low Consistent
        c_low = np.array([20, 20, 20, 20, 0.0, 1.0, 20.0])
        assert profile_cluster(c_low, names) == "Low Consistent"

    def test_evaluate_clusterings_selection(self, sample_survey_dataframe):
        matrix, feat_names = extract_composite_trajectory_matrix(
            sample_survey_dataframe
        )
        res = evaluate_clusterings(
            matrix,
            feat_names,
            k_range=[2, 3],
            n_bootstrap=5,  # fast test
            silhouette_threshold=0.10,
            random_seed=42,
        )
        assert res.best_algorithm in ["kmeans", "gmm_diag", "gmm_full", "ward"]
        assert res.best_k in [2, 3]
        assert len(res.cluster_names) == res.best_k
        assert res.centroids.shape == (res.best_k, 7)


class TestStateFeatureTransformerNoLeakage:
    """Strict Rule G3 no-leakage and pipeline integration tests."""

    def test_transformer_fit_transform_shapes(self, sample_survey_dataframe):
        transformer = StateFeatureTransformer(
            clustering_algo="kmeans",
            n_clusters=3,
            hmm_k=2,
            n_hmm_restarts=2,
            random_state=42,
        )
        transformer.fit(sample_survey_dataframe)
        features = transformer.transform(sample_survey_dataframe)

        assert isinstance(features, pd.DataFrame)
        assert len(features) == len(sample_survey_dataframe)
        # Feature names check
        assert "state_p1" in features.columns
        assert "final_state" in features.columns
        assert "n_transitions" in features.columns
        assert "time_in_state_1" in features.columns
        assert "posterior_p4_state_1" in features.columns
        assert "sequence_log_likelihood" in features.columns
        assert "cluster_0" in features.columns
        assert "dist_to_cluster_center" in features.columns
        assert "pattern_consistent" in features.columns
        assert "level_high" in features.columns

    def test_fold_wise_fitting_does_not_leak_validation_data(
        self, sample_survey_dataframe
    ):
        """Rule G3: Modifying validation rows must NOT alter fitted parameters on train fold."""
        train_df = sample_survey_dataframe.iloc[:40].copy()
        val_df = sample_survey_dataframe.iloc[40:].copy()

        # Fit transformer on original training data
        transformer1 = StateFeatureTransformer(
            clustering_algo="kmeans",
            n_clusters=3,
            hmm_k=2,
            n_hmm_restarts=2,
            random_state=42,
        )
        transformer1.fit(train_df)

        # Mutate validation data completely (extreme values)
        val_mutated = val_df.copy()
        for col in val_mutated.columns:
            if col.startswith("resource_access"):
                val_mutated[col] = 1

        # Transformer fitted on train_df must yield identical centroids regardless of validation data
        centroids_pre = transformer1.cluster_centroids_.copy()
        hmm_means_pre = transformer1.hmm_model_.means_.copy()

        # Transform both
        out1 = transformer1.transform(val_df)
        out2 = transformer1.transform(val_mutated)

        # Parameters on transformer remain unchanged
        np.testing.assert_array_equal(transformer1.cluster_centroids_, centroids_pre)
        np.testing.assert_array_equal(transformer1.hmm_model_.means_, hmm_means_pre)

        # Predictions for mutated rows differ appropriately
        assert not out1["dist_to_cluster_center"].equals(out2["dist_to_cluster_center"])

    def test_alternative_clustering_algorithms(self, sample_survey_dataframe):
        """Verify StateFeatureTransformer works with gmm_diag and ward."""
        for algo in ["gmm_diag", "ward"]:
            transformer = StateFeatureTransformer(
                clustering_algo=algo,
                n_clusters=2,
                hmm_k=2,
                n_hmm_restarts=1,
                random_state=42,
            )
            transformer.fit(sample_survey_dataframe)
            features = transformer.transform(sample_survey_dataframe)
            assert "cluster_0" in features.columns
            assert "cluster_1" in features.columns
            assert len(features) == len(sample_survey_dataframe)


class TestEvaluationAndFigures:
    """Tests for ground truth recovery, CV ablation, and publication figure generation."""

    def test_ground_truth_recovery(self, tmp_path):
        gt_file = tmp_path / "mock_gt.csv"
        df_gt = pd.DataFrame(
            {
                "student_id": ["STU_0001", "STU_0002", "STU_0003", "STU_0004"],
                "archetype": [
                    "consistent_high",
                    "consistent_high",
                    "declining",
                    "declining",
                ],
            }
        )
        df_gt.to_csv(gt_file, index=False)

        res = evaluate_ground_truth_recovery(
            student_ids=["STU_0001", "STU_0002", "STU_0003", "STU_0004"],
            cluster_labels=np.array([0, 0, 1, 1]),
            decoded_patterns=["Consistent", "Consistent", "Declining", "Declining"],
            ground_truth_path=gt_file,
        )
        assert res is not None
        assert res.archetype_pattern_ari == 1.0
        assert res.archetype_cluster_ari == 1.0
        assert res.is_valid_only_for_generated_data is True

    def test_ablation_cross_validation(self, sample_survey_dataframe, tmp_path):
        import json

        # Mock 2-fold CV scheme on 30 students
        sub_df = sample_survey_dataframe.iloc[:30].copy().reset_index(drop=True)
        agg_features = sub_df[
            [c for c in sub_df.columns if c.startswith("resource_access")]
        ].copy()

        cv_file = tmp_path / "mock_cv.json"
        cv_data = {
            "folds": [
                {
                    "fold_id": 0,
                    "train_indices": list(range(15)),
                    "val_indices": list(range(15, 30)),
                },
                {
                    "fold_id": 1,
                    "train_indices": list(range(15, 30)),
                    "val_indices": list(range(15)),
                },
            ]
        }
        cv_file.write_text(json.dumps(cv_data), encoding="utf-8")

        summary = run_state_ablation_cv(
            clean_df=sub_df,
            agg_features_df=agg_features,
            cv_folds_path=cv_file,
            clustering_algo="kmeans",
            n_clusters=2,
            hmm_k=2,
            random_seed=42,
        )
        assert "logistic_regression" in summary.results
        assert "lightgbm" in summary.results
        assert "agg" in summary.results["logistic_regression"]
        assert "agg_states" in summary.results["logistic_regression"]
        assert "states_only" in summary.results["logistic_regression"]
        assert "auc" in summary.wilcoxon_tests["logistic_regression"]

    def test_publication_figures_generation(self, sample_survey_dataframe, tmp_path):
        from mei.states.figures import (
            plot_at_risk_by_pattern,
            plot_cluster_silhouette,
            plot_emission_profiles,
            plot_k_selection_curves,
            plot_pattern_distribution,
            plot_trajectories_by_pattern,
            plot_transition_matrix,
        )

        fig_dirs = [tmp_path / "fig_out"]

        # 1. Emission profiles
        plot_emission_profiles(
            emission_means=np.array([[1.0] * 7, [3.0] * 7]),
            indicator_names=CANONICAL_INDICATORS,
            state_names={0: "Low", 1: "High"},
            output_dirs=fig_dirs,
        )
        assert (fig_dirs[0] / "state_01_emission_profiles.png").exists()
        assert (fig_dirs[0] / "state_01_emission_profiles.svg").exists()

        # 2. Transition matrix
        plot_transition_matrix(
            transmat=np.array([[0.8, 0.2], [0.1, 0.9]]),
            dwell_times={0: 5.0, 1: 10.0},
            state_names={0: "Low", 1: "High"},
            output_dirs=fig_dirs,
        )
        assert (fig_dirs[0] / "state_02_transition_matrix.png").exists()

        # 3. Trajectories by pattern
        plot_trajectories_by_pattern(
            clean_df=sample_survey_dataframe,
            patterns=["Consistent"] * 30 + ["Declining"] * 30,
            output_dirs=fig_dirs,
        )
        assert (fig_dirs[0] / "state_03_trajectories_by_pattern.png").exists()

        # 4. Pattern distribution
        plot_pattern_distribution(
            patterns=["Consistent"] * 30 + ["Declining"] * 30,
            levels=["Low"] * 20 + ["Medium"] * 20 + ["High"] * 20,
            output_dirs=fig_dirs,
        )
        assert (fig_dirs[0] / "state_04_pattern_distribution.png").exists()

        # 5. At risk by pattern
        plot_at_risk_by_pattern(
            patterns=["Consistent"] * 30 + ["Declining"] * 30,
            at_risk=np.array([0] * 40 + [1] * 20),
            output_dirs=fig_dirs,
        )
        assert (fig_dirs[0] / "state_05_at_risk_by_pattern.png").exists()

        # 6. Silhouette plot
        plot_cluster_silhouette(
            X_scaled=np.random.randn(60, 4),
            cluster_labels=np.array([0] * 30 + [1] * 30),
            cluster_names={0: "Low Consistent", 1: "High Consistent"},
            output_dirs=fig_dirs,
        )
        assert (fig_dirs[0] / "state_06_cluster_silhouette.png").exists()

        # 7. Model selection curves
        model_table = pd.DataFrame(
            {
                "k": [2, 3, 4],
                "train_log_likelihood": [-500.0, -450.0, -420.0],
                "bic": [1200.0, 1150.0, 1180.0],
                "val_log_likelihood": [-520.0, -470.0, -490.0],
            }
        )
        plot_k_selection_curves(
            model_table=model_table,
            output_dirs=fig_dirs,
        )
        assert (fig_dirs[0] / "state_07_k_selection_curves.png").exists()
