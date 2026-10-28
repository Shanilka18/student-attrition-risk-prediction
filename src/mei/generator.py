"""Realistic synthetic survey data generator from latent behavioral models.

Generates Google Forms style raw exports with authentic response anomalies,
temporal dynamics, response styles, mess, and calibrated attrition labels.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

from mei.config import CONFIG_DIR, DATA_DIR
from mei.io.codec import SurveyCodec
from mei.io.schema import load_mapping, load_schema

logger = logging.getLogger(__name__)


@dataclass
class CalibrationRecord:
    """Record of a single calibration iteration."""

    iteration: int
    signal_strength: float
    intercept_w0: float
    prevalence: float
    lr_auc: float
    lgb_auc: float
    auc_gap: float
    converged: bool


@dataclass
class RealityCheckStats:
    """Configured vs empirical anomaly rates."""

    metrics: Dict[str, Dict[str, float]] = field(default_factory=dict)


class SyntheticSurveyGenerator:
    """Generates synthetic student cohorts from hidden latent engagement dynamics."""

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        seed: int = 42,
    ) -> None:
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        # Load configurations
        self.schema = load_schema()
        self.mapping = load_mapping()
        self.codec = SurveyCodec(self.schema)

        if config is None:
            import yaml

            proj_file = CONFIG_DIR / "project.yaml"
            with open(proj_file, "r", encoding="utf-8") as f:
                full_cfg = yaml.safe_load(f)
            self.config = full_cfg.get("sample_data", {})
        else:
            self.config = config

        self.n_students: int = self.config.get("n_students", 2400)
        self.signal_strength: float = self.config.get("signal_strength", 1.0)
        self.calibration_history: List[CalibrationRecord] = []

    def generate_cohort(
        self,
    ) -> Tuple[List[str], np.ndarray, np.ndarray]:
        """Generate student IDs, archetypes, and 4-period latent engagement levels."""
        n = self.n_students
        student_ids = [f"STU_{i + 1:04d}" for i in range(n)]

        arch_cfg = self.config.get("archetypes", {})
        probs_dict = arch_cfg.get("probabilities", {})
        arch_names = list(probs_dict.keys())
        arch_probs = [probs_dict[k] for k in arch_names]

        # Normalize probabilities in case of float rounding
        arch_probs = np.array(arch_probs) / np.sum(arch_probs)
        archetypes = self.rng.choice(arch_names, size=n, p=arch_probs)

        latent_e = np.zeros((n, 4), dtype=float)

        for i in range(n):
            arch = archetypes[i]
            if arch == "consistent_high":
                b = self.rng.beta(15.6, 4.4)
                noise = self.rng.normal(0, 0.04, size=4)
                latent_e[i] = np.clip(b + noise, 0.01, 0.99)
            elif arch == "consistent_moderate":
                b = self.rng.beta(11.0, 9.0)
                noise = self.rng.normal(0, 0.04, size=4)
                latent_e[i] = np.clip(b + noise, 0.01, 0.99)
            elif arch == "consistently_low":
                b = self.rng.beta(5.0, 15.0)
                noise = self.rng.normal(0, 0.04, size=4)
                latent_e[i] = np.clip(b + noise, 0.01, 0.99)
            elif arch == "declining":
                start = self.rng.uniform(0.70, 0.90)
                fall = self.rng.uniform(0.08, 0.20)
                onset = self.rng.choice([1, 2])
                vals = [start]
                for t in range(1, 4):
                    prev = vals[-1]
                    if t >= onset:
                        prev -= fall
                    vals.append(prev + self.rng.normal(0, 0.03))
                latent_e[i] = np.clip(vals, 0.01, 0.99)
            elif arch == "sporadic":
                base = self.rng.uniform(0.50, 0.70)
                shocks = self.rng.uniform(0.15, 0.30, size=4) * self.rng.choice(
                    [-1.0, 1.0], size=4
                )
                noise = self.rng.normal(0, 0.03, size=4)
                latent_e[i] = np.clip(base + shocks + noise, 0.01, 0.99)
            elif arch == "improving":
                start = self.rng.uniform(0.15, 0.35)
                rise = self.rng.uniform(0.08, 0.20)
                vals = [start]
                for t in range(1, 4):
                    vals.append(vals[-1] + rise + self.rng.normal(0, 0.03))
                latent_e[i] = np.clip(vals, 0.01, 0.99)

        return student_ids, archetypes, latent_e

    def generate_indicator_codes(
        self, latent_e: np.ndarray
    ) -> Tuple[np.ndarray, List[str]]:
        """Generate 28 continuous indicator scores and discretize into 1..5."""
        n = latent_e.shape[0]
        ind_cfg = self.config.get("indicator_generation", {})
        diligence_sigma = ind_cfg.get("diligence_sigma", 0.08)
        noise_sigma = ind_cfg.get("noise_sigma", 0.10)
        loadings = ind_cfg.get("loadings", {})
        cutpoints_rel = np.array(ind_cfg.get("cutpoints", [0.30, 0.46, 0.62, 0.78]))

        indicators = [
            "resource_access",
            "study_time",
            "review_time",
            "quiz_practice",
            "assignment_engagement",
            "study_consistency",
            "participation",
        ]

        u = self.rng.normal(0, diligence_sigma, size=n)
        codes = np.zeros((n, 28), dtype=float)
        col_names: List[str] = []

        for j, ind in enumerate(indicators):
            lam = loadings.get(ind, 0.80)
            tau = lam * cutpoints_rel
            for t in range(4):
                col_name = f"{ind}_p{t + 1}"
                col_names.append(col_name)
                col_idx = j * 4 + t
                nu = self.rng.normal(0, noise_sigma, size=n)
                y_star = lam * latent_e[:, t] + u + nu
                disc = np.digitize(y_star, tau) + 1
                codes[:, col_idx] = disc.astype(float)

        return codes, col_names

    def apply_response_styles(
        self,
        codes: np.ndarray,
        col_names: List[str],
        latent_e: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """Apply acquiescence, extreme/midpoint styles, straightlining, and screener."""
        n = codes.shape[0]
        styles_cfg = self.config.get("response_styles", {})
        acq_rate = styles_cfg.get("acquiescent_rate", 0.10)
        ext_rate = styles_cfg.get("extreme_rate", 0.05)
        mid_rate = styles_cfg.get("midpoint_rate", 0.05)
        straight_rate = styles_cfg.get("straight_line_rate", 0.02)
        attn_fail_rate = styles_cfg.get("attention_fail_rate", 0.02)

        out_codes = codes.copy()

        # 1. Acquiescence: shift agreement items up
        acq_mask = self.rng.random(n) < acq_rate
        consistency_indices = [
            i for i, name in enumerate(col_names) if "study_consistency" in name
        ]
        for c in consistency_indices:
            boost = (self.rng.random(n) < 0.5).astype(float)
            out_codes[acq_mask, c] = np.clip(
                out_codes[acq_mask, c] + boost[acq_mask], 1.0, 5.0
            )

        # 2. Extreme responding: polarize codes 2 -> 1, 4 -> 5
        ext_mask = self.rng.random(n) < ext_rate
        for c in range(28):
            c2_to_1 = (out_codes[:, c] == 2.0) & ext_mask & (self.rng.random(n) < 0.70)
            c4_to_5 = (out_codes[:, c] == 4.0) & ext_mask & (self.rng.random(n) < 0.70)
            out_codes[c2_to_1, c] = 1.0
            out_codes[c4_to_5, c] = 5.0

        # 3. Midpoint responding: collapse codes 2 or 4 -> 3
        mid_mask = self.rng.random(n) < mid_rate
        for c in range(28):
            c_mid = (
                np.isin(out_codes[:, c], [2.0, 4.0])
                & mid_mask
                & (self.rng.random(n) < 0.70)
            )
            out_codes[c_mid, c] = 3.0

        # 4. Straight-liners: identical answer across all items
        straight_mask = self.rng.random(n) < straight_rate
        for i in np.where(straight_mask)[0]:
            val = float(self.rng.choice([3, 4]))
            out_codes[i, :] = val

        # 5. Attention check item: code 3 is "Sometimes" (expected)
        attn_check = np.full(n, 3, dtype=int)
        attn_fail_mask = self.rng.random(n) < attn_fail_rate
        # For straight-liners, their attention check matches their straight line
        attn_check[straight_mask] = out_codes[straight_mask, 0].astype(int)
        # For explicit attention check failures, select invalid options (1, 2, 4, 5)
        for i in np.where(attn_fail_mask)[0]:
            attn_check[i] = int(self.rng.choice([1, 2, 4, 5]))

        metadata = {
            "acquiescent_mask": acq_mask,
            "extreme_mask": ext_mask,
            "midpoint_mask": mid_mask,
            "straight_mask": straight_mask,
            "attn_fail_mask": (attn_check != 3),
        }

        return out_codes, attn_check, metadata

    def apply_missingness(
        self, codes: np.ndarray, latent_e: np.ndarray
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply random item missingness (MCAR) and period-level dropout (MAR)."""
        n = codes.shape[0]
        miss_cfg = self.config.get("missingness", {})
        item_rate = miss_cfg.get("item_missing_rate", 0.02)
        period_rate = miss_cfg.get("period_missing_rate", 0.03)

        out = codes.copy()

        # 1. 2% random item missingness (MCAR)
        mcar_mask = self.rng.random((n, 28)) < item_rate
        out[mcar_mask] = np.nan

        # 2. 3% period missingness (MAR: higher dropout for low engagement)
        e_mean = latent_e.mean(axis=1)
        low_eng_factor = (1.0 - e_mean) / np.mean(1.0 - e_mean)
        mar_prob = np.clip(period_rate * low_eng_factor, 0.005, 0.10)
        mar_students = np.where(self.rng.random(n) < mar_prob)[0]

        period_miss_count = 0
        for idx in mar_students:
            miss_period = self.rng.choice([2, 3])  # Period 3 or Period 4
            for j in range(7):
                out[idx, j * 4 + miss_period] = np.nan
            period_miss_count += 1

        total_cells = n * 28
        missing_cells = int(np.isnan(out).sum())

        metadata = {
            "total_cells": total_cells,
            "missing_cells": missing_cells,
            "item_missing_pct": float(missing_cells / total_cells),
            "period_missing_students": period_miss_count,
            "period_missing_pct": float(period_miss_count / n),
        }

        return out, metadata

    def generate_outcomes(
        self,
        latent_e: np.ndarray,
        signal_strength: float,
        acq_mask: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Generate outcome items, continuous score, binary at_risk, and intercept."""
        n = latent_e.shape[0]
        out_cfg = self.config.get("outcome_model", {})
        w1 = out_cfg.get("weight_w1_low_mean", 2.20)
        w2 = out_cfg.get("weight_w2_drop", 1.80)
        w3 = out_cfg.get("weight_w3_volatility", 1.00)
        w4 = out_cfg.get("weight_w4_recent_low", 1.50)
        eps_sigma = out_cfg.get("epsilon_sigma", 0.35)

        e_mean = latent_e.mean(axis=1)
        e_drop = np.maximum(0, latent_e[:, 0] - latent_e[:, 3])
        e_std = latent_e.std(axis=1)
        e_p4_inv = 1.0 - latent_e[:, 3]

        lin_comb = w1 * (1.0 - e_mean) + w2 * e_drop + w3 * e_std + w4 * e_p4_inv
        noise_eps = self.rng.normal(0, eps_sigma, size=n)
        uncentered_score = signal_strength * lin_comb + noise_eps

        # Calibrate w0 so that top ~28% exceed threshold (logit 0.405)
        # Logit(0.60) = 0.405465
        w0 = 0.405465 - float(np.percentile(uncentered_score, 72.0))
        r = expit(w0 + uncentered_score)

        A = np.zeros((n, 3), dtype=int)
        for k in range(3):
            item_noise = self.rng.normal(0, 0.35, size=n)
            a_val = np.round(1.0 + 4.0 * r + item_noise)
            # Apply acquiescent response style
            boost = (self.rng.random(n) < 0.5).astype(float)
            a_val[acq_mask] += boost[acq_mask]
            A[:, k] = np.clip(a_val, 1, 5).astype(int)

        score = ((A.mean(axis=1) - 1.0) / 4.0) * 100.0
        at_risk = (score >= 60.0).astype(int)

        return A, score, at_risk, w0

    def evaluate_cv(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, float, float]:
        """Fit 5-fold CV to evaluate LogisticRegression and LightGBM AUC."""
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.seed)
        lr_oof = np.zeros(len(y), dtype=float)
        lgb_oof = np.zeros(len(y), dtype=float)

        for train_idx, val_idx in cv.split(X, y):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_va = X[val_idx]

            imputer = SimpleImputer(strategy="median")
            X_tr_imp = imputer.fit_transform(X_tr)
            X_va_imp = imputer.transform(X_va)

            # Logistic Regression
            lr = LogisticRegression(max_iter=500, random_state=self.seed)
            lr.fit(X_tr_imp, y_tr)
            lr_oof[val_idx] = lr.predict_proba(X_va_imp)[:, 1]

            # LightGBM Classifier
            gbm = lgb.LGBMClassifier(
                n_estimators=100,
                learning_rate=0.05,
                max_depth=4,
                verbose=-1,
                random_state=self.seed,
            )
            gbm.fit(X_tr_imp, y_tr)
            lgb_oof[val_idx] = gbm.predict_proba(X_va_imp)[:, 1]

        lr_auc = float(roc_auc_score(y, lr_oof))
        lgb_auc = float(roc_auc_score(y, lgb_oof))
        gap = float(lgb_auc - lr_auc)

        return lr_auc, lgb_auc, gap

    def calibrate(self) -> float:
        """Calibrate signal_strength until LightGBM AUC in [0.86, 0.92]."""
        cal_cfg = self.config.get("calibration", {})
        max_iter = cal_cfg.get("max_iterations", 8)
        target_min = cal_cfg.get("target_auc_min", 0.86)
        target_max = cal_cfg.get("target_auc_max", 0.92)

        sig = self.signal_strength
        self.calibration_history.clear()

        # Generate base synthetic indicators once
        _, _, latent_e = self.generate_cohort()
        codes, col_names = self.generate_indicator_codes(latent_e)
        styled_codes, _, styles_meta = self.apply_response_styles(
            codes, col_names, latent_e
        )
        messy_codes, _ = self.apply_missingness(styled_codes, latent_e)
        acq_mask = styles_meta["acquiescent_mask"]

        for it in range(1, max_iter + 1):
            _, _, at_risk, w0 = self.generate_outcomes(latent_e, sig, acq_mask)
            prev = float(np.mean(at_risk))

            lr_auc, lgb_auc, gap = self.evaluate_cv(messy_codes, at_risk)
            converged = target_min <= lgb_auc <= target_max

            record = CalibrationRecord(
                iteration=it,
                signal_strength=round(sig, 4),
                intercept_w0=round(w0, 4),
                prevalence=round(prev, 4),
                lr_auc=round(lr_auc, 4),
                lgb_auc=round(lgb_auc, 4),
                auc_gap=round(gap, 4),
                converged=converged,
            )
            self.calibration_history.append(record)

            logger.info(
                "Calibration %d: sig=%.3f, p=%.3f, LR=%.4f, LGB=%.4f, gap=%.4f",
                it,
                sig,
                prev,
                lr_auc,
                lgb_auc,
                gap,
            )

            if converged:
                break

            # Adjust signal strength
            if lgb_auc < target_min:
                sig *= 1.15
            else:
                sig *= 0.88

        self.signal_strength = sig
        return sig

    def generate_all(
        self, calibrate: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, RealityCheckStats]:
        """Run cohort generation, mess injection, and return datasets."""
        if calibrate:
            self.calibrate()

        # 1. Base cohort
        student_ids, archetypes, latent_e = self.generate_cohort()
        codes, col_names = self.generate_indicator_codes(latent_e)
        styled_codes, attn_codes, styles_meta = self.apply_response_styles(
            codes, col_names, latent_e
        )
        messy_codes, miss_meta = self.apply_missingness(styled_codes, latent_e)

        # 2. Outcomes
        A_items, scores, at_risk, w0 = self.generate_outcomes(
            latent_e, self.signal_strength, styles_meta["acquiescent_mask"]
        )

        # 3. Hidden ground truth dataframe (STRICTLY ISOLATED FROM FEATURES)
        hidden_df = pd.DataFrame(
            {
                "student_id": student_ids,
                "archetype": archetypes,
                "e_1": np.round(latent_e[:, 0], 4),
                "e_2": np.round(latent_e[:, 1], 4),
                "e_3": np.round(latent_e[:, 2], 4),
                "e_4": np.round(latent_e[:, 3], 4),
                "latent_risk": np.round(
                    expit(
                        w0
                        + self.signal_strength
                        * (
                            2.20 * (1.0 - latent_e.mean(axis=1))
                            + 1.80 * np.maximum(0, latent_e[:, 0] - latent_e[:, 3])
                            + 1.00 * latent_e.std(axis=1)
                            + 1.50 * (1.0 - latent_e[:, 3])
                        )
                    ),
                    4,
                ),
                "attrition_intent_score": np.round(scores, 2),
                "at_risk": at_risk,
            }
        )

        # 4. Construct raw survey export dataframe
        n = self.n_students
        mess_cfg = self.config.get("mess", {})
        no_consent_rate = mess_cfg.get("no_consent_rate", 0.01)
        typo_rate = mess_cfg.get("typo_rate", 0.003)
        duplicate_rate = mess_cfg.get("duplicate_rate", 0.005)

        # Base timestamps (spread over 14 days)
        base_ts = pd.Timestamp("2026-09-01 09:00:00")
        timestamps = [
            (
                base_ts + pd.Timedelta(seconds=int(self.rng.uniform(0, 14 * 86400)))
            ).strftime("%Y-%m-%d %H:%M:%S")
            for _ in range(n)
        ]

        # Consent item (1% no consent)
        consent_labels = []
        no_consent_mask = self.rng.random(n) < no_consent_rate
        for i in range(n):
            if no_consent_mask[i]:
                consent_labels.append("I do not consent")
            else:
                consent_labels.append("I consent to participate")

        # Map canonical columns to long Google Forms headers from mapping.yaml
        long_headers: Dict[str, str] = {}
        for col_name in self.mapping.columns:
            variants = self.mapping.columns[col_name]
            # Select the Google Forms bracketed question variant (index 5 or longest)
            bracketed = [v for v in variants if "[" in v and "]" in v]
            if bracketed:
                long_headers[col_name] = bracketed[0]
            else:
                long_headers[col_name] = (
                    variants[1] if len(variants) > 1 else variants[0]
                )

        # Decode numeric codes to text labels
        survey_dict: Dict[str, List[Any]] = {
            long_headers["timestamp"]: timestamps,
            long_headers["student_id"]: student_ids,
            long_headers["consent"]: consent_labels,
        }

        # 28 indicator columns with decoded text
        typo_count = 0
        for idx, col_name in enumerate(col_names):
            hdr = long_headers[col_name]
            col_labels: List[Any] = []
            for i in range(n):
                val = messy_codes[i, idx]
                if np.isnan(val):
                    col_labels.append(None)
                else:
                    text_label = self.codec.decode(col_name, int(val))
                    # Inject realistic typos / formatting mess
                    if self.rng.random() < typo_rate:
                        typo_count += 1
                        if text_label == "Always":
                            text_label = "Alwys"
                        elif text_label == "Sometimes":
                            text_label = "Some times"
                        elif text_label and "–" in text_label:
                            text_label = text_label.replace("–", "-")
                        elif text_label:
                            text_label = f"  {text_label}  "
                    col_labels.append(text_label)
            survey_dict[hdr] = col_labels

        # Attention check
        attn_hdr = long_headers["attention_check"]
        attn_text_list: List[str] = []
        for i in range(n):
            code = attn_codes[i]
            label = self.codec.decode("attention_check", code)
            attn_text_list.append(label or "Sometimes")
        survey_dict[attn_hdr] = attn_text_list

        # Outcome items A1..A3
        outcome_cols = ["intent_leave", "intent_defer", "intent_not_continue"]
        for k, o_col in enumerate(outcome_cols):
            o_hdr = long_headers[o_col]
            o_labels: List[str] = []
            for i in range(n):
                code = A_items[i, k]
                label = self.codec.decode(o_col, code)
                o_labels.append(label or "Neutral")
            survey_dict[o_hdr] = o_labels

        raw_df = pd.DataFrame(survey_dict)

        # 5. Inject duplicate submissions (0.5% = 12 duplicates)
        num_dups = int(round(duplicate_rate * n))
        dup_indices = self.rng.choice(n, size=num_dups, replace=False)
        dup_rows = []
        for d_idx in dup_indices:
            row = raw_df.iloc[d_idx].copy()
            orig_ts = pd.Timestamp(row[long_headers["timestamp"]])
            new_ts = orig_ts + pd.Timedelta(hours=int(self.rng.uniform(2, 72)))
            row[long_headers["timestamp"]] = new_ts.strftime("%Y-%m-%d %H:%M:%S")
            dup_rows.append(row)

        if dup_rows:
            raw_df = pd.concat([raw_df, pd.DataFrame(dup_rows)], ignore_index=True)

        # Reality check statistics
        stats = RealityCheckStats(
            metrics={
                "prevalence": {
                    "configured": 0.28,
                    "empirical": float(np.mean(at_risk)),
                },
                "item_missing_rate": {
                    "configured": 0.02,
                    "empirical": miss_meta["item_missing_pct"],
                },
                "period_missing_rate": {
                    "configured": 0.03,
                    "empirical": miss_meta["period_missing_pct"],
                },
                "acquiescent_rate": {
                    "configured": 0.10,
                    "empirical": float(np.mean(styles_meta["acquiescent_mask"])),
                },
                "extreme_rate": {
                    "configured": 0.05,
                    "empirical": float(np.mean(styles_meta["extreme_mask"])),
                },
                "midpoint_rate": {
                    "configured": 0.05,
                    "empirical": float(np.mean(styles_meta["midpoint_mask"])),
                },
                "straight_line_rate": {
                    "configured": 0.02,
                    "empirical": float(np.mean(styles_meta["straight_mask"])),
                },
                "attention_fail_rate": {
                    "configured": 0.02,
                    "empirical": float(np.mean(styles_meta["attn_fail_mask"])),
                },
                "no_consent_rate": {
                    "configured": 0.01,
                    "empirical": float(np.mean(no_consent_mask)),
                },
                "duplicate_rate": {
                    "configured": 0.005,
                    "empirical": float(num_dups / n),
                },
                "typo_rate": {
                    "configured": 0.003,
                    "empirical": float(typo_count / (n * 28)),
                },
            }
        )

        return raw_df, hidden_df, stats

    def create_evaluation_samples(
        self, raw_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Generate sample upload files (clean.xlsx and messy.csv)."""
        # 1. Clean sample: 200 students, canonical headers, NO outcome columns
        n_sample = 200
        clean_df = pd.DataFrame()
        clean_df["student_id"] = [f"TEST_STU_{i + 1:04d}" for i in range(n_sample)]

        # Generate clean indicators without missingness
        _, _, sample_e = self.generate_cohort()
        sample_e = sample_e[:n_sample]
        sample_codes, col_names = self.generate_indicator_codes(sample_e)

        for idx, col_name in enumerate(col_names):
            clean_df[col_name] = sample_codes[:, idx].astype(int)

        # Strict assertion: NO outcome columns in clean sample
        outcome_items = ["intent_leave", "intent_defer", "intent_not_continue"]
        for o in outcome_items:
            assert o not in clean_df.columns, f"Target leakage in clean sample: {o}"

        # 2. Messy sample: 200 students, renamed headers, blanks, and extra col
        messy_df = pd.DataFrame()
        messy_df["Student_ID"] = [f"MESSY_STU_{i + 1:04d}" for i in range(n_sample)]

        for idx, col_name in enumerate(col_names):
            # Pick a variant header from mapping.yaml
            variants = self.mapping.columns[col_name]
            variant_hdr = variants[min(2, len(variants) - 1)]

            vals = sample_codes[:, idx].copy().astype(object)
            # Inject 5% blanks
            blank_mask = self.rng.random(n_sample) < 0.05
            vals[blank_mask] = None

            # Inject out-of-range values in first 3 rows
            if idx == 0:
                vals[0] = 99
            elif idx == 1:
                vals[1] = "Unknown"
            elif idx == 2:
                vals[2] = -1

            messy_df[variant_hdr] = vals

        # Inject extra unrecognized column
        messy_df["campus_branch"] = self.rng.choice(
            ["North Campus", "City Center", "West Annex"], size=n_sample
        )

        # Strict assertion: NO outcome columns in messy sample
        for o in outcome_items:
            assert o not in messy_df.columns, f"Target leakage in messy sample: {o}"

        return clean_df, messy_df

    def save_all_artifacts(
        self,
        raw_df: pd.DataFrame,
        hidden_df: pd.DataFrame,
        clean_df: pd.DataFrame,
        messy_df: pd.DataFrame,
    ) -> Dict[str, Path]:
        """Save raw data, hidden ground truth, and upload samples to disk."""
        raw_dir = DATA_DIR / "raw"
        samples_dir = DATA_DIR / "samples"
        raw_dir.mkdir(parents=True, exist_ok=True)
        samples_dir.mkdir(parents=True, exist_ok=True)

        # 1. Raw Google Forms export (.csv and .xlsx)
        raw_csv_path = raw_dir / "survey_raw_sample.csv"
        raw_xlsx_path = raw_dir / "survey_raw_sample.xlsx"
        raw_df.to_csv(raw_csv_path, index=False, encoding="utf-8")
        raw_df.to_excel(raw_xlsx_path, index=False, engine="openpyxl")

        # 2. Hidden ground truth (.csv)
        hidden_path = raw_dir / "ground_truth_hidden.csv"
        hidden_df.to_csv(hidden_path, index=False, encoding="utf-8")

        # 3. Clean and messy upload samples
        clean_path = samples_dir / "sample_upload_clean.xlsx"
        clean_df.to_excel(clean_path, index=False, engine="openpyxl")

        messy_path = samples_dir / "sample_upload_messy.csv"
        messy_df.to_csv(messy_path, index=False, encoding="utf-8")

        return {
            "raw_csv": raw_csv_path,
            "raw_xlsx": raw_xlsx_path,
            "ground_truth": hidden_path,
            "clean_sample": clean_path,
            "messy_sample": messy_path,
        }
