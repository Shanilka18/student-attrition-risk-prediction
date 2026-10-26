"""Test script to evaluate noise calibration in SCM generator."""

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from lightgbm import LGBMClassifier

from src.data_gen.generate import SocioeconomicDGP
from src.preprocessing.pipeline import (
    validate_feature_boundary,
    get_scaled_pipeline,
    get_tree_pipeline,
    ALL_ALLOWED_FEATURES,
)

def test_noise_calibration():
    # Load pristine df from existing generator to get the raw features
    dgp = SocioeconomicDGP(seed=42)
    
    # We inspect the exact arrays inside SocioeconomicDGP
    # Let's run with modified parameters
    for noise_std in [0.40, 0.45, 0.50]:
        for beta_0 in [-2.75, -2.80, -2.85, -2.90]:
            # Temporarily compute logit
            # Let's run a test by overriding generate
            rng = np.random.default_rng(42)
            
            # Recreate generation with specific noise_std and beta_0
            # 1. Demographic features
            n = 2000
            participant_ids = [f"P-{i+1:05d}" for i in range(n)]
            gender = rng.choice(["Female", "Male", "Non-binary"], size=n, p=[0.49, 0.49, 0.02])
            faculty = rng.choice(["Computing", "Engineering", "Business", "Humanities"], size=n, p=[0.38, 0.26, 0.22, 0.14])
            year_of_study = rng.integers(low=1, high=5, size=n)
            district_type = rng.choice(["urban", "semi-urban", "rural"], size=n, p=[0.35, 0.40, 0.25])
            
            household_income_bracket = np.zeros(n, dtype=int)
            for i in range(n):
                dt = district_type[i]
                probs = [0.35, 0.30, 0.20, 0.10, 0.04, 0.01] if dt == "rural" else ([0.15, 0.25, 0.30, 0.18, 0.09, 0.03] if dt == "semi-urban" else [0.08, 0.15, 0.25, 0.27, 0.17, 0.08])
                household_income_bracket[i] = rng.choice(np.arange(1, 7), p=probs)
                
            dependents_count = np.zeros(n, dtype=int)
            for i in range(n):
                dt = district_type[i]
                lam = 2.0 if dt == "rural" else (1.5 if dt == "semi-urban" else 1.1)
                dependents_count[i] = int(np.clip(rng.poisson(lam), 0, 6))
                
            financial_stress = np.zeros(n, dtype=int)
            for i in range(n):
                ls = 4.2 - 0.58 * household_income_bracket[i] + 0.28 * dependents_count[i] + rng.normal(0, 0.75)
                financial_stress[i] = 1 if ls < 1.7 else (2 if ls < 2.5 else (3 if ls < 3.3 else (4 if ls < 4.1 else 5)))
                
            scholarship_bursary = np.zeros(n, dtype=int)
            for i in range(n):
                inc = household_income_bracket[i]
                p_sch = 0.45 if inc == 1 else (0.28 if inc == 2 else (0.12 if inc == 3 else (0.04 if inc == 4 else 0.01)))
                scholarship_bursary[i] = int(rng.random() < p_sch)
                
            housing_type = []
            for i in range(n):
                dt = district_type[i]
                probs = [0.20, 0.55, 0.15, 0.10] if dt == "rural" else ([0.45, 0.35, 0.10, 0.10] if dt == "semi-urban" else [0.85, 0.05, 0.03, 0.07])
                housing_type.append(rng.choice(["home", "boarding", "hostel", "relatives"], p=probs))
            housing_type = np.array(housing_type)
            
            commute_time_min = np.zeros(n, dtype=float)
            commute_cost_lkr_week = np.zeros(n, dtype=float)
            for i in range(n):
                ht = housing_type[i]
                dt = district_type[i]
                mu_time = 15.0 if ht in ["boarding", "hostel"] else (75.0 if dt == "rural" else (50.0 if dt == "semi-urban" else 35.0))
                ct = rng.normal(mu_time, 18.0)
                commute_time_min[i] = round(float(np.clip(ct, 5.0, 240.0)), 1)
                cost_rate = 22.0 if ht == "home" else 10.0
                cost = commute_time_min[i] * cost_rate * 5.0 * rng.uniform(0.85, 1.25)
                commute_cost_lkr_week[i] = round(float(np.clip(cost, 0.0, 15000.0)), 2)
                
            internet_reliability = np.zeros(n, dtype=int)
            internet_affordability = np.zeros(n, dtype=int)
            device_quality = np.zeros(n, dtype=int)
            device_type = []
            quiet_study_space = np.zeros(n, dtype=int)
            
            for i in range(n):
                dt = district_type[i]
                inc = household_income_bracket[i]
                mu_rel = 2.4 if dt == "rural" else (3.2 if dt == "semi-urban" else 4.1)
                internet_reliability[i] = int(np.clip(round(rng.normal(mu_rel, 0.85)), 1, 5))
                mu_aff = 1.6 + 0.55 * inc
                internet_affordability[i] = int(np.clip(round(rng.normal(mu_aff, 0.85)), 1, 5))
                p_dev = [0.25, 0.40, 0.32, 0.03] if inc <= 2 else ([0.60, 0.28, 0.11, 0.01] if inc <= 4 else [0.90, 0.08, 0.02, 0.00])
                dev_choice = rng.choice(["own laptop", "shared laptop", "phone only", "none"], p=p_dev)
                device_type.append(dev_choice)
                q_mu = (2.5 + 0.45 * inc) if dev_choice == "own laptop" else (2.4 if dev_choice == "shared laptop" else (1.8 if dev_choice == "phone only" else 1.0))
                device_quality[i] = int(np.clip(round(rng.normal(q_mu, 0.8)), 1, 5))
                dep = dependents_count[i]
                p_quiet = 0.80 - (0.10 * dep) - (0.15 if housing_type[i] == "boarding" else 0.0)
                quiet_study_space[i] = int(rng.random() < np.clip(p_quiet, 0.15, 0.90))
                
            device_type = np.array(device_type)
            part_time_employment = np.zeros(n, dtype=int)
            work_hours_per_week = np.zeros(n, dtype=float)
            for i in range(n):
                inc = household_income_bracket[i]
                stress = financial_stress[i]
                p_w = float(np.clip(0.12 + 0.08 * (6 - inc) + 0.06 * (stress - 1), 0.05, 0.65))
                is_emp = int(rng.random() < p_w)
                part_time_employment[i] = is_emp
                work_hours_per_week[i] = round(float(np.clip(rng.normal(18.0 + 2.0 * stress, 6.0), 5.0, 40.0)), 1) if is_emp else 0.0
                
            gpa_choices = ["Band 1 (< 2.00)", "Band 2 (2.00 - 2.49)", "Band 3 (2.50 - 2.99)", "Band 4 (3.00 - 3.49)", "Band 5 (3.50 - 4.00)"]
            baseline_gpa_band = rng.choice(gpa_choices, size=n, p=[0.08, 0.18, 0.38, 0.26, 0.10])
            baseline_engagement_score = np.zeros(n, dtype=float)
            for i in range(n):
                g_idx = gpa_choices.index(baseline_gpa_band[i])
                baseline_engagement_score[i] = round(float(np.clip(rng.normal(52.0 + g_idx * 6.5, 14.0), 8.0, 99.0)), 1)
                
            # Derived exposures
            financial_hardship = ((household_income_bracket <= 2) & (financial_stress >= 4)).astype(int)
            long_commute = (commute_time_min > 120.0).astype(int)
            inadequate_digital_access = ((internet_reliability <= 2) | np.isin(device_type, ["phone only", "none"])).astype(int)
            heavy_part_time_work = (work_hours_per_week > 15.0).astype(int)
            
            # Mediators
            later_gpa_decline = np.zeros(n, dtype=int)
            later_engagement_decline = np.zeros(n, dtype=int)
            for i in range(n):
                p_gpa_drop = 0.12 + 0.18 * financial_hardship[i] + 0.12 * long_commute[i] + 0.14 * inadequate_digital_access[i] + 0.15 * heavy_part_time_work[i] - 0.05 * (baseline_engagement_score[i] / 100.0)
                later_gpa_decline[i] = int(rng.random() < np.clip(p_gpa_drop, 0.05, 0.85))
                p_eng_drop = 0.14 + 0.14 * long_commute[i] + 0.20 * inadequate_digital_access[i] + 0.18 * heavy_part_time_work[i] + 0.10 * (1 - quiet_study_space[i])
                later_engagement_decline[i] = int(rng.random() < np.clip(p_eng_drop, 0.05, 0.85))
                
            # Enhanced signal on core environmental determinants
            alpha_inc = -0.28 * (household_income_bracket - 3.5)
            alpha_gpa = np.array([0.85 if b == "Band 1 (< 2.00)" else (0.45 if b == "Band 2 (2.00 - 2.49)" else (0.0 if b == "Band 3 (2.50 - 2.99)" else (-0.40 if b == "Band 4 (3.00 - 3.49)" else -0.75))) for b in baseline_gpa_band])
            alpha_eng = -0.018 * (baseline_engagement_score - 60.0)
            alpha_year = 0.08 * (year_of_study - 2.5)
            senior_rural = ((year_of_study >= 3) & (district_type == "rural")).astype(float)
            
            # Synergistic non-linear compounding interactions
            synergy_hardship = 0.45 * (financial_hardship * long_commute)
            synergy_digital = 0.40 * (inadequate_digital_access * (financial_stress >= 4).astype(int))
            synergy_work_gpa = 0.35 * (heavy_part_time_work * (baseline_gpa_band == "Band 1 (< 2.00)").astype(int))
            
            noise = rng.normal(0, noise_std, size=n)
            
            latent_logit = (
                beta_0
                + alpha_inc
                + alpha_gpa
                + alpha_eng
                + alpha_year
                + 0.85 * financial_hardship
                + 0.45 * (financial_hardship * senior_rural)
                + 0.55 * long_commute
                + 0.65 * inadequate_digital_access
                + 0.48 * heavy_part_time_work
                + 0.52 * later_gpa_decline
                + 0.44 * later_engagement_decline
                + synergy_hardship
                + synergy_digital
                + synergy_work_gpa
                + noise
            )
            
            true_prob = expit(latent_logit)
            attrition = (rng.random(size=n) < true_prob).astype(int)
            prev = float(np.mean(attrition))
            
            # Bayes ceiling
            bayes_auc = roc_auc_score(attrition, true_prob)
            bayes_pr = average_precision_score(attrition, true_prob)
            
            # Model performance on allowed features
            # Let's assemble quick df
            df_test = pd.DataFrame({
                "household_income_bracket": household_income_bracket,
                "financial_stress": financial_stress,
                "scholarship_bursary": scholarship_bursary,
                "dependents_count": dependents_count,
                "housing_type": housing_type,
                "commute_time_min": commute_time_min,
                "commute_cost_lkr_week": commute_cost_lkr_week,
                "internet_reliability": internet_reliability,
                "internet_affordability": internet_affordability,
                "device_type": device_type,
                "device_quality": device_quality,
                "quiet_study_space": quiet_study_space,
                "part_time_employment": part_time_employment,
                "work_hours_per_week": work_hours_per_week,
                "baseline_gpa_band": baseline_gpa_band,
                "baseline_engagement_score": baseline_engagement_score,
                "district_type": district_type,
                "year_of_study": year_of_study,
                "faculty": faculty,
                "attrition": attrition,
            })
            
            # 5-fold CV
            X_encoded = pd.get_dummies(df_test.drop(columns=["attrition"]), drop_first=True)
            y = df_test["attrition"].values
            
            skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            cv_lgb = []
            for tr_i, te_i in skf.split(X_encoded, y):
                clf = LGBMClassifier(n_estimators=60, learning_rate=0.08, max_depth=4, random_state=42, verbose=-1)
                clf.fit(X_encoded.iloc[tr_i], y[tr_i])
                p = clf.predict_proba(X_encoded.iloc[te_i])[:, 1]
                cv_lgb.append(roc_auc_score(y[te_i], p))
                
            mean_lgb_auc = np.mean(cv_lgb)
            
            if True:
                print(f"Noise={noise_std:.2f} | Beta0={beta_0:.2f} -> Prev={prev:.4f} | Bayes AUC={bayes_auc:.4f} | LightGBM 3-Fold AUC={mean_lgb_auc:.4f}")

if __name__ == "__main__":
    test_noise_calibration()
