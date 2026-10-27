"""
Latent Process Synthetic Academic Cohort Generator.
Models longitudinal student trajectories, academic signals, realistic noise,
and attrition dynamics based on Sri Lankan higher education contexts (SLIIT/SLIATE benchmarks).

Component: Academic Intelligence Framework (IT23164680)
"""

from typing import Dict, List, Optional, Tuple, Any
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import expit

logger = logging.getLogger("AcademicIntelligence.DataGenerator")

TRAJECTORY_TYPES = [
    "stable-high",
    "stable-average",
    "slow-decline",
    "sudden-drop",
    "recovering",
    "volatile",
    "struggling-from-start"
]

TRAJECTORY_PROBABILITIES = [0.24, 0.28, 0.14, 0.09, 0.08, 0.08, 0.09]


class LatentStudentGenerator:
    """
    Simulates longitudinal student academic cohorts via latent cognitive, behavioral,
    and resilience traits, subject to semester-to-semester autocorrelation and stress shocks.
    """

    def __init__(
        self,
        seed: int = 42,
        preset: str = "realistic",
        target_attrition_rate: Optional[float] = None
    ):
        self.seed = seed
        self.preset = preset.lower()
        self.rng = np.random.default_rng(seed)

        # Set preset parameters
        if self.preset == "easy":
            self.noise_std = 0.10
            self.overlap_factor = 0.5
            self.base_hazard_offset = -4.05
            self.default_attrition = 0.18
        elif self.preset == "hard":
            self.noise_std = 0.28
            self.overlap_factor = 1.4
            self.base_hazard_offset = -3.40
            self.default_attrition = 0.24
        elif self.preset in ("high-attrition", "high_attrition"):
            self.noise_std = 0.20
            self.overlap_factor = 1.0
            self.base_hazard_offset = -2.60
            self.default_attrition = 0.40
        else:  # 'realistic' default (15-25% target cumulative attrition)
            self.noise_std = 0.18
            self.overlap_factor = 1.0
            self.base_hazard_offset = -3.70
            self.default_attrition = 0.20

        self.target_attrition_rate = target_attrition_rate or self.default_attrition

    def generate_cohort(
        self,
        n_students: int = 5000,
        max_semesters: int = 8,
        missing_rate: float = 0.03
    ) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Generates longitudinal panel data and outcome labels.
        Returns:
            academic_long_df: Observable features only (NO labels, NO hidden traits).
            outcomes_df: Outcome labels and attrition timing per student.
            audit_metadata: Ground truth latent distributions for validation & auditing.
        """
        logger.info(
            f"Generating synthetic cohort: n={n_students}, preset={self.preset}, "
            f"target_attrition={self.target_attrition_rate:.1%}, seed={self.seed}"
        )

        # 1. Sample hidden latent traits
        # Ability (A), Conscientiousness (C), Resilience (R)
        ability = self.rng.normal(0.0, 1.0, size=n_students)
        conscientiousness = self.rng.normal(0.0, 1.0, size=n_students)
        resilience = self.rng.normal(0.0, 1.0, size=n_students)

        # Assign hidden trajectory types
        traj_indices = self.rng.choice(
            len(TRAJECTORY_TYPES),
            size=n_students,
            p=TRAJECTORY_PROBABILITIES
        )
        student_trajs = [TRAJECTORY_TYPES[i] for i in traj_indices]

        records: List[Dict[str, Any]] = []
        outcomes: List[Dict[str, Any]] = []
        audit_records: List[Dict[str, Any]] = []

        # Autoregressive memory parameter (semester-to-semester momentum)
        ar_alpha = 0.65
        cohort_years = [2021, 2022, 2023, 2024]

        for i in range(n_students):
            sid = f"STU_{i+1:05d}"
            ttype = student_trajs[i]
            a_val = ability[i]
            c_val = conscientiousness[i]
            r_val = resilience[i]
            cohort_yr = int(self.rng.choice(cohort_years))

            # Determine sudden drop semester if applicable
            drop_sem = self.rng.integers(2, 5) if ttype == "sudden-drop" else -1

            # Base latent state
            if ttype == "stable-high":
                base_mu = 1.1 + 0.3 * a_val
            elif ttype == "stable-average":
                base_mu = 0.1 + 0.3 * a_val
            elif ttype == "struggling-from-start":
                base_mu = -1.2 + 0.2 * a_val
            elif ttype == "slow-decline":
                base_mu = 0.7 + 0.3 * a_val
            elif ttype == "recovering":
                base_mu = -0.8 + 0.3 * a_val
            elif ttype == "sudden-drop":
                base_mu = 0.9 + 0.3 * a_val
            else:  # volatile
                base_mu = 0.0 + 0.4 * a_val

            current_latent = base_mu
            attrited = False
            attrition_sem = None
            last_sem = 1
            prev_gpa = 3.0

            for s in range(1, max_semesters + 1):
                last_sem = s

                # Trajectory-specific temporal drift
                drift = 0.0
                if ttype == "slow-decline":
                    drift = -0.22 * (s - 1)
                elif ttype == "sudden-drop" and s >= drop_sem:
                    drift = -1.45
                elif ttype == "recovering" and s >= 3:
                    drift = +0.65
                elif ttype == "volatile":
                    drift = 0.55 * np.sin(s * 1.6)

                # Semester difficulty load effect (Sem 3 & 4 have harder technical core)
                load_effect = -0.08 if s in (3, 4) else 0.0

                # Stress shock (financial / psychosocial event)
                shock = 0.0
                if s > 1 and self.rng.random() < 0.08:
                    raw_shock = self.rng.exponential(0.7)
                    shock = raw_shock * (1.0 - 0.45 * expit(r_val))

                # Update autoregressive latent state
                target_state = base_mu + drift + load_effect + 0.25 * c_val - shock
                noise = self.rng.normal(0.0, self.noise_std)
                current_latent = ar_alpha * current_latent + (1.0 - ar_alpha) * target_state + noise

                # Map latent state to observables
                # 1. GPA: 0.00 to 4.00
                gpa_val = 2.80 + 0.55 * current_latent + self.rng.normal(0.0, 0.08 * self.overlap_factor)
                gpa_val = float(np.clip(gpa_val, 0.0, 4.0))

                # 2. Exam Marks %: 0.0 to 100.0, correlated with GPA + Ability
                exam_val = 22.0 + 19.5 * gpa_val + 3.0 * a_val + self.rng.normal(0.0, 3.5 * self.overlap_factor)
                exam_val = float(np.clip(exam_val, 0.0, 100.0))

                # 3. Attendance %: 0.0 to 100.0, strongly influenced by Conscientiousness & Latent State
                att_val = 62.0 + 14.0 * c_val + 7.5 * current_latent + self.rng.normal(0.0, 5.0 * self.overlap_factor)
                att_val = float(np.clip(att_val, 0.0, 100.0))

                # 4. Submission Delay in days: >= 0.0, exponential-like
                delay_mu = np.exp(1.1 - 0.45 * c_val - 0.40 * current_latent + self.rng.normal(0.0, 0.25)) - 0.7
                delay_val = float(np.clip(delay_mu, 0.0, 30.0))

                # 5. Late Assignments Count (0 to 8)
                late_cnt = int(np.clip(self.rng.poisson(0.25 + 0.45 * delay_val), 0, 8))

                # Compute dynamic attrition hazard for this semester
                # Factors: low GPA, negative velocity (GPA drop), attendance deficit, high delay, buffered by resilience
                gpa_drop = max(0.0, prev_gpa - gpa_val) if s > 1 else 0.0
                hazard_logit = (
                    self.base_hazard_offset
                    + 1.35 * (2.40 - gpa_val)
                    + 2.10 * gpa_drop
                    + 0.045 * (65.0 - att_val)
                    + 0.18 * (delay_val - 2.5)
                    - 0.65 * r_val
                )

                # Adjust offset for user-configured target attrition rate
                hazard_logit += (self.target_attrition_rate - 0.20) * 3.5
                hazard_prob = float(expit(hazard_logit))

                # Non-separability clause:
                # Some low-GPA students with high resilience persist (e.g. academic retake policy)
                if gpa_val < 2.0 and r_val > 0.8:
                    hazard_prob *= 0.25

                # Some students with GPA > 2.0 suffering steep drops or chronic disengagement drop out
                if gpa_drop > 0.7 and att_val < 55.0:
                    hazard_prob = max(hazard_prob, 0.45)

                records.append({
                    "student_id": sid,
                    "cohort_year": cohort_yr,
                    "semester": s,
                    "gpa": round(gpa_val, 2),
                    "exam_marks": round(exam_val, 1),
                    "attendance_pct": round(att_val, 1),
                    "assignment_delay_days": round(delay_val, 1),
                    "late_assignments_count": late_cnt
                })

                audit_records.append({
                    "student_id": sid,
                    "cohort_year": cohort_yr,
                    "semester": s,
                    "trajectory_type": ttype,
                    "latent_state": current_latent,
                    "hazard_prob": hazard_prob
                })

                prev_gpa = gpa_val

                # Evaluate probabilistic attrition event
                if self.rng.random() < hazard_prob and s < max_semesters:
                    attrited = True
                    attrition_sem = s
                    break  # Student stops producing semesters!

            outcomes.append({
                "student_id": sid,
                "cohort_year": cohort_yr,
                "attrited": 1 if attrited else 0,
                "last_semester_observed": last_sem,
                "attrition_semester": attrition_sem
            })

        df_long = pd.DataFrame(records)
        df_outcomes = pd.DataFrame(outcomes)
        df_audit = pd.DataFrame(audit_records)

        # Inject realistic Missing Values (MCAR & MAR)
        if missing_rate > 0.0:
            df_long = self._inject_missingness(df_long, missing_rate)

        # Calculate actual cohort statistics
        actual_attrition_rate = df_outcomes["attrited"].mean()
        logger.info(
            f"Cohort generation complete: {n_students} students, "
            f"{len(df_long)} student-semesters. Attrition rate: {actual_attrition_rate:.1%}"
        )

        metadata = {
            "n_students": n_students,
            "total_semesters": len(df_long),
            "actual_attrition_rate": float(actual_attrition_rate),
            "preset": self.preset,
            "seed": self.seed,
            "audit_df": df_audit,
            "provenance": "SYNTHETIC"
        }

        return df_long, df_outcomes, metadata

    def _inject_missingness(self, df: pd.DataFrame, base_rate: float) -> pd.DataFrame:
        """
        Injects realistic MCAR and MAR missingness into observations.
        Does not touch student_id or semester columns.
        """
        df_out = df.copy()
        n_rows = len(df_out)

        # 1. MCAR: Random missing attendance records (e.g. scanner glitch)
        mcar_mask = self.rng.random(size=n_rows) < base_rate
        df_out.loc[mcar_mask, "attendance_pct"] = np.nan

        # 2. MAR: Struggling students (low GPA) more likely to miss assignment delay logs
        mar_prob = np.where(df_out["gpa"] < 2.0, base_rate * 2.2, base_rate * 0.7)
        mar_mask = self.rng.random(size=n_rows) < mar_prob
        df_out.loc[mar_mask, "assignment_delay_days"] = np.nan
        df_out.loc[mar_mask, "late_assignments_count"] = np.nan

        # 3. Occasional missing exam marks (deferred medical / absent)
        exam_mask = self.rng.random(size=n_rows) < (base_rate * 0.5)
        df_out.loc[exam_mask, "exam_marks"] = np.nan

        return df_out


def save_synthetic_dataset(
    output_dir: Path,
    n_students: int = 5000,
    preset: str = "realistic",
    target_attrition_rate: Optional[float] = None,
    seed: int = 42
) -> Tuple[Path, Path, Dict[str, Any]]:
    """
    Convenience function to generate and save synthetic dataset to disk.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = LatentStudentGenerator(seed=seed, preset=preset, target_attrition_rate=target_attrition_rate)
    df_long, df_outcomes, meta = generator.generate_cohort(n_students=n_students)

    long_path = output_dir / "academic_long.csv"
    outcomes_path = output_dir / "outcomes.csv"

    df_long.to_csv(long_path, index=False)
    df_outcomes.to_csv(outcomes_path, index=False)
    logger.info(f"Saved: {long_path} ({len(df_long)} rows) and {outcomes_path} ({len(df_outcomes)} students)")

    return long_path, outcomes_path, meta
