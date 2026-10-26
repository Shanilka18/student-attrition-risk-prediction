"""Psychometric feature engineering and latent factor aggregation.

Calculates:
- 5 latent factor domain scores (Stress, Motivation, Belongingness, Self-Efficacy, Peer Support).
- Risk-oriented factor scores (protective factors reversed so higher = riskier).
- Standardized z-scores and aggregate interaction features.
- Configurable feature matrices for comparative model benchmarking (raw items vs. factor scores).
- Target extractions (continuous PRI regression, 3-class risk categorization, binary attrition).
"""

from typing import Dict, Any, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.utils.logger import get_logger

logger = get_logger("features.build_features")

RAW_ITEM_COLS: List[str] = [
    "S1", "S2", "S3",
    "M1", "M2", "M3",
    "B1", "B2", "B3",
    "SE1", "SE2", "SE3",
    "PS1", "PS2", "PS3",
]

FACTOR_MAPPING: Dict[str, List[str]] = {
    "stress": ["S1", "S2", "S3"],
    "motivation": ["M1", "M2", "M3"],
    "belongingness": ["B1", "B2", "B3"],
    "self_efficacy": ["SE1", "SE2", "SE3"],
    "peer_support": ["PS1", "PS2", "PS3"],
}

# expose mapping for API imports
FACTOR_ITEMS = FACTOR_MAPPING

FACTOR_NAMES = list(FACTOR_MAPPING.keys())

FACTOR_COLS: List[str] = [
    "factor_stress",
    "factor_motivation",
    "factor_belongingness",
    "factor_self_efficacy",
    "factor_peer_support",
]

RISK_FACTOR_COLS: List[str] = [
    "risk_stress",
    "risk_motivation",
    "risk_belongingness",
    "risk_self_efficacy",
    "risk_peer_support",
]

INTERACTION_COLS: List[str] = [
    "stress_minus_efficacy",
    "stress_to_support_ratio",
    "protective_mean",
    "resilience_deficit",
    "stress_motivation_interaction",
]


def reverse_protective_items(df: pd.DataFrame) -> pd.DataFrame:
    """Reverse 5-point Likert scale items for protective factors (Reversed = 6 - Raw).

    On a 1-5 scale:
    - 5 (Strongly Agree / Highly Protective) becomes 1 (Low Risk contribution).
    - 1 (Strongly Disagree / Low Protective) becomes 5 (High Risk contribution).

    Args:
        df: DataFrame containing protective Likert items (M*, B*, SE*, PS*).

    Returns:
        pd.DataFrame: DataFrame copy with reversed items suffixed (_rev).
    """
    df_rev = df.copy()
    protective_items = [
        "M1", "M2", "M3",
        "B1", "B2", "B3",
        "SE1", "SE2", "SE3",
        "PS1", "PS2", "PS3",
    ]
    for col in protective_items:
        if col in df_rev.columns:
            df_rev[f"{col}_rev"] = 6 - df_rev[col]
    return df_rev


def compute_factor_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean scores for each of the 5 latent psychosocial factors.

    Factors:
    - Stress (S1, S2, S3) [Raw scale 1-5, Higher = worse]
    - Motivation (M1, M2, M3) [Raw scale 1-5, Higher = better]
    - Belongingness (B1, B2, B3) [Raw scale 1-5, Higher = better]
    - Self-Efficacy (SE1, SE2, SE3) [Raw scale 1-5, Higher = better]
    - Peer Support (PS1, PS2, PS3) [Raw scale 1-5, Higher = better]

    Args:
        df: Survey DataFrame containing 15 items.

    Returns:
        pd.DataFrame: DataFrame copy with added 'factor_*' columns.
    """
    out = df.copy()
    out["factor_stress"] = out[["S1", "S2", "S3"]].mean(axis=1)
    out["factor_motivation"] = out[["M1", "M2", "M3"]].mean(axis=1)
    out["factor_belongingness"] = out[["B1", "B2", "B3"]].mean(axis=1)
    out["factor_self_efficacy"] = out[["SE1", "SE2", "SE3"]].mean(axis=1)
    out["factor_peer_support"] = out[["PS1", "PS2", "PS3"]].mean(axis=1)
    return out


def compute_risk_oriented_factors(df: pd.DataFrame) -> pd.DataFrame:
    """Compute risk-oriented factors where higher score always indicates higher risk.

    - risk_stress = factor_stress
    - risk_motivation = 6.0 - factor_motivation
    - risk_belongingness = 6.0 - factor_belongingness
    - risk_self_efficacy = 6.0 - factor_self_efficacy
    - risk_peer_support = 6.0 - factor_peer_support
    """
    out = df if "factor_stress" in df.columns else compute_factor_scores(df)
    res = out.copy()
    res["risk_stress"] = res["factor_stress"]
    res["risk_motivation"] = 6.0 - res["factor_motivation"]
    res["risk_belongingness"] = 6.0 - res["factor_belongingness"]
    res["risk_self_efficacy"] = 6.0 - res["factor_self_efficacy"]
    res["risk_peer_support"] = 6.0 - res["factor_peer_support"]
    return res


def compute_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Generate psychometric interaction and aggregate disparity features.

    Features:
    - stress_minus_efficacy: Stress - Self-Efficacy (strain vs coping capability)
    - stress_to_support_ratio: Stress / (Peer Support + 0.01)
    - protective_mean: Average of the 4 protective domains
    - resilience_deficit: Inverted protective mean - Stress
    - stress_motivation_interaction: Stress * (6.0 - Motivation)
    """
    out = compute_risk_oriented_factors(df) if "risk_stress" not in df.columns else df.copy()

    f_stress = out["factor_stress"]
    f_mot = out["factor_motivation"]
    f_bel = out["factor_belongingness"]
    f_se = out["factor_self_efficacy"]
    f_ps = out["factor_peer_support"]

    out["stress_minus_efficacy"] = f_stress - f_se
    out["stress_to_support_ratio"] = f_stress / (f_ps + 0.01)
    out["protective_mean"] = (f_mot + f_bel + f_se + f_ps) / 4.0
    out["resilience_deficit"] = (6.0 - out["protective_mean"]) - f_stress
    out["stress_motivation_interaction"] = f_stress * (6.0 - f_mot)
    return out


def engineer_all_features(
    df: pd.DataFrame,
    scaler: Optional[StandardScaler] = None,
    fit_scaler: bool = False,
) -> Tuple[pd.DataFrame, Optional[StandardScaler]]:
    """Complete psychometric feature engineering pipeline.

    Calculates:
    - Raw items
    - Factor scores
    - Risk-oriented factor scores
    - Non-linear interaction features
    - Standardized z-score variants (fitted strictly on train if fit_scaler=True)

    Args:
        df: Input survey DataFrame.
        scaler: Optional fitted StandardScaler.
        fit_scaler: If True, fits a new StandardScaler on the continuous engineered columns.

    Returns:
        Tuple[pd.DataFrame, Optional[StandardScaler]]: (Engineered DataFrame, fitted or passed scaler).
    """
    res = compute_interaction_features(df)

    # Standardize continuous factors and interactions
    cols_to_scale = FACTOR_COLS + RISK_FACTOR_COLS + INTERACTION_COLS
    if fit_scaler:
        scaler = StandardScaler()
        z_vals = scaler.fit_transform(res[cols_to_scale])
    elif scaler is not None:
        z_vals = scaler.transform(res[cols_to_scale])
    else:
        z_vals = None

    if z_vals is not None:
        for idx, col in enumerate(cols_to_scale):
            res[f"z_{col}"] = z_vals[:, idx]

    return res, scaler


def extract_feature_set(df: pd.DataFrame, feature_set: str = "raw_items") -> pd.DataFrame:
    """Extract a specific feature subset for model experimentation and leaderboards.

    Supported sets:
    - 'raw_items': 15 Likert items (S1..S3, M1..M3, B1..B3, SE1..SE3, PS1..PS3).
    - 'factor_scores': 5 factor mean scores (factor_stress, etc.).
    - 'risk_factors': 5 risk-oriented factor scores (risk_stress, etc.).
    - 'interactions': 5 interaction disparity features.
    - 'combined': 15 raw items + 5 factor scores + 5 interaction features.
    - 'all': All available raw, factor, risk, interaction, and z-score features.

    Args:
        df: Input DataFrame with engineered features.
        feature_set: Keyword selecting feature subset.

    Returns:
        pd.DataFrame: Feature matrix X.
    """
    clean_fs = feature_set.lower().strip()
    if clean_fs == "raw_items":
        return df[RAW_ITEM_COLS].copy()
    elif clean_fs == "factor_scores":
        return df[FACTOR_COLS].copy()
    elif clean_fs == "risk_factors":
        return df[RISK_FACTOR_COLS].copy()
    elif clean_fs == "interactions":
        return df[INTERACTION_COLS].copy()
    elif clean_fs == "combined":
        cols = RAW_ITEM_COLS + FACTOR_COLS + INTERACTION_COLS
        return df[cols].copy()
    elif clean_fs == "all":
        z_cols = [f"z_{c}" for c in FACTOR_COLS + RISK_FACTOR_COLS + INTERACTION_COLS if f"z_{c}" in df.columns]
        cols = RAW_ITEM_COLS + FACTOR_COLS + RISK_FACTOR_COLS + INTERACTION_COLS + z_cols
        return df[cols].copy()
    else:
        raise ValueError(
            f"Unknown feature_set '{feature_set}'. "
            f"Valid options: 'raw_items', 'factor_scores', 'risk_factors', 'interactions', 'combined', 'all'."
        )


def extract_targets(df: pd.DataFrame) -> Dict[str, pd.Series]:
    """Extract prediction targets from dataset.

    Returns dictionary with:
    - 'pri_score': Continuous PRI target (0 - 100)
    - 'risk_category': Categorical risk label ('Low', 'Moderate', 'High')
    - 'attrition_label': Binary retention outcome (0 or 1)
    - 'dropout_intention': Likert intention (1 - 5)
    """
    targets: Dict[str, pd.Series] = {}

    if "pri_true" in df.columns:
        targets["pri_score"] = df["pri_true"]
    elif "pri_score" in df.columns:
        targets["pri_score"] = df["pri_score"]
    else:
        # Compute deterministic baseline
        targets["pri_score"] = compute_pri_score(compute_factor_scores(df))

    if "risk_category" in df.columns:
        targets["risk_category"] = df["risk_category"]
    else:
        targets["risk_category"] = categorize_risk(targets["pri_score"])

    if "attrition_label" in df.columns:
        targets["attrition_label"] = df["attrition_label"]

    if "dropout_intention" in df.columns:
        targets["dropout_intention"] = df["dropout_intention"]

    return targets


def compute_pri_score(df: pd.DataFrame) -> pd.Series:
    """Compute Psychosocial Risk Index (PRI) on a 0 to 100 continuous scale."""
    out = df if "factor_stress" in df.columns else compute_factor_scores(df)
    stress_risk = out["factor_stress"]
    mot_risk = 6.0 - out["factor_motivation"]
    bel_risk = 6.0 - out["factor_belongingness"]
    se_risk = 6.0 - out["factor_self_efficacy"]
    ps_risk = 6.0 - out["factor_peer_support"]

    composite_risk = (stress_risk + mot_risk + bel_risk + se_risk + ps_risk) / 5.0
    pri = (composite_risk - 1.0) / 4.0 * 100.0
    return pri.clip(0.0, 100.0)


def categorize_risk(pri_series: pd.Series, low_cut: float = 34.0, mod_cut: float = 66.0) -> pd.Series:
    """Discretize continuous PRI into standardized risk tiers: Low, Moderate, High."""
    bins = [-np.inf, low_cut, mod_cut, np.inf]
    labels = ["Low", "Moderate", "High"]
    return pd.cut(pri_series, bins=bins, labels=labels)
