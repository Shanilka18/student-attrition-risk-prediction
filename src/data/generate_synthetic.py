"""Realistic Synthetic Survey Data Generator for the Psychosocial Intelligence Framework.

Generates a representative Sri Lankan undergraduate dataset mimicking psychometric properties:
1. 5 correlated latent traits sampled via a Multivariate Normal distribution.
2. Demographic subgroup shifts (year of study, gender, programme).
3. Ordinal discretization to 15 Likert items (3 per factor) calibrated to achieve
   Cronbach's alpha between 0.70 and 0.88.
4. Real-world survey artifacts: MCAR + MAR missingness, straight-lining respondents,
   and a dedicated 'dirty' dataset for validation testing.
5. Probabilistic logistic attrition outcomes with unobserved confounders.
6. Weighted continuous PRI ground-truth index and tri-tier risk categorization.
7. Diagnostic data profiling visualizations exported to /artifacts/plots.
"""

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import yaml

from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger("data.generate_synthetic")

FACTOR_ITEMS: Dict[str, List[str]] = {
    "stress": ["S1", "S2", "S3"],
    "motivation": ["M1", "M2", "M3"],
    "belongingness": ["B1", "B2", "B3"],
    "self_efficacy": ["SE1", "SE2", "SE3"],
    "peer_support": ["PS1", "PS2", "PS3"],
}


def load_synthetic_config(config_path: str = "config/synthetic.yaml") -> Dict[str, Any]:
    """Load generator configuration from YAML file."""
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Synthetic config not found at: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def compute_cronbach_alpha(df: pd.DataFrame, items: List[str]) -> float:
    """Calculate Cronbach's alpha internal consistency coefficient for a set of items.

    Formula:
        alpha = (k / (k - 1)) * (1 - sum(var(y_i)) / var(sum(y_i)))

    Args:
        df: DataFrame containing the item responses.
        items: List of column names for the factor items.

    Returns:
        float: Calculated Cronbach's alpha.
    """
    sub_df = df[items].dropna()
    k = len(items)
    if k <= 1 or len(sub_df) < 5:
        return 0.0
    item_variances = sub_df.var(axis=0, ddof=1).sum()
    total_score = sub_df.sum(axis=1)
    total_variance = total_score.var(ddof=1)
    if total_variance <= 0:
        return 0.0
    alpha = (k / (k - 1.0)) * (1.0 - (item_variances / total_variance))
    return float(alpha)


def sample_latent_traits(
    n_samples: int,
    mean: List[float],
    cov_matrix: List[List[float]],
    seed: int = 42,
) -> np.ndarray:
    """Sample correlated latent traits from a Multivariate Normal distribution."""
    rng = np.random.default_rng(seed)
    return rng.multivariate_normal(mean=mean, cov=cov_matrix, size=n_samples)


def apply_demographic_shifts(
    latents: np.ndarray,
    factors: List[str],
    cfg_demo: Dict[str, Any],
    rng: np.random.Generator,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Assign demographic attributes and inject modest realistic mean shifts."""
    n_samples = len(latents)
    shifted_latents = latents.copy()
    demo_records: Dict[str, List[Any]] = {
        "year_of_study": [],
        "gender": [],
        "programme": [],
        "age_group": [],
    }

    yos_cfg = cfg_demo["year_of_study"]
    gender_cfg = cfg_demo["gender"]
    prog_cfg = cfg_demo["programme"]
    age_cfg = cfg_demo["age_group"]

    yos_cats = yos_cfg["categories"]
    yos_probs = yos_cfg["probabilities"]
    gen_cats = gender_cfg["categories"]
    gen_probs = gender_cfg["probabilities"]
    prog_cats = prog_cfg["categories"]
    prog_probs = prog_cfg["probabilities"]
    age_cats = age_cfg["categories"]
    age_probs = age_cfg["probabilities"]

    sampled_yos = rng.choice(yos_cats, size=n_samples, p=yos_probs)
    sampled_gen = rng.choice(gen_cats, size=n_samples, p=gen_probs)
    sampled_prog = rng.choice(prog_cats, size=n_samples, p=prog_probs)
    sampled_age = rng.choice(age_cats, size=n_samples, p=age_probs)

    demo_records["year_of_study"] = list(sampled_yos)
    demo_records["gender"] = list(sampled_gen)
    demo_records["programme"] = list(sampled_prog)
    demo_records["age_group"] = list(sampled_age)

    factor_idx_map = {f: i for i, f in enumerate(factors)}

    for i in range(n_samples):
        y = str(sampled_yos[i])
        g = str(sampled_gen[i])

        if y in yos_cfg.get("effects", {}):
            for f_name, shift in yos_cfg["effects"][y].items():
                if f_name in factor_idx_map:
                    shifted_latents[i, factor_idx_map[f_name]] += shift

        if g in gender_cfg.get("effects", {}):
            for f_name, shift in gender_cfg["effects"][g].items():
                if f_name in factor_idx_map:
                    shifted_latents[i, factor_idx_map[f_name]] += shift

    return shifted_latents, pd.DataFrame(demo_records)


def discretize_to_likert(
    latent_factor: np.ndarray,
    thresholds: List[float],
    noise_sd: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Transform continuous latent traits to 1-5 ordinal Likert scale via noisy thresholds."""
    noisy_val = latent_factor + rng.normal(0, noise_sd, size=len(latent_factor))
    # Threshold discretization:
    # < t0 -> 1
    # t0 to t1 -> 2
    # t1 to t2 -> 3
    # t2 to t3 -> 4
    # > t3 -> 5
    likert = np.digitize(noisy_val, bins=thresholds) + 1
    return np.clip(likert, 1, 5).astype(int)


def generate_synthetic_dataset(
    n_samples: int = 3000,
    config: Optional[Dict[str, Any]] = None,
    seed: int = 42,
    include_missing: bool = True,
    include_straight_lining: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Synthesize a complete survey dataset with psychometric validity and realistic noise.

    Args:
        n_samples: Number of student survey respondents to generate.
        config: Configuration dictionary (or loads default config/synthetic.yaml).
        seed: Random seed for exact reproducibility.
        include_missing: Whether to inject realistic MCAR/MAR null entries.
        include_straight_lining: Whether to inject ~1% straight-lining respondents.

    Returns:
        Tuple[pd.DataFrame, Dict[str, float]]: (DataFrame, achieved_alphas_dict).
    """
    if config is None:
        config = load_synthetic_config()

    set_seed(seed)
    rng = np.random.default_rng(seed)

    latent_cfg = config["latent_factors"]
    factors = latent_cfg["names"]
    mean = latent_cfg["mean"]
    cov_matrix = latent_cfg["correlation_matrix"]

    # 1. Latent trait sampling
    raw_latents = sample_latent_traits(n_samples, mean, cov_matrix, seed=seed)

    # 2. Demographic subgroup shifts
    latents, demo_df = apply_demographic_shifts(
        raw_latents, factors, config["demographics"], rng=rng
    )

    factor_idx_map = {f: i for i, f in enumerate(factors)}

    # 3. Discretize items per factor
    items_cfg = config["items"]
    noise_sd = items_cfg["item_noise_sd"]
    thresholds = items_cfg["ordinal_thresholds"]

    item_data: Dict[str, np.ndarray] = {}
    for factor_name, item_names in FACTOR_ITEMS.items():
        f_idx = factor_idx_map[factor_name]
        f_latent = latents[:, f_idx]
        for it in item_names:
            item_data[it] = discretize_to_likert(f_latent, thresholds, noise_sd, rng=rng)

    df_items = pd.DataFrame(item_data)

    # Calculate achieved Cronbach's alpha
    achieved_alphas: Dict[str, float] = {}
    for factor_name, item_names in FACTOR_ITEMS.items():
        alpha = compute_cronbach_alpha(df_items, item_names)
        achieved_alphas[factor_name] = round(alpha, 4)
        logger.info("Achieved Cronbach's alpha for %s: %.3f", factor_name, alpha)

    # 4. Probabilistic Outcomes (Attrition & Dropout Intention)
    outcomes_cfg = config["outcomes"]
    att_cfg = outcomes_cfg["attrition"]
    att_weights = att_cfg["weights"]

    log_odds = np.full(n_samples, att_cfg["intercept"], dtype=float)
    for f_name, w in att_weights.items():
        log_odds += w * latents[:, factor_idx_map[f_name]]

    # Unobserved confounders
    unobserved = rng.normal(0, att_cfg["unobserved_confounder_sd"], size=n_samples)
    log_odds += unobserved

    # Sigmoid probability
    attrition_probs = 1.0 / (1.0 + np.exp(-log_odds))
    attrition_label = (rng.uniform(0, 1, size=n_samples) < attrition_probs).astype(int)

    # Dropout intention (1-5 Likert scale)
    di_cfg = outcomes_cfg["dropout_intention"]
    di_weights = di_cfg["weights"]
    di_propensity = np.full(n_samples, di_cfg["intercept"], dtype=float)
    for f_name, w in di_weights.items():
        di_propensity += w * latents[:, factor_idx_map[f_name]]
    di_propensity += rng.normal(0, di_cfg["noise_sd"], size=n_samples)
    dropout_intention = np.clip(np.round(di_propensity), 1, 5).astype(int)

    # 5. Continuous PRI Target & Risk Category
    pri_cfg = config["pri_target"]
    pri_weights = pri_cfg["weights"]

    # Compute observed factor scores from clean items
    f_stress = df_items[FACTOR_ITEMS["stress"]].mean(axis=1)
    f_mot = df_items[FACTOR_ITEMS["motivation"]].mean(axis=1)
    f_bel = df_items[FACTOR_ITEMS["belongingness"]].mean(axis=1)
    f_se = df_items[FACTOR_ITEMS["self_efficacy"]].mean(axis=1)
    f_ps = df_items[FACTOR_ITEMS["peer_support"]].mean(axis=1)

    # Invert protective factors to risk orientation (6 - score)
    r_stress = f_stress
    r_mot = 6.0 - f_mot
    r_bel = 6.0 - f_bel
    r_se = 6.0 - f_se
    r_ps = 6.0 - f_ps

    composite_raw_risk = (
        pri_weights["stress"] * r_stress
        + pri_weights["motivation"] * r_mot
        + pri_weights["belongingness"] * r_bel
        + pri_weights["self_efficacy"] * r_se
        + pri_weights["peer_support"] * r_ps
    )

    # Map from [1.0, 5.0] to [0.0, 100.0] + small noise
    pri_true = ((composite_raw_risk - 1.0) / 4.0) * 100.0
    pri_true += rng.normal(0, pri_cfg["noise_sd"], size=n_samples)
    pri_true = np.clip(pri_true, 0.0, 100.0).round(2)

    # Discretize into Risk Categories
    low_cut = pri_cfg["cut_points"]["low_upper"]
    mod_cut = pri_cfg["cut_points"]["moderate_upper"]
    risk_category = []
    for val in pri_true:
        if val <= low_cut:
            risk_category.append("Low")
        elif val <= mod_cut:
            risk_category.append("Moderate")
        else:
            risk_category.append("High")

    # Combine into main DataFrame
    df = pd.DataFrame({"study_id": [f"SLIIT_{i+1:05d}" for i in range(n_samples)]})
    df = pd.concat([df, df_items, demo_df], axis=1)
    df["attrition_label"] = attrition_label
    df["dropout_intention"] = dropout_intention
    df["pri_true"] = pri_true
    df["risk_category"] = risk_category
    df["is_synthetic"] = True

    # 6. Realism Artifacts: Straight-lining
    if include_straight_lining:
        sl_rate = config["realism_noise"]["straight_lining_rate"]
        n_straight = int(np.round(n_samples * sl_rate))
        if n_straight > 0:
            sl_indices = rng.choice(n_samples, size=n_straight, replace=False)
            all_items = [it for item_group in FACTOR_ITEMS.values() for it in item_group]
            for idx in sl_indices:
                uniform_choice = int(rng.choice([1, 2, 3, 4, 5], p=[0.05, 0.15, 0.60, 0.15, 0.05]))
                df.loc[idx, all_items] = uniform_choice
            logger.info("Injected %d straight-lining respondents.", n_straight)

    # 7. Realism Artifacts: Missing values (MCAR + MAR)
    if include_missing:
        miss_rate = config["realism_noise"]["missing_rate"]
        all_items = [it for item_group in FACTOR_ITEMS.values() for it in item_group]
        stress_scores = latents[:, factor_idx_map["stress"]]
        # Standardize stress to [0, 1] range to drive MAR
        p_mar = (stress_scores - stress_scores.min()) / (stress_scores.max() - stress_scores.min() + 1e-6)

        for col in all_items:
            # Baseline MCAR probability
            prob_missing = miss_rate * (0.6 + 0.8 * p_mar)
            mask_missing = rng.uniform(0, 1, size=n_samples) < prob_missing
            df.loc[mask_missing, col] = np.nan

        logger.info("Injected MCAR/MAR missing values across survey items.")

    return df, achieved_alphas


def create_dirty_dataset(clean_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Create a corrupted dataset with out-of-range, non-integer, and duplicate records for testing."""
    rng = np.random.default_rng(seed)
    dirty = clean_df.copy()

    # 1. Out of range values (6, 0, -1)
    dirty.loc[10, "S1"] = 6
    dirty.loc[11, "M2"] = 0
    dirty.loc[12, "SE3"] = -1

    # 2. Non-integer values (floats, strings)
    dirty["B1"] = dirty["B1"].astype(object)
    dirty.loc[20, "B1"] = 3.5
    dirty.loc[21, "B1"] = "four"

    # 3. Duplicate study_ids
    dirty.loc[30, "study_id"] = dirty.loc[29, "study_id"]

    # 4. Invalid outcome values
    dirty.loc[40, "attrition_label"] = 2
    dirty.loc[41, "dropout_intention"] = 8

    return dirty


def generate_data_profiling_plots(
    df: pd.DataFrame,
    output_dir: str = "artifacts/plots",
) -> List[str]:
    """Generate diagnostic data profiling visualizations and save to output_dir."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    saved_plots: List[str] = []

    sns.set_theme(style="whitegrid", palette="muted")
    all_items = [it for item_group in FACTOR_ITEMS.values() for it in item_group]

    # 1. Item & Factor Correlation Heatmap
    plt.figure(figsize=(11, 9))
    corr_matrix = df[all_items].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        mask=mask,
        cmap="coolwarm",
        vmax=0.8,
        vmin=-0.8,
        annot=True,
        fmt=".2f",
        square=True,
        linewidths=0.5,
        cbar_kws={"shrink": 0.8},
    )
    plt.title("Psychosocial Survey Item Correlation Heatmap (15 Items)", fontsize=14, weight="bold")
    plt.tight_layout()
    p1 = str(out_path / "correlation_heatmap.png")
    plt.savefig(p1, dpi=300)
    plt.close()
    saved_plots.append(p1)

    # 2. Factor Distributions (Density + Histogram)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for idx, (f_name, items) in enumerate(FACTOR_ITEMS.items()):
        f_score = df[items].mean(axis=1).dropna()
        sns.histplot(f_score, kde=True, ax=axes[idx], color="steelblue", bins=15, stat="density")
        axes[idx].set_title(f"Factor: {f_name.replace('_', ' ').title()}", fontsize=12, weight="bold")
        axes[idx].set_xlabel("Mean Score (1 - 5)")
        axes[idx].set_xlim(1.0, 5.0)

    # Remove extra subplot
    fig.delaxes(axes[5])
    plt.suptitle("Latent Factor Empirical Score Distributions", fontsize=16, weight="bold")
    plt.tight_layout()
    p2 = str(out_path / "factor_distributions.png")
    plt.savefig(p2, dpi=300)
    plt.close()
    saved_plots.append(p2)

    # 3. PRI Distribution by Risk Category
    plt.figure(figsize=(10, 6))
    palette_dict = {"Low": "#10B981", "Moderate": "#F59E0B", "High": "#EF4444"}
    sns.histplot(
        data=df,
        x="pri_true",
        hue="risk_category",
        palette=palette_dict,
        multiple="stack",
        bins=30,
        kde=True,
    )
    plt.axvline(34.0, color="gray", linestyle="--", linewidth=1.5, label="Low/Mod Cutoff (34)")
    plt.axvline(66.0, color="darkred", linestyle="--", linewidth=1.5, label="Mod/High Cutoff (66)")
    plt.title("Ground-Truth Psychosocial Risk Index (PRI) Distribution", fontsize=14, weight="bold")
    plt.xlabel("PRI Score (0.0 - 100.0)")
    plt.ylabel("Student Count")
    plt.legend()
    plt.tight_layout()
    p3 = str(out_path / "pri_distribution.png")
    plt.savefig(p3, dpi=300)
    plt.close()
    saved_plots.append(p3)

    # 4. Attrition Class Balance & Dropout Intention
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    att_counts = df["attrition_label"].value_counts(normalize=True) * 100
    categories = ["Retained (0)", "Attrited (1)"]
    pcts = [att_counts.get(0, 0), att_counts.get(1, 0)]
    sns.barplot(
        x=categories,
        y=pcts,
        hue=categories,
        palette=["#3B82F6", "#EF4444"],
        legend=False,
        ax=ax1,
    )
    ax1.set_title("Attrition Outcome Class Balance", fontsize=12, weight="bold")
    ax1.set_ylabel("Percentage (%)")
    for p in ax1.patches:
        ax1.annotate(f"{p.get_height():.1f}%", (p.get_x() + 0.30, p.get_height() / 2), color="white", weight="bold")

    sns.countplot(
        x="dropout_intention",
        data=df,
        hue="dropout_intention",
        palette="Blues_d",
        legend=False,
        ax=ax2,
    )
    ax2.set_title("Self-Reported Dropout Intention (1-5)", fontsize=12, weight="bold")
    ax2.set_xlabel("Likert Rating (1=Low, 5=High)")
    ax2.set_ylabel("Student Count")

    plt.suptitle("Attrition & Dropout Intention Characteristics", fontsize=14, weight="bold")
    plt.tight_layout()
    p4 = str(out_path / "attrition_class_balance.png")
    plt.savefig(p4, dpi=300)
    plt.close()
    saved_plots.append(p4)

    logger.info("Saved %d profiling plots to %s", len(saved_plots), output_dir)
    return saved_plots


def main() -> None:
    """Execute complete generation pipeline for default (3000), small (300), and dirty datasets."""
    cfg = load_synthetic_config()
    seed = cfg.get("seed", 42)
    out_dir = Path(cfg["outputs"]["synthetic_dir"])
    plots_dir = Path(cfg["outputs"]["plots_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Starting Phase 2 Synthetic Generation Pipeline (Seed=%d)...", seed)

    # 1. Default 3000 dataset (with realistic missingness & straight-lining)
    df_3000, alphas_3000 = generate_synthetic_dataset(
        n_samples=cfg["sample_sizes"]["default"],
        config=cfg,
        seed=seed,
        include_missing=True,
        include_straight_lining=True,
    )
    p_3000 = out_dir / cfg["outputs"]["clean_dataset_name"]
    df_3000.to_csv(p_3000, index=False)
    logger.info("Generated %d-row dataset at %s", len(df_3000), p_3000)

    # 2. Small 300-row dataset (mimics realistic survey collection size)
    df_300, alphas_300 = generate_synthetic_dataset(
        n_samples=cfg["sample_sizes"]["small"],
        config=cfg,
        seed=seed + 1,
        include_missing=True,
        include_straight_lining=True,
    )
    p_300 = out_dir / cfg["outputs"]["small_dataset_name"]
    df_300.to_csv(p_300, index=False)
    logger.info("Generated %d-row small dataset at %s", len(df_300), p_300)

    # 3. Clean version of 3000 (without missingness/straight-lining) for plotting & modeling baseline
    df_clean_3000, _ = generate_synthetic_dataset(
        n_samples=cfg["sample_sizes"]["default"],
        config=cfg,
        seed=seed,
        include_missing=False,
        include_straight_lining=False,
    )

    # 4. Dirty dataset with deliberate errors to test validation engine
    df_dirty = create_dirty_dataset(df_3000, seed=seed)
    p_dirty = out_dir / cfg["outputs"]["dirty_dataset_name"]
    df_dirty.to_csv(p_dirty, index=False)
    logger.info("Generated dirty dataset with validation errors at %s", p_dirty)

    # 5. Data Profiling Plots
    saved_plots = generate_data_profiling_plots(df_clean_3000, output_dir=str(plots_dir))

    # 6. Save Metadata JSON
    all_items = [it for item_group in FACTOR_ITEMS.values() for it in item_group]
    item_corr = df_clean_3000[all_items].corr().round(3).to_dict()

    att_rate = float(df_3000["attrition_label"].mean())
    risk_dist = df_3000["risk_category"].value_counts(normalize=True).round(3).to_dict()

    metadata: Dict[str, Any] = {
        "generator": "src.data.generate_synthetic",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "sample_size_default": len(df_3000),
        "sample_size_small": len(df_300),
        "achieved_cronbach_alphas_default": alphas_3000,
        "achieved_cronbach_alphas_small": alphas_300,
        "attrition_base_rate": round(att_rate, 4),
        "risk_category_distribution": risk_dist,
        "pri_mean": round(float(df_3000["pri_true"].mean()), 2),
        "pri_std": round(float(df_3000["pri_true"].std()), 2),
        "correlation_matrix_15_items": item_corr,
        "generated_files": {
            "default_dataset": str(p_3000.resolve()),
            "small_dataset": str(p_300.resolve()),
            "dirty_dataset": str(p_dirty.resolve()),
        },
        "profiling_plots": saved_plots,
        "is_synthetic": True,
        "disclaimer": "SYNTHETIC DATA NOTICE: All records are mathematically simulated for pipeline validation and do not represent actual student clinical or academic records.",
    }

    meta_path = out_dir / cfg["outputs"]["metadata_name"]
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    logger.info("Synthetic dataset metadata persisted to %s", meta_path)
    print("\n" + "=" * 60)
    print(f"Synthetic Generation Complete! (Seed={seed})")
    print(f"Default Dataset (N={len(df_3000)}): {p_3000}")
    print(f"Small Dataset   (N={len(df_300)}):  {p_300}")
    print(f"Dirty Dataset   (N={len(df_dirty)}): {p_dirty}")
    print(f"Metadata JSON:                      {meta_path}")
    print(f"Attrition Base Rate:               {att_rate:.1%}")
    print(f"Achieved Cronbach's Alphas:         {alphas_3000}")
    print(f"Risk Category Distribution:         {risk_dist}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
