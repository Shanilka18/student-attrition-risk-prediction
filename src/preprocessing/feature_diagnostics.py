"""Exploratory and fold-wise diagnostics for engineered EFI features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from sklearn.metrics import roc_auc_score

from src.preprocessing.feature_transformers import build_feature_pipeline
from src.utils.config_loader import load_yaml
from src.utils.paths import get_project_root


def _cramers_v(feature: pd.Series, target: pd.Series) -> float:
    table = pd.crosstab(feature, target)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    statistic = chi2_contingency(table, correction=False)[0]
    denominator = table.to_numpy().sum() * min(table.shape[0] - 1, table.shape[1] - 1)
    return float(np.sqrt(statistic / denominator)) if denominator else 0.0


def feature_effect_sizes(features: pd.DataFrame, target: pd.Series) -> pd.DataFrame:
    """Measure binary features with Cramer's V and continuous features with Cohen's d."""
    rows = []
    labels = target.astype(int)
    for column in features.columns:
        values = pd.to_numeric(features[column], errors="coerce")
        valid = values.notna() & labels.notna()
        observed = values.loc[valid]
        observed_labels = labels.loc[valid]
        if observed.nunique() < 2 or observed_labels.nunique() != 2:
            effect, measure = 0.0, "Cramer's V"
        elif observed.nunique() <= 2:
            effect, measure = _cramers_v(observed, observed_labels), "Cramer's V"
        else:
            positive = observed.loc[observed_labels == 1]
            negative = observed.loc[observed_labels == 0]
            pooled = np.sqrt(
                (
                    (len(positive) - 1) * positive.var(ddof=1)
                    + (len(negative) - 1) * negative.var(ddof=1)
                )
                / max(len(positive) + len(negative) - 2, 1)
            )
            effect = (
                float((positive.mean() - negative.mean()) / pooled)
                if pooled > 0
                else 0.0
            )
            measure = "Cohen's d"
        rows.append(
            {
                "Feature": column,
                "Effect size": float(effect),
                "Measure": measure,
                "At-risk mean": float(observed.loc[observed_labels == 1].mean()),
                "Lower-risk mean": float(observed.loc[observed_labels == 0].mean()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "Effect size", key=lambda values: values.abs(), ascending=False
    )


def leakage_audit(
    features: pd.DataFrame,
    target: pd.Series,
    source_columns: list[str],
    maximum_absolute_correlation: float = 0.95,
) -> dict[str, Any]:
    """Check outcome separation and flag suspiciously label-correlated features."""
    outcome_fields = {
        "at_risk",
        "attrition_intent_score",
        "student_id",
        "timestamp",
        "consent",
        "attention_check",
    }
    absent_outcomes = sorted(outcome_fields - set(features.columns))
    values = features.apply(pd.to_numeric, errors="coerce")
    correlations = values.corrwith(target.astype(float)).dropna()
    high_correlations = {
        str(name): float(value)
        for name, value in correlations.items()
        if abs(float(value)) > maximum_absolute_correlation
    }
    outcome_inputs = sorted(set(source_columns).intersection(outcome_fields))
    passed = not high_correlations and len(absent_outcomes) == len(outcome_fields)
    return {
        "passed": passed,
        "outcome_columns_present_in_features": sorted(
            set(features.columns).intersection(outcome_fields)
        ),
        "outcome_columns_available_in_source_frame": outcome_inputs,
        "absolute_correlation_threshold": maximum_absolute_correlation,
        "high_label_correlations": high_correlations,
        "maximum_observed_absolute_correlation": (
            float(correlations.abs().max()) if not correlations.empty else None
        ),
    }


def mutual_information_ranking(
    features: pd.DataFrame, target: pd.Series, random_state: int = 42
) -> pd.DataFrame:
    """Rank numeric features by cross-validated-input mutual information estimate."""
    from sklearn.feature_selection import mutual_info_classif

    values = features.astype(float)
    discrete = values.nunique(dropna=False).le(2).to_numpy()
    scores = mutual_info_classif(
        values,
        target.astype(int),
        discrete_features=discrete,
        random_state=random_state,
    )
    return pd.DataFrame(
        {"Feature": values.columns, "Mutual information": scores}
    ).sort_values("Mutual information", ascending=False)


def variance_inflation_report(features: pd.DataFrame) -> pd.DataFrame:
    """Report multicollinearity for numeric linear-model inputs without dropping them."""
    values = features.apply(pd.to_numeric, errors="coerce")
    values = values.loc[:, values.nunique(dropna=True) > 1]
    values = values.fillna(values.median()).astype(float)
    if values.empty:
        return pd.DataFrame(columns=["Feature", "VIF"])

    matrix = values.to_numpy()
    rows = []
    for column_index, name in enumerate(values.columns):
        response = matrix[:, column_index]
        predictors = np.delete(matrix, column_index, axis=1)
        design = np.column_stack([np.ones(len(response)), predictors])
        coefficients, *_ = np.linalg.lstsq(design, response, rcond=None)
        residuals = response - design @ coefficients
        total = np.square(response - response.mean()).sum()
        residual = np.square(residuals).sum()
        r_squared = 1.0 - residual / total if total else 1.0
        vif = float("inf") if 1.0 - r_squared <= 1e-12 else 1.0 / (1.0 - r_squared)
        rows.append({"Feature": name, "VIF": vif})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False)


def _save_diagnostic_figures(
    effects: pd.DataFrame,
    mutual_information: pd.DataFrame,
    features: pd.DataFrame,
    output_dir: Path,
    palette: dict[str, str],
) -> list[Path]:
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    figures = []
    plot_effects = effects.head(30).sort_values("Effect size")
    fig, axis = plt.subplots(figsize=(10, max(6, len(plot_effects) * 0.27)))
    axis.barh(
        plot_effects["Feature"],
        plot_effects["Effect size"],
        color=palette.get("emerald", "#059669"),
    )
    axis.set(
        title="Engineered feature effect sizes by risk group", xlabel="Effect size"
    )
    fig.tight_layout()
    effect_path = output_dir / "feature_effect_sizes.png"
    fig.savefig(effect_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    figures.append(effect_path)

    correlation = features.corr(method="spearman")
    size = max(10, min(22, len(correlation.columns) * 0.32))
    fig, axis = plt.subplots(figsize=(size, size * 0.82))
    image = axis.imshow(
        correlation.to_numpy(),
        cmap="PiYG",
        vmin=-1,
        vmax=1,
        aspect="auto",
        interpolation="nearest",
    )
    axis.set_xticks(np.arange(len(correlation.columns)))
    axis.set_xticklabels(correlation.columns, rotation=90, fontsize=6)
    axis.set_yticks(np.arange(len(correlation.index)))
    axis.set_yticklabels(correlation.index, fontsize=6)
    fig.colorbar(image, ax=axis, label="Spearman correlation", fraction=0.025)
    axis.set_title("Spearman correlation among engineered features")
    fig.tight_layout()
    correlation_path = output_dir / "feature_spearman_heatmap.png"
    fig.savefig(correlation_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    figures.append(correlation_path)

    plot_mi = mutual_information.head(30).sort_values("Mutual information")
    fig, axis = plt.subplots(figsize=(10, max(6, len(plot_mi) * 0.27)))
    axis.barh(
        plot_mi["Feature"],
        plot_mi["Mutual information"],
        color=palette.get("teal", "#0f766e"),
    )
    axis.set(title="Mutual-information feature ranking", xlabel="Mutual information")
    fig.tight_layout()
    mi_path = output_dir / "feature_mutual_information.png"
    fig.savefig(mi_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    figures.append(mi_path)
    return figures


def run_feature_diagnostics(
    features: pd.DataFrame,
    target: pd.Series,
    source_columns: list[str],
    output_dir: str | Path,
    palette: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Write effects, correlations, MI, leakage report, and branded figures."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    colors = palette or {
        "emerald": "#059669",
        "teal": "#0f766e",
        "purple": "#7c3aed",
        "amber": "#d97706",
    }
    effects = feature_effect_sizes(features, target)
    mutual_information = mutual_information_ranking(features, target)
    vif = variance_inflation_report(features)
    audit = leakage_audit(features, target, source_columns)
    effects.to_csv(output / "feature_effect_sizes.csv", index=False)
    mutual_information.to_csv(output / "feature_mutual_information.csv", index=False)
    vif.to_csv(output / "linear_model_vif.csv", index=False)
    (output / "leakage_audit.json").write_text(
        json.dumps(audit, indent=2),
        encoding="utf-8",
    )
    figures = _save_diagnostic_figures(
        effects, mutual_information, features, output, colors
    )
    return {
        "effects": effects,
        "mutual_information": mutual_information,
        "variance_inflation": vif,
        "leakage_audit": audit,
        "figures": figures,
    }


def run_foldwise_model_comparison(
    data: pd.DataFrame,
    target: pd.Series,
    folds: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
    output_dir: str | Path,
    model_zoo_path: str | None = None,
) -> tuple[pd.DataFrame, dict[str, str]]:
    """Compare weighted logistic regression and LightGBM on raw and aggregate sets."""
    from sklearn.linear_model import LogisticRegression

    root = get_project_root()
    zoo_path = (
        Path(model_zoo_path)
        if model_zoo_path is not None
        else root / "config" / "model_zoo.yaml"
    )
    model_zoo = load_yaml(zoo_path)
    registry = model_zoo["model_registry"]
    model_errors: dict[str, str] = {}
    model_factories = {
        "logistic_regression": lambda: LogisticRegression(
            class_weight=registry["logistic_regression"]["class_weight"],
            max_iter=int(registry["logistic_regression"]["max_iter"]),
            random_state=42,
        ),
    }
    try:
        from lightgbm import LGBMClassifier

        model_factories["lightgbm"] = lambda: LGBMClassifier(
            **{
                key: value
                for key, value in registry["lightgbm"].items()
                if key not in {"family", "estimator"}
            },
            n_jobs=1,
            random_state=42,
        )
    except (ImportError, OSError) as exc:
        model_errors["lightgbm"] = f"Could not load LightGBM: {exc}"

    rows = []
    for feature_set in ("raw", "agg"):
        for model_name, factory in model_factories.items():
            if model_name in model_errors:
                continue
            for fold_number, (train_indices, validation_indices) in enumerate(
                folds, start=1
            ):
                train_rows = list(train_indices)
                validation_rows = list(validation_indices)
                feature_pipeline = build_feature_pipeline(
                    feature_set, model_name, model_zoo_path=str(zoo_path)
                )
                training_features = feature_pipeline.fit_transform(
                    data.iloc[train_rows]
                )
                validation_features = feature_pipeline.transform(
                    data.iloc[validation_rows]
                )
                model = factory()
                try:
                    model.fit(
                        np.ascontiguousarray(
                            training_features.to_numpy(dtype=np.float64, copy=True)
                        ),
                        target.iloc[train_rows].to_numpy(dtype=np.int32, copy=True),
                    )
                    probabilities = model.predict_proba(
                        np.ascontiguousarray(
                            validation_features.to_numpy(dtype=np.float64, copy=True)
                        )
                    )[:, 1]
                except OSError as exc:
                    model_errors[model_name] = (
                        f"Fold {fold_number} failed while fitting or predicting: {exc}"
                    )
                    break
                score = roc_auc_score(
                    target.iloc[validation_rows].astype(int),
                    probabilities,
                )
                rows.append(
                    {
                        "feature_set": feature_set,
                        "model": model_name,
                        "fold": fold_number,
                        "roc_auc": float(score),
                    }
                )

    results = pd.DataFrame(rows)
    summary = (
        results.groupby(["feature_set", "model"])["roc_auc"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    summary["ci_95_low"] = (
        results.groupby(["feature_set", "model"])["roc_auc"].quantile(0.025).to_numpy()
    )
    summary["ci_95_high"] = (
        results.groupby(["feature_set", "model"])["roc_auc"].quantile(0.975).to_numpy()
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "foldwise_model_comparison.csv", index=False)
    summary.to_csv(output / "model_comparison_summary.csv", index=False)
    (output / "model_comparison_errors.json").write_text(
        json.dumps(model_errors, indent=2),
        encoding="utf-8",
    )

    import matplotlib.pyplot as plt

    palette = {"raw": "#059669", "agg": "#0f766e"}
    fig, axis = plt.subplots(figsize=(9, 5))
    labels = []
    means = []
    errors = []
    colors = []
    for row in summary.itertuples(index=False):
        labels.append(f"{row.feature_set} / {row.model}")
        means.append(row.mean)
        errors.append([row.mean - row.ci_95_low, row.ci_95_high - row.mean])
        colors.append(palette[row.feature_set])
    axis.bar(
        labels,
        means,
        yerr=np.asarray(errors).T,
        color=colors,
        capsize=4,
    )
    axis.set(
        title="Repeated-fold ROC-AUC: raw and aggregate feature sets",
        ylabel="ROC-AUC",
        ylim=(0.4, 1.0),
    )
    axis.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    figure_path = output / "raw_vs_aggregate_auc.png"
    fig.savefig(figure_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return summary, model_errors
