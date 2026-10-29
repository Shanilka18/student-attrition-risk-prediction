"""Fold-wise profile-feature ablation and paired statistical comparisons."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    roc_auc_score,
)

from src.preprocessing.feature_transformers import (
    FeatureScaler,
    NearConstantFeatureRemover,
    build_feature_pipeline,
)
from src.profiles.transformer import ProfileFeatureTransformer
from src.utils.config_loader import load_yaml
from src.utils.paths import get_project_root

ABLATION_FEATURE_SETS = ("agg", "agg_profiles", "profiles_only")
ABLATION_MODELS = ("logistic_regression", "lightgbm")
ABLATION_METRICS = ("f1", "roc_auc", "mcc")


def _model_factories(
    model_registry: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, str]]:
    factories = {
        "logistic_regression": lambda: LogisticRegression(
            class_weight=model_registry["logistic_regression"]["class_weight"],
            max_iter=int(model_registry["logistic_regression"]["max_iter"]),
            random_state=42,
        )
    }
    errors = {}
    try:
        from lightgbm import LGBMClassifier

        factories["lightgbm"] = lambda: LGBMClassifier(
            **{
                key: value
                for key, value in model_registry["lightgbm"].items()
                if key not in {"family", "estimator"}
            },
            n_jobs=1,
            random_state=42,
        )
    except (ImportError, OSError) as exc:
        errors["lightgbm"] = f"Could not load LightGBM: {exc}"
    return factories, errors


def paired_wilcoxon_tests(
    fold_metrics: pd.DataFrame,
    expected_folds: int | None = None,
) -> list[dict[str, Any]]:
    """Compare each profile set with agg using paired two-sided Wilcoxon tests."""
    comparisons = []
    for model in ABLATION_MODELS:
        baseline = fold_metrics.loc[
            (fold_metrics["model"] == model) & (fold_metrics["feature_set"] == "agg")
        ].set_index("fold")
        for feature_set in ABLATION_FEATURE_SETS:
            if feature_set == "agg":
                continue
            comparison = fold_metrics.loc[
                (fold_metrics["model"] == model)
                & (fold_metrics["feature_set"] == feature_set)
            ].set_index("fold")
            for metric in ABLATION_METRICS:
                paired = (
                    baseline[[metric]]
                    .join(
                        comparison[[metric]],
                        how="inner",
                        lsuffix="_agg",
                        rsuffix="_profile",
                    )
                    .dropna()
                )
                p_value = None
                complete_pairing = (
                    expected_folds is None or len(paired) == expected_folds
                )
                if len(paired) >= 2 and complete_pairing:
                    p_value = float(
                        wilcoxon(
                            paired[f"{metric}_agg"],
                            paired[f"{metric}_profile"],
                            alternative="two-sided",
                        ).pvalue
                    )
                comparisons.append(
                    {
                        "model": model,
                        "baseline": "agg",
                        "comparison": feature_set,
                        "metric": metric,
                        "paired_folds": len(paired),
                        "p_value": p_value,
                        "status": (
                            "available" if p_value is not None else "incomplete_folds"
                        ),
                    }
                )
    return comparisons


def run_profile_ablation(
    data: pd.DataFrame,
    folds: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...],
    output_dir: str | Path,
    model_zoo_path: str | None = None,
) -> dict[str, Any]:
    """Evaluate profile feature sets with identical persisted folds and models."""
    root = get_project_root()
    zoo_path = (
        Path(model_zoo_path)
        if model_zoo_path is not None
        else root / "config" / "model_zoo.yaml"
    )
    model_zoo = load_yaml(zoo_path)
    registry = model_zoo["model_registry"]
    factories, model_errors = _model_factories(registry)
    settings_path = root / "config" / "profile_selection.yaml"
    threshold = float(
        model_zoo.get("preprocessing", {}).get(
            "near_constant_variance_threshold", 0.005
        )
    )
    fold_rows = []

    for fold_number, (train_indices, validation_indices) in enumerate(folds, start=1):
        train_rows = list(train_indices)
        validation_rows = list(validation_indices)
        train_data = data.iloc[train_rows]
        validation_data = data.iloc[validation_rows]
        y_train = train_data["at_risk"].astype(int)
        y_validation = validation_data["at_risk"].astype(int)

        aggregate_pipeline = build_feature_pipeline(
            "agg", "lightgbm", model_zoo_path=str(zoo_path)
        )
        aggregate_train = aggregate_pipeline.fit_transform(train_data)
        aggregate_validation = aggregate_pipeline.transform(validation_data)
        profile_transformer = ProfileFeatureTransformer(
            selection_config_path=str(settings_path)
        ).fit(aggregate_train)
        profiled_train = profile_transformer.transform(aggregate_train)
        profiled_validation = profile_transformer.transform(aggregate_validation)
        profile_columns = list(profile_transformer.feature_names_out_)
        feature_data = {
            "agg": (aggregate_train, aggregate_validation),
            "agg_profiles": (profiled_train, profiled_validation),
            "profiles_only": (
                profiled_train.loc[:, profile_columns],
                profiled_validation.loc[:, profile_columns],
            ),
        }

        for model_name, factory in factories.items():
            if model_name in model_errors:
                continue
            for feature_set in ABLATION_FEATURE_SETS:
                train_features, validation_features = feature_data[feature_set]
                remover = NearConstantFeatureRemover(threshold).fit(train_features)
                train_features = remover.transform(train_features)
                validation_features = remover.transform(validation_features)
                if (
                    registry[model_name]["family"]
                    in model_zoo["preprocessing"]["scaled_model_families"]
                ):
                    scaler = FeatureScaler().fit(train_features)
                    train_features = scaler.transform(train_features)
                    validation_features = scaler.transform(validation_features)

                model = factory()
                try:
                    train_matrix = np.ascontiguousarray(
                        train_features.to_numpy(dtype=np.float64, copy=True)
                    )
                    validation_matrix = np.ascontiguousarray(
                        validation_features.to_numpy(dtype=np.float64, copy=True)
                    )
                    model.fit(train_matrix, y_train.to_numpy(dtype=np.int32, copy=True))
                    probabilities = model.predict_proba(validation_matrix)[:, 1]
                except OSError as exc:
                    model_errors[model_name] = (
                        f"Fold {fold_number}, feature set {feature_set} failed "
                        f"while fitting or predicting: {exc}"
                    )
                    break
                predictions = probabilities >= 0.5
                fold_rows.append(
                    {
                        "fold": fold_number,
                        "model": model_name,
                        "feature_set": feature_set,
                        "f1": float(f1_score(y_validation, predictions)),
                        "roc_auc": float(roc_auc_score(y_validation, probabilities)),
                        "mcc": float(matthews_corrcoef(y_validation, predictions)),
                        "chosen_k": profile_transformer.n_profiles_,
                        "profile_algorithm": profile_transformer.clusterer_.algorithm,
                    }
                )

    results = pd.DataFrame(fold_rows)
    summaries = []
    for model_name in ABLATION_MODELS:
        for feature_set in ABLATION_FEATURE_SETS:
            subset = results.loc[
                (results["model"] == model_name)
                & (results["feature_set"] == feature_set)
            ]
            summary: dict[str, Any] = {
                "model": model_name,
                "feature_set": feature_set,
                "n_folds": len(subset),
                "status": (
                    "available"
                    if len(subset) == len(folds)
                    else ("incomplete" if len(subset) else "unavailable")
                ),
                "expected_folds": len(folds),
            }
            for metric in ABLATION_METRICS:
                summary[f"{metric}_mean"] = (
                    float(subset[metric].mean()) if len(subset) else None
                )
                summary[f"{metric}_std"] = (
                    float(subset[metric].std(ddof=1))
                    if len(subset) > 1
                    else (0.0 if len(subset) else None)
                )
            summaries.append(summary)

    comparisons = (
        paired_wilcoxon_tests(results, expected_folds=len(folds))
        if not results.empty
        else []
    )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "ablation_fold_metrics.csv", index=False)
    pd.DataFrame(summaries).to_csv(output / "ablation_summary.csv", index=False)
    report = {
        "models": list(ABLATION_MODELS),
        "feature_sets": list(ABLATION_FEATURE_SETS),
        "metrics": list(ABLATION_METRICS),
        "classification_threshold": 0.5,
        "fold_metrics": results.to_dict(orient="records"),
        "summary": summaries,
        "paired_wilcoxon_tests": comparisons,
        "model_errors": model_errors,
        "interpretation_note": (
            "Cross-validation folds overlap across repeats; paired Wilcoxon "
            "p-values are descriptive and should not be treated as independent "
            "replicate evidence."
        ),
    }
    (output / "ablation.json").write_text(
        json.dumps(report, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return report
