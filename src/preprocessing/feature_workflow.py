"""Split-first feature-table generation and fold-wise diagnostics."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.cleaning.envfin import EnvironmentalFinancialCleaner
from src.preprocessing.feature_diagnostics import (
    run_feature_diagnostics,
    run_foldwise_model_comparison,
)
from src.preprocessing.feature_transformers import build_feature_pipeline
from src.preprocessing.splits import FeatureSplits, create_feature_splits
from src.utils.config_loader import load_yaml
from src.utils.paths import get_project_root

FEATURE_SETS = ("raw", "agg", "agg_profiles", "single_signal")


def _configured_input_path() -> Path:
    root = get_project_root()
    project = load_yaml(root / "config" / "project.yaml")
    pipeline = project.get("feature_pipeline", {})
    source = project.get("project", {}).get("data_source")
    if source not in {"sample", "survey"}:
        raise ValueError(
            "config/project.yaml project.data_source must be sample or survey"
        )
    input_path = pipeline.get("inputs", {}).get(source)
    if not input_path:
        raise ValueError(f"No feature_pipeline input is configured for {source}")
    return root / input_path


def _save_feature_tables(
    data: pd.DataFrame,
    splits: FeatureSplits,
    run_dir: Path,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame, pd.Series]:
    train_rows = list(splits.train_indices)
    validation_rows = list(splits.validation_indices)
    train = data.iloc[train_rows]
    validation = data.iloc[validation_rows]
    target = data["at_risk"].astype(int)
    table_dir = run_dir / "feature_tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    target.iloc[train_rows].to_frame().to_csv(
        table_dir / "train_labels.csv", index=False
    )
    target.iloc[validation_rows].to_frame().to_csv(
        table_dir / "validation_labels.csv", index=False
    )

    shapes: dict[str, dict[str, Any]] = {}
    agg_training = None
    agg_validation = None
    for feature_set in FEATURE_SETS:
        pipeline = build_feature_pipeline(feature_set, "lightgbm")
        training_features = pipeline.fit_transform(train)
        validation_features = pipeline.transform(validation)
        training_features.to_csv(table_dir / f"train_{feature_set}.csv", index=False)
        validation_features.to_csv(
            table_dir / f"validation_{feature_set}.csv",
            index=False,
        )
        shapes[feature_set] = {
            "train": list(training_features.shape),
            "validation": list(validation_features.shape),
            "columns": list(training_features.columns),
            "removed_near_constant": list(
                pipeline.named_steps["near_constant"].removed_features_
            ),
        }
        if feature_set == "agg":
            agg_training = training_features
            agg_validation = validation_features

    if agg_training is None or agg_validation is None:
        raise RuntimeError("The aggregate feature tables were not produced")
    return (
        shapes,
        pd.concat([agg_training, agg_validation]).reset_index(drop=True),
        target,
    )


def run_feature_workflow(
    input_path: str | Path | None = None,
    run_id: str | None = None,
    output_root: str | Path | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Clean, split, fit train-only feature transforms, and write diagnostics."""
    project_root = get_project_root()
    root = project_root if output_root is None else Path(output_root)
    project_config = load_yaml(project_root / "config" / "project.yaml")
    pipeline_settings = project_config.get("feature_pipeline", {})
    selected_seed = int(
        pipeline_settings.get("split_seed", 42) if seed is None else seed
    )
    current_run_id = run_id or datetime.now(UTC).strftime("features_%Y%m%d_%H%M%S_%f")
    source = _configured_input_path() if input_path is None else Path(input_path)
    if not source.is_absolute():
        source = project_root / source
    if not source.is_file():
        raise FileNotFoundError(f"Feature input file does not exist: {source}")

    cleaned = EnvironmentalFinancialCleaner().clean(source, mode="training")
    if cleaned.data.empty:
        raise ValueError(
            "Feature input contains no usable training rows after cleaning"
        )
    if cleaned.report["traffic_light"]["status"] == "red":
        raise ValueError(
            "Feature input failed training cleaning requirements: "
            + "; ".join(cleaned.report["traffic_light"]["messages"])
        )

    splits = create_feature_splits(
        cleaned.data,
        run_id=current_run_id,
        output_root=root,
        seed=selected_seed,
    )
    run_dir = root / "artifacts" / "runs" / current_run_id
    feature_dir = run_dir / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)

    shapes, agg_features, target = _save_feature_tables(
        splits.data,
        splits,
        feature_dir,
    )
    train_validation_rows = [
        *splits.train_indices,
        *splits.validation_indices,
    ]
    diagnostics = run_feature_diagnostics(
        agg_features,
        target.iloc[train_validation_rows].reset_index(drop=True),
        source_columns=list(splits.data.columns),
        output_dir=feature_dir / "diagnostics",
        palette=project_config["sample_data"]["brand_palette"],
    )
    if not diagnostics["leakage_audit"]["passed"]:
        raise RuntimeError(
            "Feature leakage audit failed: "
            + json.dumps(diagnostics["leakage_audit"], sort_keys=True)
        )
    cv_summary, model_errors = run_foldwise_model_comparison(
        splits.data,
        target,
        splits.folds,
        feature_dir / "diagnostics",
    )

    summary = {
        "run_id": current_run_id,
        "data_source": project_config["project"]["data_source"],
        "cleaning": {
            "rows_in": cleaned.report["rows_in"],
            "rows_out": cleaned.report["rows_out"],
            "traffic_light": cleaned.report["traffic_light"],
        },
        "splits_path": str(splits.path),
        "feature_shapes": shapes,
        "leakage_audit": diagnostics["leakage_audit"],
        "model_comparison": cv_summary.to_dict(orient="records"),
        "model_errors": model_errors,
        "summary_path": str(feature_dir / "feature_run_summary.json"),
        "figures": [str(path) for path in diagnostics["figures"]]
        + [str(feature_dir / "diagnostics" / "raw_vs_aggregate_auc.png")],
        "warning": splits.warning,
    }
    summary_path = feature_dir / "feature_run_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
