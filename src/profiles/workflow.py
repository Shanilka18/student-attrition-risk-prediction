"""Profile discovery, generated-data recovery, and feature-ablation workflow."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

from src.cleaning.envfin import EnvironmentalFinancialCleaner
from src.preprocessing.feature_transformers import build_feature_pipeline
from src.preprocessing.splits import create_feature_splits
from src.profiles.evaluation import run_profile_ablation
from src.profiles.figures import profile_statistics, save_profile_figures
from src.profiles.selection import DEFAULT_INDEX_COLUMNS
from src.profiles.transformer import ProfileFeatureTransformer
from src.utils.config_loader import load_yaml
from src.utils.paths import get_project_root


def _input_path(config: dict[str, Any], input_path: str | Path | None) -> Path:
    root = get_project_root()
    if input_path is not None:
        selected = Path(input_path)
        return selected if selected.is_absolute() else root / selected
    data_source = config.get("project", {}).get("data_source")
    if data_source not in {"sample", "survey"}:
        raise ValueError(
            "config/project.yaml project.data_source must be sample or survey"
        )
    configured = config.get("feature_pipeline", {}).get("inputs", {}).get(data_source)
    if not configured:
        raise ValueError(f"No profiles input is configured for {data_source}")
    return root / configured


def _profile_rows(
    data: pd.DataFrame,
    aggregate: pd.DataFrame,
    profile_features: pd.DataFrame,
    profile_ids: np.ndarray,
    profile_names: tuple[str, ...],
    partition: str,
) -> pd.DataFrame:
    result = data.loc[:, ["student_id", "at_risk"]].copy()
    result["partition"] = partition
    result["profile_id"] = profile_ids
    result["profile_name"] = [profile_names[index] for index in profile_ids]
    for column in (*DEFAULT_INDEX_COLUMNS, "overall_pressure_index"):
        result[column] = aggregate[column].to_numpy()
    return pd.concat(
        [result.reset_index(drop=True), profile_features.reset_index(drop=True)], axis=1
    )


def _hidden_type_recovery(
    profile_table: pd.DataFrame,
    source: Path,
    configured_sample_path: Path,
    config: dict[str, Any],
    root: Path,
) -> dict[str, Any]:
    if source.resolve() != configured_sample_path.resolve():
        return {
            "status": "not_run",
            "validity": "valid only for generated data",
            "reason": "The input was not the configured full generated sample file.",
        }
    truth_path = root / config["sample_data"]["outputs"]["hidden_truth"]
    if not truth_path.is_file():
        return {
            "status": "not_run",
            "validity": "valid only for generated data",
            "reason": f"Hidden generated-data truth is unavailable: {truth_path}",
        }
    truth = pd.read_csv(truth_path)
    if "student_id" not in truth.columns:
        id_config = config["sample_data"]["identifiers"]
        prefix = str(id_config["prefix"])
        digits = int(id_config["digits"])
        truth.insert(
            0,
            "student_id",
            [f"{prefix}{index:0{digits}d}" for index in range(1, len(truth) + 1)],
        )
    if "type" not in truth.columns:
        raise ValueError("Hidden generated-data truth must include a type column")
    joined = profile_table[["student_id", "profile_id"]].merge(
        truth[["student_id", "type"]],
        on="student_id",
        how="inner",
        validate="one_to_one",
    )
    if joined.empty:
        return {
            "status": "not_run",
            "validity": "valid only for generated data",
            "reason": "No generated student identifiers matched hidden truth.",
        }
    confusion = pd.crosstab(
        joined["profile_id"],
        joined["type"],
        dropna=False,
    )
    return {
        "status": "available",
        "validity": "valid only for generated data",
        "students_matched": len(joined),
        "ari": float(adjusted_rand_score(joined["type"], joined["profile_id"])),
        "nmi": float(
            normalized_mutual_info_score(joined["type"], joined["profile_id"])
        ),
        "confusion_matrix": {
            "profile_ids": [int(index) for index in confusion.index],
            "student_types": [str(column) for column in confusion.columns],
            "counts": confusion.to_numpy(dtype=int).tolist(),
        },
    }


def run_profiles_workflow(
    input_path: str | Path | None = None,
    run_id: str | None = None,
    output_root: str | Path | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Fit profiles on outer-train rows and run repeated-fold ablation."""
    project_root = get_project_root()
    root = project_root if output_root is None else Path(output_root)
    project_config = load_yaml(project_root / "config" / "project.yaml")
    source = _input_path(project_config, input_path)
    if not source.is_file():
        raise FileNotFoundError(f"Profile input file does not exist: {source}")
    data_source = project_config.get("project", {}).get("data_source")
    selected_seed = int(
        project_config.get("feature_pipeline", {}).get("split_seed", 42)
        if seed is None
        else seed
    )
    current_run_id = run_id or datetime.now(UTC).strftime("profiles_%Y%m%d_%H%M%S_%f")

    cleaned = EnvironmentalFinancialCleaner().clean(source, mode="training")
    if cleaned.data.empty:
        raise ValueError("Profile input contains no usable training rows")
    if cleaned.report["traffic_light"]["status"] == "red":
        raise ValueError(
            "Profile input failed training cleaning requirements: "
            + "; ".join(cleaned.report["traffic_light"]["messages"])
        )
    splits = create_feature_splits(
        cleaned.data,
        run_id=current_run_id,
        output_root=root,
        seed=selected_seed,
    )
    run_dir = root / "artifacts" / "runs" / current_run_id
    profiles_dir = run_dir / "profiles"
    profiles_dir.mkdir(parents=True, exist_ok=True)

    train_rows = list(splits.train_indices)
    validation_rows = list(splits.validation_indices)
    train_data = splits.data.iloc[train_rows]
    validation_data = splits.data.iloc[validation_rows]
    aggregate_pipeline = build_feature_pipeline("agg", "lightgbm")
    aggregate_train = aggregate_pipeline.fit_transform(train_data)
    aggregate_validation = aggregate_pipeline.transform(validation_data)

    transformer = ProfileFeatureTransformer().fit(aggregate_train)
    profiled_train = transformer.transform(aggregate_train)
    profiled_validation = transformer.transform(aggregate_validation)
    profile_columns = list(transformer.feature_names_out_)
    train_ids = transformer.training_labels_
    validation_ids = np.argmax(
        profiled_validation.loc[
            :, [f"profile_id_{index}" for index in range(transformer.n_profiles_)]
        ].to_numpy(),
        axis=1,
    )
    train_table = _profile_rows(
        train_data,
        aggregate_train,
        profiled_train.loc[:, profile_columns],
        train_ids,
        transformer.profile_names_,
        "train",
    )
    validation_table = _profile_rows(
        validation_data,
        aggregate_validation,
        profiled_validation.loc[:, profile_columns],
        validation_ids,
        transformer.profile_names_,
        "validation",
    )
    profile_table = pd.concat([train_table, validation_table], ignore_index=True)
    profile_table.to_csv(profiles_dir / "profile_feature_table.csv", index=False)

    all_ids = np.concatenate([train_ids, validation_ids])
    all_profile_indices = pd.concat(
        [aggregate_train, aggregate_validation],
        ignore_index=True,
    )
    all_profile_indices["at_risk"] = pd.concat(
        [train_data["at_risk"], validation_data["at_risk"]],
        ignore_index=True,
    ).to_numpy()
    statistics = profile_statistics(
        all_profile_indices,
        all_ids,
        transformer.profile_names_,
    )
    statistics.to_csv(profiles_dir / "profile_summary.csv", index=False)
    candidate_metrics = transformer.selection_summary_["candidate_metrics"]
    standardized_train = transformer.scaler_.transform(
        aggregate_train.loc[
            :,
            (*DEFAULT_INDEX_COLUMNS, "overall_pressure_index"),
        ].to_numpy(dtype=float)
    )
    figures = save_profile_figures(
        standardized_training=standardized_train,
        training_labels=train_ids,
        index_means=statistics.set_index("profile_id"),
        profile_statistics=statistics,
        candidate_metrics=candidate_metrics,
        profile_names=transformer.profile_names_,
        output_dir=profiles_dir / "figures",
        palette=project_config["sample_data"]["brand_palette"],
    )

    joblib.dump(
        {
            "aggregate_pipeline": aggregate_pipeline,
            "profile_transformer": transformer,
        },
        profiles_dir / "profile_pipeline.joblib",
    )
    joblib.dump(
        transformer.clusterer_,
        profiles_dir / "selected_cluster_model.joblib",
    )

    configured_sample = (
        project_root / project_config["feature_pipeline"]["inputs"]["sample"]
    )
    recovery = _hidden_type_recovery(
        profile_table,
        source,
        configured_sample,
        project_config,
        project_root,
    )
    ablation = run_profile_ablation(
        splits.data,
        splits.folds,
        profiles_dir / "ablation",
    )
    summary = {
        "run_id": current_run_id,
        "data_source": data_source,
        "cleaning": {
            "rows_in": cleaned.report["rows_in"],
            "rows_out": cleaned.report["rows_out"],
            "traffic_light": cleaned.report["traffic_light"],
        },
        "splits_path": str(splits.path),
        "fit_rows": len(train_rows),
        "descriptive_rows": len(profile_table),
        "chosen_k": transformer.n_profiles_,
        "chosen_algorithm": transformer.clusterer_.algorithm,
        "profile_names": list(transformer.profile_names_),
        "selection": transformer.selection_summary_,
        "hidden_type_recovery": recovery,
        "ablation": ablation,
        "profile_table": str(profiles_dir / "profile_feature_table.csv"),
        "profile_model": str(profiles_dir / "profile_pipeline.joblib"),
        "selected_cluster_model": str(profiles_dir / "selected_cluster_model.joblib"),
        "profile_statistics": statistics.to_dict(orient="records"),
        "figures": [str(path) for path in figures],
        "warning": splits.warning,
    }
    summary_path = profiles_dir / "summary.json"
    summary["summary_path"] = str(summary_path)
    summary_path.write_text(
        json.dumps(summary, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    return summary
