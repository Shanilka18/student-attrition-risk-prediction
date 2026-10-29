"""Build artifact-backed viva documentation metrics and publication figures."""

from __future__ import annotations

import base64
import json
import math
import re
import shutil
import struct
import zlib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
RUNS = ROOT / "artifacts" / "runs"
SOURCE_FIGURES = DOCS / "figures"
FINAL_FIGURES = SOURCE_FIGURES / "final"
NUMBER_FILE = DOCS / "_numbers.json"
SUMMARY_FILES = (DOCS / "RESULTS_SUMMARY.md", DOCS / "19_RESULTS_SUMMARY.md")
CHEAT_SHEET_FILE = DOCS / "23_PRESENTATION_GUIDE.md"
SOURCE_DPI = 300


def read_json(path: Path, *, required: bool = True) -> Any:
    if not path.is_file():
        if required:
            raise FileNotFoundError(
                f"Required documentation artifact is missing: {path}"
            )
        return None
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def active_run() -> tuple[str, Path]:
    pointer_path = ROOT / "artifacts" / "active_run.txt"
    if not pointer_path.is_file():
        raise FileNotFoundError(f"Active run pointer is missing: {pointer_path}")
    run_id = pointer_path.read_text(encoding="utf-8").strip()
    if not isinstance(run_id, str) or not re.fullmatch(r"run_[A-Za-z0-9_-]+", run_id):
        raise ValueError(
            "artifacts/active_run.txt does not contain a valid run identifier"
        )
    run_dir = (RUNS / run_id).resolve()
    if RUNS.resolve() not in run_dir.parents or not run_dir.is_dir():
        raise FileNotFoundError(f"Active run directory is unavailable: {run_id}")
    return run_id, run_dir


def get_nested(data: Any, *keys: str) -> Any:
    current = data
    for key in keys:
        if not isinstance(current, dict) or key not in current:
            return None
        current = current[key]
    return current


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, list):
        return [json_safe(item) for item in value]
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    return value


def as_rows(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        for key in ("models", "leaderboard", "results", "rows"):
            if isinstance(data.get(key), list):
                return [row for row in data[key] if isinstance(row, dict)]
    raise ValueError("Leaderboard artifact does not contain a model-row list")


def short_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "not recorded in the active-run artifacts"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, int):
        return f"{value:,}"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def md_cell(value: Any) -> str:
    if value is None:
        return "not recorded"
    return str(value).replace("|", "\\|").replace("\n", " ")


def summarize_category_results(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [row for row in data if isinstance(row, dict)]
    if isinstance(data, dict):
        rows: list[dict[str, Any]] = []
        for category, result in data.items():
            if isinstance(result, dict):
                rows.append({"category": category, **result})
        return rows
    return []


def read_quality_audit(
    run_dir: Path,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    training_report = read_json(
        run_dir / "data" / "data_quality_report.json", required=False
    )
    if training_report is not None:
        return training_report, None

    latest = read_json(
        ROOT / "data" / "interim" / "data_quality_report.json", required=False
    )
    if latest is None:
        return None, None
    return None, latest


def report_stats(
    run_id: str,
    run_dir: Path,
    classification_rows: list[dict[str, Any]],
    regression_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    model_card = read_json(run_dir / "champion" / "model_card.json")
    feature_manifest = read_json(run_dir / "champion" / "features.json")
    if not isinstance(feature_manifest, list) or not all(
        isinstance(feature, str) for feature in feature_manifest
    ):
        raise ValueError("Champion feature manifest must be a list of feature names")
    outcome_names = {
        "intent_leave",
        "intent_defer",
        "intent_not_continue",
        "attrition_intent_score",
        "mei",
    }
    leaked_features = [
        feature
        for feature in feature_manifest
        if feature.casefold() in outcome_names
        or re.match(r"^a[123](?:_|$)", feature.casefold())
    ]
    if leaked_features:
        raise ValueError(
            "Champion feature manifest contains outcome fields: "
            + ", ".join(leaked_features)
        )
    states = read_json(run_dir / "states" / "summary.json")
    importance = read_json(run_dir / "explain" / "global_importance.json")
    consistency = read_json(run_dir / "explain" / "shap_consistency.json")
    bands = read_json(run_dir / "champion" / "bands.json")
    split = read_json(run_dir / "splits" / "split_summary.json")
    cis = read_json(run_dir / "test_bootstrap_cis.json", required=False)
    class_categories = summarize_category_results(
        read_json(run_dir / "per_category_best.json")
    )
    regression_categories = summarize_category_results(
        read_json(run_dir / "per_category_best_regression.json")
    )
    training_quality, separate_quality = read_quality_audit(run_dir)

    classification_rows.sort(key=lambda row: row.get("rank", 10**9))
    regression_rows.sort(key=lambda row: row.get("rank", 10**9))
    if not classification_rows or not regression_rows:
        raise ValueError("Both active-run leaderboards must contain at least one model")
    champion = classification_rows[0] if classification_rows else {}
    regression_champion = regression_rows[0] if regression_rows else {}

    def has_recorded_test_metric(row: dict[str, Any], keys: tuple[str, ...]) -> bool:
        for key in keys:
            value = row.get(key)
            if value is not None and not (
                isinstance(value, str) and value.casefold().startswith("not evaluated")
            ):
                return True
        return False

    classification_test_metric_models = sorted(
        {
            str(row["model_id"])
            for row in classification_rows
            if row.get("model_id") != model_card.get("model_id")
            and has_recorded_test_metric(row, ("test_f1", "test_auc_roc", "test_mcc"))
        }
    )
    regression_test_metric_models = sorted(
        {
            str(row["model_id"])
            for row in regression_rows
            if row.get("model_id") != regression_champion.get("model_id")
            and has_recorded_test_metric(row, ("test_r2", "test_rmse", "test_mae"))
        }
    )

    source = str(model_card.get("data_source", ""))
    if not source:
        raise ValueError("Champion model card is missing data_source")
    if "survey" in source.casefold() and "sample" not in source.casefold():
        source_label = "survey"
    elif (
        "sample" in source.casefold()
        or "synthetic" in source.casefold()
        or "generated" in source.casefold()
    ):
        source_label = "sample (generated)"
    else:
        raise ValueError(f"Unrecognized model-card data_source: {source}")

    split_counts = {
        "train": get_nested(split, "train", "count"),
        "validation": get_nested(split, "val", "count"),
        "test": get_nested(split, "test", "count"),
    }
    if not all(isinstance(value, int) for value in split_counts.values()):
        raise ValueError("Split summary is missing train, validation, or test counts")

    total_samples = split.get("total_samples")
    if not isinstance(total_samples, int):
        total_samples = sum(split_counts.values())

    test_cis = {}
    if isinstance(cis, dict):
        test_cis = cis.get(f"track_a_champion_{model_card.get('model_id')}", {})
    if not test_cis:
        test_cis = model_card.get("bootstrap_95_ci", {})
    regression_test_cis = {}
    if isinstance(cis, dict):
        regression_test_cis = next(
            (
                result
                for key, result in cis.items()
                if key.startswith("track_b_champion_") and isinstance(result, dict)
            ),
            {},
        )

    shap_rows = get_nested(importance, "validation_ranking")
    if not isinstance(shap_rows, list):
        raise ValueError("SHAP global-importance artifact lacks validation_ranking")

    ablation_raw = get_nested(states, "ablation_benchmark")
    if not isinstance(ablation_raw, dict):
        raise ValueError("State summary lacks ablation_benchmark")
    ablation = ablation_raw.get("wilcoxon_tests")
    if not isinstance(ablation, dict):
        raise ValueError("State summary lacks ablation significance tests")
    clustering_raw = states.get("clustering")
    hmm_raw = states.get("hmm")
    if not isinstance(clustering_raw, dict) or not isinstance(hmm_raw, dict):
        raise ValueError("State summary lacks clustering or HMM results")

    training_clean_rows_in = None
    training_clean_rows_out = None
    training_removed = None
    training_missing_percent = None
    if training_quality:
        training_clean_rows_in = training_quality.get("rows_in")
        training_clean_rows_out = training_quality.get("rows_out")
        if isinstance(training_clean_rows_in, int) and isinstance(
            training_clean_rows_out, int
        ):
            training_removed = training_clean_rows_in - training_clean_rows_out
        training_missing_percent = get_nested(
            training_quality, "missingness_summary", "overall_missing_pct"
        )

    scoring_quality = None
    if separate_quality:
        scoring_quality = {
            "run_id": separate_quality.get("run_id"),
            "mode": separate_quality.get("mode"),
            "rows_in": separate_quality.get("rows_in"),
            "rows_out": separate_quality.get("rows_out"),
            "retention_rate": separate_quality.get("retention_rate"),
            "status": separate_quality.get("status"),
            "missing_percent": get_nested(
                separate_quality, "missingness_summary", "overall_missing_pct"
            ),
            "rows_flagged": sum(
                rule.get("rows_affected", 0)
                for rule in separate_quality.get("rules_applied", [])
                if isinstance(rule, dict)
                and rule.get("rule_id") == "R07_CARELESS_RESPONDING"
            ),
            "data_quality_report_is_for_training_run": separate_quality.get("run_id")
            == run_id,
        }

    return {
        "active_run_id": run_id,
        "data_source": source_label,
        "data_source_artifact_value": source,
        "sample_count": total_samples,
        "feature_manifest": {
            "feature_count": len(feature_manifest),
            "outcome_items_excluded": True,
            "outcome_fields_found": leaked_features,
        },
        "split_counts": split_counts,
        "test_at_risk_count": get_nested(split, "test", "at_risk_count"),
        "test_at_risk_percent": get_nested(split, "test", "at_risk_pct"),
        "test_evaluation_audit": {
            "marker_present": (run_dir / "TEST_USED.lock").is_file(),
            "classification_nonchampion_test_metric_models": classification_test_metric_models,
            "regression_nonchampion_test_metric_models": regression_test_metric_models,
        },
        "training_cleaning": {
            "report_available": training_quality is not None,
            "rows_in": training_clean_rows_in,
            "rows_out": training_clean_rows_out,
            "rows_removed": training_removed,
            "overall_missing_percent": training_missing_percent,
        },
        "separate_scoring_quality_audit": scoring_quality,
        "classification_model_count": len(classification_rows),
        "regression_model_count": len(regression_rows),
        "classification_leaderboard_top10": classification_rows[:10],
        "regression_leaderboard_top10": regression_rows[:10],
        "classification_leaderboard": classification_rows,
        "regression_leaderboard": regression_rows,
        "best_by_category_classification": class_categories,
        "best_by_category_regression": regression_categories,
        "champion": {
            "model_id": model_card.get("model_id"),
            "display_name": model_card.get("display_name"),
            "category": model_card.get("category"),
            "leaderboard_rank": champion.get("rank"),
            "cv_score": champion.get("leaderboard_score"),
            "validation_f1": champion.get("val_f1_optimal"),
            "validation_auc": champion.get("val_auc_roc"),
            "test_f1": model_card.get("test_f1", champion.get("test_f1")),
            "test_auc": model_card.get("test_auc_roc", champion.get("test_auc_roc")),
            "test_mcc": model_card.get("test_mcc", champion.get("test_mcc")),
            "leaderboard_threshold": champion.get("optimal_threshold"),
            "model_card_threshold": model_card.get("optimal_threshold"),
            "calibrator": model_card.get("calibrator"),
            "test_ci_f1": get_nested(test_cis, "f1"),
            "test_ci_auc": get_nested(test_cis, "auc_roc"),
            "test_ci_mcc": get_nested(test_cis, "mcc"),
            "test_cohort_size": model_card.get("test_cohort_size"),
            "training_cohort_size": model_card.get("training_cohort_size"),
        },
        "regression_champion": {
            "model_id": regression_champion.get("model_id"),
            "display_name": regression_champion.get("display_name"),
            "category": regression_champion.get("category"),
            "validation_r2": regression_champion.get("val_r2"),
            "validation_rmse": regression_champion.get("val_rmse"),
            "validation_mae": regression_champion.get("val_mae"),
            "test_r2": regression_champion.get("test_r2"),
            "test_rmse": regression_champion.get("test_rmse"),
            "test_mae": regression_champion.get("test_mae"),
            "test_r2_ci": regression_test_cis.get("r2"),
        },
        "calibration": {
            "method": model_card.get("calibrator"),
            "validation_ece": champion.get("val_ece"),
            "test_ece": champion.get("test_ece"),
            "test_calibration_metric_present": any(
                key in champion for key in ("test_ece", "test_brier", "test_log_loss")
            ),
        },
        "states": {
            "train_samples": states.get("n_train_samples"),
            "validation_samples": states.get("n_val_samples"),
            "test_samples": states.get("n_test_samples"),
            "clustering": {
                "selected_algorithm": clustering_raw.get("selected_algorithm"),
                "selected_k": clustering_raw.get("selected_k"),
                "silhouette": clustering_raw.get("silhouette"),
                "bootstrap_stability": clustering_raw.get("bootstrap_stability"),
                "cluster_names": clustering_raw.get("cluster_names"),
            },
            "hmm": {
                "selected_k": hmm_raw.get("selected_k"),
                "bic": hmm_raw.get("bic"),
                "converged": hmm_raw.get("converged"),
                "state_names": hmm_raw.get("state_names"),
            },
            "pattern_distribution": states.get("pattern_distribution"),
            "ablation_benchmark": ablation,
        },
        "risk_bands": {
            "definitions": [
                {
                    "label": definition.get("label"),
                    "min": definition.get("min"),
                    "max": definition.get("max"),
                }
                for definition in bands.get("definitions", {}).values()
                if isinstance(definition, dict)
            ],
            "test_holdout_verification": get_nested(
                bands, "test_holdout_verification", "bands"
            ),
            "strictly_monotonic": get_nested(
                bands, "test_holdout_verification", "is_strictly_monotonic"
            ),
            "monotonicity_status": get_nested(
                bands, "test_holdout_verification", "monotonicity_status"
            ),
        },
        "shap_top10": shap_rows[:10],
        "shap_consistency": consistency,
        "feature_leakage_note": "A1-A3 and named outcome fields are absent from the saved champion feature manifest.",
    }


def leaderboard_table(rows: list[dict[str, Any]], fields: list[tuple[str, str]]) -> str:
    lines = ["| " + " | ".join(label for label, _ in fields) + " |"]
    lines.append("| " + " | ".join("---" for _ in fields) + " |")
    for row in rows:
        lines.append(
            "| " + " | ".join(md_cell(row.get(key)) for _, key in fields) + " |"
        )
    return "\n".join(lines)


def category_table(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "_Per-category selection is not recorded in the active-run artifacts._"
    keys = list(dict.fromkeys(key for row in rows for key in row))
    chosen = [
        key
        for key in (
            "category",
            "model_id",
            "display_name",
            "leaderboard_score",
            "val_auc_roc",
            "val_r2",
        )
        if key in keys
    ]
    if not chosen:
        chosen = keys[:5]
    return leaderboard_table(
        rows, [(key.replace("_", " ").title(), key) for key in chosen]
    )


def confidence_interval(value: Any) -> str:
    if not isinstance(value, dict):
        return "not recorded"
    lower = value.get("ci_lower")
    upper = value.get("ci_upper")
    if lower is None or upper is None:
        return "not recorded"
    return f"[{short_number(lower)}, {short_number(upper)}]"


def risk_band_table(data: dict[str, Any]) -> str:
    rows = data.get("test_holdout_verification")
    if not isinstance(rows, list) or not rows:
        return "_Held-out risk-band verification is not recorded._"
    lines = [
        "| Band | Score range | Test students | Observed at-risk count | Observed at-risk percentage |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        if not isinstance(row, dict):
            continue
        lines.append(
            f"| {md_cell(row.get('band_label'))} | {md_cell(row.get('score_range'))} | "
            f"{short_number(row.get('n_students'))} | "
            f"{short_number(row.get('observed_at_risk_count'))} | "
            f"{short_number(row.get('observed_at_risk_pct'), 2)}% |"
        )
    return "\n".join(lines)


def ablation_tables(tests: dict[str, Any]) -> tuple[str, str]:
    auc_lines = [
        "| Model | AUC mean difference (states added minus aggregate) | p-value | Verdict |",
        "|---|---:|---:|---|",
    ]
    for model in ("logistic_regression", "lightgbm"):
        result = get_nested(tests, model, "auc") or {}
        p_value = result.get("p_value")
        if not isinstance(p_value, (int, float)):
            raise ValueError(f"Ablation AUC p-value is missing for {model}")
        verdict = (
            "not statistically significant"
            if p_value >= 0.05
            else "statistically significant"
        )
        auc_lines.append(
            f"| {model} | {short_number(result.get('mean_diff'))} | "
            f"{short_number(p_value)} | {verdict} |"
        )

    lgbm_mcc = get_nested(tests, "lightgbm", "mcc") or {}
    p_value = lgbm_mcc.get("p_value")
    if not isinstance(p_value, (int, float)):
        raise ValueError("Ablation MCC p-value is missing for lightgbm")
    verdict = (
        "statistically significant decline"
        if p_value < 0.05
        else "not statistically significant"
    )
    mcc = (
        "| Model | MCC mean difference (states added minus aggregate) | p-value | Verdict |\n"
        "|---|---:|---:|---|\n"
        f"| lightgbm | {short_number(lgbm_mcc.get('mean_diff'))} | "
        f"{short_number(p_value)} | {verdict} |"
    )
    return "\n".join(auc_lines), mcc


def results_markdown(data: dict[str, Any]) -> str:
    champion = data["champion"]
    regression = data["regression_champion"]
    quality = data["training_cleaning"]
    scoring = data["separate_scoring_quality_audit"]
    states = data["states"]
    calibration = data["calibration"]
    ablation = states["ablation_benchmark"]
    auc_ablation, mcc_ablation = ablation_tables(ablation)

    cleaning_summary = (
        f"Training data quality report records {short_number(quality['rows_in'])} input rows, "
        f"{short_number(quality['rows_out'])} output rows, and "
        f"{short_number(quality['rows_removed'])} removed rows."
        if quality["report_available"]
        else "A training-run cleaning report is absent, so raw input and dropped-row counts "
        "cannot be verified from the active run."
    )
    scoring_summary = (
        f"The separate scoring audit ({scoring['run_id']}) covers "
        f"{short_number(scoring['rows_in'])} rows: "
        f"{short_number(scoring['rows_flagged'])} were flagged for straight-lining, "
        f"{short_number(scoring['rows_out'])} remained in the scoring output, and feature "
        f"missingness was {short_number(scoring['missing_percent'], 2)}%. This is not the "
        "training-run cleaning report."
        if scoring
        else "No separate scoring audit is available."
    )
    test_audit = data["test_evaluation_audit"]
    nonchampion_test_models = (
        test_audit["classification_nonchampion_test_metric_models"]
        + test_audit["regression_nonchampion_test_metric_models"]
    )
    regression_ids = [
        row.get("model_id") for row in data["regression_leaderboard_top10"]
    ]
    regression_top_has_repeated_ids = len(regression_ids) != len(set(regression_ids))
    test_ece = (
        short_number(calibration["test_ece"])
        if calibration["test_ece"] is not None
        else "not recorded; do not infer test calibration quality from the validation ECE"
    )

    sections = [
        "> Summary: This is the generated, artifact-backed snapshot of the active run.\n"
        f"> Data source: {data['data_source']}.\n"
        f"> The active run contains {short_number(data['sample_count'])} split-cohort records.\n"
        f"> Test marker present: {short_number(test_audit['marker_present'])}; strict one-time "
        "use cannot be independently confirmed from saved artifacts.\n"
        "> Missing evidence is stated as unavailable, not estimated.",
        "# Results Summary",
        "",
        "_Generated by `scripts/build_docs_numbers.py`. Regenerate with `python -m mei.cli docs`._",
        "",
        f"**Data source: {data['data_source']}**",
        "",
        "## Data and sample sizes",
        "",
        f"- Active run: `{data['active_run_id']}`.",
        f"- Split-cohort total: {short_number(data['sample_count'])}.",
        f"- Train / validation / test: {short_number(data['split_counts']['train'])} / "
        f"{short_number(data['split_counts']['validation'])} / "
        f"{short_number(data['split_counts']['test'])}.",
        f"- Test at-risk count and percentage: {short_number(data['test_at_risk_count'])} / "
        f"{short_number(data['test_at_risk_percent'], 2)}%.",
        f"- Test-set use audit: `TEST_USED.lock` is "
        f"{'present' if test_audit['marker_present'] else 'absent'}, but saved leaderboard test "
        f"metrics exist for non-champion models: "
        f"`{json.dumps(nonchampion_test_models)}`. The marker alone does not establish that "
        "the test was accessed only once for a final frozen model.",
        "- Raw recruited/input cohort size: not recorded in the active-run artifacts.",
        f"- Champion feature manifest: {short_number(data['feature_manifest']['feature_count'])} "
        "features; direct outcome item names absent: "
        f"{short_number(data['feature_manifest']['outcome_items_excluded'])}.",
        "",
        "## Cleaning and missingness",
        "",
        f"- {cleaning_summary}",
        f"- {scoring_summary}",
        "",
        "## States and sequence model",
        "",
        f"- K-means clustering artifact: `{json.dumps(states['clustering'], sort_keys=True)}`.",
        f"- HMM artifact: `{json.dumps(states['hmm'], sort_keys=True)}`.",
        f"- Training pattern counts: `{json.dumps(states['pattern_distribution'], sort_keys=True)}`.",
        "",
        "## Classification leaderboard: top ten",
        "",
        leaderboard_table(
            data["classification_leaderboard_top10"],
            [
                ("Rank", "rank"),
                ("Model", "model_id"),
                ("Category", "category"),
                ("Feature set", "feature_set"),
                ("CV score", "leaderboard_score"),
                ("Validation F1", "val_f1_optimal"),
                ("Validation ROC AUC", "val_auc_roc"),
                ("Generalization gap", "generalization_gap"),
                ("Test F1", "test_f1"),
                ("Test ROC AUC", "test_auc_roc"),
            ],
        ),
        "",
        f"- Models in classification leaderboard: {short_number(data['classification_model_count'])}.",
        "",
        "## Regression leaderboard: top ten",
        "",
        (
            "- The table preserves repeated remediated model IDs as stored in the leaderboard; "
            "it is a row ranking, not a deduplicated model-family ranking."
            if regression_top_has_repeated_ids
            else "- The table preserves the active artifact's top rows."
        ),
        "",
        leaderboard_table(
            data["regression_leaderboard_top10"],
            [
                ("Rank", "rank"),
                ("Model", "model_id"),
                ("Category", "category"),
                ("Feature set", "feature_set"),
                ("CV score", "leaderboard_score"),
                ("Validation R-squared", "val_r2"),
                ("Validation RMSE", "val_rmse"),
                ("Validation MAE", "val_mae"),
                ("Test R-squared", "test_r2"),
                ("Test RMSE", "test_rmse"),
                ("Test MAE", "test_mae"),
            ],
        ),
        "",
        f"- Models in regression leaderboard: {short_number(data['regression_model_count'])}.",
        "",
        "## Best model by category",
        "",
        "### Classification",
        "",
        category_table(data["best_by_category_classification"]),
        "",
        "### Regression",
        "",
        category_table(data["best_by_category_regression"]),
        "",
        "## Champion, uncertainty, calibration, and risk bands",
        "",
        f"- Classification champion: **{md_cell(champion['display_name'])}** "
        f"(`{md_cell(champion['model_id'])}`); its cross-validation leaderboard threshold is "
        f"{short_number(champion['leaderboard_threshold'])}, while the model card records "
        f"{short_number(champion['model_card_threshold'])}. These artifact fields differ; "
        "the frozen bundle follows its model card and the discrepancy should be reconciled "
        "before another release.",
        f"- Test F1: {short_number(champion['test_f1'])}; bootstrap 95% CI "
        f"{confidence_interval(champion['test_ci_f1'])}.",
        f"- Test ROC AUC: {short_number(champion['test_auc'])}; bootstrap 95% CI "
        f"{confidence_interval(champion['test_ci_auc'])}.",
        f"- Test MCC: {short_number(champion['test_mcc'])}; bootstrap 95% CI "
        f"{confidence_interval(champion['test_ci_mcc'])}.",
        "| Calibration field | Result |",
        "|---|---|",
        f"| Method | {md_cell(calibration['method'])} |",
        f"| Validation ECE | {short_number(calibration['validation_ece'])} |",
        f"| Test ECE | {test_ece} |",
        "",
        f"Band monotonicity check: **{md_cell(data['risk_bands'].get('monotonicity_status'))}** "
        f"(strictly monotonic: {short_number(data['risk_bands'].get('strictly_monotonic'))}).",
        "",
        risk_band_table(data["risk_bands"]),
        "",
        "## Regression champion and generalization",
        "",
        f"- Champion: **{md_cell(regression['display_name'])}** (`{md_cell(regression['model_id'])}`).",
        f"- Validation R-squared / RMSE / MAE: {short_number(regression['validation_r2'])} / "
        f"{short_number(regression['validation_rmse'])} / {short_number(regression['validation_mae'])}.",
        f"- Test R-squared / RMSE / MAE: {short_number(regression['test_r2'])} / "
        f"{short_number(regression['test_rmse'])} / {short_number(regression['test_mae'])}.",
        f"- Test R-squared bootstrap 95% CI: {confidence_interval(regression['test_r2_ci'])}. "
        "A low test R-squared and an interval crossing zero do not support strong "
        "out-of-sample regression claims.",
        "",
        "## SHAP drivers and consistency",
        "",
        "| Rank | Feature | Mean absolute SHAP | Mean absolute points |",
        "|---:|---|---:|---:|",
    ]
    for rank, row in enumerate(data["shap_top10"], start=1):
        feature = row.get(
            "feature", row.get("feature_name", row.get("name", "not recorded"))
        )
        shap_value = row.get("mean_abs_shap", row.get("mean_absolute_shap"))
        points = row.get("mean_abs_points", row.get("mean_absolute_points"))
        sections.append(
            f"| {rank} | {md_cell(feature)} | {short_number(shap_value)} | {short_number(points)} |"
        )
    consistency = data["shap_consistency"]
    sections.extend(
        [
            "",
            f"- Across-fold SHAP consistency artifact: {short_number(consistency.get('n_folds'))} "
            f"folds; mean Spearman {short_number(consistency.get('mean_spearman'))}; "
            f"Kendall W {short_number(consistency.get('kendall_w'))}; mean top-ten Jaccard "
            f"{short_number(consistency.get('mean_top10_jaccard'))}. These values indicate "
            "weak rank stability; do not describe the explanations as robust or causal.",
            "",
            "## State-feature ablation verdict",
            "",
            "The Wilcoxon tests compare aggregate-plus-state features with aggregate-only features.",
            "",
            auc_ablation,
            "",
            mcc_ablation,
            "",
            "Neither tested classifier shows a statistically significant AUC gain from adding "
            "states. The LightGBM MCC comparison shows a statistically significant decline. "
            "These tests do not establish causality or a universal effect.",
            "",
            "## What changes when real data arrive",
            "",
            "The `survey` source label will replace `sample (generated)` only when the active "
            "model card identifies real survey data. Cohort sizes, cleaning counts, split "
            "metrics, intervals, state selection, bands, SHAP rankings, and leaderboard rows "
            "will be recalculated from the new run artifacts. Generated-data results must not "
            "be carried over or presented as real-survey evidence. Preserve the split-before-fit "
            "rule, fit transformations inside CV, exclude A1-A3 from predictors, and use the "
            "held-out test once.",
            "",
            "## Common mistakes",
            "",
            "- Do not confuse validation estimates with held-out test performance.",
            "- Do not merge the separate scoring audit with training cleaning counts.",
            "- Do not claim significance when the recorded p-value is at least 0.05.",
            "- Do not explain a threshold discrepancy as if the two artifacts agree.",
            "",
            "## If the examiner asks",
            "",
            "Point to the active-run artifacts and explain which file supports each claim. "
            "For missing training-cleaning counts or test calibration metrics, say they are "
            "not recorded instead of estimating them.",
            "",
        ]
    )
    return "\n".join(sections)


def png_with_300_dpi(source: Path, destination: Path) -> tuple[int, int]:
    blob = source.read_bytes()
    if not blob.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"Not a PNG file: {source}")
    pos = 8
    chunks: list[tuple[bytes, bytes]] = []
    dimensions: tuple[int, int] | None = None
    while pos < len(blob):
        length = struct.unpack(">I", blob[pos : pos + 4])[0]
        kind = blob[pos + 4 : pos + 8]
        data = blob[pos + 8 : pos + 8 + length]
        end = pos + 12 + length
        if kind == b"IHDR":
            dimensions = struct.unpack(">II", data[:8])
        if kind != b"pHYs":
            chunks.append((kind, data))
        pos = end
        if kind == b"IEND":
            break
    if dimensions is None:
        raise ValueError(f"PNG has no IHDR chunk: {source}")

    output = bytearray(b"\x89PNG\r\n\x1a\n")
    inserted = False
    for kind, data in chunks:
        if kind == b"IHDR" and not inserted:
            output.extend(_png_chunk(kind, data))
            output.extend(_png_chunk(b"pHYs", struct.pack(">IIB", 11811, 11811, 1)))
            inserted = True
        else:
            output.extend(_png_chunk(kind, data))
    destination.write_bytes(output)
    return dimensions


def _png_chunk(kind: bytes, data: bytes) -> bytes:
    body = kind + data
    return (
        struct.pack(">I", len(data))
        + body
        + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    )


def make_svg_wrapper(png_path: Path, svg_path: Path, width: int, height: int) -> None:
    encoded = base64.b64encode(png_path.read_bytes()).decode("ascii")
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width / SOURCE_DPI:.4f}in" '
        f'height="{height / SOURCE_DPI:.4f}in" viewBox="0 0 {width} {height}">'
        f'<image width="{width}" height="{height}" '
        f'href="data:image/png;base64,{encoded}"/></svg>\n'
    )
    svg_path.write_text(svg, encoding="utf-8")


def _index_descriptions() -> dict[str, str]:
    index = SOURCE_FIGURES / "INDEX.md"
    descriptions: dict[str, str] = {}
    if not index.is_file():
        return descriptions
    for line in index.read_text(encoding="utf-8").splitlines():
        if not line.startswith("|"):
            continue
        match = re.search(r"\[([^]]+\.png)\]\(", line)
        if not match:
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) >= 3:
            descriptions[match.group(1)] = cells[2]
    return descriptions


def figure_purpose(filename: str) -> tuple[str, str, str]:
    stem = Path(filename).stem
    words = stem.replace("_", " ").replace("-", " ")
    title = re.sub(r"\s+", " ", words).strip().capitalize()
    lowered = stem.casefold()
    if lowered.startswith("ui") or "screen" in lowered or "dashboard" in lowered:
        topic, use = (
            "interface screen or interaction",
            "Use as a static demo reference.",
        )
        how = (
            "Read the visible page labels and controls as interface elements; this screenshot "
            "is not model-performance evidence."
        )
        return title, f"This screenshot shows the {title} interface view.", how
    if lowered.startswith("eda"):
        topic, use = (
            "exploratory data patterns",
            "Use when introducing the sample and survey inputs.",
        )
    elif lowered.startswith("state") or "hmm" in lowered or "cluster" in lowered:
        topic, use = (
            "engagement-state or sequence structure",
            "Use when explaining state construction and interpretation.",
        )
    elif lowered.startswith("class") or "roc" in lowered or "pr_curve" in lowered:
        topic, use = (
            "classification model performance",
            "Use when presenting model comparison and classification evidence.",
        )
    elif lowered.startswith("reg"):
        topic, use = (
            "continuous-outcome regression evidence",
            "Use when discussing the weaker regression generalization.",
        )
    elif lowered.startswith("deep"):
        topic, use = (
            "deep-model training or validation diagnostics",
            "Use when comparing deep and simpler models.",
        )
    elif "shap" in lowered or "importance" in lowered:
        topic, use = (
            "feature attribution or explanation stability",
            "Use when discussing interpretation and its uncertainty.",
        )
    elif "calibr" in lowered or "reliability" in lowered:
        topic, use = (
            "probability calibration",
            "Use when explaining probability reliability and threshold choices.",
        )
    else:
        topic, use = (
            "the project figure named by the file",
            "Use alongside the corresponding method or result.",
        )
    how = (
        "Read the title, axes, units, and legend first; compare only the quantities shown, "
        "and treat the plot as descriptive rather than causal."
    )
    return title, f"This figure presents {topic}: {title}.", f"{how} {use}"


def export_figures() -> list[dict[str, str]]:
    sources = sorted(SOURCE_FIGURES.glob("*.png"))
    if not sources:
        raise FileNotFoundError(f"No source PNG figures found in {SOURCE_FIGURES}")
    FINAL_FIGURES.mkdir(parents=True, exist_ok=True)
    indexed = _index_descriptions()
    figures = []
    for source in sources:
        png = FINAL_FIGURES / source.name
        width, height = png_with_300_dpi(source, png)
        svg = FINAL_FIGURES / f"{source.stem}.svg"
        source_svg = SOURCE_FIGURES / f"{source.stem}.svg"
        if source_svg.is_file():
            shutil.copyfile(source_svg, svg)
            svg_provenance = "source SVG"
        else:
            make_svg_wrapper(png, svg, width, height)
            svg_provenance = "SVG wrapper around the 300-dpi PNG"
        title, inferred, how = figure_purpose(source.name)
        description = indexed.get(source.name, inferred)
        description = re.sub(r"\$[^$]*\$", "", description)
        description = re.sub(r"\([^)]*\d[^)]*\)", "", description)
        description = re.sub(
            r"\ball\s+\d+\s+indicators\b", "indicators", description, flags=re.I
        )
        description = re.sub(r"\b\d+(?:\.\d+)?\b", "", description)
        description = re.sub(r"\s+", " ", description).strip()
        figures.append(
            {
                "filename": source.name,
                "svg_filename": svg.name,
                "title": title,
                "description": description,
                "how_to_read": how,
                "svg_provenance": svg_provenance,
            }
        )
    return figures


def figure_catalog_markdown(figures: list[dict[str, str]]) -> str:
    rows = [
        "> Summary: This catalog lists every PNG figure in the project figure folder.\n"
        f"> It contains {len(figures)} figure entries.\n"
        "> Each final PNG is tagged at 300 dpi.\n"
        "> Existing vector SVGs are preserved; missing SVGs are documented raster wrappers.\n"
        "> Use the slide guidance to select figures and avoid overclaiming what they show.",
        "# Figure Catalog",
        "",
        "_Generated by `scripts/build_docs_numbers.py`._",
        "",
        "| Figure file | What it shows | How to read it; when to use it |",
        "|---|---|---|",
    ]
    for figure in figures:
        png_link = f"[{figure['filename']}](figures/final/{figure['filename']})"
        svg_link = f"[SVG](figures/final/{figure['svg_filename']})"
        rows.append(
            f"| {png_link} ({svg_link}) | {figure['description']} | "
            f"{figure['how_to_read']} ({figure['svg_provenance']}) |"
        )
    rows.extend(
        [
            "",
            "Final files live in `docs/figures/final/`. Raster exports carry 300-dpi metadata. "
            "A source SVG is retained when available. When no vector source exists, the SVG is "
            "an SVG container around the same 300-dpi PNG; it is not a newly redrawn vector plot.",
            "",
            "## Common mistakes",
            "",
            "- Do not read a plotted association as a cause.",
            "- Do not crop away axes, labels, or uncertainty intervals in slides.",
            "- Do not describe a raster-wrapper SVG as vectorized chart content.",
            "",
            "## If the examiner asks",
            "",
            "Explain the axis and comparison first. Then state the limitation shown by the figure "
            "and connect it to the relevant result artifact.",
            "",
        ]
    )
    return "\n".join(rows)


def cheat_sheet_section(data: dict[str, Any]) -> str:
    champion = data["champion"]
    regression = data["regression_champion"]
    clustering = data["states"].get("clustering") or {}
    consistency = data["shap_consistency"]
    items = [
        ("Split-cohort total", short_number(data["sample_count"])),
        ("Training cohort", short_number(data["split_counts"]["train"])),
        ("Validation cohort", short_number(data["split_counts"]["validation"])),
        ("Test cohort", short_number(data["split_counts"]["test"])),
        ("Test at-risk percentage", short_number(data["test_at_risk_percent"], 2)),
        ("Champion CV leaderboard score", short_number(champion["cv_score"])),
        ("Classification test F1", short_number(champion["test_f1"])),
        ("Classification test F1 95% CI", confidence_interval(champion["test_ci_f1"])),
        ("Classification test ROC AUC", short_number(champion["test_auc"])),
        ("Regression test R-squared", short_number(regression["test_r2"])),
        ("K-means clusters", short_number(clustering.get("selected_k"))),
        ("SHAP mean Spearman", short_number(consistency.get("mean_spearman"))),
    ]
    return "\n".join(
        ["| Remember | Current artifact value |", "|---|---|"]
        + [f"| {label} | {md_cell(value)} |" for label, value in items]
    )


def update_cheat_sheet(data: dict[str, Any]) -> None:
    if not CHEAT_SHEET_FILE.is_file():
        raise FileNotFoundError(
            f"Presentation guide is required before building docs: {CHEAT_SHEET_FILE}"
        )
    text = CHEAT_SHEET_FILE.read_text(encoding="utf-8")
    start = "<!-- GENERATED CHEAT SHEET START -->"
    end = "<!-- GENERATED CHEAT SHEET END -->"
    if text.count(start) != 1 or text.count(end) != 1:
        raise ValueError(
            "Presentation guide must contain one pair of generated cheat-sheet markers"
        )
    before, remainder = text.split(start, 1)
    _, after = remainder.split(end, 1)
    text = before + start + "\n" + cheat_sheet_section(data) + "\n" + end + after
    CHEAT_SHEET_FILE.write_text(text, encoding="utf-8")


def main() -> None:
    run_id, run_dir = active_run()
    classification = as_rows(read_json(run_dir / "leaderboard_classification.json"))
    regression = as_rows(read_json(run_dir / "leaderboard_regression.json"))
    numbers = json_safe(report_stats(run_id, run_dir, classification, regression))
    figures = export_figures()
    numbers["figure_count"] = len(figures)

    DOCS.mkdir(parents=True, exist_ok=True)
    NUMBER_FILE.write_text(
        json.dumps(numbers, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    rendered = results_markdown(numbers)
    for path in SUMMARY_FILES:
        path.write_text(rendered, encoding="utf-8")
    (DOCS / "20_FIGURE_CATALOG.md").write_text(
        figure_catalog_markdown(figures), encoding="utf-8"
    )
    update_cheat_sheet(numbers)
    print(
        f"Built artifact-backed docs for {run_id}: "
        f"{len(classification)} classification models, {len(regression)} regression models, "
        f"{len(figures)} figures."
    )


if __name__ == "__main__":
    main()
