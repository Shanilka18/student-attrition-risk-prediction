"""Frontend JSON export pipeline for ERI Visual Dashboard.

Exports structured data for frontend rendering:
- artifacts/frontend_data/leaderboard.json
- artifacts/frontend_data/category_winners.json
- artifacts/frontend_data/all_splits_metrics.json
- artifacts/frontend_data/histories.json
- artifacts/frontend_data/curves.json
- artifacts/frontend_data/confusion_matrices.json
- artifacts/frontend_data/final_model_card.json
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from src.utils.logging_config import setup_logger
from src.utils.paths import get_artifacts_dir

logger = setup_logger("export_frontend")


def export_all_frontend_data(
    leaderboard_c: pd.DataFrame,
    leaderboard_r: pd.DataFrame,
    category_winners: Dict[str, Any],
    selection_decision: Dict[str, Any],
    audit_report: Dict[str, Any],
) -> Path:
    """Consolidate and write all JSON assets required by the web UI."""
    frontend_dir = get_artifacts_dir("frontend_data")
    frontend_dir.mkdir(parents=True, exist_ok=True)

    # 1. Leaderboard export
    lb_data = {
        "classification": leaderboard_c.to_dict(orient="records"),
        "regression": leaderboard_r.to_dict(orient="records") if not leaderboard_r.empty else [],
    }
    with open(frontend_dir / "leaderboard.json", "w", encoding="utf-8") as f:
        json.dump(lb_data, f, indent=2)

    # 2. Category winners
    with open(frontend_dir / "category_winners.json", "w", encoding="utf-8") as f:
        json.dump(category_winners, f, indent=2)

    # 3. All splits metrics
    metrics_csv = get_artifacts_dir("metrics") / "all_splits_metrics.csv"
    if metrics_csv.exists():
        df_metrics = pd.read_csv(metrics_csv)
        with open(frontend_dir / "all_splits_metrics.json", "w", encoding="utf-8") as f:
            json.dump(df_metrics.to_dict(orient="records"), f, indent=2)

    # 4. Histories
    histories = {}
    history_dir = get_artifacts_dir("history")
    if history_dir.exists():
        for h_file in history_dir.glob("*.json"):
            try:
                histories[h_file.stem] = json.loads(h_file.read_text(encoding="utf-8"))
            except Exception:
                pass
    with open(frontend_dir / "histories.json", "w", encoding="utf-8") as f:
        json.dump(histories, f, indent=2)

    # 5. Curves
    curves = {}
    curves_dir = get_artifacts_dir("curves")
    if curves_dir.exists():
        for c_file in curves_dir.glob("*.json"):
            try:
                curves[c_file.stem] = json.loads(c_file.read_text(encoding="utf-8"))
            except Exception:
                pass
    with open(frontend_dir / "curves.json", "w", encoding="utf-8") as f:
        json.dump(curves, f, indent=2)

    # 6. Confusion Matrices
    conf_matrices = {}
    # Read validation predictions for selected finalist
    final_model = selection_decision.get("selected_model", "logistic_l2")
    preds_dir = get_artifacts_dir("preds")
    val_pred_file = preds_dir / f"{final_model}_C_val_preds.npy"
    if val_pred_file.exists():
        val_probs = np.load(str(val_pred_file))
        from src.preprocessing.split import load_split_dataset
        _, val_df, _ = load_split_dataset()
        y_val = val_df["attrition"].values
        # Use optimal validation threshold if tuned
        tau = 0.5
        th_file = get_artifacts_dir("leaderboard") / "thresholds.json"
        if th_file.exists():
            try:
                with open(th_file, "r", encoding="utf-8") as tf:
                    td = json.load(tf)
                    if final_model in td:
                        tau = float(td[final_model].get("threshold", 0.5))
            except Exception:
                pass
        y_pred = (val_probs >= tau).astype(int)
        cm = confusion_matrix(y_val, y_pred)
        conf_matrices[final_model] = {
            "tn": int(cm[0, 0]),
            "fp": int(cm[0, 1]),
            "fn": int(cm[1, 0]),
            "tp": int(cm[1, 1]),
            "matrix": cm.tolist(),
        }
    with open(frontend_dir / "confusion_matrices.json", "w", encoding="utf-8") as f:
        json.dump(conf_matrices, f, indent=2)

    # 7. Final Model Card summary
    model_card = {
        "model_name": final_model,
        "selection_decision": selection_decision,
        "audit_summary": audit_report.get("disparity_summary", {}),
        "pipeline_version": "1.0.0",
        "intended_use": "Institutional early-warning for student environmental and financial risk.",
        "headline_metrics": leaderboard_c[leaderboard_c["model"] == final_model].to_dict(orient="records"),
    }
    with open(frontend_dir / "final_model_card.json", "w", encoding="utf-8") as f:
        json.dump(model_card, f, indent=2)

    logger.info(f"Frontend JSON assets successfully written to {frontend_dir}")
    return frontend_dir
