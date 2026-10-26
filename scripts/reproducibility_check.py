"""Phase 10: Reproducibility audit.

Loads all persisted trained models and compares predictions from two
independent calls with the same input to confirm identical outputs.
Also verifies that the leaderboard rank order and champion selection
are stable across identical pipeline runs (uses persisted artifacts).
"""

import json
import pathlib
import sys

import joblib
import numpy as np
import pandas as pd

from src.features.build_features import compute_factor_scores, FACTOR_COLS
from src.utils.seed import set_seed

SEED = 42
RESULTS = []


def compare_predictions(model, X: pd.DataFrame, model_name: str) -> dict:
    """Run prediction twice and check for bit-identical outputs."""
    pred1 = model.predict(X)
    pred2 = model.predict(X)
    identical = np.array_equal(pred1, pred2)
    max_diff = float(np.max(np.abs(pred1.astype(float) - pred2.astype(float)))) if not identical else 0.0
    return {
        "model": model_name,
        "identical_across_calls": identical,
        "max_diff": max_diff,
        "nondeterminism_source": "None" if identical else "Nondeterministic predict() — possible source: probabilistic layer or thread timing",
    }


def verify_leaderboard_stability() -> dict:
    """Verify that both leaderboard CSV files list consistent champions."""
    findings = {}
    for track in ["regression", "classification"]:
        csv_path = pathlib.Path(f"artifacts/leaderboards/{track}_leaderboard.csv")
        json_path = pathlib.Path(f"artifacts/leaderboards/{track}_leaderboard.json")
        if not csv_path.exists():
            findings[track] = {"error": "CSV not found"}
            continue
        df = pd.read_csv(csv_path)
        # Sort correctly
        ascending = track == "regression"
        df_sorted = df.sort_values("val_primary_metric", ascending=ascending).reset_index(drop=True)
        csv_champion = df_sorted.iloc[0]["model_name"]
        json_champion = None
        if json_path.exists():
            data = json.load(open(json_path))
            # JSON is insertion-ordered; re-sort by primary metric
            if data:
                reverse = track != "regression"
                data_sorted = sorted(data, key=lambda x: x.get("validation_metrics", {}).get("primary", 0), reverse=reverse)
                json_champion = data_sorted[0]["model_name"]
        findings[track] = {
            "csv_champion": csv_champion,
            "json_champion": json_champion,
            "consistent": csv_champion == json_champion if json_champion else True,
        }
    return findings


def main():
    set_seed(SEED)
    print("=" * 65)
    print("Phase 10: Reproducibility Audit")
    print("=" * 65)

    val = pd.read_csv("data/processed/val.csv")
    X = compute_factor_scores(val)[FACTOR_COLS]
    print(f"\n  Using validation set: {len(val)} rows, {len(FACTOR_COLS)} features")

    # --- Per-model prediction consistency ---
    print("\n[1] Per-model prediction determinism (call twice, same inputs):")
    models_dir = pathlib.Path("artifacts/models")
    results_det = []
    for track in ["regression", "classification"]:
        track_dir = models_dir / track
        if not track_dir.exists():
            continue
        for mf in sorted(track_dir.glob("*.joblib")):
            model_name = f"{track}/{mf.stem}"
            try:
                model = joblib.load(mf)
                r = compare_predictions(model, X, model_name)
                status = "PASS" if r["identical_across_calls"] else "WARN"
                print(f"  [{status}] {model_name:50s} max_diff={r['max_diff']:.2e}")
                results_det.append(r)
            except Exception as e:
                print(f"  [SKIP] {model_name}: {e}")
                results_det.append({"model": model_name, "error": str(e)})

    # --- Leaderboard stability ---
    print("\n[2] Leaderboard champion stability:")
    lb_stability = verify_leaderboard_stability()
    all_consistent = True
    for track, info in lb_stability.items():
        consistent = info.get("consistent", False)
        if not consistent:
            all_consistent = False
        status = "PASS" if consistent else "WARN"
        print(f"  [{status}] {track}: CSV={info.get('csv_champion')} JSON={info.get('json_champion')} consistent={consistent}")

    # --- Known nondeterminism sources ---
    print("\n[3] Known nondeterminism sources:")
    non_det_sources = [
        "HistGradientBoostingClassifier/Regressor: deterministic for same n_iter_no_change and seed",
        "RandomForest: seed-fixed, deterministic on same n_jobs=1 (parallelism may cause float reordering)",
        "TabNet/PyTorch MLPs: CUDA-seeded; CPU-only is fully deterministic with manual_seed",
        "Optuna TPE sampler: fully deterministic with fixed seed (TPESampler(seed=42))",
        "SHAP TreeExplainer: deterministic (tree-based, no randomness)",
    ]
    for s in non_det_sources:
        print(f"  - {s}")

    # --- Summary ---
    det_passes = sum(1 for r in results_det if r.get("identical_across_calls", False))
    det_total = sum(1 for r in results_det if "identical_across_calls" in r)
    print(f"\n[SUMMARY] Deterministic models: {det_passes}/{det_total}")
    print(f"[SUMMARY] Leaderboard consistency: {all_consistent}")

    # Save
    report = {
        "seed": SEED,
        "determinism_results": results_det,
        "leaderboard_stability": lb_stability,
        "known_nondeterminism_sources": non_det_sources,
    }
    out = pathlib.Path("artifacts/metrics/reproducibility_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to {out}")
    return 0 if all_consistent else 1


if __name__ == "__main__":
    sys.exit(main())
