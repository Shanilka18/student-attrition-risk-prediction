"""Phase 5 CLI entry point for full evaluation pipeline.

Usage:
    python -m src.evaluation.run_evaluation [--tracks regression classification]
                                            [--no-test]
                                            [--seed 42]
"""

import argparse
import json
from pathlib import Path

from src.evaluation.leaderboard import run_evaluation, print_leaderboard
from src.utils.logger import get_logger
from src.utils.seed import set_seed

logger = get_logger("evaluation.run_evaluation")


def main() -> None:
    """CLI entrypoint for Phase 5 evaluation."""
    parser = argparse.ArgumentParser(
        description="PRI Phase 5: Full evaluation, leaderboard, and champion selection."
    )
    parser.add_argument(
        "--tracks", nargs="+",
        choices=["regression", "classification", "both"],
        default=["both"],
        help="Which track(s) to evaluate.",
    )
    parser.add_argument(
        "--no-test", action="store_true",
        help="Skip final test-set evaluation (validation-only mode).",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    tracks = (
        ["regression", "classification"]
        if "both" in args.tracks
        else args.tracks
    )

    set_seed(args.seed)

    print("\n" + "=" * 65)
    print("  PSYCHOSOCIAL RISK INDEX — PHASE 5 EVALUATION ENGINE")
    print(f"  Tracks:    {tracks}")
    print(f"  Test Set:  {'ENABLED — sealed test set will be touched ONCE' if not args.no_test else 'DISABLED (validation only)'}")
    print(f"  Seed:      {args.seed}")
    print("=" * 65 + "\n")

    results = run_evaluation(
        tracks=tracks,
        seed=args.seed,
        run_test_set=not args.no_test,
    )

    for track, out in results.items():
        lb_df = out["leaderboard"]
        print_leaderboard(lb_df, track)

        champion = out["champion"]
        print(f"\n  [CHAMPION] ({track.upper()}): {champion['champion']}")
        print(f"     {champion['justification']}")

        if "test_metrics" in champion:
            tm = champion["test_metrics"]
            if track == "regression":
                print(f"\n  FINAL TEST-SET RESULTS (champion={champion['champion']}):")
                print(f"     R^2         : {tm.get('r2', 'N/A')}")
                print(f"     RMSE        : {tm.get('rmse', 'N/A')}")
                print(f"     MAE         : {tm.get('mae', 'N/A')}")
                print(f"     Spearman r  : {tm.get('spearman_r', 'N/A')}")
                print(f"     Within +/-10: {tm.get('within_pm10_acc', 'N/A')}")
            else:
                print(f"\n  FINAL TEST-SET RESULTS (champion={champion['champion']}):")
                print(f"     Macro F1       : {tm.get('macro_f1', 'N/A')}")
                print(f"     Accuracy       : {tm.get('accuracy', 'N/A')}")
                print(f"     High-Risk Rec  : {tm.get('high_risk_recall', 'N/A')}")
                print(f"     MCC            : {tm.get('mcc', 'N/A')}")
                print(f"     ROC-AUC (OvR)  : {tm.get('roc_auc_ovr_macro', 'N/A')}")
                print(f"     ECE            : {tm.get('ece', 'N/A')}")

        if champion.get("statistical_comparisons"):
            print(f"\n  Statistical Comparisons (Wilcoxon, alpha=0.05):")
            for c in champion["statistical_comparisons"][:4]:
                sig = "SIGNIFICANT" if c.get("significant") else "NOT SIGNIFICANT"
                print(
                    f"     {champion['champion']} vs {c['challenger']:25s} "
                    f"p={c.get('p_value', 'N/A')}  [{sig}]"
                )

    print("\n  Artifacts saved:")
    print("    artifacts/leaderboards/  - CSV, JSON leaderboards + champion.json")
    print("    artifacts/models/final/  - PRI deployable bundle")
    print("    artifacts/plots/         - ROC, PR, calibration, confusion matrix\n")


if __name__ == "__main__":
    main()
