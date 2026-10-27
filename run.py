"""
Unified Pipeline and Project Runner for Academic Intelligence Framework.
Usage:
    python run.py setup
    python run.py test
    python run.py data
    python run.py preprocess
    python run.py train
    python run.py evaluate
    python run.py explain
    python run.py serve
    python run.py frontend
    python run.py all
"""

import sys
import subprocess
import argparse
from pathlib import Path
import logging
import platform
from typing import Optional, List, Dict, Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("Runner")

PROJECT_ROOT = Path(__file__).resolve().parent


def _npm_cmd() -> str:
    """Return 'npm.cmd' on Windows (needed for subprocess without shell=True), 'npm' elsewhere."""
    return "npm.cmd" if platform.system() == "Windows" else "npm"


def run_cmd(cmd: list[str], cwd: Path = PROJECT_ROOT) -> int:
    """Executes a command and returns the exit code."""
    cmd_str = " ".join(cmd)
    logger.info(f"Executing: {cmd_str} (in {cwd})")
    res = subprocess.run(cmd, cwd=str(cwd))
    return res.returncode


def target_setup() -> int:
    """Verifies runtime environment, directories, and config."""
    logger.info("=== Running Target: setup ===")
    from backend.src.config import config
    logger.info(f"Verified Project ID: {config.project_id}")
    logger.info(f"Active Device: {config.device}")
    logger.info(f"Global Seed: {config.global_seed}")
    
    # Ensure all directories exist
    dirs = [
        config.raw_data_dir,
        config.synthetic_data_dir,
        config.processed_data_dir,
        config.splits_dir,
        config.uploads_dir,
        config.models_dir,
        config.metrics_dir,
        config.leaderboard_dir,
        config.plots_dir,
        config.explanations_dir,
        config.docs_dir,
    ]
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        logger.info(f"Verified directory: {d}")

    # Check node dependencies if frontend package.json exists
    frontend_pkg = PROJECT_ROOT / "frontend" / "package.json"
    if frontend_pkg.exists():
        logger.info("Installing frontend dependencies...")
        npm = _npm_cmd()
        rc = run_cmd([npm, "install"], cwd=PROJECT_ROOT / "frontend")
        if rc != 0:
            logger.warning(
                "npm install returned a non-zero exit code. "
                "Frontend build may fail. Ensure Node.js >= 18 is installed."
            )

    logger.info("Setup target completed successfully.")
    return 0


def target_test() -> int:
    """Executes pytest suite on backend/tests."""
    logger.info("=== Running Target: test ===")
    return run_cmd([sys.executable, "-m", "pytest", "backend/tests", "-v"])


def target_data() -> int:
    """Generates synthetic student cohort data, task samples, naive baseline check, and audit plots."""
    logger.info("=== Running Target: data (Phase 2) ===")
    from backend.src.config import config
    from backend.src.data.generator import LatentStudentGenerator
    from backend.src.data.task import create_prefix_task_samples
    from backend.src.evaluation.naive_baseline import evaluate_naive_gpa_cutoff
    from backend.src.data.audit import run_data_audit_suite
    import json

    # 1. Main Headline Dataset: 5,000 students (realistic preset)
    gen = LatentStudentGenerator(seed=config.global_seed, preset="realistic")
    df_long, df_outcomes, meta = gen.generate_cohort(n_students=5000, max_semesters=8)

    long_path = config.synthetic_data_dir / "academic_long.csv"
    outcomes_path = config.synthetic_data_dir / "outcomes.csv"
    df_long.to_csv(long_path, index=False)
    df_outcomes.to_csv(outcomes_path, index=False)
    logger.info(f"Saved headline synthetic cohort: {long_path} ({len(df_long)} rows), {outcomes_path} ({len(df_outcomes)} students)")

    # 2. Small variant: 1,000 students for rapid tests
    small_dir = config.synthetic_data_dir / "small"
    small_dir.mkdir(parents=True, exist_ok=True)
    gen_small = LatentStudentGenerator(seed=config.global_seed, preset="realistic")
    df_long_sm, df_outcomes_sm, _ = gen_small.generate_cohort(n_students=1000, max_semesters=8)
    df_long_sm.to_csv(small_dir / "academic_long.csv", index=False)
    df_outcomes_sm.to_csv(small_dir / "outcomes.csv", index=False)
    logger.info(f"Saved small variant cohort: {small_dir}")

    # 3. Create Prefix Task Samples (t in [2, 3, 4, 5, 6])
    df_task = create_prefix_task_samples(df_long, df_outcomes, min_window=2, max_window=6)
    task_path = config.synthetic_data_dir / "task_samples.csv"
    df_task.to_csv(task_path, index=False)
    logger.info(f"Saved task prefix samples: {task_path} ({len(df_task)} window samples)")

    # 4. Evaluate Naive Baseline (GPA < 2.0 cutoff)
    naive_results = evaluate_naive_gpa_cutoff(df_long, df_task, gpa_threshold=2.0)
    naive_path = config.metrics_dir / "naive_baseline_metrics.json"
    naive_path.parent.mkdir(parents=True, exist_ok=True)
    with open(naive_path, "w", encoding="utf-8") as f:
        json.dump(naive_results, f, indent=2)
    logger.info(f"Saved naive baseline evaluation to: {naive_path}")
    print("\n--- MOTIVATING NAIVE BASELINE RESULT ---")
    print(f"Rule: {naive_results['rule']}")
    print(f"Accuracy:  {naive_results['accuracy']:.4f}")
    print(f"Precision: {naive_results['precision']:.4f}")
    print(f"Recall:    {naive_results['recall']:.4f}  <-- Mediocre Recall confirms research motivation!")
    print(f"F1-Score:  {naive_results['f1_score']:.4f}")
    print(f"Details:   {naive_results['motivating_finding']}\n")

    # 5. Run Data Audit & Visualization Suite
    audit_plots_dir = config.plots_dir / "data_audit"
    run_data_audit_suite(df_long, df_outcomes, meta, audit_plots_dir)
    logger.info(f"Completed Phase 2 data generation and auditing successfully.")
    return 0


def target_preprocess() -> int:
    """Preprocesses academic sequences, enforces grouped splits, and builds feature representations."""
    logger.info("=== Running Target: preprocess (Phase 3) ===")
    from backend.src.config import config
    from backend.src.data.splits import create_stratified_group_splits, create_temporal_split
    from backend.src.features.preprocessor import AcademicPreprocessor
    from backend.src.features.builder import (
        build_tabular_engineered_features,
        build_static_snapshot_features,
        build_sequence_tensors
    )
    from backend.src.evaluation.static_baseline import evaluate_static_snapshot_baseline
    import pandas as pd
    import torch
    import json

    # 1. Load Synthetic Cohort
    long_path = config.synthetic_data_dir / "academic_long.csv"
    outcomes_path = config.synthetic_data_dir / "outcomes.csv"
    task_path = config.synthetic_data_dir / "task_samples.csv"

    if not long_path.exists() or not outcomes_path.exists() or not task_path.exists():
        logger.warning("Synthetic dataset not found. Running target_data first...")
        target_data()

    df_long = pd.read_csv(long_path)
    df_outcomes = pd.read_csv(outcomes_path)
    df_task = pd.read_csv(task_path)

    # 2. Generate Leakage-Safe Splits
    logger.info("Step 1: Generating grouped stratified and temporal splits...")
    split_stats = create_stratified_group_splits(
        df_outcomes, config.splits_dir, seed=config.global_seed
    )
    temporal_stats = create_temporal_split(df_outcomes, config.splits_dir)

    with open(config.splits_dir / "train_students.json", "r", encoding="utf-8") as f:
        train_sids = json.load(f)
    with open(config.splits_dir / "test_students.json", "r", encoding="utf-8") as f:
        test_sids = json.load(f)

    # 3. Fit Preprocessor STRICTLY on Training Students
    logger.info("Step 2: Fitting preprocessor on training partition records...")
    df_train_long = df_long[df_long["student_id"].isin(train_sids)]
    preprocessor = AcademicPreprocessor()
    preprocessor.fit(df_train_long)
    preprocessor.save(config.models_dir / "preprocessor.joblib")

    # Transform all records using fitted preprocessor
    df_clean_long = preprocessor.transform(df_long)

    # 4. Build Tabular Engineered Features
    logger.info("Step 3: Extracting longitudinal trend features (Tabular)...")
    df_tabular = build_tabular_engineered_features(df_clean_long, df_task)
    tab_csv_path = config.processed_data_dir / "tabular_features.csv"
    df_tabular.to_csv(tab_csv_path, index=False)
    try:
        df_tabular.to_parquet(config.processed_data_dir / "tabular_features.parquet", index=False)
    except Exception:
        pass
    logger.info(f"Saved tabular features to: {tab_csv_path} (Shape: {df_tabular.shape})")

    # 5. Build Static Snapshot Features (Proposal Baseline Comparator)
    logger.info("Step 4: Building static snapshot baseline features...")
    df_static = build_static_snapshot_features(df_tabular)
    static_csv_path = config.processed_data_dir / "static_features.csv"
    df_static.to_csv(static_csv_path, index=False)
    logger.info(f"Saved static snapshot features to: {static_csv_path}")

    # 6. Build Sequence Tensors for PyTorch Models
    logger.info("Step 5: Building 3D padded sequence tensors and masks...")
    seq_payload = build_sequence_tensors(df_clean_long, df_task, preprocessor, max_window=6)
    seq_path = config.processed_data_dir / "sequence_tensors.pt"
    torch.save(seq_payload, seq_path)
    logger.info(f"Saved sequence tensors to: {seq_path} (Tensor Shape: {seq_payload['tensors'].shape})")

    # 7. Evaluate Static Snapshot Logistic Regression Comparator
    logger.info("Step 6: Running static snapshot baseline benchmark...")
    static_results = evaluate_static_snapshot_baseline(
        df_static, train_sids, test_sids,
        config.metrics_dir / "static_snapshot_benchmark.json",
        seed=config.global_seed
    )
    print("\n--- STATIC SNAPSHOT BASELINE RESULT (PROPOSAL COMPARATOR) ---")
    print(f"Model:     {static_results['model']}")
    print(f"Features:  {static_results['features_used']}")
    print(f"ROC-AUC:   {static_results['roc_auc']:.4f}")
    print(f"Recall:    {static_results['recall']:.4f}")
    print(f"Precision: {static_results['precision']:.4f}")
    print(f"F1-Score:  {static_results['f1_score']:.4f}")
    print(f"Accuracy:  {static_results['accuracy']:.4f}\n")

    logger.info("Phase 3 Preprocessing and Feature Extraction completed successfully.")
    return 0


def target_train(
    budget: str = "quick",
    force: bool = False,
    models: Optional[List[str]] = None,
    categories: Optional[List[str]] = None,
) -> int:
    """Trains, tunes, calibrates, and validates all registered models (Phase 5)."""
    logger.info("=== Running Target: train (Phase 5 Training & Validation Engine) ===")
    from backend.src.training.orchestrator import run_training_pipeline

    result = run_training_pipeline(
        budget=budget,
        force=force,
        target_models=models,
        target_categories=categories,
    )
    if result["n_failed"] > 0 and result["n_completed"] == 0:
        logger.error(f"Training pipeline finished with 0 completed and {result['n_failed']} failed models.")
        return 1

    logger.info(
        f"Training pipeline finished: {result['n_completed']} completed, "
        f"{result['n_failed']} failed out of {result['n_models']} models."
    )
    return 0


def target_evaluate() -> int:
    """Phase 6: Final one-touch test-set evaluation, leaderboard, and champion selection."""
    logger.info("=== Running Target: evaluate (Phase 6 Final Evaluation) ===")
    from backend.src.evaluation.evaluator import run_full_evaluation
    from backend.src.evaluation.leaderboard import build_leaderboard

    # Step 1: Evaluate every trained model ONCE on the held-out test set
    eval_results = run_full_evaluation()

    n_ok = len(eval_results.get("models", {}))
    n_fail = len(eval_results.get("failures", []))
    logger.info(f"Evaluation complete: {n_ok} succeeded, {n_fail} failed")

    if n_ok == 0:
        logger.error("No models were successfully evaluated. Aborting leaderboard.")
        return 1

    # Step 2: Build leaderboard, select champion, answer research question
    lb_results = build_leaderboard(eval_results)

    # Step 3: Print summary
    # Step 3: Run 4-stage progressive modality ablation on sequence champion
    try:
        from backend.src.evaluation.ablation import run_champion_ablation_study
        from backend.src.training.data_loader import TrainingDataLoader
        from backend.src.evaluation.evaluator import _load_sequence_test, _load_student_ids
        from backend.src.config import config
        import pandas as pd

        seq_champ = eval_results.get("key_models", {}).get("sequence_champion", "gru")
        logger.info(f"Running 4-stage modality ablation for sequence champion '{seq_champ}'...")

        loader = TrainingDataLoader()
        seq_train_val = loader.load_sequence_data()
        test_sids = _load_student_ids("test")
        X_seq_test, y_seq_test, df_seq_meta = _load_sequence_test(test_sids)
        df_outcomes = pd.read_csv(config.root_dir / "data" / "synthetic" / "outcomes.csv")

        ablation_res = run_champion_ablation_study(
            champion_key=seq_champ,
            train_data={"X_train": seq_train_val["X_train"], "y_train": seq_train_val["y_train"]},
            val_data={"X_val": seq_train_val["X_val"], "y_val": seq_train_val["y_val"]},
            test_data={"X_test": X_seq_test, "y_test": y_seq_test},
            df_test_samples=df_seq_meta,
            df_outcomes=df_outcomes,
            output_dir=config.leaderboard_dir,
            epochs=5,
            seed=config.global_seed,
        )
        logger.info(f"Modality ablation complete: {ablation_res.get('conclusion')}")
    except Exception as e:
        logger.warning(f"Modality ablation skipped or encountered error: {e}", exc_info=True)

    # Step 4: Print summary
    champion = lb_results.get("champion")
    rq = lb_results.get("research_question_answer")

    print("\n" + "=" * 72)
    print("PHASE 6 FINAL EVALUATION SUMMARY (SYNTHETIC DATA)")
    print("=" * 72)

    if champion:
        tm = champion.get("test_metrics", {})
        print(f"\n[CHAMPION MODEL]: {champion['model_key']}")
        print(f"   Category:       {champion['category']} ({champion['label']})")
        print(f"   ROC-AUC:        {tm.get('roc_auc', 0):.4f}")
        print(f"   PR-AUC:         {tm.get('pr_auc', 0):.4f}")
        print(f"   F2 Score:       {tm.get('f2_score', 0):.4f}")
        print(f"   Recall:         {tm.get('recall', 0):.4f}")
        print(f"   Brier Score:    {tm.get('brier_score', 0):.4f}")
        print(f"   ECE:            {tm.get('ece', 0):.4f}")
        print(f"   Lead Time:      {tm.get('lead_time_semesters', 0):.1f} semesters")
        print(f"   Inference:      {champion.get('efficiency', {}).get('inference_time_ms_per_student', 0):.1f} ms/student")

    if rq:
        print(f"\n[RESEARCH QUESTION ANSWER]: {rq.get('answer', 'N/A')}")
        print(f"   {rq.get('practical_significance', '')}")
        print(f"\n   [ASSESSMENT]: {rq.get('honest_assessment', '')}")

    # Category champions
    cats = lb_results.get("category_champions", {})
    if cats:
        print("\n[CATEGORY CHAMPIONS]:")
        for cat, info in sorted(cats.items()):
            print(f"   Category {cat}: {info['model_key']} (ROC-AUC={info['test_roc_auc']:.4f})")

    print("\n" + "=" * 72)
    logger.info("Phase 6 evaluation completed successfully.")
    return 0


def target_explain() -> int:
    """Phase 7: Generates SHAP attributions, temporal attention, risk bands, and Fusion contract."""
    logger.info("=== Running Target: explain (Phase 7 Explainability Layer) ===")
    from backend.src.explain.orchestrator import run_explainability_pipeline
    import json

    result = run_explainability_pipeline()

    print("\n" + "=" * 72)
    print("PHASE 7 EXPLAINABILITY & RISK INDEX SUMMARY (SYNTHETIC DATA)")
    print("=" * 72)
    print(f"\n[CHAMPION EXPLAINED]: {result['champion_model']}")
    print(f"   Total Students Explained: {result['n_students_explained']}")
    print(f"   Execution Runtime:        {result['runtime_seconds']}s")

    bs = result.get("band_summary", {})
    if bs:
        pcts = bs.get("band_percentages", {})
        print(f"\n[CLINICAL RISK BANDS]:")
        for band, pct in pcts.items():
            print(f"   - {band:<10}: {pct:.1f}% of cohort (Count: {bs['band_counts'].get(band, 0):,})")

    faith = result.get("faithfulness", {})
    if faith:
        print(f"\n[EXPLANATION FAITHFULNESS]:")
        print(f"   Monotonic Risk Drop:     {faith.get('is_monotonic_decrease')}")
        print(f"   Comprehensiveness Score: {faith.get('comprehensiveness_score')}")
        print(f"   Conclusion:              {faith.get('conclusion')}")

    stab = result.get("stability", {})
    if stab:
        print(f"\n[EXPLANATION STABILITY]:")
        print(f"   Rank Stability (Spearman rho): {stab.get('mean_spearman_rank_correlation')} (Threshold >= {stab.get('stability_threshold')})")
        print(f"   Is Stable:                    {stab.get('is_stable')}")

    print(f"\n[FUSION FRAMEWORK CONTRACT]:")
    print(f"   Export Destination:  {result.get('fusion_json')}")
    print(f"   Sample Contract:     {result.get('sample_fusion_json')}")

    print(f"\n[EXPLAINABILITY PLOTS]:")
    for p in result.get("plots", []):
        print(f"   - artifacts/plots/explainability/{p}")

    print("\n" + "=" * 72)
    logger.info("Phase 7 explainability completed successfully.")
    return 0


def target_serve() -> int:
    """Starts the FastAPI backend server using Uvicorn."""
    logger.info("=== Running Target: serve (FastAPI backend) ===")
    return run_cmd([
        sys.executable, "-m", "uvicorn", "backend.src.api.main:app",
        "--host", "127.0.0.1", "--port", "8000", "--reload"
    ])


def target_frontend() -> int:
    """Starts the React frontend development server."""
    logger.info("=== Running Target: frontend ===")
    return run_cmd([_npm_cmd(), "run", "dev"], cwd=PROJECT_ROOT / "frontend")


def target_all() -> int:
    """Runs setup, test, and all pipeline stages sequentially."""
    logger.info("=== Running Target: all ===")
    for stage_func in [target_setup, target_test, target_data, target_preprocess, target_train, target_evaluate, target_explain]:
        code = stage_func()
        if code != 0:
            logger.error(f"Stage {stage_func.__name__} failed with exit code {code}")
            return code
    logger.info("All pipeline targets completed successfully.")
    return 0


TARGETS = {
    "setup": target_setup,
    "test": target_test,
    "data": target_data,
    "preprocess": target_preprocess,
    "train": target_train,
    "evaluate": target_evaluate,
    "explain": target_explain,
    "serve": target_serve,
    "frontend": target_frontend,
    "all": target_all,
}


def main():
    parser = argparse.ArgumentParser(description="Academic Intelligence Framework Command Runner")
    parser.add_argument("target", choices=list(TARGETS.keys()), default="test", nargs="?", help="Target stage to run")
    parser.add_argument("--budget", choices=["quick", "standard", "full"], default="quick", help="Hyperparameter tuning budget")
    parser.add_argument("--force", action="store_true", help="Force retrain models, skipping cache")
    parser.add_argument("--models", nargs="*", default=None, help="Specific model keys to process")
    parser.add_argument("--categories", nargs="*", default=None, help="Specific categories to process (e.g. A B C D E F)")
    args = parser.parse_args()

    fn = TARGETS.get(args.target)
    if not fn:
        logger.error(f"Unknown target: {args.target}")
        sys.exit(1)

    if args.target == "train":
        code = fn(budget=args.budget, force=args.force, models=args.models, categories=args.categories)
    else:
        code = fn()
    sys.exit(code)


if __name__ == "__main__":
    main()
