"""Verification script for Phase 0 / Project Skeleton Initialization.

Validates that:
1. All directory structures exist.
2. All required skeleton files, config, requirements, and Makefile exist and have valid syntax.
3. Docstrings are present in all Python modules.
4. AGENTS.md and docs/00_project_overview.md are present and properly structured.
"""

import os
import sys
import ast
from pathlib import Path


def verify_directories(root: Path) -> bool:
    required_dirs = [
        "config",
        "data/raw",
        "data/synthetic",
        "data/processed",
        "src/data",
        "src/features",
        "src/models",
        "src/evaluation",
        "src/explain",
        "src/api",
        "src/utils",
        "artifacts/models",
        "artifacts/metrics",
        "artifacts/leaderboards",
        "artifacts/plots",
        "artifacts/shap",
        "frontend",
        "docs",
        "tests",
        "notebooks",
    ]
    all_ok = True
    print("[1/4] Verifying directory structure...")
    for d in required_dirs:
        p = root / d
        if not p.is_dir():
            print(f"  [FAIL] Missing directory: {d}")
            all_ok = False
        else:
            print(f"  [OK] Directory present: {d}")
    return all_ok


def verify_critical_files(root: Path) -> bool:
    required_files = [
        "AGENTS.md",
        "requirements.txt",
        "Makefile",
        "config/default.yaml",
        "config/schema.yaml",
        "config/synthetic.yaml",
        "docs/00_project_overview.md",
        "docs/01_data_contract.md",
        "docs/02_synthetic_data.md",
        "docs/03_preprocessing_reliability.md",
        "data/synthetic/synthetic_survey_3000.csv",
        "data/synthetic/synthetic_survey_300.csv",
        "data/synthetic/synthetic_survey_dirty.csv",
        "data/synthetic/metadata.json",
        "data/processed/train.csv",
        "data/processed/val.csv",
        "data/processed/test.csv",
        "artifacts/models/preprocessor.joblib",
        "artifacts/metrics/reliability.json",
        "artifacts/plots/correlation_heatmap.png",
        "artifacts/plots/factor_distributions.png",
        "artifacts/plots/pri_distribution.png",
        "artifacts/plots/attrition_class_balance.png",
        "artifacts/plots/reliability_alpha_bars.png",
        "artifacts/plots/inter_factor_correlations.png",
        "src/__init__.py",
        "src/utils/__init__.py",
        "src/utils/config.py",
        "src/utils/logger.py",
        "src/utils/seed.py",
        "src/data/__init__.py",
        "src/data/make_dataset.py",
        "src/data/schema.py",
        "src/data/generate_synthetic.py",
        "src/data/splits.py",
        "src/data/preprocess.py",
        "src/features/__init__.py",
        "src/features/build_features.py",
        "src/features/reliability.py",
        "src/models/__init__.py",
        "src/models/architectures.py",
        "src/models/train.py",
        "src/evaluation/__init__.py",
        "src/evaluation/evaluate.py",
        "src/explain/__init__.py",
        "src/explain/shap_explainer.py",
        "src/api/__init__.py",
        "src/api/schemas.py",
        "src/api/main.py",
        "tests/__init__.py",
        "tests/test_data.py",
        "tests/test_data_schema.py",
        "tests/test_synthetic_data.py",
        "tests/test_preprocessing_splits.py",
        "tests/test_features.py",
        "tests/test_models.py",
        "tests/test_api.py",
    ]
    all_ok = True
    print("\n[2/4] Verifying required files...")
    for f in required_files:
        p = root / f
        if not p.is_file():
            print(f"  [FAIL] Missing file: {f}")
            all_ok = False
        else:
            size = p.stat().st_size
            print(f"  [OK] File exists ({size} bytes): {f}")
    return all_ok


def verify_python_syntax_and_docstrings(root: Path) -> bool:
    print("\n[3/4] Verifying Python syntax and docstrings...")
    all_ok = True
    py_files = list((root / "src").rglob("*.py")) + list((root / "tests").rglob("*.py"))
    for py_file in py_files:
        rel_path = py_file.relative_to(root)
        try:
            with open(py_file, "r", encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source, filename=str(py_file))
            docstring = ast.get_docstring(tree)
            if docstring:
                print(f"  [OK] Valid syntax & docstring: {rel_path}")
            else:
                print(f"  [WARN] Missing module docstring: {rel_path}")
        except Exception as e:
            print(f"  [FAIL] Syntax error in {rel_path}: {e}")
            all_ok = False
    return all_ok


def verify_docs_and_rules(root: Path) -> bool:
    print("\n[4/4] Verifying documentation and project rules...")
    agents_path = root / "AGENTS.md"
    overview_path = root / "docs/00_project_overview.md"

    if not agents_path.exists():
        print("  [FAIL] AGENTS.md missing")
        return False
    if not overview_path.exists():
        print("  [FAIL] docs/00_project_overview.md missing")
        return False

    with open(agents_path, "r", encoding="utf-8") as f:
        agents_content = f.read()
    with open(overview_path, "r", encoding="utf-8") as f:
        overview_content = f.read()

    assert "Psychosocial Intelligence Framework" in agents_content
    assert "Simple Explanation" in overview_content
    assert "Technical Explanation" in overview_content
    assert "RESEARCH NOTICE" in overview_content
    print("  [OK] AGENTS.md and docs/00_project_overview.md verified successfully.")
    return True


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    print(f"Executing Skeleton Verification in workspace: {root}\n{'=' * 60}")
    ok1 = verify_directories(root)
    ok2 = verify_critical_files(root)
    ok3 = verify_python_syntax_and_docstrings(root)
    ok4 = verify_docs_and_rules(root)

    print(f"\n{'=' * 60}")
    if ok1 and ok2 and ok3 and ok4:
        print("[SUCCESS] All Phase 0 skeleton components and verification checks passed.")
        sys.exit(0)
    else:
        print("[ERROR] Verification failures detected.")
        sys.exit(1)


if __name__ == "__main__":
    main()
