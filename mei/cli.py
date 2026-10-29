"""CLI interface for Environmental & Financial Framework (mei compatibility layer).

Commands
--------
setup    -- Validate YAML schemas, mapping configs, and codecs.
profiles -- Show latent risk profile configuration.
info     -- Print project identity and active config.
make-sample-data -- Calibrate and generate synthetic survey files and EDA figures.
clean -- Clean an EF CSV/XLSX/XLS file and save canonical parquet plus report.
"""

import argparse
import sys
from pathlib import Path
from typing import Any

from src.preprocessing.codec import EnvFinCodec
from src.utils.config_loader import load_yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _load_project_yaml() -> dict[str, Any]:
    """Load the project configuration from the repository root."""
    return load_yaml(PROJECT_ROOT / "config" / "project.yaml")


def cmd_setup(args):
    """Validate schema, mapping, and codec for all 17 items."""
    schema_path = Path(args.schema)
    if not schema_path.exists():
        schema_path = Path("configs/schema_envfin.yaml")
    mapping_path = Path(args.mapping)
    if not mapping_path.exists():
        mapping_path = Path("configs/mapping_envfin.yaml")

    codec = EnvFinCodec(schema_path=schema_path, mapping_path=mapping_path)
    n_items = len(codec.items)
    n_indices = len(codec.derived_indices)
    n_outcomes = len(codec.outcomes)

    assert n_items == 17, f"Expected 17 canonical survey items, found {n_items}"
    assert n_indices == 4, f"Expected 4 derived indices, found {n_indices}"
    assert n_outcomes >= 2, f"Expected at least 2 outcomes, found {n_outcomes}"

    for name, meta in codec.items.items():
        assert "pressure_direction" in meta, f"Item '{name}' missing pressure_direction"
        assert meta.get("role") == "feature", f"Item '{name}' role must be 'feature'"
        for lvl in meta.get("levels", []):
            label = lvl["label"]
            enc = codec.encode(name, label)
            dec = codec.decode(name, enc)
            assert (
                dec == label
            ), f"Codec mismatch for '{name}': {label} -> {enc} -> {dec}"

    print(
        f"[SUCCESS] Environmental & Financial Schema Validated: "
        f"{n_items} items, {n_indices} derived indices, {n_outcomes} outcomes. "
        f"All codecs verified."
    )
    sys.exit(0)


def cmd_profiles(args):
    """Show latent risk profile configuration from project.yaml."""
    cfg = _load_project_yaml()
    design = cfg.get("design", {})
    project = cfg.get("project", {})
    method = design.get("profile_method", "gmm")
    n_profiles = design.get("n_profiles", 4)
    data_source = project.get("data_source", "sample")

    print(
        f"Component   : {project.get('component', 'Environmental & Financial Factors')}"
    )
    print(f"Index       : {project.get('index_name', 'EFI')} (0–100)")
    print(f"Data source : {data_source}")
    print()
    print(f"Profile method  : {method}  (cross-sectional; fitted inside CV folds)")
    print(f"N profiles      : {n_profiles}  (tuned by BIC/AIC)")
    sys.exit(0)


def cmd_info(args):
    """Print project identity and active config values."""
    cfg = _load_project_yaml()
    project = cfg.get("project", {})
    efi = cfg.get("efi", {})
    print(f"Project     : {project.get('id', 'KND-CDAP-IT-05')}")
    print(
        f"Component   : {project.get('component', 'Environmental & Financial Factors')}"
    )
    print(
        f"Index       : {project.get('index_name', 'EFI')} ({project.get('index_abbr', 'EFI')})"
    )
    print(
        f"Product     : {project.get('website_product_name', 'Study Conditions Index')}"
    )
    print(f"Data source : {project.get('data_source', 'sample')}")
    bands = efi.get("bands", {})
    print(
        f"EFI bands   : Low {bands.get('low', [0, 34])} / "
        f"Moderate {bands.get('moderate', [35, 64])} / "
        f"High {bands.get('high', [65, 100])}"
    )
    sys.exit(0)


def cmd_make_sample_data(args):
    """Generate configured synthetic survey files and exploratory figures."""
    from src.data_gen.sample_survey import generate_sample_data

    result = generate_sample_data()
    metrics = result["calibration"]["metrics"]
    print(
        "Calibration: "
        f"signal_strength={result['calibration']['signal_strength']:.2f}, "
        f"at_risk={metrics['at_risk_prevalence']:.1%}, "
        f"LightGBM 5-fold AUC={metrics['lightgbm_auc']:.3f}, "
        f"Logistic 5-fold AUC={metrics['logistic_auc']:.3f}"
    )
    print("Generated sample files:")
    for path in result["outputs"].values():
        print(f"  {path.relative_to(PROJECT_ROOT)}")
    print(f"Generated {len(result['figures'])} EDA figures.")


def cmd_clean(args):
    """Run the reusable EF cleaning pipeline and persist its quality report."""
    from src.cleaning.cleaner import EnvironmentalFinancialCleaner

    cleaner = EnvironmentalFinancialCleaner()
    result = cleaner.clean(args.input, mode=args.mode)
    saved = cleaner.write_result(
        result,
        output_root=Path(args.output_root) if args.output_root else None,
    )
    for suggestion in saved.report["mapping"]["suggestions"]:
        print(
            "Header suggestion: "
            f"{suggestion['source']} -> {suggestion['suggested_canonical']} "
            f"({suggestion['confidence']:.1f}%; not auto-accepted)"
        )
    if args.mode == "training" and result.report["traffic_light"]["status"] == "red":
        print(
            "Cleaning completed with blocking data-quality errors: "
            + "; ".join(result.report["traffic_light"]["messages"])
        )
        print(f"Warnings: {result.report['warnings']}")
        sys.exit(2)
    print(
        f"Cleaning status: {saved.report['traffic_light']['status']} "
        f"({saved.report['rows_in']} rows in, {saved.report['rows_out']} rows out)"
    )
    print(f"Clean parquet: {saved.parquet_path}")
    print(f"Quality report: {saved.report_path}")
    rows_with_reasons = sum(
        bool(reasons) for reasons in saved.report["row_reasons"].values()
    )
    if rows_with_reasons:
        print(f"Rows with quality reasons: {rows_with_reasons}")
    for warning in saved.report["warnings"]:
        print(f"Warning: {warning}")


def main():
    parser = argparse.ArgumentParser(
        prog="python -m mei.cli",
        description="Environmental & Financial Intelligence CLI",
    )
    subparsers = parser.add_subparsers(dest="command", help="Sub-commands")

    # setup
    setup_p = subparsers.add_parser(
        "setup", help="Validate YAML schemas, mapping configs, and item codecs"
    )
    setup_p.add_argument(
        "--schema", default="config/schema_envfin.yaml", help="Path to schema YAML"
    )
    setup_p.add_argument(
        "--mapping", default="config/mapping_envfin.yaml", help="Path to mapping YAML"
    )

    subparsers.add_parser(
        "profiles",
        help="Show latent risk profile configuration",
    )

    # info
    subparsers.add_parser("info", help="Print project identity and active config")
    subparsers.add_parser(
        "make-sample-data",
        help="Calibrate and generate synthetic survey files and EDA figures",
    )
    clean_parser = subparsers.add_parser(
        "clean",
        help="Clean a survey file and write canonical parquet and quality JSON",
    )
    clean_parser.add_argument("--input", required=True, help="CSV, XLSX, or XLS path")
    clean_parser.add_argument(
        "--mode",
        choices=("training", "scoring"),
        default="training",
        help="Training filters low-quality rows; scoring preserves rows with reasons",
    )
    clean_parser.add_argument(
        "--output-root",
        help="Optional project root for parquet and JSON output (primarily for tests)",
    )

    args = parser.parse_args()

    dispatch = {
        "setup": cmd_setup,
        "profiles": cmd_profiles,
        "info": cmd_info,
        "make-sample-data": cmd_make_sample_data,
        "clean": cmd_clean,
    }

    if args.command in dispatch:
        dispatch[args.command](args)
    else:
        parser.print_help()
        sys.exit(0)


if __name__ == "__main__":
    main()
