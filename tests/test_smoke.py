"""Smoke tests for MEI environment, CLI stub commands, and leakage rules."""

from pathlib import Path

from typer.testing import CliRunner

from mei.cli import app
from mei.config import load_project_config

runner = CliRunner()


def test_cli_help() -> None:
    """CLI help must display usage and list all planned subcommands."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    commands = [
        "setup",
        "make-sample-data",
        "clean",
        "features",
        "states",
        "train-classic",
        "train-deep",
        "train-regression",
        "finalize",
        "explain",
        "serve",
        "retrain",
        "docs",
        "all",
    ]
    for cmd in commands:
        assert cmd in result.stdout


def test_cli_setup_command() -> None:
    """setup command must validate configs and print schema summary."""
    result = runner.invoke(app, ["setup"])
    assert result.exit_code == 0
    assert "MEI Canonical Questionnaire Schema Summary" in result.stdout
    assert "Total Columns" in result.stdout
    assert "35" in result.stdout


def test_cli_make_sample_data_command() -> None:
    """make-sample-data command must generate datasets, artifacts, and summary."""
    result = runner.invoke(app, ["make-sample-data"])
    assert result.exit_code == 0
    assert "Signal Strength Calibration Log" in result.stdout
    assert "Synthetic Cohort Reality Check" in result.stdout
    assert "Generated Dataset Artifacts" in result.stdout


def test_cli_clean_command() -> None:
    """clean command must process training data and scoring data cleanly."""
    # Training mode
    res_train = runner.invoke(
        app,
        ["clean", "--input", "data/raw/survey_raw_sample.csv", "--mode", "training"],
    )
    assert res_train.exit_code == 0
    assert "Data Quality & Cleaning Summary" in res_train.stdout
    assert "GREEN" in res_train.stdout

    # Scoring mode
    res_score = runner.invoke(
        app,
        [
            "clean",
            "--input",
            "data/samples/sample_upload_messy.csv",
            "--mode",
            "scoring",
        ],
    )
    assert res_score.exit_code == 0
    assert "Data Quality & Cleaning Summary" in res_score.stdout
    assert "AMBER" in res_score.stdout


def test_cli_features_command() -> None:
    """features command must partition cohort, fit transformers, and audit leakage."""
    active_file = Path("artifacts/active_run.txt")
    orig_run = (
        active_file.read_text(encoding="utf-8").strip()
        if active_file.exists()
        else None
    )

    res = runner.invoke(app, ["features", "--seed", "42"])
    assert res.exit_code == 0
    assert "Cohort Splitting & Stratification Summary" in res.stdout
    assert "Named Feature Sets Inventory" in res.stdout
    assert "Anti-Leakage Audit" in res.stdout
    assert "PASS" in res.stdout
    assert "Top Predictive Longitudinal Features" in res.stdout

    if orig_run:
        active_file.write_text(orig_run, encoding="utf-8")


def test_cli_states_command() -> None:
    """states command must evaluate clustering, select HMM, and run ablation."""
    active_file = Path("artifacts/active_run.txt")
    orig_run = (
        active_file.read_text(encoding="utf-8").strip()
        if active_file.exists()
        else None
    )

    res = runner.invoke(app, ["states", "--n-bootstrap", "5", "--n-hmm-restarts", "2"])
    assert res.exit_code == 0
    assert "Longitudinal Clustering Evaluation" in res.stdout
    assert "Gaussian HMM Model Selection" in res.stdout
    assert "15-Fold Repeated CV Feature Set Ablation Benchmark" in res.stdout
    assert "Paired Wilcoxon Signed-Rank Test Across 15 Folds" in res.stdout

    if orig_run:
        active_file.write_text(orig_run, encoding="utf-8")


def test_cli_serve_command() -> None:
    """Serve CLI command must provide server options."""
    res = runner.invoke(app, ["serve", "--help"])
    assert res.exit_code == 0
    assert "Run FastAPI server" in res.stdout
    assert "--port" in res.stdout


def test_cli_remaining_stub_commands() -> None:
    """Unimplemented commands must return 0 and output 'not implemented yet'."""
    stubs = [
        "retrain",
        "docs",
        "all",
    ]
    for cmd in stubs:
        result = runner.invoke(app, [cmd])
        assert result.exit_code == 0
        assert "not implemented yet" in result.stdout


def test_project_config() -> None:
    """Project configuration must load and define mandatory keys."""
    cfg = load_project_config()
    assert isinstance(cfg, dict)
    assert "project" in cfg
    assert cfg["project"]["random_seed"] == 42
    assert "questionnaire" in cfg
    assert len(cfg["questionnaire"]["periods"]) == 4
    assert len(cfg["questionnaire"]["indicators"]) == 7


def test_no_target_leakage_rule() -> None:
    """Rule G3: outcome items A1-A3 must NEVER be included in model features."""
    cfg = load_project_config()
    indicators = cfg["questionnaire"]["indicators"]
    outcome_items = cfg["questionnaire"]["outcome_items"]

    # Verify outcome items and indicators are disjoint
    overlap = set(indicators).intersection(set(outcome_items))
    assert len(overlap) == 0, f"Target leakage detected in config: {overlap}"
