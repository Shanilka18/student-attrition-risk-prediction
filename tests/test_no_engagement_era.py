"""Guard active source, scripts, and configuration against temporal designs."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ACTIVE_ROOTS = ("src", "mei", "tests", "config", "configs", "scripts", "frontend")
ACTIVE_FILES = ("Makefile", "tasks.ps1")
SCAN_EXTENSIONS = {
    ".js",
    ".jsx",
    ".json",
    ".mk",
    ".py",
    ".ps1",
    ".toml",
    ".ts",
    ".tsx",
    ".yaml",
    ".yml",
}
LEGACY_DIRECTORY = "legacy"
IGNORED_DIRECTORIES = {".venv", "__pycache__", "build", "dist", "node_modules"}
SELF = Path(__file__).resolve()

FORBIDDEN_PATTERNS: list[tuple[str, str]] = [
    (r"\bperiods?\b", "temporal period design"),
    (r"\bPeriodReshaper\b", "period reshaping implementation"),
    (r"\bn_periods?\b|\bperiod_id\b|\bperiod_label\b", "period-specific fields"),
    (r"\bHiddenMarkov\b|\bHMM\b|\bhmmlearn\b|\bpomegranate\b", "HMM implementation"),
    (
        r"\b(?:sequence|sequential)[-_ ](?:models?|networks?|classifiers?)\b|\bSequenceModel\b|\bLSTM\b|\bGRU\b",
        "sequence model implementation",
    ),
]


def _active_files() -> list[Path]:
    files = [PROJECT_ROOT / name for name in ACTIVE_FILES]
    for root_name in ACTIVE_ROOTS:
        root = PROJECT_ROOT / root_name
        if not root.exists():
            continue
        files.extend(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix.lower() in SCAN_EXTENSIONS
            and not any(part.lower() == LEGACY_DIRECTORY for part in path.parts)
            and path.resolve() != SELF
            and not any(
                part.startswith(".") or part in IGNORED_DIRECTORIES
                for part in path.parts
            )
        )
    return [path for path in files if path.is_file()]


def test_active_code_has_no_temporal_or_sequence_design():
    patterns = [
        (re.compile(pattern, re.IGNORECASE), description)
        for pattern, description in FORBIDDEN_PATTERNS
    ]
    violations = []

    for path in _active_files():
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
        for line_number, line in enumerate(lines, start=1):
            for pattern, description in patterns:
                if pattern.search(line):
                    violations.append(
                        f"{path.relative_to(PROJECT_ROOT)}:{line_number}: "
                        f"{description}: {line.strip()}"
                    )

    if violations:
        pytest.fail(
            "Active files reference a disallowed longitudinal design:\n"
            + "\n".join(violations)
        )


def test_project_config_uses_supported_data_and_profile_modes():
    import yaml

    config_path = PROJECT_ROOT / "config" / "project.yaml"
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    project = config.get("project", {})
    design = config.get("design", {})

    assert project.get("data_source") in {"sample", "survey"}
    assert design.get("cross_sectional") is True
    assert design.get("profile_method") in {"gmm", "kmeans", "latent_class"}
