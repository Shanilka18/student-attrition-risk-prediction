"""Guard tracked repository content against removed project terminology."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ALLOWLIST_PATH = Path("tests/allowlist_terms.txt")
PHASE_REPORTS_PREFIX = "docs/phase_reports/"
FORBIDDEN_TERMS = (
    "m" + "e" + "i",
    "Moo" + "dle",
    "H" + "M" + "M",
    "Period" + "Reshaper",
    "Engage" + "ment",
)


def _tracked_repository_files() -> list[tuple[str, Path]]:
    result = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
    )
    paths = [path.decode("utf-8") for path in result.stdout.split(b"\0") if path]
    return [
        (relative_path, PROJECT_ROOT / relative_path)
        for relative_path in paths
        if not relative_path.startswith(PHASE_REPORTS_PREFIX)
    ]


def test_tracked_repository_has_no_disallowed_terms():
    allowlist = {
        line.strip().casefold()
        for line in (PROJECT_ROOT / ALLOWLIST_PATH)
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    patterns = [
        (
            term,
            re.compile(
                rb"(?<![A-Za-z0-9])"
                + re.escape(term.encode("ascii"))
                + rb"(?![A-Za-z0-9])",
                re.IGNORECASE,
            ),
        )
        for term in FORBIDDEN_TERMS
    ]
    violations = []

    for relative_path, path in _tracked_repository_files():
        contents = path.read_bytes()
        if b"\0" in contents:
            # Compressed binary payloads can contain accidental byte matches.
            continue
        for term, pattern in patterns:
            for match in pattern.finditer(contents):
                matched_term = match.group().decode("ascii").casefold()
                if matched_term not in allowlist:
                    violations.append(
                        f"{relative_path}: disallowed term {matched_term!r}"
                    )

    if violations:
        pytest.fail(
            "Tracked files contain disallowed project terminology:\n"
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
