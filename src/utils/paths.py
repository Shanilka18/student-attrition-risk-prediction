"""Project path helper utilities.

Provides reliable, dynamic absolute path resolution for the entire project
without relying on machine-specific hardcoded directories.
"""

from pathlib import Path


def get_project_root() -> Path:
    """Return the absolute Path to the repository root directory.

    Infers root from this file location: <project_root>/src/utils/paths.py.
    """
    return Path(__file__).resolve().parent.parent.parent


def ensure_dir(path: str | Path) -> Path:
    """Ensure directory exists; creates parent directories if necessary."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_configs_dir() -> Path:
    """Return absolute path to configs/ directory."""
    return ensure_dir(get_project_root() / "configs")


def get_config_path(config_filename: str = "config.yaml") -> Path:
    """Return absolute path to a specific config file."""
    return get_configs_dir() / config_filename


def get_schema_path(schema_filename: str = "schema_envfin.yaml") -> Path:
    """Return absolute path to schema configuration."""
    return get_project_root() / "config" / schema_filename


def get_leaderboard_weights_path(
    weights_filename: str = "leaderboard_weights.yaml",
) -> Path:
    """Return absolute path to leaderboard weights file."""
    return get_configs_dir() / weights_filename


def get_data_dir(stage: str = "raw") -> Path:
    """Return absolute path to specified data subdirectory (raw, interim, processed, synthetic)."""
    valid_stages = {"raw", "interim", "processed", "synthetic"}
    if stage not in valid_stages:
        raise ValueError(f"Unknown data stage '{stage}'. Must be one of {valid_stages}")
    return ensure_dir(get_project_root() / "data" / stage)


def get_artifacts_dir(subdir: str = "") -> Path:
    """Return absolute path to artifacts/ or a specific subfolder."""
    base = get_project_root() / "artifacts"
    if subdir:
        return ensure_dir(base / subdir)
    return ensure_dir(base)


def get_docs_dir() -> Path:
    """Return absolute path to docs/ directory."""
    return ensure_dir(get_project_root() / "docs")


def get_reports_dir(subdir: str = "figures") -> Path:
    """Return absolute path to reports/ directory."""
    base = get_project_root() / "reports"
    if subdir:
        return ensure_dir(base / subdir)
    return ensure_dir(base)
