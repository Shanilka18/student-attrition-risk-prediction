"""Central configuration loader and filesystem paths for MEI."""

import os
from pathlib import Path
from typing import Any, Dict

import yaml

# Base project directories (using pathlib only, relative resolution)
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
RUNS_DIR = ARTIFACTS_DIR / "runs"
FIGURES_DIR = ARTIFACTS_DIR / "figures"
DOCS_DIR = PROJECT_ROOT / "docs"
DOCS_FIGURES_DIR = DOCS_DIR / "figures"

# Prevent Matplotlib cache permission warnings on Windows
MPL_DIR = PROJECT_ROOT / ".matplotlib_cache"
MPL_DIR.mkdir(parents=True, exist_ok=True)
os.environ["MPLCONFIGDIR"] = str(MPL_DIR)


def load_project_config() -> Dict[str, Any]:
    """Load configuration from config/project.yaml."""
    cfg_file = CONFIG_DIR / "project.yaml"
    if not cfg_file.exists():
        return {}
    with open(cfg_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
