"""Configuration loader utilities.

Parses YAML configurations for pipeline parameters, variable schemas,
and model selection leaderboard weights.
"""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from src.utils.paths import (
    get_config_path,
    get_leaderboard_weights_path,
    get_schema_path,
)


def load_yaml(file_path: str | Path) -> dict[str, Any]:
    """Safely parse a YAML file into a Python dictionary.

    Parameters
    ----------
    file_path : Union[str, Path]
        Path to the YAML file to parse.

    Returns
    -------
    Dict[str, Any]
        Parsed dictionary contents.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if data is None:
        return {}
    if not isinstance(data, dict):
        raise TypeError(f"YAML file {path} did not parse into a dictionary.")
    return data


@lru_cache(maxsize=4)
def load_config(filename: str = "config.yaml") -> dict[str, Any]:
    """Load and cache the primary pipeline configuration file."""
    path = get_config_path(filename)
    return load_yaml(path)


@lru_cache(maxsize=4)
def load_schema(filename: str = "schema_envfin.yaml") -> dict[str, Any]:
    """Load and cache the data schema specification."""
    path = get_schema_path(filename)
    return load_yaml(path)


@lru_cache(maxsize=4)
def load_leaderboard_weights(
    filename: str = "leaderboard_weights.yaml",
) -> dict[str, Any]:
    """Load and cache the model evaluation composite scoring weights."""
    path = get_leaderboard_weights_path(filename)
    return load_yaml(path)
