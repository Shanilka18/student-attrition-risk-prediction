"""Configuration loader for the Psychosocial Intelligence Framework.

Provides centralized access to project parameters defined in YAML configuration files.
"""

from pathlib import Path
from typing import Any, Dict
import yaml


def load_config(config_path: str = "config/default.yaml") -> Dict[str, Any]:
    """Load configuration from a YAML file.

    Args:
        config_path (str): Relative or absolute path to the configuration YAML file.
            Defaults to 'config/default.yaml'.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file cannot be found at config_path.
        yaml.YAMLError: If the YAML file contains syntax errors.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {path.resolve()}")
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)
