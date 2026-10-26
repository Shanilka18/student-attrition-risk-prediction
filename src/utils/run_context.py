"""Run context, execution identifiers, and artifact archival utilities."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional

from src.utils.logging_config import setup_logger
from src.utils.paths import get_artifacts_dir

logger = setup_logger("run_context")

_ACTIVE_RUN_ID: Optional[str] = None
_ACTIVE_RUN_TIMESTAMP: Optional[str] = None


def get_run_timestamp() -> str:
    """Return ISO 8601 UTC timestamp."""
    global _ACTIVE_RUN_TIMESTAMP
    if _ACTIVE_RUN_TIMESTAMP is None:
        _ACTIVE_RUN_TIMESTAMP = datetime.now(timezone.utc).isoformat()
    return _ACTIVE_RUN_TIMESTAMP


def get_or_create_run_id(prefix: str = "run") -> str:
    """Return active run_id or generate a new timestamped identifier."""
    global _ACTIVE_RUN_ID
    if _ACTIVE_RUN_ID is None:
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        _ACTIVE_RUN_ID = f"{prefix}_{ts}"
    return _ACTIVE_RUN_ID


def set_run_id(run_id: str) -> None:
    """Explicitly set active run_id."""
    global _ACTIVE_RUN_ID
    _ACTIVE_RUN_ID = run_id


def get_current_run_info() -> Dict[str, str]:
    """Retrieve current run_id and timestamp."""
    return {
        "run_id": get_or_create_run_id(),
        "timestamp": get_run_timestamp(),
    }


def archive_stale_artifacts(run_id: Optional[str] = None) -> Optional[Path]:
    """Archive existing leaderboard and evaluation artifacts before a new evaluation run.

    Copies existing files from artifacts/leaderboard, artifacts/metrics, etc.
    into artifacts/archive/{previous_run_id}/
    """
    artifacts_root = get_artifacts_dir()
    manifest_path = artifacts_root / "manifest.json"

    previous_run_id = "stale_run"
    if manifest_path.exists():
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                prev_manifest = json.load(f)
                previous_run_id = prev_manifest.get("run_id", "stale_run")
        except Exception as e:
            logger.warning(f"Failed reading existing manifest: {e}")

    archive_target = artifacts_root / "archive" / previous_run_id
    dirs_to_archive = ["leaderboard", "metrics", "calibrators", "frontend_data"]

    archived_any = False
    for d_name in dirs_to_archive:
        src_dir = artifacts_root / d_name
        if src_dir.exists() and any(src_dir.iterdir()):
            dst_dir = archive_target / d_name
            dst_dir.mkdir(parents=True, exist_ok=True)
            for item in src_dir.iterdir():
                if item.is_file():
                    shutil.copy2(item, dst_dir / item.name)
                    archived_any = True

    if archived_any:
        logger.info(f"Stale artifacts archived successfully to {archive_target}")
        return archive_target
    return None


def write_manifest(metadata: Optional[Dict[str, Any]] = None) -> Path:
    """Write system run manifest documenting active run_id, timestamp, and generated files."""
    run_info = get_current_run_info()
    manifest_data = {
        "run_id": run_info["run_id"],
        "timestamp": run_info["timestamp"],
        "split_used_for_ranking": "validation",
        "description": "Model zoo leaderboard, probability calibration, and parsimonious selection run",
        "metadata": metadata or {},
    }
    manifest_path = get_artifacts_dir() / "manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)
    return manifest_path
