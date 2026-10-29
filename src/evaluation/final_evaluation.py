"""Single-use access to the held-out test partition."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from src.utils.paths import get_project_root


def evaluate_final_on_test(
    data: pd.DataFrame,
    run_id: str,
    evaluator: Callable[[pd.DataFrame], Any],
    output_root: str | Path | None = None,
) -> Any:
    """Read test positions once, seal the run, and evaluate only those rows."""
    root = get_project_root() if output_root is None else Path(output_root)
    split_dir = root / "artifacts" / "runs" / run_id / "splits"
    split_path = split_dir / "indices.json"
    if not split_path.is_file():
        raise FileNotFoundError(f"Split index artifact does not exist: {split_path}")

    access_path = split_dir / "final_evaluation_access.json"
    try:
        with access_path.open("x", encoding="utf-8") as file:
            json.dump(
                {
                    "run_id": run_id,
                    "accessed_at": datetime.now(UTC).isoformat(),
                },
                file,
                indent=2,
            )
    except FileExistsError as exc:
        raise PermissionError(
            f"The held-out test partition for run {run_id} has already been accessed"
        ) from exc

    split_payload = json.loads(split_path.read_text(encoding="utf-8"))
    test_indices = split_payload.get("test_indices")
    if not isinstance(test_indices, list) or not test_indices:
        raise ValueError(f"Split artifact for run {run_id} has no test positions")
    if any(
        not isinstance(index, int) or index < 0 or index >= len(data)
        for index in test_indices
    ):
        raise ValueError(
            f"Split artifact for run {run_id} contains invalid test positions"
        )
    return evaluator(data.iloc[test_indices].copy())
