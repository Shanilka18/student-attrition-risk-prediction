import json

import pytest

from src.utils import run_context


def test_run_info_reads_and_validates_configured_data_source(monkeypatch):
    monkeypatch.setattr(
        run_context,
        "load_yaml",
        lambda path: {"project": {"data_source": "survey"}},
    )
    monkeypatch.setattr(run_context, "get_or_create_run_id", lambda: "run_test")
    monkeypatch.setattr(
        run_context, "get_run_timestamp", lambda: "2026-10-29T00:00:00Z"
    )

    assert run_context.get_current_run_info() == {
        "run_id": "run_test",
        "timestamp": "2026-10-29T00:00:00Z",
        "data_source": "survey",
    }


def test_run_info_rejects_invalid_data_source(monkeypatch):
    monkeypatch.setattr(
        run_context,
        "load_yaml",
        lambda path: {"project": {"data_source": "synthetic"}},
    )

    with pytest.raises(ValueError, match="project.data_source"):
        run_context.get_current_run_info()


def test_manifest_records_data_source(tmp_path, monkeypatch):
    monkeypatch.setattr(
        run_context,
        "get_current_run_info",
        lambda: {
            "run_id": "run_test",
            "timestamp": "2026-10-29T00:00:00Z",
            "data_source": "sample",
        },
    )
    monkeypatch.setattr(run_context, "get_artifacts_dir", lambda: tmp_path)

    manifest_path = run_context.write_manifest()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["data_source"] == "sample"
