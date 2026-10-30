"""
Tests for Config and Directory Resolution.
"""

from backend.src.config import config, get_project_root


def test_project_root():
    """Verify that project root directory is correctly resolved."""
    root = get_project_root()
    assert root.exists()
    assert (root / "configs" / "config.yaml").exists()


def test_config_paths():
    """Verify that required directory paths exist or are resolvable."""
    assert config.synthetic_data_dir.exists()
    assert config.processed_data_dir.exists()
    assert config.models_dir.exists()
    assert config.metrics_dir.exists()
    assert config.docs_dir.exists()


def test_config_values():
    """Verify key configuration parameters match proposal requirements."""
    assert config.global_seed == 42
    assert config.cohort_size >= 500
    assert "gpa" in config.sequence_features
    assert "exam_mark_avg" in config.sequence_features
    assert "attendance_rate" in config.sequence_features
    assert "submission_delay_avg_days" in config.sequence_features
    assert config.target_attrition_rate > 0.0
    assert config.target_attrition_rate < 0.50
