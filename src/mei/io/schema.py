"""Pydantic schema definitions and validation loaders for survey configs."""

from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import yaml
from pydantic import BaseModel, ConfigDict, Field

from mei.config import CONFIG_DIR

ColumnRole = Literal["id", "consent", "attention_check", "feature", "outcome", "meta"]


class ColumnSchema(BaseModel):
    """Schema definition for a single questionnaire column."""

    model_config = ConfigDict(extra="ignore")

    name: str
    type: str = "integer"
    role: ColumnRole
    period: Optional[int] = None
    indicator: Optional[str] = None
    item_id: Optional[str] = None
    statement: Optional[str] = None
    description: Optional[str] = None
    allowed_codes: Optional[List[int]] = None
    expected_code: Optional[int] = None
    text_labels: Optional[Dict[int, str]] = None


class SurveySchema(BaseModel):
    """Comprehensive questionnaire schema model."""

    model_config = ConfigDict(extra="ignore")

    version: str = "1.0.0"
    num_columns: int = 35
    num_features: int = 28
    num_outcomes: int = 3
    columns: List[ColumnSchema]

    def get_column(self, name: str) -> Optional[ColumnSchema]:
        """Look up column schema by canonical name."""
        for col in self.columns:
            if col.name == name:
                return col
        return None

    def get_feature_names(self) -> List[str]:
        """Return canonical names of all feature columns (28 items)."""
        return [col.name for col in self.columns if col.role == "feature"]

    def get_outcome_names(self) -> List[str]:
        """Return canonical names of all outcome columns (3 items)."""
        return [col.name for col in self.columns if col.role == "outcome"]

    def get_columns_by_role(self, role: ColumnRole) -> List[ColumnSchema]:
        """Filter columns by assigned role."""
        return [col for col in self.columns if col.role == role]

    def get_role_counts(self) -> Dict[str, int]:
        """Count columns by role."""
        counts: Dict[str, int] = {}
        for col in self.columns:
            counts[col.role] = counts.get(col.role, 0) + 1
        return counts


class FuzzyMatchingConfig(BaseModel):
    """Configuration for RapidFuzz fallback matching."""

    model_config = ConfigDict(extra="ignore")

    enabled: bool = True
    threshold: float = 88.0
    scorer: str = "token_sort_ratio"
    auto_accept: bool = False
    description: Optional[str] = None


class MappingConfig(BaseModel):
    """Header variant mapping configuration for Google Forms ingestion."""

    model_config = ConfigDict(extra="ignore")

    version: str = "1.0.0"
    fuzzy_matching: FuzzyMatchingConfig = Field(default_factory=FuzzyMatchingConfig)
    columns: Dict[str, List[str]]

    def suggest_mapping(
        self, header: str, threshold: Optional[float] = None
    ) -> List[Tuple[str, float]]:
        """Suggest canonical column matches using rapidfuzz token_sort_ratio.

        Per Global Rule & Step 3: suggestions only, never automatically accepted.
        """
        try:
            from rapidfuzz import fuzz
        except ImportError:
            return []

        limit_threshold = (
            threshold if threshold is not None else self.fuzzy_matching.threshold
        )
        suggestions: List[Tuple[str, float]] = []

        for canonical_name, variants in self.columns.items():
            best_score = 0.0
            for variant in variants:
                score = fuzz.token_sort_ratio(header, variant)
                if score > best_score:
                    best_score = score
            if best_score >= limit_threshold:
                suggestions.append((canonical_name, round(best_score, 2)))

        # Sort descending by similarity score
        suggestions.sort(key=lambda x: x[1], reverse=True)
        return suggestions


class OutcomeItemConfig(BaseModel):
    """Details for an outcome item."""

    model_config = ConfigDict(extra="ignore")

    item_id: str
    statement: str
    scale_min: int = 1
    scale_max: int = 5


class LabelsConfig(BaseModel):
    """Target and risk band definition schema."""

    model_config = ConfigDict(extra="ignore")

    version: str = "1.0.0"
    outcome_items: Dict[str, OutcomeItemConfig]
    targets: Dict[str, Any]
    mei_index: Dict[str, Any]


def load_schema(path: Optional[Path] = None) -> SurveySchema:
    """Load and validate config/schema.yaml."""
    schema_path = path or (CONFIG_DIR / "schema.yaml")
    if not schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {schema_path}")
    with open(schema_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return SurveySchema.model_validate(data)


def load_mapping(path: Optional[Path] = None) -> MappingConfig:
    """Load and validate config/mapping.yaml."""
    mapping_path = path or (CONFIG_DIR / "mapping.yaml")
    if not mapping_path.exists():
        raise FileNotFoundError(f"Mapping file not found: {mapping_path}")
    with open(mapping_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return MappingConfig.model_validate(data)


def load_labels(path: Optional[Path] = None) -> LabelsConfig:
    """Load and validate config/labels.yaml."""
    labels_path = path or (CONFIG_DIR / "labels.yaml")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")
    with open(labels_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return LabelsConfig.model_validate(data)


def validate_all_configs() -> Dict[str, Any]:
    """Validate all three YAML configuration files and verify integrity rules."""
    schema = load_schema()
    mapping = load_mapping()
    labels = load_labels()

    # Rule G3 check: outcome items must NEVER be features
    feature_names = set(schema.get_feature_names())
    outcome_names = set(schema.get_outcome_names())
    overlap = feature_names.intersection(outcome_names)
    if overlap:
        raise ValueError(f"Target leakage violation in schema: {overlap}")

    # Verify mapping covers all canonical columns
    canonical_in_schema = {col.name for col in schema.columns}
    canonical_in_mapping = set(mapping.columns.keys())
    missing_mappings = canonical_in_schema - canonical_in_mapping
    if missing_mappings:
        raise ValueError(f"Mapping missing canonical columns: {missing_mappings}")

    # Verify outcome items in labels match schema outcomes
    outcome_items_in_labels = set(labels.outcome_items.keys())
    if outcome_items_in_labels != outcome_names:
        raise ValueError(
            f"Labels outcomes {outcome_items_in_labels} != schema {outcome_names}"
        )

    return {
        "status": "VALID",
        "num_columns": len(schema.columns),
        "num_features": len(feature_names),
        "num_outcomes": len(outcome_names),
        "role_counts": schema.get_role_counts(),
        "fuzzy_threshold": mapping.fuzzy_matching.threshold,
        "at_risk_threshold": labels.targets["classification"]["threshold"],
    }
