"""Generate configuration-driven Environmental & Financial survey samples."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.utils.config_loader import load_yaml
from src.utils.paths import get_project_root

FACTOR_NAMES = (
    "financial_strain",
    "access_burden",
    "digital_gap",
    "home_pressure",
)


def load_sample_configuration() -> (
    tuple[dict[str, Any], dict[str, Any], dict[str, Any]]
):
    """Load project settings, canonical questionnaire schema, and header mapping."""
    root = get_project_root()
    project = load_yaml(root / "config" / "project.yaml")
    schema = load_yaml(root / "config" / "schema_envfin.yaml")
    mapping = load_yaml(root / "config" / "mapping_envfin.yaml")
    return project["sample_data"], schema, mapping["canonical_mappings"]


def _level_label(schema: dict[str, Any], item: str, code: int) -> str:
    for level in schema["items"][item]["levels"]:
        if int(level["code"]) == int(code):
            return str(level["label"])
    special_fields = ("structural_na", "prefer_not_to_say")
    for field in special_fields:
        special = schema["items"][item].get(field)
        if special and int(special["code"]) == int(code):
            return str(special["label"])
    raise ValueError(f"No response label for {item} code {code}")


class EnvironmentalFinancialSampleGenerator:
    """Generate survey-like responses and isolated latent ground truth."""

    def __init__(
        self,
        settings: dict[str, Any],
        schema: dict[str, Any],
        mappings: dict[str, Any],
    ) -> None:
        self.settings = settings
        self.schema = schema
        self.mappings = mappings
        self.item_settings = settings["response_generation"]
        self.type_names = list(settings["student_types"]["proportions"])
        self.type_probabilities = np.asarray(
            list(settings["student_types"]["proportions"].values()), dtype=float
        )
        self.latent = settings["latent_factors"]
        self.means = self.latent["type_means"]
        self.correlation = np.asarray(self.latent["correlation"], dtype=float)
        np.linalg.cholesky(self.correlation)
        if not np.isclose(self.type_probabilities.sum(), 1.0):
            raise ValueError("sample_data student type proportions must sum to 1")
        if set(self.type_names) != set(self.means):
            raise ValueError("Every student type must have configured latent means")

    def generate(
        self, n: int, seed: int | None = None, signal_strength: float | None = None
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return Google-Forms-style raw responses and separate hidden truth."""
        rng = np.random.default_rng(self.settings["seed"] if seed is None else seed)
        outcome_settings = self.settings["outcomes"]
        if signal_strength is None:
            signal_strength = outcome_settings["signal_strength"]["initial"]

        type_codes = rng.choice(len(self.type_names), size=n, p=self.type_probabilities)
        student_types = np.asarray(self.type_names, dtype=object)[type_codes]
        mean_matrix = np.asarray(
            [self.means[student_type] for student_type in student_types], dtype=float
        )
        correlated_noise = rng.multivariate_normal(
            np.zeros(len(FACTOR_NAMES)), self.correlation, size=n
        )
        latent_values = np.clip(
            mean_matrix + correlated_noise * float(self.latent["standard_deviation"]),
            0.0,
            1.0,
        )
        factors = pd.DataFrame(latent_values, columns=FACTOR_NAMES)

        codes: dict[str, np.ndarray] = {}
        online = rng.random(n) < float(self.item_settings["online_rate"])
        for item, item_config in self.item_settings.items():
            if item == "online_rate":
                continue
            factor_values = factors[item_config["factor"]].to_numpy()
            if item_config.get("secondary_factor"):
                factor_values = (
                    factor_values
                    + float(item_config["secondary_weight"])
                    * factors[item_config["secondary_factor"]].to_numpy()
                )
            item_signal = factor_values + rng.normal(
                0.0, float(item_config["noise_sd"]), size=n
            )
            order = item_config["pressure_order"]
            if item in {"family_support_duty"}:
                ranks = (item_signal >= float(item_config["threshold"])).astype(int)
            else:
                ranks = np.digitize(
                    item_signal, np.asarray(item_config["cutpoints"], dtype=float)
                )
            item_codes = np.asarray(order, dtype=int)[ranks]

            if item == "part_time_job":
                status_probabilities = item_config["status_probabilities"]
                item_codes = np.asarray(
                    [
                        rng.choice(
                            order,
                            p=status_probabilities[student_type],
                        )
                        for student_type in student_types
                    ],
                    dtype=int,
                )
            elif item == "job_hours":
                employed = codes["part_time_job"] >= 1
                item_codes = np.where(employed, item_codes.clip(1), 0)
            elif item in {"commute_time", "commute_spend", "transport_mode"}:
                item_codes = np.where(online, -2, item_codes)
            codes[item] = item_codes.astype(int)

        outcome_linear = float(outcome_settings["intercept"])
        weighted_risk = np.zeros(n, dtype=float)
        for factor_name, weight in outcome_settings["factor_weights"].items():
            weighted_risk += float(weight) * factors[factor_name].to_numpy()
        weighted_risk += float(outcome_settings["interaction_weight"]) * (
            factors["financial_strain"].to_numpy() * factors["access_burden"].to_numpy()
        )
        outcome_logit = (
            outcome_linear
            + float(signal_strength) * weighted_risk
            + rng.normal(0.0, float(outcome_settings["latent_noise_sd"]), size=n)
        )
        latent_risk = 1.0 / (1.0 + np.exp(-outcome_logit))
        score = np.clip(
            100.0 * latent_risk
            + rng.normal(0.0, float(outcome_settings["score_noise_sd"]), size=n),
            0.0,
            100.0,
        )
        threshold = float(outcome_settings["at_risk_threshold"])
        at_risk = (score >= threshold).astype(int)

        timestamp_settings = self.settings["timestamps"]
        start = datetime.fromisoformat(timestamp_settings["start"])
        interval = int(timestamp_settings["interval_minutes"])
        id_settings = self.settings["identifiers"]
        ids = [
            f"{id_settings['prefix']}{index:0{int(id_settings['digits'])}d}"
            for index in range(1, n + 1)
        ]
        raw = pd.DataFrame()
        raw[self.mappings["student_id"]["google_forms_header"]] = ids
        raw[self.mappings["timestamp"]["google_forms_header"]] = [
            (start + timedelta(minutes=interval * index)).isoformat(
                sep=" ", timespec="seconds"
            )
            for index in range(n)
        ]
        raw[self.mappings["consent"]["google_forms_header"]] = "Yes / I agree"
        raw[self.mappings["attention_check"]["google_forms_header"]] = "Agree"
        for item in self.schema["items"]:
            raw[self.mappings[item]["google_forms_header"]] = [
                _level_label(self.schema, item, int(code)) for code in codes[item]
            ]
        raw[self.mappings["attrition_intent_score"]["google_forms_header"]] = np.rint(
            score
        ).astype(int)
        raw[self.mappings["at_risk"]["google_forms_header"]] = np.where(
            at_risk == 1, "At-Risk of Withdrawal", "Persisting / Low Risk"
        )

        hidden = pd.DataFrame(
            {
                "type": student_types,
                "F": factors["financial_strain"],
                "A": factors["access_burden"],
                "D": factors["digital_gap"],
                "H": factors["home_pressure"],
                "latent_risk": latent_risk,
            }
        )
        return raw, hidden

    def canonical_feature_frame(self, raw: pd.DataFrame) -> pd.DataFrame:
        """Return only the 17 canonical feature columns, without shared fields."""
        columns = {
            item: raw[self.mappings[item]["google_forms_header"]]
            for item in self.schema["items"]
        }
        return pd.DataFrame(columns)


def sample_output_path(settings: dict[str, Any], key: str) -> Path:
    """Resolve a configured sample output path under the repository root."""
    relative = Path(settings["outputs"][key])
    return get_project_root() / relative
