"""Generate configuration-driven Environmental & Financial survey samples."""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.config_loader import load_yaml
from src.utils.paths import get_project_root

FACTOR_NAMES = (
    "financial_strain",
    "access_burden",
    "digital_gap",
    "home_pressure",
)
_LIGHTGBM_DLL_DIRECTORY_HANDLES: list[object] = []


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

    def add_response_quality_effects(
        self, raw: pd.DataFrame, seed: int | None = None
    ) -> pd.DataFrame:
        """Inject configured survey response styles and raw-export imperfections."""
        quality = self.settings["response_quality"]
        rng = np.random.default_rng(quality["random_state"] if seed is None else seed)
        result = raw.copy()
        n_rows = len(result)
        item_names = list(self.schema["items"])
        item_headers = {
            item: self.mappings[item]["google_forms_header"] for item in item_names
        }
        ordinal_items = [
            item
            for item, metadata in self.schema["items"].items()
            if metadata["type"] == "ordinal"
        ]
        style_names = list(quality["response_style_rates"])
        style_probabilities = [
            float(quality["response_style_rates"][style]) for style in style_names
        ]
        no_style_probability = 1.0 - sum(style_probabilities)
        if no_style_probability < 0:
            raise ValueError("Configured response style rates must sum to at most 1")
        style_codes = rng.choice(
            len(style_names) + 1,
            size=n_rows,
            p=style_probabilities + [no_style_probability],
        )
        styles = {
            style: style_codes == style_index
            for style_index, style in enumerate(style_names)
        }

        def set_code(item: str, row: int, code: int) -> None:
            result.at[row, item_headers[item]] = _level_label(self.schema, item, code)

        def code_for(item: str, row: int) -> int | None:
            value = result.at[row, item_headers[item]]
            if pd.isna(value):
                return None
            for level in self.schema["items"][item]["levels"]:
                if str(level["label"]) == str(value):
                    return int(level["code"])
            special = self.schema["items"][item].get("structural_na")
            if special and str(special["label"]) == str(value):
                return int(special["code"])
            return None

        for style, row_codes in styles.items():
            rows = np.flatnonzero(row_codes)
            for row in rows:
                if style == "acquiescence_money_worry":
                    codes = quality["acquiescence_codes"]
                    set_code(
                        "money_worry",
                        int(row),
                        int(rng.choice(codes)),
                    )
                elif style == "midpoint":
                    for item in ordinal_items:
                        if code_for(item, int(row)) == 0 and item == "job_hours":
                            continue
                        current_code = code_for(item, int(row))
                        if current_code is None or current_code < 0:
                            continue
                        levels = self.schema["items"][item]["levels"]
                        midpoint_index = round(
                            (len(levels) - 1)
                            * float(quality["straight_liner_level_fraction"])
                        )
                        set_code(
                            item,
                            int(row),
                            int(levels[midpoint_index]["code"]),
                        )
                elif style == "extreme":
                    high_extreme = bool(rng.integers(0, 2))
                    for item in ordinal_items:
                        current_code = code_for(item, int(row))
                        if current_code is None or current_code < 0:
                            continue
                        levels = self.schema["items"][item]["levels"]
                        if item == "job_hours" and current_code == 0:
                            continue
                        if item == "job_hours" and not high_extreme:
                            set_code(item, int(row), int(levels[1]["code"]))
                        else:
                            level = levels[-1] if high_extreme else levels[0]
                            set_code(item, int(row), int(level["code"]))
        # Enforce the questionnaire's employment gate after style responses.
        for row in range(n_rows):
            status_code = code_for("part_time_job", row)
            hours_code = code_for("job_hours", row)
            if status_code == 0:
                set_code("job_hours", row, 0)
            elif status_code is not None and status_code > 0 and hours_code == 0:
                set_code("job_hours", row, 1)

        straight_count = round(n_rows * float(quality["straight_liner_rate"]))
        no_style_rows = np.flatnonzero(style_codes == len(style_names))
        straight_rows = rng.choice(no_style_rows, size=straight_count, replace=False)
        for row in straight_rows:
            for item in ordinal_items:
                current_code = code_for(item, int(row))
                if current_code is None or current_code < 0:
                    continue
                levels = self.schema["items"][item]["levels"]
                midpoint_index = round(
                    (len(levels) - 1) * float(quality["straight_liner_level_fraction"])
                )
                set_code(item, int(row), int(levels[midpoint_index]["code"]))
            status_code = code_for("part_time_job", int(row))
            if status_code == 0:
                set_code("job_hours", int(row), 0)
            elif code_for("job_hours", int(row)) == 0:
                set_code("job_hours", int(row), 1)

        consent_header = self.mappings["consent"]["google_forms_header"]
        no_consent_count = round(n_rows * float(quality["no_consent_rate"]))
        no_consent_rows = rng.choice(n_rows, size=no_consent_count, replace=False)
        result.loc[no_consent_rows, consent_header] = "No / I decline"

        attention_header = self.mappings["attention_check"]["google_forms_header"]
        failed_count = round(n_rows * float(quality["attention_fail_rate"]))
        failed_rows = rng.choice(n_rows, size=failed_count, replace=False)
        result.loc[failed_rows, attention_header] = quality["attention_fail_response"]

        missing_count = round(
            n_rows * len(item_names) * float(quality["random_missing_rate"])
        )
        missing_positions = rng.choice(
            n_rows * len(item_names), size=missing_count, replace=False
        )
        for position in missing_positions:
            row, item_index = divmod(int(position), len(item_names))
            result.at[row, item_headers[item_names[item_index]]] = np.nan

        response_count = n_rows * len(item_names)
        typo_count = max(1, round(response_count * float(quality["typo_rate"])))
        typo_positions = rng.choice(response_count, size=typo_count, replace=False)
        for position in typo_positions:
            row, item_index = divmod(int(position), len(item_names))
            header = item_headers[item_names[item_index]]
            value = result.at[row, header]
            if pd.isna(value):
                continue
            if rng.random() < 0.5:
                result.at[row, header] = f"  {value}  "
            else:
                text = str(value)
                result.at[row, header] = text[:-1] if len(text) > 1 else text

        # Preserve the explicit non-disclosure response after injecting blanks.
        pns_count = round(n_rows * float(quality["prefer_not_to_say_rate"]))
        pns_rows = rng.choice(n_rows, size=pns_count, replace=False)
        pns = self.schema["items"]["cost_difficulty"]["prefer_not_to_say"]
        result.loc[pns_rows, item_headers["cost_difficulty"]] = pns["label"]

        duplicate_count = round(n_rows * float(quality["duplicate_rate"]))
        duplicate_rows = rng.choice(n_rows, size=duplicate_count, replace=False)
        result = pd.concat(
            [result, result.iloc[duplicate_rows].copy()], ignore_index=True
        )
        return result


def sample_output_path(settings: dict[str, Any], key: str) -> Path:
    """Resolve a configured sample output path under the repository root."""
    relative = Path(settings["outputs"][key])
    return get_project_root() / relative


def _encode_item_codes(
    generator: EnvironmentalFinancialSampleGenerator, raw: pd.DataFrame
) -> pd.DataFrame:
    """Encode questionnaire labels with their schema codes for model validation."""
    features = generator.canonical_feature_frame(raw)
    encoded = pd.DataFrame(index=features.index)
    for item, metadata in generator.schema["items"].items():
        label_codes = {
            str(level["label"]): int(level["code"]) for level in metadata["levels"]
        }
        structural = metadata.get("structural_na")
        if structural:
            label_codes[str(structural["label"])] = int(structural["code"])
        pns = metadata.get("prefer_not_to_say")
        if pns:
            label_codes[str(pns["label"])] = int(pns["code"])
        encoded[item] = features[item].map(label_codes)
    return encoded


def _calibration_models(
    generator: EnvironmentalFinancialSampleGenerator,
) -> tuple[Pipeline, Pipeline]:
    """Create fold-fitted LightGBM and logistic pipelines from survey features."""
    if os.name == "nt":
        import sklearn

        sklearn_dll_directory = Path(sklearn.__file__).resolve().parent / ".libs"
        if sklearn_dll_directory.is_dir():
            _LIGHTGBM_DLL_DIRECTORY_HANDLES.append(
                os.add_dll_directory(str(sklearn_dll_directory))
            )
    from lightgbm import LGBMClassifier

    schema_items = generator.schema["items"]
    numeric_columns = [
        item for item, metadata in schema_items.items() if metadata["type"] != "nominal"
    ]
    nominal_columns = [
        item for item, metadata in schema_items.items() if metadata["type"] == "nominal"
    ]
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_columns,
            ),
            (
                "nominal",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "one_hot",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                nominal_columns,
            ),
        ],
        remainder="drop",
    )
    calibration = generator.settings["calibration"]
    lgbm_settings = calibration["lightgbm"]
    lightgbm_model = Pipeline(
        steps=[
            ("preprocess", clone(preprocessor)),
            (
                "classifier",
                LGBMClassifier(
                    **lgbm_settings,
                    random_state=int(calibration["random_state"]),
                ),
            ),
        ]
    )
    logistic_settings = calibration["logistic_regression"]
    logistic_model = Pipeline(
        steps=[
            ("preprocess", clone(preprocessor)),
            (
                "classifier",
                LogisticRegression(
                    C=float(logistic_settings["c_value"]),
                    max_iter=int(logistic_settings["max_iter"]),
                    random_state=int(calibration["random_state"]),
                ),
            ),
        ]
    )
    return lightgbm_model, logistic_model


def calibrate_signal_strength(
    generator: EnvironmentalFinancialSampleGenerator,
    n: int | None = None,
    seed: int | None = None,
) -> dict[str, Any]:
    """Tune signal strength against configured prevalence and five-fold AUC gates."""
    settings = generator.settings
    sizes = settings["sizes"]
    n_rows = int(sizes["full"] if n is None else n)
    seed_value = int(settings["seed"] if seed is None else seed)
    outcome_settings = settings["outcomes"]
    signal_settings = outcome_settings["signal_strength"]
    calibration_settings = settings["calibration"]
    lower_auc, upper_auc = map(float, outcome_settings["lightgbm_auc_range"])
    lower_prevalence, upper_prevalence = map(
        float, outcome_settings["prevalence_range"]
    )
    history: list[dict[str, float | int | bool]] = []
    lightgbm_model, logistic_model = _calibration_models(generator)

    for iteration in range(int(signal_settings["maximum_iterations"])):
        strength = float(signal_settings["initial"]) + (
            iteration * float(signal_settings["increment"])
        )
        raw, _ = generator.generate(n_rows, seed=seed_value, signal_strength=strength)
        encoded = _encode_item_codes(generator, raw)
        target_column = generator.mappings["attrition_intent_score"][
            "google_forms_header"
        ]
        y = (
            raw[target_column].astype(float)
            >= float(outcome_settings["at_risk_threshold"])
        ).astype(int)
        prevalence = float(y.mean())
        if y.nunique() != 2:
            raise ValueError(
                f"Calibration at signal_strength={strength:.3f} produced one class"
            )
        folds = StratifiedKFold(
            n_splits=int(calibration_settings["folds"]),
            shuffle=True,
            random_state=int(calibration_settings["random_state"]),
        )
        lightgbm_probabilities = cross_val_predict(
            lightgbm_model,
            encoded,
            y,
            cv=folds,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        logistic_probabilities = cross_val_predict(
            logistic_model,
            encoded,
            y,
            cv=folds,
            method="predict_proba",
            n_jobs=1,
        )[:, 1]
        lightgbm_auc = float(roc_auc_score(y, lightgbm_probabilities))
        logistic_auc = float(roc_auc_score(y, logistic_probabilities))
        in_range = lower_auc <= lightgbm_auc <= upper_auc
        prevalence_in_range = lower_prevalence <= prevalence <= upper_prevalence
        lr_gap_ok = logistic_auc <= lightgbm_auc + float(
            outcome_settings["max_logistic_auc_delta"]
        )
        history.append(
            {
                "iteration": iteration + 1,
                "signal_strength": strength,
                "at_risk_prevalence": prevalence,
                "lightgbm_auc": lightgbm_auc,
                "logistic_auc": logistic_auc,
                "lightgbm_auc_in_range": in_range,
                "prevalence_in_range": prevalence_in_range,
                "logistic_gap_ok": lr_gap_ok,
            }
        )
        if in_range and prevalence_in_range and lr_gap_ok:
            return {
                "signal_strength": strength,
                "metrics": history[-1],
                "history": history,
                "raw": raw,
            }

    raise RuntimeError(
        "Calibration did not meet configured at-risk prevalence and CV AUC "
        f"requirements after {len(history)} iterations: {history}"
    )
