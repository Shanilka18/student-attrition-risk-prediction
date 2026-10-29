"""Generate configuration-driven Environmental & Financial survey samples."""

from __future__ import annotations

import os
import re
import zipfile
from datetime import datetime, timedelta
from io import BytesIO
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
        score = np.rint(
            np.clip(
                100.0 * latent_risk
                + rng.normal(0.0, float(outcome_settings["score_noise_sd"]), size=n),
                0.0,
                100.0,
            )
        ).astype(int)
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
        raw[self.mappings["attrition_intent_score"]["google_forms_header"]] = score
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
    ) -> tuple[pd.DataFrame, dict[str, int | float]]:
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

        # Preserve non-disclosure responses after injecting random blanks.
        pns_count = round(n_rows * float(quality["prefer_not_to_say_rate"]))
        pns_rows = rng.choice(n_rows, size=pns_count, replace=False)
        pns = self.schema["items"]["cost_difficulty"]["prefer_not_to_say"]
        result.loc[pns_rows, item_headers["cost_difficulty"]] = pns["label"]

        response_count = n_rows * len(item_names)
        typo_count = max(1, round(response_count * float(quality["typo_rate"])))
        item_header_list = [item_headers[item] for item in item_names]
        eligible = result[item_header_list].notna().to_numpy()
        cost_item_index = item_names.index("cost_difficulty")
        eligible[pns_rows, cost_item_index] = False
        eligible_positions = np.flatnonzero(eligible.ravel())
        typo_count = min(typo_count, len(eligible_positions))
        typo_positions = rng.choice(eligible_positions, size=typo_count, replace=False)
        for position in typo_positions:
            row, item_index = divmod(int(position), len(item_names))
            header = item_headers[item_names[item_index]]
            value = result.at[row, header]
            if rng.random() < 0.5:
                result.at[row, header] = f"  {value}  "
            else:
                text = str(value)
                result.at[row, header] = text[:-1] if len(text) > 1 else text

        duplicate_count = round(n_rows * float(quality["duplicate_rate"]))
        duplicate_rows = rng.choice(n_rows, size=duplicate_count, replace=False)
        result = pd.concat(
            [result, result.iloc[duplicate_rows].copy()], ignore_index=True
        )
        achieved = {
            "base_rows": n_rows,
            "output_rows": len(result),
            "duplicates": duplicate_count,
            "online_rows": int(
                raw[item_headers["commute_time"]]
                .astype(str)
                .str.startswith("Not applicable")
                .sum()
            ),
            "prefer_not_to_say_rows": pns_count,
            "no_consent_rows": no_consent_count,
            "attention_fail_rows": failed_count,
            "random_missing_cells": missing_count,
            "typo_cells": typo_count,
            "straight_liner_rows": straight_count,
        }
        achieved.update(
            {
                f"style_{style}_rows": int(np.count_nonzero(style_codes == index))
                for index, style in enumerate(style_names)
            }
        )
        return result, achieved


def sample_output_path(
    settings: dict[str, Any], key: str, output_root: Path | None = None
) -> Path:
    """Resolve a configured sample output path under the repository root."""
    relative = Path(settings["outputs"][key])
    root = get_project_root() if output_root is None else output_root
    return root / relative


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


def _write_deterministic_excel(frame: pd.DataFrame, path: Path) -> None:
    """Write an XLSX with stable workbook properties and ZIP timestamps."""
    path.parent.mkdir(parents=True, exist_ok=True)
    workbook_buffer = BytesIO()
    fixed_timestamp = datetime(2000, 1, 1)
    with pd.ExcelWriter(workbook_buffer, engine="openpyxl") as writer:
        frame.to_excel(writer, index=False, sheet_name="Responses")
        writer.book.properties.created = fixed_timestamp
        writer.book.properties.modified = fixed_timestamp

    normalized_buffer = BytesIO()
    with zipfile.ZipFile(workbook_buffer, "r") as source, zipfile.ZipFile(
        normalized_buffer, "w", compression=zipfile.ZIP_DEFLATED
    ) as normalized:
        for name in sorted(source.namelist()):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o600 << 16
            content = source.read(name)
            if name == "docProps/core.xml":
                content = re.sub(
                    rb"(<dcterms:modified[^>]*>).*(</dcterms:modified>)",
                    rb"\g<1>2000-01-01T00:00:00Z\g<2>",
                    content,
                )
            normalized.writestr(info, content)
    path.write_bytes(normalized_buffer.getvalue())


def _write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8", lineterminator="\n")


def _pressure_coded_features(
    generator: EnvironmentalFinancialSampleGenerator, raw: pd.DataFrame
) -> pd.DataFrame:
    """Map all item levels to pressure ranks for descriptive correlation plots."""
    encoded = _encode_item_codes(generator, raw)
    pressure = pd.DataFrame(index=encoded.index)
    for item, item_settings in generator.item_settings.items():
        if item == "online_rate":
            continue
        item_codes = encoded[item]
        order = item_settings["pressure_order"]
        ranks = {int(code): rank for rank, code in enumerate(order)}
        pressure[item] = item_codes.map(ranks)
        if item in {"commute_time", "commute_spend", "transport_mode"}:
            pressure.loc[item_codes == -2, item] = 0
    return pressure


def _reality_check_rows(
    generator: EnvironmentalFinancialSampleGenerator,
    clean_raw: pd.DataFrame,
    achieved: dict[str, int | float],
    calibration: dict[str, Any],
) -> list[tuple[str, str, str]]:
    settings = generator.settings
    quality = settings["response_quality"]
    base_rows = int(achieved["base_rows"])
    item_cells = base_rows * len(generator.schema["items"])
    rows = [
        (
            "Online commute responses",
            f"{settings['response_generation']['online_rate']:.1%}",
            f"{int(achieved['online_rows']) / base_rows:.1%}",
        ),
        (
            "Cost non-disclosure",
            f"{quality['prefer_not_to_say_rate']:.1%}",
            f"{int(achieved['prefer_not_to_say_rows']) / base_rows:.1%}",
        ),
        (
            "Random item missingness",
            f"{quality['random_missing_rate']:.1%} of item cells",
            f"{int(achieved['random_missing_cells']) / item_cells:.1%} of item cells",
        ),
        (
            "Typos / formatting variants",
            f"{quality['typo_rate']:.1%} of item cells",
            f"{int(achieved['typo_cells']) / item_cells:.1%} of item cells",
        ),
        (
            "Attention-check failures",
            f"{quality['attention_fail_rate']:.1%}",
            f"{int(achieved['attention_fail_rows']) / base_rows:.1%}",
        ),
        (
            "No consent",
            f"{quality['no_consent_rate']:.1%}",
            f"{int(achieved['no_consent_rows']) / base_rows:.1%}",
        ),
        (
            "Duplicate submissions",
            f"{quality['duplicate_rate']:.1%}",
            f"{int(achieved['duplicates']) / base_rows:.1%}",
        ),
        (
            "Straight-liners",
            f"{quality['straight_liner_rate']:.1%}",
            f"{int(achieved['straight_liner_rows']) / base_rows:.1%}",
        ),
    ]
    for style, rate in quality["response_style_rates"].items():
        rows.append(
            (
                f"Response style: {style.replace('_', ' ')}",
                f"{float(rate):.1%}",
                f"{int(achieved[f'style_{style}_rows']) / base_rows:.1%}",
            )
        )
    at_risk_header = generator.mappings["at_risk"]["google_forms_header"]
    prevalence = float(clean_raw[at_risk_header].eq("At-Risk of Withdrawal").mean())
    rows.extend(
        [
            ("At-risk prevalence", "24%-32%", f"{prevalence:.1%}"),
            (
                "5-fold LightGBM ROC-AUC",
                "0.78-0.88",
                f"{calibration['metrics']['lightgbm_auc']:.3f}",
            ),
        ]
    )
    return rows


def _write_eda_figures(
    generator: EnvironmentalFinancialSampleGenerator,
    clean_raw: pd.DataFrame,
    messy_raw: pd.DataFrame,
    hidden: pd.DataFrame,
    achieved: dict[str, int | float],
    calibration: dict[str, Any],
    output_root: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_directory = sample_output_path(
        generator.settings, "figures_dir", output_root
    )
    figure_directory.mkdir(parents=True, exist_ok=True)
    colors = list(generator.settings["brand_palette"].values())
    dpi = int(generator.settings["figure_dpi"])
    item_names = list(generator.schema["items"])
    feature_frame = generator.canonical_feature_frame(clean_raw)
    output_paths: list[Path] = []

    fig, axes = plt.subplots(5, 4, figsize=(18, 19))
    for index, (axis, item) in enumerate(zip(axes.flat, item_names, strict=False)):
        counts = feature_frame[item].value_counts(normalize=True)
        axis.bar(
            range(len(counts)),
            counts.values,
            color=colors[index % len(colors)],
        )
        axis.set_title(item.replace("_", " "))
        axis.set_ylim(0, 1)
        axis.tick_params(axis="x", labelrotation=55, labelsize=7)
        axis.set_xticks(range(len(counts)), counts.index.astype(str))
        axis.set_ylabel("Share")
    for axis in axes.flat[len(item_names) :]:
        axis.set_visible(False)
    fig.suptitle("Environmental & Financial Item Response Distributions")
    fig.tight_layout()
    path = figure_directory / "01_item_distributions.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    output_paths.append(path)

    pressure = _pressure_coded_features(generator, clean_raw)
    correlation = pressure.corr()
    fig, axis = plt.subplots(figsize=(13, 11))
    image = axis.imshow(correlation, cmap="PuBuGn", vmin=-1, vmax=1)
    axis.set_xticks(range(len(item_names)), item_names, rotation=75, ha="right")
    axis.set_yticks(range(len(item_names)), item_names)
    for row in range(len(item_names)):
        for column in range(len(item_names)):
            axis.text(
                column,
                row,
                f"{correlation.iloc[row, column]:.2f}",
                ha="center",
                va="center",
                fontsize=6,
            )
    fig.colorbar(image, ax=axis, label="Pearson correlation of pressure ranks")
    axis.set_title("Correlation of Pressure-Oriented Item Ranks")
    fig.tight_layout()
    path = figure_directory / "02_item_correlations.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    output_paths.append(path)

    messy_features = pd.DataFrame(
        {
            item: messy_raw[generator.mappings[item]["google_forms_header"]]
            for item in item_names
        }
    ).iloc[: int(generator.settings["sizes"]["upload"])]
    missingness = messy_features.isna().to_numpy(dtype=int)
    fig, axis = plt.subplots(figsize=(16, 7))
    axis.imshow(missingness.T, aspect="auto", interpolation="nearest", cmap="Greys")
    axis.set_yticks(range(len(item_names)), item_names)
    axis.set_xlabel("Upload record")
    axis.set_title("Missingness Map: Messy Survey Export")
    fig.tight_layout()
    path = figure_directory / "03_missingness_map.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    output_paths.append(path)

    at_risk_header = generator.mappings["at_risk"]["google_forms_header"]
    risk = clean_raw[at_risk_header].eq("At-Risk of Withdrawal").to_numpy()
    type_rates = (
        pd.DataFrame({"type": hidden["type"], "at_risk": risk})
        .groupby("type", sort=False)["at_risk"]
        .mean()
    )
    fig, axis = plt.subplots(figsize=(11, 6))
    axis.bar(
        type_rates.index,
        type_rates.values,
        color=[colors[index % len(colors)] for index in range(len(type_rates))],
    )
    axis.set_ylim(0, 1)
    axis.set_ylabel("At-risk share")
    axis.set_title("At-Risk Prevalence by Hidden Student Type")
    axis.tick_params(axis="x", labelrotation=25)
    fig.tight_layout()
    path = figure_directory / "04_at_risk_by_type.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    output_paths.append(path)

    income_header = generator.mappings["income_band"]["google_forms_header"]
    income_rates = (
        clean_raw.assign(_at_risk=risk).groupby(income_header)["_at_risk"].mean()
    )
    income_levels = [
        str(level["label"])
        for level in generator.schema["items"]["income_band"]["levels"]
    ]
    income_rates = income_rates.reindex(income_levels)
    fig, axis = plt.subplots(figsize=(12, 6))
    axis.bar(income_rates.index, income_rates.values, color=colors[1])
    axis.set_ylim(0, 1)
    axis.set_ylabel("At-risk share")
    axis.set_title("At-Risk Prevalence by Household Income Band")
    axis.tick_params(axis="x", labelrotation=25)
    fig.tight_layout()
    path = figure_directory / "05_at_risk_by_income_band.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    output_paths.append(path)

    reality_rows = _reality_check_rows(generator, clean_raw, achieved, calibration)
    fig, axis = plt.subplots(figsize=(13, 7))
    axis.axis("off")
    table = axis.table(
        cellText=reality_rows,
        colLabels=["Measure", "Configured", "Achieved"],
        loc="center",
        cellLoc="left",
        colWidths=[0.48, 0.24, 0.24],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)
    for column in range(3):
        table[(0, column)].set_facecolor(colors[0])
        table[(0, column)].get_text().set_color("white")
    axis.set_title("Configured Targets and Achieved Sample Rates", pad=20)
    fig.tight_layout()
    path = figure_directory / "06_configured_vs_achieved.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    output_paths.append(path)
    return output_paths


def generate_sample_data(output_root: Path | None = None) -> dict[str, Any]:
    """Calibrate and write all requested sample datasets and six EDA figures."""
    settings, schema, mappings = load_sample_configuration()
    generator = EnvironmentalFinancialSampleGenerator(settings, schema, mappings)
    root = get_project_root() if output_root is None else Path(output_root)
    calibration = calibrate_signal_strength(generator)
    full_raw = calibration["raw"]
    _, hidden = generator.generate(
        int(settings["sizes"]["full"]),
        seed=int(settings["seed"]),
        signal_strength=float(calibration["signal_strength"]),
    )
    small_raw, _ = generator.generate(
        int(settings["sizes"]["small"]),
        seed=int(settings["seed"]),
        signal_strength=float(calibration["signal_strength"]),
    )
    full_messy, achieved = generator.add_response_quality_effects(
        full_raw, seed=int(settings["seed"])
    )
    small_messy, _ = generator.add_response_quality_effects(
        small_raw, seed=int(settings["seed"])
    )

    clean_features = generator.canonical_feature_frame(full_raw).iloc[
        : int(settings["sizes"]["upload"])
    ]
    messy_upload = clean_features.copy()
    header_offset = int(settings["messy_upload"]["header_variant_offset"])
    messy_upload = messy_upload.rename(
        columns={
            item: mappings[item]["accepted_variants"][
                min(header_offset, len(mappings[item]["accepted_variants"]) - 1)
            ]
            for item in clean_features
        }
    )
    messy_rng = np.random.default_rng(int(settings["seed"]))
    blank_count = round(
        len(messy_upload)
        * len(clean_features.columns)
        * float(settings["messy_upload"]["blank_rate"])
    )
    blank_positions = messy_rng.choice(
        len(messy_upload) * len(messy_upload.columns),
        size=blank_count,
        replace=False,
    )
    for position in blank_positions:
        row, column = divmod(int(position), len(messy_upload.columns))
        messy_upload.iat[row, column] = np.nan
    invalid_count = max(
        1,
        round(
            len(messy_upload) * float(settings["messy_upload"]["invalid_value_rate"])
        ),
    )
    invalid_values = settings["messy_upload"]["invalid_values"]
    invalid_items = list(invalid_values)
    invalid_rows = messy_rng.choice(
        len(messy_upload), size=min(invalid_count, len(messy_upload)), replace=False
    )
    for index, row in enumerate(invalid_rows):
        item = invalid_items[index % len(invalid_items)]
        header = mappings[item]["accepted_variants"][
            min(header_offset, len(mappings[item]["accepted_variants"]) - 1)
        ]
        messy_upload.at[int(row), header] = invalid_values[item]
    messy_upload[settings["messy_upload"]["unknown_column_name"]] = (
        "unmapped synthetic field"
    )

    output_paths = {
        "raw_full": sample_output_path(settings, "raw_full", root),
        "raw_full_xlsx": sample_output_path(settings, "raw_full_xlsx", root),
        "raw_small": sample_output_path(settings, "raw_small", root),
        "hidden_truth": sample_output_path(settings, "hidden_truth", root),
        "clean_upload": sample_output_path(settings, "clean_upload", root),
        "messy_upload": sample_output_path(settings, "messy_upload", root),
    }
    _write_csv(full_messy, output_paths["raw_full"])
    _write_deterministic_excel(full_messy, output_paths["raw_full_xlsx"])
    _write_csv(small_messy, output_paths["raw_small"])
    _write_csv(hidden, output_paths["hidden_truth"])
    _write_deterministic_excel(clean_features, output_paths["clean_upload"])
    _write_csv(messy_upload, output_paths["messy_upload"])

    figures = _write_eda_figures(
        generator,
        full_raw,
        full_messy,
        hidden,
        achieved,
        calibration,
        root,
    )
    return {
        "settings": settings,
        "calibration": calibration,
        "quality": achieved,
        "outputs": output_paths,
        "figures": figures,
    }
