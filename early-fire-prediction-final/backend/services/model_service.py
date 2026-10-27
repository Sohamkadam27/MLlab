import json
import math
from functools import lru_cache
from pathlib import Path

import joblib

from services.feature_mapping import FEATURE_NAMES, build_feature_matrix

MODEL_DIR = Path(__file__).resolve().parent.parent / "model"
MODEL_PATH = MODEL_DIR / "fire_model.pkl"
CONFIG_PATH = MODEL_DIR / "model_config.json"   # created by calibrate_thresholds.py

# Feature groups used to explain a prediction ("what drives this score?").
DRIVER_GROUPS = {
    "Surface pressure":      ["surface_pressure"],
    "Soil temperature":      ["soil_temperature_0_to_7cm", "soil_temperature_7_to_28cm"],
    "Soil moisture":         ["soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm"],
    "Air temp / humidity / wind": ["temperature_2m", "relative_humidity_2m", "wind_speed_10m"],
    "Cloud & rain":          ["cloud_cover", "precipitation"],
}

BASIC_MODE_NOTE = (
    "Low-reliability estimate: with only temperature/humidity/wind the model is barely better "
    "than chance on held-out data. Use /zones for live, full-feature scoring."
)


@lru_cache(maxsize=1)
def get_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at: {MODEL_PATH}. Run train_xgboost.py.")
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def get_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"{CONFIG_PATH} not found. Run `python calibrate_thresholds.py` from the project root."
        )
    return json.loads(CONFIG_PATH.read_text())


def _cuts(mode: str):
    c = get_config()[mode]
    return float(c["low_max"]), float(c["high_min"])


def _to_risk_level(p: float, mode: str) -> str:
    low_max, high_min = _cuts(mode)
    if p < low_max:
        return "Low"
    if p < high_min:
        return "Medium"
    return "High"


def _to_display_score(p: float, mode: str) -> float:
    """Piecewise-linear 0-100 score: the learned cutoffs land exactly on 40 and 75."""
    low_max, high_min = _cuts(mode)
    p = max(0.0, min(1.0, p))
    if p < low_max:
        return 40.0 * p / low_max
    if p < high_min:
        return 40.0 + 35.0 * (p - low_max) / (high_min - low_max)
    return 75.0 + 25.0 * (p - high_min) / (1.0 - high_min)


def _logit(p: float) -> float:
    p = min(max(p, 1e-4), 1 - 1e-4)
    return math.log(p / (1 - p))


def _contributions(row: list, p: float):
    """
    Which inputs drive this score? Replace each feature group with its typical
    (training-median) value and measure how far the model's score moves
    (in logit space, so saturated scores still show differences).
    Returns {group: percent} or None if medians are unavailable.
    """
    medians = get_config().get("medians")
    if not medians:
        return None
    model = get_model()
    base = _logit(p)
    shifts = {}
    for group, names in DRIVER_GROUPS.items():
        alt = list(row[0])
        for n in names:
            alt[FEATURE_NAMES.index(n)] = medians[n]
        shifts[group] = abs(base - _logit(float(model.predict_proba([alt])[0][1])))
    total = sum(shifts.values())
    if total < 1e-9:
        return {g: 0 for g in shifts}
    return {g: round(100 * v / total) for g, v in shifts.items()}


def predict_fire_risk(temperature: float, humidity: float, wind: float,
                      vegetation: float = 0.0, full_features: dict | None = None):
    """
    full_features: dict with ALL 10 keys of FEATURE_NAMES (from live weather) -> "full" mode,
    the reliable one. Otherwise only temperature/humidity/wind are used ("basic" mode, weak).
    `vegetation` is accepted for API compatibility but is NOT a model feature.
    """
    model = get_model()

    if full_features is not None:
        missing = [k for k in FEATURE_NAMES if k not in full_features]
        if missing:
            raise ValueError(f"full_features missing keys: {missing}")
        row = [[float(full_features[k]) for k in FEATURE_NAMES]]
        mode = "full"
    else:
        row = build_feature_matrix(temperature, humidity, wind, get_config()["imputation"]).tolist()
        mode = "basic"

    p = float(model.predict_proba(row)[0][1])

    out = {
        "risk_score": round(_to_display_score(p, mode), 2),
        "risk_level": _to_risk_level(p, mode),
        "model_score": round(p, 4),
        "input_mode": mode,
    }
    if mode == "full":
        out["contributions"] = _contributions(row, p)
    else:
        out["note"] = BASIC_MODE_NOTE
    return out


def get_model_info() -> dict:
    """Validation numbers for the UI (honest, from the held-out year)."""
    cfg = get_config()
    full = cfg["full"]
    return {
        "algorithm": "XGBoost classifier",
        "trained_on": cfg["split"]["train"],
        "validated_on": cfg["split"]["test"],
        "validation_rows": cfg["split"]["test_rows"],
        "roc_auc": full["roc_auc"],
        "pr_auc": full["pr_auc"],
        "precision_at_high": full["precision_at_high"],
        "recall_at_high": full["recall_at_high"],
        "low_max": full["low_max"],
        "high_min": full["high_min"],
        "bucket_summary": full["test_bucket_summary"],
    }
