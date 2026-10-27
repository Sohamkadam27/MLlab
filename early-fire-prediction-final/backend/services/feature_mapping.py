"""
Single source of truth for turning the app's 3 weather inputs
(temperature, humidity, wind) into the model's 10 features.
Used by BOTH backend/services/model_service.py and calibrate_thresholds.py,
so the thresholds are calibrated on exactly what the app computes.
"""

import numpy as np

FEATURE_NAMES = [
    "temperature_2m",
    "relative_humidity_2m",
    "precipitation",
    "surface_pressure",
    "cloud_cover",
    "wind_speed_10m",
    "soil_temperature_0_to_7cm",
    "soil_temperature_7_to_28cm",
    "soil_moisture_0_to_7cm",
    "soil_moisture_7_to_28cm",
]


def build_feature_matrix(temperature, humidity, wind, imputation):
    """
    imputation: dict from model_config.json["imputation"].
      - {"value": x}                          -> constant (training median)
      - {"intercept": b, "coef": [a_t,a_h,a_w]} -> linear estimate from t, h, w
    Accepts scalars or arrays; returns shape (n, 10) in FEATURE_NAMES order.
    """
    t = np.atleast_1d(np.asarray(temperature, dtype=float))
    h = np.atleast_1d(np.asarray(humidity, dtype=float))
    w = np.atleast_1d(np.asarray(wind, dtype=float))

    cols = []
    for name in FEATURE_NAMES:
        if name == "temperature_2m":
            cols.append(t)
        elif name == "relative_humidity_2m":
            cols.append(h)
        elif name == "wind_speed_10m":
            cols.append(w)
        else:
            spec = imputation[name]
            if "coef" in spec:
                c = spec["coef"]
                col = spec["intercept"] + c[0] * t + c[1] * h + c[2] * w
                if "moisture" in name:
                    col = np.clip(col, 0.0, 1.0)
                cols.append(col)
            else:
                cols.append(np.full_like(t, spec["value"]))
    return np.column_stack(cols)
