"""
Builds backend/model/model_config.json from the trained model + the 2023 holdout.

It stores:
  imputation  how to estimate the 7 features the 3-slider mode does not measure
  medians     typical values, used to explain which inputs drive a prediction
  full/basic  Low / Medium / High cutoffs for each input mode, plus honest metrics

How the cutoffs are chosen (on the held-out year, never on training data):
  LOW  -> below the highest score that still catches >= TARGET_RECALL of real fire rows
          ("Low" misses at most 1 - TARGET_RECALL of fires)
  HIGH -> at/above the score with the best F1 (balanced precision and recall)
  MEDIUM is everything in between: elevated, keep watching.

Usage (project root, AFTER train_xgboost.py):
    python calibrate_thresholds.py
"""

import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

from data_utils import FEATURE_COLUMNS, load_time_split

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "backend"))
from services.feature_mapping import build_feature_matrix  # noqa: E402

DATA_PATH = ROOT / "Trihourly_Weather_Dataset.csv"
MODEL_PATH = ROOT / "backend" / "model" / "fire_model.pkl"
OUT_PATH = ROOT / "backend" / "model" / "model_config.json"

TARGET_RECALL = 0.97   # "Low" may miss at most 3% of real fire rows. At 0.99 about 70% of ordinary days
                       # fell into "Medium" (useless); 0.97 keeps Medium a rare, genuinely elevated band.


def fit_imputation(X_train):
    imp = {}
    base = X_train[["temperature_2m", "relative_humidity_2m", "wind_speed_10m"]].values
    for name in ["precipitation", "surface_pressure", "cloud_cover"]:
        imp[name] = {"value": float(X_train[name].median())}
    for name in ["soil_temperature_0_to_7cm", "soil_temperature_7_to_28cm",
                 "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm"]:
        lr = LinearRegression().fit(base, X_train[name].values)
        imp[name] = {"intercept": float(lr.intercept_), "coef": [float(c) for c in lr.coef_]}
    return imp


def derive(probs, y_test):
    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    precision, recall = precision[:-1], recall[:-1]

    ok = thresholds[recall >= TARGET_RECALL]
    low_max = float(ok.max()) if len(ok) else float(thresholds.min())

    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-9)
    best = int(f1.argmax())
    high_min = float(thresholds[best])

    if low_max >= high_min:                       # keep a real Medium band
        low_max = high_min / 2.0

    levels = np.where(probs < low_max, "Low", np.where(probs < high_min, "Medium", "High"))
    summary = {}
    for lvl in ("Low", "Medium", "High"):
        mask = levels == lvl
        n = int(mask.sum())
        fires = int(y_test.values[mask].sum()) if n else 0
        summary[lvl] = {"samples": n, "actual_fires": fires,
                        "fire_rate": round(fires / n, 4) if n else None}

    return {
        "low_max": round(low_max, 4),
        "high_min": round(high_min, 4),
        "roc_auc": round(float(roc_auc_score(y_test, probs)), 4),
        "pr_auc": round(float(average_precision_score(y_test, probs)), 4),
        "precision_at_high": round(float(precision[best]), 4),
        "recall_at_high": round(float(recall[best]), 4),
        "test_bucket_summary": summary,
    }


def main():
    X_train, X_test, y_train, y_test, _ = load_time_split(DATA_PATH)
    model = joblib.load(MODEL_PATH)
    imputation = fit_imputation(X_train)

    full_probs = model.predict_proba(X_test)[:, 1]
    basic_feats = build_feature_matrix(
        X_test["temperature_2m"].values, X_test["relative_humidity_2m"].values,
        X_test["wind_speed_10m"].values, imputation)
    basic_probs = model.predict_proba(basic_feats)[:, 1]

    config = {
        "split": {"train": "2020-2022", "test": "2023", "train_rows": int(len(X_train)),
                  "test_rows": int(len(X_test)), "test_fire_rate": round(float(y_test.mean()), 4)},
        "target_recall": TARGET_RECALL,
        "imputation": imputation,
        "medians": {c: float(X_train[c].median()) for c in FEATURE_COLUMNS},
        "full": derive(full_probs, y_test),
        "basic": derive(basic_probs, y_test),
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(json.dumps(config, indent=2))

    for mode, label in (("full", "ALL 10 REAL FEATURES (used by the map and Prediction Panel)"),
                        ("basic", "ONLY TEMP + HUMIDITY + WIND (legacy /predict without a zone)")):
        c = config[mode]
        print(f"=== {label} ===")
        print(f"  ROC AUC {c['roc_auc']} | PR AUC {c['pr_auc']} | at High cutoff: precision {c['precision_at_high']}, recall {c['recall_at_high']}")
        print(f"  Low < {c['low_max']} <= Medium < {c['high_min']} <= High")
        for lvl, sm in c["test_bucket_summary"].items():
            print(f"    {lvl:<7}{sm['samples']:>5} rows | {sm['actual_fires']:>4} real fires | fire rate {sm['fire_rate']}")
        print()
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()
