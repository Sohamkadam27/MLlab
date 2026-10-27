"""
Train the FINAL model (XGBoost) with an honest time-based split:
train on 2020-2022, hold out 2023 for evaluation.

Usage (project root):
    pip install pandas scikit-learn xgboost joblib
    python train_xgboost.py
    python calibrate_thresholds.py      # <- always run after training

Output: backend/model/fire_model.pkl
"""

from pathlib import Path

import joblib
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from data_utils import load_time_split

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "Trihourly_Weather_Dataset.csv"
MODEL_OUT = ROOT / "backend" / "model" / "fire_model.pkl"


def main():
    X_train, X_test, y_train, y_test, _ = load_time_split(DATA_PATH)
    print(f"Train: {len(X_train)} rows (2020-2022, fire rate {y_train.mean():.1%})")
    print(f"Test : {len(X_test)} rows (2023 holdout, fire rate {y_test.mean():.1%})\n")

    model = XGBClassifier(
        eval_metric="logloss", max_depth=4, n_estimators=200, learning_rate=0.05,
        min_child_weight=5, subsample=0.8, colsample_bytree=0.8, random_state=42,
    )
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    print("=== XGBoost on the held-out year (honest numbers) ===")
    print(f"ROC AUC : {roc_auc_score(y_test, probs):.4f}")
    print(f"PR  AUC : {average_precision_score(y_test, probs):.4f}   (a random model would score {y_test.mean():.3f})")

    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUT)
    print(f"\nSaved model to {MODEL_OUT}\nNext: python calibrate_thresholds.py")


if __name__ == "__main__":
    main()
