"""
Train several OTHER classifiers purely for comparison with XGBoost, using the same honest
time-based split (train 2020-2022, test 2023). Nothing here is saved or deployed.

Usage:  python train_other_models.py
"""

from pathlib import Path

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from data_utils import load_time_split

DATA_PATH = Path(__file__).resolve().parent / "Trihourly_Weather_Dataset.csv"


def evaluate(name, model, X_train, X_test, y_train, y_test, scale=False):
    if scale:
        scaler = StandardScaler()
        X_train, X_test = scaler.fit_transform(X_train), scaler.transform(X_test)
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]
    preds = (probs >= 0.5).astype(int)
    tp = int(((preds == 1) & (y_test.values == 1)).sum())
    return {
        "Model": name,
        "ROC AUC": roc_auc_score(y_test, probs),
        "PR AUC": average_precision_score(y_test, probs),
        "Recall@0.5": tp / max(1, int((y_test == 1).sum())),
        "Precision@0.5": tp / max(1, int(preds.sum())),
    }


def main():
    X_train, X_test, y_train, y_test, _ = load_time_split(DATA_PATH)
    print(f"Train {len(X_train)} rows (2020-22) | Test {len(X_test)} rows (2023, fire rate {y_test.mean():.1%})")
    models = [
        ("XGBoost (final model)", XGBClassifier(eval_metric="logloss", max_depth=4, n_estimators=200,
                                               learning_rate=0.05, min_child_weight=5, subsample=0.8,
                                               colsample_bytree=0.8, random_state=42), False),
        ("Gradient Boosting", GradientBoostingClassifier(n_estimators=200, max_depth=3, random_state=42), False),
        ("Random Forest", RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=5,
                                                 random_state=42), False),
        ("K-Nearest Neighbors", KNeighborsClassifier(n_neighbors=25), True),
        ("SVM (RBF)", SVC(probability=True, random_state=42), True),
        ("Logistic Regression", LogisticRegression(max_iter=2000), True),
    ]
    rows = [evaluate(n, m, X_train, X_test, y_train, y_test, sc) for n, m, sc in models]
    df = pd.DataFrame(rows).sort_values("PR AUC", ascending=False)
    print("\n=== Held-out 2023 comparison (sorted by PR AUC; random guessing would score "
          f"{y_test.mean():.3f}) ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))


if __name__ == "__main__":
    main()
