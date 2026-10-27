"""
Dataset audit: reproduces the data-leakage finding and the corrected evaluation.
Run it live in your presentation, or just quote its output.

Usage (project root; needs Trihourly_Weather_Dataset.csv and backend/data/Wildfire_Data_2020-2023.csv):
    python audit_dataset.py
Writes backend/model/audit.json, which the app shows in the Prediction Panel.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from xgboost import XGBClassifier

from data_utils import FEATURE_COLUMNS as F

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "backend" / "model" / "audit.json"
BOX = dict(lat=(37.1, 37.5), lon=(-121.8, -121.2))      # study box used by the original notebook

# ---------------------------------------------------------------- load
w = pd.read_csv(ROOT / "Trihourly_Weather_Dataset.csv").dropna(subset=["forest_fire"] + F)
w["ts"] = pd.to_datetime(w.iloc[:, 0], format="%m/%d/%Y %H:%M", errors="coerce")
w = w.dropna(subset=["ts"])
w["y"] = (w.forest_fire == "Y").astype(int)
N, Y = w[w.y == 0], w[w.y == 1]

print("=" * 70)
print("PART 1: WHAT IS WRONG WITH THE ORIGINAL LABELS")
print("=" * 70)
slots = 1461 * 8
print(f"'N' rows: {len(N)}  | unique timestamps: {N.ts.nunique()}  | 3-hour slots in 4 years: {slots}")
print("  -> the N rows are ONE complete weather series from a single reference site.")
pairs = Y.merge(N, on="ts", suffixes=("_y", "_n"))
gap = float((pairs.surface_pressure_y - pairs.surface_pressure_n).abs().median())
print(f"'Y' rows: {len(Y)}  | {len(pairs)} share a timestamp with an N row")
print(f"  -> at the SAME moment their pressure differs by a median of {gap:.1f} hPa: different places.")
te = w[w.ts.dt.year == 2023]
site = float(N.surface_pressure.median())
auc_p = roc_auc_score(te.y, te.surface_pressure)
auc_d = roc_auc_score(te.y, (te.surface_pressure - site).abs())
print(f"AUC of surface_pressure ALONE on 2023:            {auc_p:.3f}   (the full model scored ~0.987)")
print(f"AUC of |pressure - {site:.0f} hPa| ALONE on 2023:       {auc_d:.3f}")
print("  -> the model learned 'reference site vs elsewhere', not fire weather.")

# ---------------------------------------------------------------- corrected labels
print("\n" + "=" * 70)
print("PART 2: CORRECTED EVALUATION (clean single-site weather + FIRMS labels)")
print("=" * 70)
ref = N.drop_duplicates("ts").copy()
ref["day"] = ref.ts.dt.date
ref["doy"] = ref.ts.dt.dayofyear
f = pd.read_csv(ROOT / "backend" / "data" / "Wildfire_Data_2020-2023.csv")
f = f[f.confidence.isin(["n", "h"])]
f["d"] = pd.to_datetime(f.acq_date, format="%m/%d/%y").dt.date
box = f[f.latitude.between(*BOX["lat"]) & f.longitude.between(*BOX["lon"])]
ref["y"] = ref.day.isin(set(box.d)).astype(int)

tr, te = ref[ref.ts.dt.year <= 2021], ref[ref.ts.dt.year >= 2022]
y = te.y.values
print(f"Label: any NASA FIRMS detection inside the study box that day ({box.d.nunique()} such days overall)")
print(f"Train 2020-2021: {len(tr)} rows | Test 2022-2023: {len(te)} rows, {te[te.y == 1].day.nunique()} fire days "
      f"(base rate {y.mean():.2f})")

mk = lambda: XGBClassifier(eval_metric="logloss", max_depth=3, n_estimators=150, learning_rate=0.05,
                           min_child_weight=10, subsample=0.8, colsample_bytree=0.8, random_state=42)
ang = lambda d: np.c_[np.sin(2 * np.pi * d.doy / 365.25), np.cos(2 * np.pi * d.doy / 365.25)]
W = ["temperature_2m", "relative_humidity_2m", "wind_speed_10m", "precipitation", "cloud_cover"]

p_model = mk().fit(tr[F], tr.y).predict_proba(te[F])[:, 1]
p_season = LogisticRegression().fit(ang(tr), tr.y).predict_proba(ang(te))[:, 1]
p_weather = mk().fit(tr[W], tr.y).predict_proba(te[W])[:, 1]

# uncertainty: resample whole days (neighbouring rows are not independent)
days = te.day.values
ud = np.unique(days)
idx = {d: np.where(days == d)[0] for d in ud}
rng = np.random.default_rng(0)
boot = []
for _ in range(400):
    ii = np.concatenate([idx[d] for d in rng.choice(ud, len(ud))])
    if 0 < y[ii].sum() < len(ii):
        boot.append(roc_auc_score(y[ii], p_model[ii]))
lo, hi = np.percentile(boot, [2.5, 97.5])

res = {
    "train": "2020-2021", "test": "2022-2023",
    "label": "any NASA FIRMS detection in the study box that day",
    "test_rows": int(len(te)), "test_fire_days": int(te[te.y == 1].day.nunique()),
    "base_rate": round(float(y.mean()), 3),
    "roc_auc": round(float(roc_auc_score(y, p_model)), 3),
    "pr_auc": round(float(average_precision_score(y, p_model)), 3),
    "auc_ci95": [round(float(lo), 3), round(float(hi), 3)],
    "season_only_auc": round(float(roc_auc_score(y, p_season)), 3),
    "weather_only_auc": round(float(roc_auc_score(y, p_weather)), 3),
}
print(f"\n{'Method':42}{'ROC AUC':>9}")
print(f"{'Random guessing':42}{0.5:>9.3f}")
print(f"{'Calendar season only (day-of-year)':42}{res['season_only_auc']:>9.3f}")
print(f"{'Weather only (temp, humidity, wind, rain, cloud)':42}{res['weather_only_auc']:>9.3f}")
print(f"{'Model: all 10 weather + soil features':42}{res['roc_auc']:>9.3f}   95% CI {lo:.3f}-{hi:.3f}")
print(f"PR AUC {res['pr_auc']} (random = {res['base_rate']})")

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps({
    "original": {"pressure_only_auc": round(float(auc_p), 3), "distance_from_site_auc": round(float(auc_d), 3),
                 "median_pressure_gap_hpa": round(gap, 1), "n_rows": int(len(N)), "y_rows": int(len(Y))},
    "corrected": res}, indent=2))
print(f"\nSaved {OUT}")
