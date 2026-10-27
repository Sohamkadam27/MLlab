"""
Shared data loading for all training/calibration scripts.

Two fixes over the naive version, both important for honest results:

1. DE-DUPLICATION. The CSV repeats the same weather row once per satellite
   detection (18,109 rows but only ~12,400 unique timestamps; August 2020 alone
   has thousands of copies). With a random split, copies of the same row land in
   both train and test, which inflates accuracy to ~99%. We keep ONE row per
   timestamp (label = a fire was detected at that time).

2. TIME-BASED SPLIT. We train on earlier years and test on a later one
   (default: train 2020-2022, test 2023). That matches real use: predict the
   future from the past.

Do not add 'datetime' as a feature: in this file that column is a leaked
severity-like value that almost perfectly separates the classes.
"""

import pandas as pd

FEATURE_COLUMNS = [
    "temperature_2m", "relative_humidity_2m", "precipitation", "surface_pressure",
    "cloud_cover", "wind_speed_10m", "soil_temperature_0_to_7cm", "soil_temperature_7_to_28cm",
    "soil_moisture_0_to_7cm", "soil_moisture_7_to_28cm",
]


def load_dataset(csv_path, test_year: int = 2023, **_ignored):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["forest_fire"] + FEATURE_COLUMNS)

    # In this file the real timestamp lives in the first (unnamed) column.
    df["ts"] = pd.to_datetime(df.iloc[:, 0], errors="coerce")
    df = df.dropna(subset=["ts"])
    df["fire"] = (df["forest_fire"] == "Y").astype(int)

    # One row per timestamp; label = any detection at that time.
    agg = {c: "first" for c in FEATURE_COLUMNS}
    agg["fire"] = "max"
    df = df.groupby("ts", as_index=False).agg(agg)
    df["year"] = df["ts"].dt.year

    train = df[df["year"] < test_year]
    test = df[df["year"] == test_year]

    X_train, y_train = train[FEATURE_COLUMNS], train["fire"]
    X_test, y_test = test[FEATURE_COLUMNS], test["fire"]

    imbalance_ratio = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    return X_train, X_test, y_train, y_test, imbalance_ratio


def load_time_split(csv_path, test_year=2023):
    """
    Honest split for time-series data: train on every year BEFORE `test_year`,
    test on `test_year`. A random split lets near-identical neighbouring
    3-hourly rows land on both sides and inflates every metric.
    Returns X_train, X_test, y_train, y_test, imbalance_ratio.
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=["forest_fire"] + FEATURE_COLUMNS)
    # the first (unnamed) column holds the real timestamp in this file
    first = df.columns[0]
    ts = pd.to_datetime(df[first], format="%m/%d/%Y %H:%M", errors="coerce")
    df = df.assign(_ts=ts).dropna(subset=["_ts"])
    y = df["forest_fire"].map({"Y": 1, "N": 0}).astype(int)
    train_mask = df["_ts"].dt.year < test_year
    test_mask = df["_ts"].dt.year == test_year
    X = df[FEATURE_COLUMNS]
    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]
    imbalance_ratio = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    return X_train, X_test, y_train, y_test, imbalance_ratio
