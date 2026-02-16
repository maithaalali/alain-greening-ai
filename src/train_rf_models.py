import argparse
import json
import os

import joblib
import pandas as pd
from data_utils import load_master
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def build_rf(X_train, y_train, X_test, y_test, num_features, cat_features):
    """Build, train and evaluate a Random Forest regressor."""
    pre = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_features),
            ("cat", OneHotEncoder(drop="first"), cat_features),
        ]
    )

    rf = Pipeline(
        steps=[
            ("preprocess", pre),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=300,
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)

    # Older sklearn may not support squared=False
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    return rf, rmse, r2


def main(args):
    os.makedirs("models", exist_ok=True)
    os.makedirs(os.path.dirname(args.metrics_out), exist_ok=True)

    df = load_master(args.input)
    metrics = {}

    # =======================
    # TEMPERATURE RF MODEL
    # =======================
    cols_T = ["temperature", "green_fraction", "humidity", "year", "station_clean"]
    df_T = df[cols_T].dropna().copy()

    train_T = df_T[df_T["year"] <= 2018]
    test_T = df_T[df_T["year"] > 2018]

    X_train_T = train_T[["green_fraction", "humidity", "year", "station_clean"]]
    y_train_T = train_T["temperature"]
    X_test_T = test_T[["green_fraction", "humidity", "year", "station_clean"]]
    y_test_T = test_T["temperature"]

    rf_T, rmse_T, r2_T = build_rf(
        X_train_T,
        y_train_T,
        X_test_T,
        y_test_T,
        num_features=["green_fraction", "humidity", "year"],
        cat_features=["station_clean"],
    )
    joblib.dump(rf_T, args.temp_model)
    metrics["temperature"] = {"rmse": rmse_T, "r2": r2_T}

    # =======================
    # AQI RF MODEL
    # =======================
    cols_A = [
        "AQI",
        "green_fraction",
        "humidity",
        "temperature",
        "PM10",
        "PM25",
        "year",
        "station_clean",
    ]
    df_A = df[cols_A].dropna().copy()

    train_A = df_A[df_A["year"] <= 2018]
    test_A = df_A[df_A["year"] > 2018]

    X_train_A = train_A[
        [
            "green_fraction",
            "humidity",
            "temperature",
            "PM10",
            "PM25",
            "year",
            "station_clean",
        ]
    ]
    y_train_A = train_A["AQI"]
    X_test_A = test_A[
        [
            "green_fraction",
            "humidity",
            "temperature",
            "PM10",
            "PM25",
            "year",
            "station_clean",
        ]
    ]
    y_test_A = test_A["AQI"]

    rf_A, rmse_A, r2_A = build_rf(
        X_train_A,
        y_train_A,
        X_test_A,
        y_test_A,
        num_features=[
            "green_fraction",
            "humidity",
            "temperature",
            "PM10",
            "PM25",
            "year",
        ],
        cat_features=["station_clean"],
    )
    joblib.dump(rf_A, args.aqi_model)
    metrics["AQI"] = {"rmse": rmse_A, "r2": r2_A}

    # =======================
    # SAVE METRICS
    # =======================
    with open(args.metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    print("RF models trained. Metrics:")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="data/AlAin_Stations_2007_2025_clean.csv"
    )
    parser.add_argument(
        "--temp_model", default="models/rf_temperature.joblib"
    )
    parser.add_argument(
        "--aqi_model", default="models/rf_aqi.joblib"
    )
    parser.add_argument(
        "--metrics_out", default="results/tables/rf_metrics.json"
    )
    args = parser.parse_args()
    main(args)


