import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

from data_utils import load_master

# Candidate feature columns (we will keep only those that actually exist in df)
FEATURE_CANDIDATES = [
    "green_fraction",
    "PM10",
    "PM25",
    "SO2",
    "CO",
    "O3",
    "NO2",
    "humidity",
    "year",
]


def evaluate_models(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    # Keep only rows with all features + target
    available_feats = [c for c in FEATURE_CANDIDATES if c in df.columns]
    cols_needed = available_feats + [target_col]
    df_clean = df.dropna(subset=cols_needed).copy()

    if df_clean.empty:
        raise ValueError(f"No data available to model {target_col}.")

    X = df_clean[available_feats].values
    y = df_clean[target_col].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    models = {
        "LinearRegression": make_pipeline(StandardScaler(), LinearRegression()),
        "Ridge": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "RandomForest": RandomForestRegressor(
            n_estimators=200, random_state=42
        ),
        "GradientBoosting": GradientBoostingRegressor(random_state=42),
    }

    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Your sklearn version doesn't accept squared=False, so do it manually:
        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        r2 = r2_score(y_test, y_pred)

        rows.append(
            {
                "target": target_col,
                "model": name,
                "rmse": rmse,
                "r2": r2,
                "n_train": len(y_train),
                "n_test": len(y_test),
                "n_features": len(available_feats),
            }
        )

    return pd.DataFrame(rows)


def main():
    os.makedirs("results/tables", exist_ok=True)

    df = load_master("data/AlAin_Stations_2007_2025_clean.csv")

    results_temp = evaluate_models(df, target_col="temperature")
    results_aqi = evaluate_models(df, target_col="AQI")

    res = pd.concat([results_temp, results_aqi], ignore_index=True)

    out_csv = "results/tables/model_comparison_metrics.csv"
    res.to_csv(out_csv, index=False)
    print(f"Saved model comparison metrics to: {out_csv}")
    print(res)


if __name__ == "__main__":
    main()

