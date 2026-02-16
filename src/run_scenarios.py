import argparse
import os

import joblib
import pandas as pd
from data_utils import load_master


def main(args):
    # Load data and models
    df = load_master(args.input)
    os.makedirs(os.path.dirname(args.out_detailed), exist_ok=True)

    rf_T = joblib.load(args.temp_model)
    rf_A = joblib.load(args.aqi_model)

    detailed_rows = []
    summary_rows = []

    # Use the most recent year in the data (should be 2025)
    latest_year = int(df["year"].max())
    stations_latest = df[df["year"] == latest_year]["station_clean"].unique()

    for st in stations_latest:
        row = df[(df["station_clean"] == st) & (df["year"] == latest_year)].iloc[0]

        g0 = row["green_fraction"]
        hum = row["humidity"]
        temp_real = row["temperature"]
        pm25_real = row["PM25"]
        pm10_real = row["PM10"]
        aqi_real = row["AQI"]
        year = int(row["year"])

        # Scenario: +75% relative increase in green_fraction, capped at 1.0
        g1 = min(1.0, g0 * 1.75)

        # ---------- Predict temperature ----------
        X_T = pd.DataFrame(
            [
                {
                    "green_fraction": g0,
                    "humidity": hum,
                    "year": year,
                    "station_clean": st,
                },
                {
                    "green_fraction": g1,
                    "humidity": hum,
                    "year": year,
                    "station_clean": st,
                },
            ]
        )
        T_pred = rf_T.predict(X_T)

        # ---------- Predict AQI ----------
        X_A = pd.DataFrame(
            [
                {
                    "green_fraction": g0,
                    "humidity": hum,
                    "temperature": temp_real,
                    "PM10": pm10_real,
                    "PM25": pm25_real,
                    "year": year,
                    "station_clean": st,
                },
                {
                    "green_fraction": g1,
                    "humidity": hum,
                    "temperature": temp_real,
                    "PM10": pm10_real,
                    "PM25": pm25_real,
                    "year": year,
                    "station_clean": st,
                },
            ]
        )
        AQI_pred = rf_A.predict(X_A)

        # Detailed rows: baseline and scenario
        cases = ["baseline_latest", "scenario_latest_+75pct_green"]
        g_vals = [g0, g1]

        for i in range(2):
            detailed_rows.append(
                {
                    "station": st,
                    "case": cases[i],
                    "year": year,
                    "green_fraction": g_vals[i],
                    "humidity": hum,
                    "temperature_real": temp_real,
                    "PM25_real": pm25_real,
                    "AQI_real": aqi_real,
                    "PM10_real": pm10_real,
                    "temperature_pred": T_pred[i],
                    "AQI_pred": AQI_pred[i],
                }
            )

        # Summary row: scenario - baseline
        dT = T_pred[1] - T_pred[0]
        dAQI = AQI_pred[1] - AQI_pred[0]
        pctAQI = 100 * dAQI / AQI_pred[0] if AQI_pred[0] != 0 else float("nan")

        summary_rows.append(
            {
                "station": st,
                "year": year,
                "green_fraction_baseline": g0,
                "green_fraction_scenario": g1,
                "delta_green_fraction": g1 - g0,
                "temperature_pred_baseline": T_pred[0],
                "temperature_pred_scenario": T_pred[1],
                "delta_temperature_pred": dT,
                "AQI_pred_baseline": AQI_pred[0],
                "AQI_pred_scenario": AQI_pred[1],
                "delta_AQI_pred": dAQI,
                "delta_AQI_percent": pctAQI,
            }
        )

    # Save outputs
    detailed_df = pd.DataFrame(detailed_rows)
    summary_df = pd.DataFrame(summary_rows)

    detailed_df.to_csv(args.out_detailed, index=False)
    summary_df.to_csv(args.out_summary, index=False)

    print("Detailed scenarios saved to:", args.out_detailed)
    print("Summary scenarios saved to:", args.out_summary)


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
        "--out_detailed",
        default="results/tables/scenarios_latest_detailed.csv",
    )
    parser.add_argument(
        "--out_summary",
        default="results/tables/scenarios_latest_summary.csv",
    )
    args = parser.parse_args()
    main(args)

