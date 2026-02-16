import argparse
import os
import pandas as pd
import statsmodels.formula.api as smf

from data_utils import load_master


def main(args):
    df = load_master(args.input)

    os.makedirs(os.path.dirname(args.out_table), exist_ok=True)

    rows = []

    # === TEMPERATURE MODEL ===
    df_T = df.dropna(subset=["temperature", "green_fraction", "humidity"]).copy()
    model_T = smf.ols(
        "temperature ~ green_fraction + humidity + C(station_clean) + C(year)",
        data=df_T,
    )
    res_T = model_T.fit()

    rows.append({
        "model": "temperature",
        "variable": "green_fraction",
        "coef": res_T.params.get("green_fraction", float("nan")),
        "p_value": res_T.pvalues.get("green_fraction", float("nan")),
    })
    rows.append({
        "model": "temperature",
        "variable": "humidity",
        "coef": res_T.params.get("humidity", float("nan")),
        "p_value": res_T.pvalues.get("humidity", float("nan")),
    })

    # === AQI MODEL ===
    df_AQI = df.dropna(subset=["AQI","green_fraction","humidity","temperature"]).copy()
    model_AQI = smf.ols(
        "AQI ~ green_fraction + humidity + temperature + C(station_clean) + C(year)",
        data=df_AQI,
    )
    res_AQI = model_AQI.fit()

    for var in ["green_fraction", "humidity", "temperature"]:
        rows.append({
            "model": "AQI",
            "variable": var,
            "coef": res_AQI.params.get(var, float("nan")),
            "p_value": res_AQI.pvalues.get(var, float("nan")),
        })

    out_df = pd.DataFrame(rows)
    out_df.to_csv(args.out_table, index=False)

    # Also save detailed summaries (for thesis appendix)
    with open(args.out_temp_summary, "w") as f:
        f.write(res_T.summary().as_text())
    with open(args.out_aqi_summary, "w") as f:
        f.write(res_AQI.summary().as_text())

    print("Regression key coefficients saved to:", args.out_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", default="data/AlAin_Stations_2007_2025_clean.csv"
    )
    parser.add_argument(
        "--out_table",
        default="results/tables/regression_key_coefficients.csv",
    )
    parser.add_argument(
        "--out_temp_summary",
        default="results/tables/temperature_model_summary.txt",
    )
    parser.add_argument(
        "--out_aqi_summary",
        default="results/tables/aqi_model_summary.txt",
    )
    args = parser.parse_args()
    main(args)

