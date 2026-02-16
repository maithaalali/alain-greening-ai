import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_utils import load_master

STATION_LABELS = {
    "Sweihan": "Sweihan",
    "AlAin_Street": "Al Ain Street",
    "AlTawia": "Al Tawia",
    "AlAin_IslamicInstitute": "Al Ain Islamic Institute",
    "Zakher": "Zakher",
    "AlQuaa": "Al Quaa",
}


def add_hotspot_flags(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add boolean flags:
      - heat_hotspot: temperature unusually high for that station
      - pm25_hotspot: PM2.5 unusually high for that station
    Thresholds are station-specific: mean + 0.5 * std.
    """
    df = df.copy()
    df["heat_hotspot"] = False
    df["pm25_hotspot"] = False

    for st, g in df.groupby("station_clean"):
        # Temperature hotspot
        if "temperature" in g.columns and g["temperature"].notna().sum() > 3:
            mean_T = g["temperature"].mean()
            std_T = g["temperature"].std()
            thr_T = mean_T + 0.5 * std_T
            idx_hot_T = g.index[g["temperature"] > thr_T]
            df.loc[idx_hot_T, "heat_hotspot"] = True

        # PM2.5 hotspot
        if "PM25" in g.columns and g["PM25"].notna().sum() > 3:
            mean_PM = g["PM25"].mean()
            std_PM = g["PM25"].std()
            thr_PM = mean_PM + 0.5 * std_PM
            idx_hot_PM = g.index[g["PM25"] > thr_PM]
            df.loc[idx_hot_PM, "pm25_hotspot"] = True

    return df


def make_hotspot_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Count number of hotspot years per station."""
    grp = df.groupby("station_clean")
    summary = pd.DataFrame({
        "n_heat_hotspot_years": grp["heat_hotspot"].sum(),
        "n_pm25_hotspot_years": grp["pm25_hotspot"].sum(),
        "n_years_with_data": grp["year"].nunique(),
    }).reset_index()

    # Pretty labels
    summary["station_label"] = summary["station_clean"].map(STATION_LABELS)
    return summary


def plot_hotspot_bars(summary: pd.DataFrame, out_dir: str = "results/figures"):
    os.makedirs(out_dir, exist_ok=True)

    # Sort by number of hotspot years
    summary = summary.sort_values("n_heat_hotspot_years", ascending=False)

    # Heat hotspots
    plt.figure(figsize=(6, 4))
    plt.bar(summary["station_label"], summary["n_heat_hotspot_years"])
    plt.ylabel("Number of heat-hotspot years")
    plt.title("Heat anomaly hotspots by station")
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.grid(axis="y", alpha=0.3)
    plt.savefig(os.path.join(out_dir, "heat_hotspot_years_per_station.png"), dpi=300)
    plt.close()

    # PM2.5 hotspots (only where we have PM2.5 data)
    summary_pm = summary[summary["n_pm25_hotspot_years"] > 0].copy()
    if len(summary_pm) > 0:
        summary_pm = summary_pm.sort_values("n_pm25_hotspot_years", ascending=False)
        plt.figure(figsize=(6, 4))
        plt.bar(summary_pm["station_label"], summary_pm["n_pm25_hotspot_years"])
        plt.ylabel("Number of PM2.5-hotspot years")
        plt.title("PM2.5 hotspots by station")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.grid(axis="y", alpha=0.3)
        plt.savefig(
            os.path.join(out_dir, "pm25_hotspot_years_per_station.png"), dpi=300
        )
        plt.close()


def main():
    os.makedirs("results/tables", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    # Load cleaned master file
    df = load_master("data/AlAin_Stations_2007_2025_clean.csv")

    df_hot = add_hotspot_flags(df)
    summary = make_hotspot_summary(df_hot)

    # Save summary table
    out_csv = "results/tables/hotspot_counts_stationlevel.csv"
    summary.to_csv(out_csv, index=False)
    print(f"Saved hotspot summary to: {out_csv}")

    # Make bar plots
    plot_hotspot_bars(summary, out_dir="results/figures")
    print("Saved hotspot bar plots to results/figures")


if __name__ == "__main__":
    main()

