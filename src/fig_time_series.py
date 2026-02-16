import os

import matplotlib.pyplot as plt
import pandas as pd

from data_utils import load_master

# Nice display names for stations
STATION_LABELS = {
    "Sweihan": "Sweihan",
    "AlAin_Street": "Al Ain Street",
    "AlTawia": "Al Tawia",
    "AlAin_IslamicInstitute": "Al Ain Islamic Institute",
    "Zakher": "Zakher",
    "AlQuaa": "Al Quaa",
}

def make_time_series_plots(df: pd.DataFrame, out_dir: str = "results/figures"):
    os.makedirs(out_dir, exist_ok=True)

    stations = sorted(df["station_clean"].dropna().unique())

    for st in stations:
        df_st = df[df["station_clean"] == st].sort_values("year")
        label = STATION_LABELS.get(st, st)

        years = df_st["year"].values

        # --- 1) Green fraction ---
        plt.figure(figsize=(6, 4))
        plt.plot(years, df_st["green_fraction"], marker="o")
        plt.xlabel("Year")
        plt.ylabel("Green fraction")
        plt.title(f"{label} – Green fraction vs. Year")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"{st}_green_fraction_timeseries.png")
        plt.savefig(out_path, dpi=300)
        plt.close()

        # --- 2) AQI ---
        if "AQI" in df_st.columns:
            plt.figure(figsize=(6, 4))
            plt.plot(years, df_st["AQI"], marker="o")
            plt.xlabel("Year")
            plt.ylabel("AQI")
            plt.title(f"{label} – AQI vs. Year")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            out_path = os.path.join(out_dir, f"{st}_AQI_timeseries.png")
            plt.savefig(out_path, dpi=300)
            plt.close()

        # --- 3) Temperature ---
        if "temperature" in df_st.columns:
            plt.figure(figsize=(6, 4))
            plt.plot(years, df_st["temperature"], marker="o")
            plt.xlabel("Year")
            plt.ylabel("Temperature (°C)")
            plt.title(f"{label} – Temperature vs. Year")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            out_path = os.path.join(out_dir, f"{st}_temperature_timeseries.png")
            plt.savefig(out_path, dpi=300)
            plt.close()

    print(f"Saved time-series figures to: {out_dir}")


def main():
    # Load your master dataset
    df = load_master("data/AlAin_Stations_2007_2025_clean.csv")
    make_time_series_plots(df, out_dir="results/figures")


if __name__ == "__main__":
    main()

