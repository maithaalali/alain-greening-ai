import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from data_utils import load_master

# Nice labels for plots
STATION_LABELS = {
    "Sweihan": "Sweihan",
    "AlAin_Street": "Al Ain Street",
    "AlTawia": "Al Tawia",
    "AlAin_IslamicInstitute": "Al Ain Islamic Institute",
    "Zakher": "Zakher",
    "AlQuaa": "Al Quaa",
}


def scatter_with_fit(df, x_col, y_col, out_path, y_label, title):
    """
    Scatter of y vs x (colored by station) + global linear fit line.
    """
    plt.figure(figsize=(7, 5))

    # Scatter per station
    for st, g in df.groupby("station_clean"):
        g_plot = g[[x_col, y_col]].dropna()
        if len(g_plot) == 0:
            continue
        plt.scatter(
            g_plot[x_col],
            g_plot[y_col],
            alpha=0.7,
            label=STATION_LABELS.get(st, st),
        )

    # Global linear fit (all stations together)
    df_fit = df[[x_col, y_col]].dropna()
    if len(df_fit) >= 3:
        m, b = np.polyfit(df_fit[x_col], df_fit[y_col], 1)
        xs = np.linspace(df_fit[x_col].min(), df_fit[x_col].max(), 100)
        ys = m * xs + b
        plt.plot(xs, ys, linewidth=2)

    plt.xlabel("Green fraction (vegetated area / total buffer)")
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend(fontsize=8)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def main():
    os.makedirs("results/figures", exist_ok=True)

    # 1) Load cleaned master dataset
    df = load_master("data/AlAin_Stations_2007_2025_clean.csv")

    # 2) Keep only rows where we actually have green_fraction and targets
    df_T = df.dropna(subset=["green_fraction", "temperature"]).copy()
    df_AQI = df.dropna(subset=["green_fraction", "AQI"]).copy()

    # 3) Scatter plots with regression line
    scatter_with_fit(
        df_T,
        x_col="green_fraction",
        y_col="temperature",
        out_path="results/figures/green_vs_temperature_allstations.png",
        y_label="Temperature (°C)",
        title="Effect of green fraction on temperature (all stations)",
    )

    scatter_with_fit(
        df_AQI,
        x_col="green_fraction",
        y_col="AQI",
        out_path="results/figures/green_vs_AQI_allstations.png",
        y_label="AQI",
        title="Effect of green fraction on AQI (all stations)",
    )

    # 4) Print key regression slopes from the regression results
    coef_path_1 = "results/tables/regression_key_coefficients.csv"
    coef_path_2 = "results/tables/regression_key_coefficients_temp_AQI.csv"

    if os.path.exists(coef_path_1):
        coef = pd.read_csv(coef_path_1)
    elif os.path.exists(coef_path_2):
        coef = pd.read_csv(coef_path_2)
    else:
        print("\n[WARN] Could not find regression coefficient table.")
        return

    # Show columns so we know what we have
    print("\nAvailable columns in regression table:", list(coef.columns))

    # Use whichever column name exists for the coefficient
    if "coefficient" in coef.columns:
        coef_col = "coefficient"
    elif "coef" in coef.columns:
        coef_col = "coef"
    else:
        print("[WARN] No 'coefficient' or 'coef' column found; skipping slope print.")
        return

    # Filter just green_fraction effects
    cf_green = coef[coef["variable"] == "green_fraction"].copy()
    if cf_green.empty:
        print("\n[WARN] No rows for variable 'green_fraction' in regression table.")
        return

    print("\nKey regression slopes for green_fraction:")
    for _, row in cf_green.iterrows():
        print(
            f"  {row['model']}: d({row['model']})/d(green_fraction) = "
            f"{row[coef_col]:.3f}"
        )


if __name__ == "__main__":
    main()

