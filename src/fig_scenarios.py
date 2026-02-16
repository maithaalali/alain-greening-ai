import os
import pandas as pd
import matplotlib.pyplot as plt

STATION_LABELS = {
    "Sweihan": "Sweihan",
    "AlAin_Street": "Al Ain Street",
    "AlTawia": "Al Tawia",
    "AlAin_IslamicInstitute": "Al Ain Islamic Institute",
    "Zakher": "Zakher",
    "AlQuaa": "Al Quaa",
}


def main():
    os.makedirs("results/figures", exist_ok=True)

    scen_path = "results/tables/scenarios_latest_summary.csv"
    df = pd.read_csv(scen_path)
    print("Scenario table columns:", list(df.columns))

    # Standardize station name & nice label
    df["station_clean"] = df["station"]
    df["station_label"] = df["station_clean"].map(STATION_LABELS).fillna(df["station_clean"])

    stations = sorted(df["station_clean"].unique())

    # Helper: extract values in consistent order
    def get_by_station(col):
        vals = []
        for st in stations:
            sub = df[df["station_clean"] == st]
            if len(sub) == 0:
                vals.append(float("nan"))
            else:
                vals.append(sub[col].iloc[0])
        return vals

    aqi_base = get_by_station("AQI_pred_baseline")
    aqi_scen = get_by_station("AQI_pred_scenario")
    temp_base = get_by_station("temperature_pred_baseline")
    temp_scen = get_by_station("temperature_pred_scenario")
    dA = get_by_station("delta_AQI_pred")
    dA_pct = get_by_station("delta_AQI_percent")
    dT = get_by_station("delta_temperature_pred")

    labels = [df[df["station_clean"] == st]["station_label"].iloc[0] for st in stations]
    x = range(len(stations))
    width = 0.35

    # 1) Baseline vs scenario AQI
    plt.figure(figsize=(8, 5))
    xs1 = [xi - width / 2 for xi in x]
    xs2 = [xi + width / 2 for xi in x]
    plt.bar(xs1, aqi_base, width=width, label="Baseline 2025")
    plt.bar(xs2, aqi_scen, width=width, label="Scenario (+75% green)")
    plt.xticks(x, labels, rotation=20)
    plt.ylabel("AQI")
    plt.title("Predicted AQI under +75% green scenario (latest year)")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/scenarios_AQI_baseline_vs_scenario.png", dpi=300)
    plt.close()

    # 2) Baseline vs scenario Temperature
    plt.figure(figsize=(8, 5))
    plt.bar(xs1, temp_base, width=width, label="Baseline 2025")
    plt.bar(xs2, temp_scen, width=width, label="Scenario (+75% green)")
    plt.xticks(x, labels, rotation=20)
    plt.ylabel("Temperature (°C)")
    plt.title("Predicted temperature under +75% green scenario (latest year)")
    plt.grid(axis="y", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/figures/scenarios_temperature_baseline_vs_scenario.png", dpi=300)
    plt.close()

    # 3) Delta AQI (%) by station
    plt.figure(figsize=(8, 5))
    plt.bar(x, dA_pct, width=0.4)
    plt.xticks(x, labels, rotation=20)
    plt.ylabel("ΔAQI (%)")
    plt.title("Relative change in AQI under +75% green scenario")
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/figures/scenarios_delta_AQI_percent_by_station.png", dpi=300)
    plt.close()

    # Also save a small summary CSV with just the key changes
    summary = pd.DataFrame(
        {
            "station_clean": stations,
            "station_label": labels,
            "AQI_baseline": aqi_base,
            "AQI_scenario": aqi_scen,
            "delta_AQI": dA,
            "delta_AQI_percent": dA_pct,
            "T_baseline": temp_base,
            "T_scenario": temp_scen,
            "delta_T": dT,
        }
    )
    out_summary = "results/tables/scenarios_station_changes_75pct_green.csv"
    summary.to_csv(out_summary, index=False)
    print(f"Saved concise scenario summary to: {out_summary}")


if __name__ == "__main__":
    main()

