import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from data_utils import load_master

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

STATION_LABELS = {
    "Sweihan": "Sweihan",
    "AlAin_Street": "Al Ain Street",
    "AlTawia": "Al Tawia",
    "AlAin_IslamicInstitute": "Al Ain Islamic Institute",
    "Zakher": "Zakher",
    "AlQuaa": "Al Quaa",
}


def main(n_boot: int = 30):
    os.makedirs("results/tables", exist_ok=True)

    df = load_master("data/AlAin_Stations_2007_2025_clean.csv")

    # Prepare training data for AQI
    feat_cols = [c for c in FEATURE_CANDIDATES if c in df.columns]
    cols_needed = feat_cols + ["AQI"]
    df_train = df.dropna(subset=cols_needed).copy()

    if df_train.empty:
        raise ValueError("No data available to train RF for AQI.")

    X = df_train[feat_cols].values
    y = df_train["AQI"].values

    results = []

    for st in sorted(df["station_clean"].dropna().unique()):
        df_st = df[df["station_clean"] == st].dropna(subset=["green_fraction"])
        if df_st.empty:
            continue

        # Use last available year as baseline
        last_row = df_st.sort_values("year").iloc[-1].copy()
        base_green = float(last_row["green_fraction"])
        scen_green = min(base_green * 1.75, 0.9)  # +75%, capped at 0.9

        base_feats = last_row[feat_cols].copy()
        scen_feats = base_feats.copy()
        if "green_fraction" in scen_feats.index:
            scen_feats["green_fraction"] = scen_green

        preds_base = []
        preds_scen = []

        for seed in range(n_boot):
            rf = RandomForestRegressor(
                n_estimators=200,
                random_state=seed,
            )
            rf.fit(X, y)

            Xb = pd.DataFrame([base_feats])
            Xs = pd.DataFrame([scen_feats])

            pb = rf.predict(Xb)[0]
            ps = rf.predict(Xs)[0]

            preds_base.append(pb)
            preds_scen.append(ps)

        base_arr = np.array(preds_base)
        scen_arr = np.array(preds_scen)
        delta_arr = scen_arr - base_arr

        results.append(
            {
                "station_clean": st,
                "station_label": STATION_LABELS.get(st, st),
                "n_bootstraps": n_boot,
                "AQI_baseline_mean": base_arr.mean(),
                "AQI_baseline_std": base_arr.std(ddof=1),
                "AQI_scenario_mean": scen_arr.mean(),
                "AQI_scenario_std": scen_arr.std(ddof=1),
                "delta_AQI_mean": delta_arr.mean(),
                "delta_AQI_std": delta_arr.std(ddof=1),
            }
        )

    res_df = pd.DataFrame(results)
    out_csv = "results/tables/scenarios_uncertainty_AQI_75pct_RF.csv"
    res_df.to_csv(out_csv, index=False)
    print(f"Saved scenario uncertainty table to: {out_csv}")
    print(res_df)


if __name__ == "__main__":
    main()

