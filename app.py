import os
import sys
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import streamlit as st
import joblib

import folium
from streamlit_folium import st_folium

# Make sure we can import from src/
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from data_utils import load_master  # noqa: E402

# -------------------------------------------------------------------
# Station metadata
# -------------------------------------------------------------------

STATION_LABELS = {
    "Sweihan": "Sweihan",
    "AlAin_Street": "Al Ain Street",
    "AlTawia": "Al Tawia",
    "AlAin_IslamicInstitute": "Al Ain Islamic Institute",
    "Zakher": "Zakher",
    "AlQuaa": "Al Quaa",
}

# APPROXIMATE coordinates; replace with precise coordinates if needed
STATION_COORDS: Dict[str, Tuple[float, float]] = {
    "Sweihan": (24.45, 55.35),
    "AlAin_Street": (24.22, 55.74),
    "AlTawia": (24.23, 55.55),
    "AlAin_IslamicInstitute": (24.24, 55.75),
    "Zakher": (24.17, 55.71),
    "AlQuaa": (23.56, 55.30),
}


# -------------------------------------------------------------------
# Cached loaders
# -------------------------------------------------------------------

@st.cache_data
def get_data():
    df = load_master("data/AlAin_Stations_2007_2025_clean.csv")
    if "station_clean" not in df.columns:
        df["station_clean"] = df["station"]
    return df


@st.cache_resource
def get_models():
    rf_T = joblib.load("models/rf_temperature.joblib")
    rf_AQI = joblib.load("models/rf_aqi.joblib")

    feat_T = list(getattr(rf_T, "feature_names_in_", []))
    feat_AQI = list(getattr(rf_AQI, "feature_names_in_", []))
    return rf_T, rf_AQI, feat_T, feat_AQI


@st.cache_data
def get_hotspots():
    path = "results/tables/hotspot_counts_stationlevel.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data
def get_uncertainty():
    """
    Load RF ensemble uncertainty for +75% greening scenario, if available.
    """
    path = "results/tables/scenarios_uncertainty_AQI_75pct_RF.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data
def compute_green_labels(df: pd.DataFrame, recent_years: int = 10):
    """
    Classify stations as Low / Medium / High green based on mean green_fraction
    over the last `recent_years`.
    """
    max_year = df["year"].max()
    cut_year = max_year - recent_years + 1
    recent = df[df["year"] >= cut_year]
    mean_green = recent.groupby("station_clean")["green_fraction"].mean()
    if mean_green.empty:
        return {}
    q1 = mean_green.quantile(0.33)
    q2 = mean_green.quantile(0.66)
    labels = {}
    for st, val in mean_green.items():
        if val <= q1:
            labels[st] = "Low green"
        elif val <= q2:
            labels[st] = "Medium green"
        else:
            labels[st] = "High green"
    return labels


# -------------------------------------------------------------------
# Helper functions
# -------------------------------------------------------------------

def get_baseline_row(df, station, year):
    """
    For a given station and year:
      - If that year exists in the data, return that row.
      - If year > max available, use the last available year for that station
        and just update the 'year' field (approximate future scenario).
    """
    station_rows = df[df["station_clean"] == station].sort_values("year")
    if station_rows.empty:
        return None

    hist = station_rows[station_rows["year"] == year]
    if not hist.empty:
        return hist.iloc[0].copy()

    last_row = station_rows.iloc[-1].copy()
    last_row["year"] = year
    return last_row


def predict_with_model(model, feature_cols, row: pd.Series):
    X = pd.DataFrame([row])
    if feature_cols:
        cols = [c for c in feature_cols if c in X.columns]
        if not cols:
            st.error("No overlapping feature columns between model and input.")
            return np.nan
        X = X[cols]
    return float(model.predict(X)[0])


def make_map(selected_station: str, green_label: str):
    m = folium.Map(location=[24.22, 55.74], zoom_start=9)
    for st, (lat, lon) in STATION_COORDS.items():
        label = STATION_LABELS.get(st, st)
        if st == selected_station:
            color = "red"
            radius = 10
        else:
            color = "green"
            radius = 6
        popup = f"{label}"
        if st == selected_station and green_label:
            popup += f" ({green_label})"
        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fill_opacity=0.8,
            popup=popup,
        ).add_to(m)
    return m


# -------------------------------------------------------------------
# Streamlit app
# -------------------------------------------------------------------

def main():
    st.set_page_config(
        page_title="Al Ain Greening AI Planner",
        layout="wide",
    )

    st.title("🌱 Al Ain Greening AI Planner")
    st.markdown(
        """
        This tool uses real sensor data (2007–2025), satellite-derived green cover,
        and machine learning models to explore how increasing greenery around
        air-quality stations in Al Ain could affect **AQI** and **temperature**.
        """
    )

    df = get_data()
    rf_T, rf_AQI, feat_T, feat_AQI = get_models()
    hotspots = get_hotspots()
    unc = get_uncertainty()
    green_labels = compute_green_labels(df, recent_years=10)

    min_year = int(df["year"].min())
    max_year = int(df["year"].max())

    # Sidebar controls
    st.sidebar.header("Controls")

    station_options = sorted(df["station_clean"].unique())
    station = st.sidebar.selectbox(
        "Station",
        station_options,
        format_func=lambda s: STATION_LABELS.get(s, s),
    )

    year = st.sidebar.slider(
        "Year",
        min_value=min_year,
        max_value=2050,
        value=max_year,
        help="Up to 2025 uses historical data; beyond 2025 uses approximate future scenario.",
    )

    # Greening presets
    st.sidebar.markdown("### Greening scenario")
    preset = st.sidebar.radio(
        "Preset increase",
        ["Baseline (0%)", "+25%", "+50%", "+75%", "Max (90% green)"],
        index=3,
    )
    preset_map = {
        "Baseline (0%)": 0.0,
        "+25%": 0.25,
        "+50%": 0.5,
        "+75%": 0.75,
        "Max (90% green)": 1.0,
    }
    rel_increase = preset_map[preset]

    baseline_row = get_baseline_row(df, station, year)
    if baseline_row is None:
        st.error("No data available for selected station.")
        return

    baseline_green = float(baseline_row.get("green_fraction", 0.0))

    st.sidebar.write(f"Baseline green fraction: **{baseline_green:.3f}**")

    scenario_green_default = min(baseline_green * (1.0 + rel_increase), 0.9)

    # Allow fine tuning
    scenario_green = st.sidebar.slider(
        "Scenario green fraction (absolute)",
        min_value=0.0,
        max_value=0.9,
        value=float(scenario_green_default),
        step=0.01,
        help="Final vegetated fraction within the station buffer under the greening scenario.",
    )

    # If ensemble uncertainty exists, show global insights in sidebar
    if unc is not None and not unc.empty:
        st.sidebar.markdown("### +75% greening – AQI impact (RF ensemble)")
        try:
            best_idx = unc["delta_AQI_mean"].idxmin()
            worst_idx = unc["delta_AQI_mean"].idxmax()
            best_row = unc.loc[best_idx]
            worst_row = unc.loc[worst_idx]

            st.sidebar.write(
                f"**Largest AQI improvement:** {best_row['station_label']}  \n"
                f"ΔAQI = {best_row['delta_AQI_mean']:.1f} ± {best_row['delta_AQI_std']:.1f}"
            )
            st.sidebar.write(
                f"**Smallest / no improvement:** {worst_row['station_label']}  \n"
                f"ΔAQI = {worst_row['delta_AQI_mean']:.1f} ± {worst_row['delta_AQI_std']:.1f}"
            )
        except Exception:
            pass

    # ----------------------------------------------------------------
    # Main layout
    # ----------------------------------------------------------------
    col_left, col_right = st.columns([2, 2])

    with col_left:
        st.subheader("📊 Station-level predictions")

        st.write(
            f"**Station:** {STATION_LABELS.get(station, station)}  "
            f"| **Year:** {year}"
        )

        green_label = green_labels.get(station, "Unknown")
        st.write(f"Green status (last 10 years): **{green_label}**")

        if hotspots is not None and "station_clean" in hotspots.columns:
            hs_row = hotspots[hotspots["station_clean"] == station]
            if not hs_row.empty:
                n_heat = int(hs_row["n_heat_hotspot_years"].iloc[0])
                n_pm = int(hs_row["n_pm25_hotspot_years"].iloc[0])
                st.write(
                    f"Heat-hotspot years: **{n_heat}**, "
                    f"PM₂.₅-hotspot years: **{n_pm}** (2007–2025)"
                )

        # Baseline predictions
        baseline_pred_T = predict_with_model(rf_T, feat_T, baseline_row)
        baseline_pred_AQI = predict_with_model(rf_AQI, feat_AQI, baseline_row)

        # Scenario predictions
        scenario_row = baseline_row.copy()
        scenario_row["green_fraction"] = scenario_green

        scenario_pred_T = predict_with_model(rf_T, feat_T, scenario_row)
        scenario_pred_AQI = predict_with_model(rf_AQI, feat_AQI, scenario_row)

        dT = scenario_pred_T - baseline_pred_T
        dAQI = scenario_pred_AQI - baseline_pred_AQI
        dAQI_pct = (
            100.0 * dAQI / baseline_pred_AQI if baseline_pred_AQI != 0 else np.nan
        )

        mc1, mc2 = st.columns(2)
        with mc1:
            st.metric(
                "Baseline AQI",
                f"{baseline_pred_AQI:.1f}",
                help="Model-predicted AQI for baseline greenness.",
            )
            st.metric(
                "Scenario AQI",
                f"{scenario_pred_AQI:.1f}",
                f"{dAQI:+.1f} ({dAQI_pct:+.1f}%)",
            )
        with mc2:
            st.metric(
                "Baseline Temperature (°C)",
                f"{baseline_pred_T:.2f}",
                help="Model-predicted annual mean temperature.",
            )
            st.metric(
                "Scenario Temperature (°C)",
                f"{scenario_pred_T:.2f}",
                f"{dT:+.2f} °C",
            )

        # --- NEW: station-level ensemble uncertainty summary ---
        if unc is not None and "station_clean" in unc.columns:
            unc_row = unc[unc["station_clean"] == station]
            if not unc_row.empty:
                row_u = unc_row.iloc[0]
                d_mean = row_u["delta_AQI_mean"]
                d_std = row_u["delta_AQI_std"]
                significance = (
                    "larger than the ensemble spread (|Δ| > 1σ)"
                    if abs(d_mean) > d_std
                    else "similar to the ensemble spread (|Δ| ≈ 1σ)"
                )

                st.markdown("---")
                st.markdown("### RF ensemble result for +75% greening (last observed year)")
                st.write(
                    f"For this station, the RF ensemble (n = {int(row_u['n_bootstraps'])}) "
                    f"predicts for a **+75% green** scenario:\n\n"
                    f"- ΔAQI_mean = **{d_mean:.1f}**  \n"
                    f"- ΔAQI_std = **{d_std:.1f}**  \n\n"
                    f"This means the predicted improvement is **{significance}**."
                )

        st.markdown("---")
        st.markdown("### Historical trends at this station")
        df_hist = df[df["station_clean"] == station].sort_values("year")
        cols_to_plot = [
            c for c in ["green_fraction", "AQI", "temperature"] if c in df_hist.columns
        ]
        if cols_to_plot:
            ts_df = df_hist[["year"] + cols_to_plot].set_index("year")
            st.line_chart(ts_df)
        else:
            st.info("No time-series variables available to plot.")

        st.markdown("---")
        st.markdown("### Scenario time series (download)")

        # Build timeseries from latest real year to 2050 for current preset
        years = list(range(max_year, 2051))
        records = []
        for y in years:
            row_y = get_baseline_row(df, station, y)
            if row_y is None:
                continue
            base_T_y = predict_with_model(rf_T, feat_T, row_y)
            base_AQI_y = predict_with_model(rf_AQI, feat_AQI, row_y)

            scen_row_y = row_y.copy()
            base_green_y = float(row_y.get("green_fraction", 0.0))
            scen_green_y = min(base_green_y * (1.0 + rel_increase), 0.9)
            scen_row_y["green_fraction"] = scen_green_y

            scen_T_y = predict_with_model(rf_T, feat_T, scen_row_y)
            scen_AQI_y = predict_with_model(rf_AQI, feat_AQI, scen_row_y)

            dT_y = scen_T_y - base_T_y
            dAQI_y = scen_AQI_y - base_AQI_y
            dAQI_pct_y = (
                100.0 * dAQI_y / base_AQI_y if base_AQI_y != 0 else np.nan
            )

            records.append(
                {
                    "station": station,
                    "year": y,
                    "green_fraction_baseline": base_green_y,
                    "green_fraction_scenario": scen_green_y,
                    "AQI_baseline": base_AQI_y,
                    "AQI_scenario": scen_AQI_y,
                    "delta_AQI": dAQI_y,
                    "delta_AQI_percent": dAQI_pct_y,
                    "T_baseline": base_T_y,
                    "T_scenario": scen_T_y,
                    "delta_T": dT_y,
                }
            )

        if records:
            ts_scen_df = pd.DataFrame(records)
            csv_bytes = ts_scen_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download scenario time series (latest year–2050)",
                data=csv_bytes,
                file_name=f"{station}_scenario_timeseries_{max_year}_2050.csv",
                mime="text/csv",
            )

        st.markdown("---")
        st.markdown(
            "**Interpretation (for your own analysis):**\n\n"
            f"- Baseline green fraction: **{baseline_green:.3f}**, "
            f"scenario green fraction: **{scenario_green:.3f}**.\n"
            f"- Under this scenario, the RF models predict AQI changes by "
            f"**{dAQI:+.1f} units** ({dAQI_pct:+.1f}%), and temperature by "
            f"**{dT:+.2f} °C** at this station and year."
        )

        if year > max_year:
            st.info(
                "Note: This is an approximate future scenario. "
                "We assume meteorology and pollution levels similar to the last observed year, "
                "and only change the year index and green fraction."
            )

    with col_right:
        st.subheader("🗺️ Station map")
        m = make_map(station, green_labels.get(station, ""))
        st_folium(m, width=700, height=450)


if __name__ == "__main__":
    main()

