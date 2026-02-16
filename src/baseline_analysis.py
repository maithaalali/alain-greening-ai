import argparse
import os
import pandas as pd

from data_utils import load_master


def main(args):
    # Load cleaned master dataset
    df = load_master(args.input)

    # Make sure output folders exist
    os.makedirs(os.path.dirname(args.out_summary), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_hotspots), exist_ok=True)

    # ===== 1) Descriptive summary per station =====
    summary = (
        df.groupby('station_clean')[['AQI', 'PM25', 'PM10',
                                     'temperature', 'humidity',
                                     'green_fraction']]
          .describe()
    )
    summary.to_csv(args.out_summary)

    # ===== 2) Heat & PM2.5 anomalies (hotspots) =====
    df['T_mean_year'] = df.groupby('year')['temperature'].transform('mean')
    df['T_anomaly'] = df['temperature'] - df['T_mean_year']

    df['PM25_mean_year'] = df.groupby('year')['PM25'].transform('mean')
    df['PM25_anomaly'] = df['PM25'] - df['PM25_mean_year']

    # Thresholds (you can tune these later)
    heat_hot = df[df['T_anomaly'] > 1.0]
    pm25_hot = df[df['PM25_anomaly'] > 10]

    heat_counts = heat_hot.groupby('station_clean').size().rename('heat_hotspot_years')
    pm25_counts = pm25_hot.groupby('station_clean').size().rename('pm25_hotspot_years')

    hotspot_table = pd.concat([heat_counts, pm25_counts], axis=1).fillna(0).astype(int)
    hotspot_table.to_csv(args.out_hotspots)

    print("Baseline summary saved to:", args.out_summary)
    print("Hotspot summary saved to:", args.out_hotspots)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/AlAin_Stations_2007_2025_clean.csv",
        help="Path to cleaned master CSV",
    )
    parser.add_argument(
        "--out_summary",
        default="results/tables/baseline_summary.csv",
        help="Output CSV for per-station descriptive stats",
    )
    parser.add_argument(
        "--out_hotspots",
        default="results/tables/hotspots_summary.csv",
        help="Output CSV for heat/PM25 hotspot counts",
    )
    args = parser.parse_args()
    main(args)
import argparse
import os
import pandas as pd

from data_utils import load_master


def main(args):
    # Load cleaned master dataset
    df = load_master(args.input)

    # Make sure output folders exist
    os.makedirs(os.path.dirname(args.out_summary), exist_ok=True)
    os.makedirs(os.path.dirname(args.out_hotspots), exist_ok=True)

    # ===== 1) Descriptive summary per station =====
    summary = (
        df.groupby('station_clean')[['AQI', 'PM25', 'PM10',
                                     'temperature', 'humidity',
                                     'green_fraction']]
          .describe()
    )
    summary.to_csv(args.out_summary)

    # ===== 2) Heat & PM2.5 anomalies (hotspots) =====
    df['T_mean_year'] = df.groupby('year')['temperature'].transform('mean')
    df['T_anomaly'] = df['temperature'] - df['T_mean_year']

    df['PM25_mean_year'] = df.groupby('year')['PM25'].transform('mean')
    df['PM25_anomaly'] = df['PM25'] - df['PM25_mean_year']

    # Thresholds (you can tune these later)
    heat_hot = df[df['T_anomaly'] > 1.0]
    pm25_hot = df[df['PM25_anomaly'] > 10]

    heat_counts = heat_hot.groupby('station_clean').size().rename('heat_hotspot_years')
    pm25_counts = pm25_hot.groupby('station_clean').size().rename('pm25_hotspot_years')

    hotspot_table = pd.concat([heat_counts, pm25_counts], axis=1).fillna(0).astype(int)
    hotspot_table.to_csv(args.out_hotspots)

    print("Baseline summary saved to:", args.out_summary)
    print("Hotspot summary saved to:", args.out_hotspots)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="data/AlAin_Stations_2007_2025_clean.csv",
        help="Path to cleaned master CSV",
    )
    parser.add_argument(
        "--out_summary",
        default="results/tables/baseline_summary.csv",
        help="Output CSV for per-station descriptive stats",
    )
    parser.add_argument(
        "--out_hotspots",
        default="results/tables/hotspots_summary.csv",
        help="Output CSV for heat/PM25 hotspot counts",
    )
    args = parser.parse_args()
    main(args)

