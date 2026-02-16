import argparse
import joblib
import pandas as pd

from data_utils import load_master


def main(args):
    df = load_master(args.input)

    # Load trained models
    rf_T = joblib.load(args.temp_model)
    rf_A = joblib.load(args.aqi_model)

    # Get one baseline row
    row = df[(df["station_clean"] == args.station) & (df["year"] == args.year)].iloc[0]

    hum = row["humidity"]
    temp_real = row["temperature"]
    pm25_real = row["PM25"]
    pm10_real = row["PM10"]

    g = args.green_fraction

    # Predict temperature
    X_T = pd.DataFrame([{
        "green_fraction": g,
        "humidity": hum,
        "year": args.year,
        "station_clean": args.station,
    }])
    T_pred = rf_T.predict(X_T)[0]

    # Predict AQI
    X_A = pd.DataFrame([{
        "green_fraction": g,
        "humidity": hum,
        "temperature": temp_real,
        "PM10": pm10_real,
        "PM25": pm25_real,
        "year": args.year,
        "station_clean": args.station,
    }])
    AQI_pred = rf_A.predict(X_A)[0]

    print(f"Station: {args.station}, Year: {args.year}")
    print(f"Input green_fraction: {g:.3f}")
    print(f"Predicted Temperature: {T_pred:.3f} °C")
    print(f"Predicted AQI: {AQI_pred:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/AlAin_Stations_2007_2025_clean.csv")
    parser.add_argument("--temp_model", default="models/rf_temperature.joblib")
    parser.add_argument("--aqi_model", default="models/rf_aqi.joblib")
    parser.add_argument("--station", required=True,
                        help="Station name, e.g. Sweihan, AlTawia, Zakher, AlAin_Street, AlAin_IslamicInstitute, AlQuaa")
    parser.add_argument("--year", type=int, default=2025)
    parser.add_argument("--green_fraction", type=float, required=True)
    args = parser.parse_args()
    main(args)

