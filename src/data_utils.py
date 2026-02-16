import pandas as pd

STATION_MAP = {
    'Sweihan': 'Sweihan',
    'al ain street': 'AlAin_Street',
    'altawia': 'AlTawia',
    'alain islamic institue': 'AlAin_IslamicInstitute',
    'zakher': 'Zakher',
    'zakherÊ': 'Zakher',
    'al quaa': 'AlQuaa',
}

NUM_COLS = [
    'AQI','PM10','PM25','SO2','CO','O3','NO2',
    'temperature','humidity','green_fraction',
    'veg_area_m2','bufferArea_m2','year'
]

def load_master(path: str) -> pd.DataFrame:
    """Load and clean the master Al Ain station-year dataset."""
    df = pd.read_csv(path)

    # Clean station names
    df['station_clean'] = df['station'].map(STATION_MAP)

    # Numeric types
    for c in NUM_COLS:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors='coerce')

    # Drop rows missing station or year
    df = df.dropna(subset=['year','station_clean']).copy()
    df['year'] = df['year'].astype(int)

    return df

