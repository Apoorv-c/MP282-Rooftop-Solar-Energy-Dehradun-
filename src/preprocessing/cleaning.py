import pandas as pd


def clean_data(df, fix_sunshine=False):
    df = df.copy()

    # Fix sunshine values recorded in minutes
    if fix_sunshine:
        industrial = df["area"] == "Industrial"
        df.loc[industrial, "sunshine_hours"] /= 60

    # Replace invalid cloud cover values
    df["cloud_cover_pct"] = df["cloud_cover_pct"].replace(-1, pd.NA)

    # Standardize roof names
    df["roof_type"] = (
        df["roof_type"]
        .astype("string")
        .str.strip()
        .str.lower()
        .str.replace(".", "", regex=False)
        .replace({"fla": "flat", "slo": "sloped"})
    )

    # Remove data leakage column
    df = df.drop(columns=["meter_reading_kwh"], errors="ignore")

    return df
