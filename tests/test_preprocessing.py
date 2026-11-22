import pandas as pd
from src.preprocessing.cleaning import clean_data


def create_sample():
    return pd.DataFrame({
        "meter_reading_kwh": [10, 20],
        "energy_kwh": [12, 22],
        "roof_type": ["Fla.", "Slo."],
        "cloud_cover_pct": [-1, 50]
    })


def test_leakage_column_is_removed():
    cleaned = clean_data(create_sample())
    assert "meter_reading_kwh" not in cleaned.columns


def test_roof_type_is_standardized():
    cleaned = clean_data(create_sample())
    assert cleaned["roof_type"].tolist() == ["flat", "sloped"]


def test_invalid_cloud_cover_is_replaced():
    cleaned = clean_data(create_sample())
    assert pd.isna(cleaned.loc[0, "cloud_cover_pct"])
    assert cleaned.loc[1, "cloud_cover_pct"] == 50
