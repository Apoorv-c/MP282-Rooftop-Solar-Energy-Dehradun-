from functools import lru_cache
import joblib
import pandas as pd

FEATURES = ["panel_area_m2", "sunshine_hours", "temp_c", "dust_days",
            "panel_age_yr", "cloud_cover_pct", "roof_type", "area"]

@lru_cache(maxsize=1)
def get_model(path: str = "models/model.joblib"):
    return joblib.load(path)

def predict_energy(payload: dict) -> float:
    row = pd.DataFrame([payload], columns=FEATURES)
    return float(get_model().predict(row)[0])