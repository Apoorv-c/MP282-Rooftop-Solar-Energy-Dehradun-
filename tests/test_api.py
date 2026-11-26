"""
API integration tests using FastAPI TestClient.
These tests do NOT read any CSV files; all data is inline.
"""
import warnings
import pytest

# Suppress the starlette/httpx deprecation warning that appears with httpx 0.28+
warnings.filterwarnings(
    "ignore",
    message=".*Using `httpx` with `starlette.testclient` is deprecated.*",
)

from fastapi.testclient import TestClient  # noqa: E402
from app.main import app  # noqa: E402

client = TestClient(app)

VALID_PAYLOAD = {
    "panel_area_m2": 20.0,
    "sunshine_hours": 6.0,
    "temp_c": 30.0,
    "dust_days": 5.0,
    "panel_age_yr": 2.0,
    "cloud_cover_pct": 10.0,
    "roof_type": "flat",
    "area": "New Colony",
}


# ------------------------------------------------------------------
# /health
# ------------------------------------------------------------------
def test_health_returns_ok():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


# ------------------------------------------------------------------
# /predict - happy path
# ------------------------------------------------------------------
def test_predict_valid_request_returns_prediction():
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert "prediction" in body
    assert isinstance(body["prediction"], float)


# ------------------------------------------------------------------
# /predict - client rule 1: panel_area_m2 > 100 must get 422
# ------------------------------------------------------------------
def test_predict_rejects_panel_area_above_100():
    payload = {**VALID_PAYLOAD, "panel_area_m2": 101.0}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


# ------------------------------------------------------------------
# /predict - client rule 2: dust_days < 0 must get 422
# ------------------------------------------------------------------
def test_predict_rejects_negative_dust_days():
    payload = {**VALID_PAYLOAD, "dust_days": -1.0}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


# ------------------------------------------------------------------
# /predict - extra field must get 422 (extra="forbid" in schema)
# ------------------------------------------------------------------
def test_predict_rejects_extra_fields():
    payload = {**VALID_PAYLOAD, "unknown_field": "oops"}
    r = client.post("/predict", json=payload)
    assert r.status_code == 422
