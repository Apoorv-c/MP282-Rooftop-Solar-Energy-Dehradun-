
import logging
from contextlib import asynccontextmanager
from typing import Literal, Optional

import mlflow
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from mlflow.tracking import MlflowClient

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)
logger = logging.getLogger(__name__)

TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "MP282_Solar_Energy_Model"
MODEL_ALIAS = "champion"


class SolarPredictionInput(BaseModel):
    panel_area_m2: float = Field(gt=0)
    sunshine_hours: float = Field(ge=0)
    temp_c: float
    dust_days: int = Field(ge=0)
    panel_age_yr: float = Field(ge=0)
    cloud_cover_pct: Optional[float] = Field(default=None, ge=0, le=100)
    roof_type: Literal["flat", "tin", "sloped"]
    area: Literal["Old City", "New Colony", "Industrial"]


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = None
    app.state.model_version = None

    try:
        mlflow.set_tracking_uri(TRACKING_URI)
        client = MlflowClient()

        version = client.get_model_version_by_alias(
            MODEL_NAME, MODEL_ALIAS
        )

        app.state.model = mlflow.pyfunc.load_model(
            f"models:/{MODEL_NAME}@{MODEL_ALIAS}"
        )
        app.state.model_version = version.version

        logger.info(
            "Champion model loaded successfully. Version: %s",
            version.version
        )
    except Exception:
        logger.exception("Failed to load MLflow champion model")

    yield
    logger.info("API shutting down")


app = FastAPI(
    title="MP282 Solar Energy Prediction API",
    description="Solar energy prediction using the MLflow champion model.",
    version="1.0.0",
    lifespan=lifespan
)


@app.get("/")
def home():
    return {
        "message": "MP282 Solar Energy Prediction API is running",
        "health": "/health",
        "prediction": "/predict",
        "documentation": "/docs"
    }


@app.get("/health")
def health():
    loaded = app.state.model is not None

    return {
        "status": "healthy" if loaded else "model_unavailable",
        "model_loaded": loaded,
        "model_name": MODEL_NAME,
        "model_version": app.state.model_version
    }


@app.post("/predict")
def predict(data: SolarPredictionInput):
    if app.state.model is None:
        raise HTTPException(
            status_code=503,
            detail="MLflow model is not available"
        )

    try:
        input_data = data.model_dump()

        if input_data["cloud_cover_pct"] is None:
            input_data["cloud_cover_pct"] = np.nan

        columns = [
            "panel_area_m2",
            "sunshine_hours",
            "temp_c",
            "dust_days",
            "panel_age_yr",
            "cloud_cover_pct",
            "roof_type",
            "area"
        ]

        input_df = pd.DataFrame([input_data], columns=columns)
        prediction = app.state.model.predict(input_df)
        value = float(np.asarray(prediction).reshape(-1)[0])

        logger.info("Prediction completed successfully")

        return {
            "predicted_energy_kwh": round(value, 2),
            "model_version": app.state.model_version
        }

    except Exception:
        logger.exception("Prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed")
