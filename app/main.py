import json, logging, sys
from fastapi import FastAPI
from src.service.schemas import SolarRequest
from src.service.predictor import predict_energy

class JsonFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({"level": record.levelname,
                           "message": record.getMessage()})

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())
logger = logging.getLogger("mp282")
logger.setLevel(logging.INFO)
logger.addHandler(handler)

app = FastAPI(title="MP282 Solar API")

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: SolarRequest):
    value = predict_energy(req.model_dump())
    logger.info(f"prediction={value:.3f}")
    return {"prediction": value}