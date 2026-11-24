import os
import joblib
import mlflow
import mlflow.sklearn

mlflow.set_tracking_uri("sqlite:///mlflow.db")
model = mlflow.sklearn.load_model("models:/MP282_Solar_Energy_Model@champion")

os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.joblib")
print("Saved models/model.joblib")