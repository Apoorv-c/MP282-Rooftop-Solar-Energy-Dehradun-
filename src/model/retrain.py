import hashlib
import yaml
import pandas as pd
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error
import sqlite3, pathlib
MODEL_NAME = "MP282_Solar_Energy_Model"
PROMOTION_FACTOR = 0.95          # v2 ko v1 se kam se kam 5% behtar hona chahiye
DROP = ["energy_kwh", "record_id", "date"]


def should_promote(mae_v1, mae_v2, factor=PROMOTION_FACTOR):
    return mae_v2 <= factor * mae_v1


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def build_pipeline(X, params):
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()
    pre = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical),
    ])
    gb = GradientBoostingRegressor(
        **params["models"]["gradient_boosting"],
        random_state=params["random_state"])
    return Pipeline([("preprocessor", pre), ("model", gb)])


def main():
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    train = pd.read_csv("data/processed/train.csv")
    later = pd.read_csv("data/processed/later.csv")

    # --- v1: train.py wala hi setup (train ka 80%) ---
    X, y = train.drop(columns=DROP), train["energy_kwh"]
    X_tr, _, y_tr, _ = train_test_split(
        X, y, test_size=params["test_size"], random_state=params["random_state"])
    v1 = build_pipeline(X, params)
    v1.fit(X_tr, y_tr)

    # --- later.csv ko date se sort karo, 70/30 todo ---
    later = later.sort_values("date", kind="stable").reset_index(drop=True)
    n_fit = int(0.7 * len(later))
    later_fit, later_check = later.iloc[:n_fit], later.iloc[n_fit:]

    # --- v2: poora cleaned train + later ka pehla 70% ---
    v2_data = pd.concat([train, later_fit], ignore_index=True)
    X2, y2 = v2_data.drop(columns=DROP), v2_data["energy_kwh"]
    v2 = build_pipeline(X2, params)
    v2.fit(X2, y2)

    # --- last 30% pe dono ka MAE ---
    Xc, yc = later_check.drop(columns=DROP), later_check["energy_kwh"]
    mae_v1 = mean_absolute_error(yc, v1.predict(Xc))
    mae_v2 = mean_absolute_error(yc, v2.predict(Xc))
    promote = should_promote(mae_v1, mae_v2)

    print(f"Rows: v2 train={len(v2_data)}, check={len(later_check)}")
    print(f"MAE v1: {mae_v1:.4f} | MAE v2: {mae_v2:.4f}")
    print(f"Threshold (0.95 * v1): {PROMOTION_FACTOR * mae_v1:.4f}")
    print("Promote v2:", promote)

    # --- MLflow ---
    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    art = pathlib.Path("mlruns").resolve().as_uri()
    con = sqlite3.connect("mlflow.db")
    con.execute("UPDATE experiments SET artifact_location=? WHERE name=?",
                (art, params["experiment_name"]))
    con.commit()
    con.close()

    mlflow.set_experiment(params["experiment_name"])
    with mlflow.start_run(run_name="v2_gradient_boosting") as run:
        mlflow.log_param("model_name", "gradient_boosting")
        mlflow.log_param("data_sha256", sha256("data/raw/train.csv"))
        mlflow.log_param("later_sha256", sha256("data/raw/later.csv"))
        mlflow.log_param("random_state", params["random_state"])
        mlflow.log_param("train_rows", len(v2_data))
        mlflow.log_param("check_rows", len(later_check))
        mlflow.log_params(params["models"]["gradient_boosting"])
        mlflow.log_metric("mae", mae_v2)
        mlflow.log_metric("mae_v1_on_check", mae_v1)
        mlflow.sklearn.log_model(v2, artifact_path="model",
                                 serialization_format="cloudpickle")
        run_id = run.info.run_id

    mv = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
    client = MlflowClient()
    if promote:
        client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
    champion = client.get_model_version_by_alias(MODEL_NAME, "champion").version
    print(f"Registered v2 as version {mv.version} | champion is now version {champion}")


if __name__ == "__main__":
    main()