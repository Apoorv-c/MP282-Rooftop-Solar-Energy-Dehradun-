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
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error


# Load configuration
with open("params.yaml", "r") as file:
    params = yaml.safe_load(file)

# Load cleaned training data only
df = pd.read_csv("data/processed/train.csv")

# Separate features and target
X = df.drop(columns=["energy_kwh", "record_id", "date"])
y = df["energy_kwh"]

# Identify column types
numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

# Preprocessing
preprocessor = ColumnTransformer([
    ("numeric", SimpleImputer(strategy="median"), numeric_features),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=params["test_size"],
    random_state=params["random_state"]
)

# Model configurations
models = {
    "random_forest_100": RandomForestRegressor(
        **params["models"]["random_forest_100"],
        random_state=params["random_state"]
    ),
    "random_forest_200": RandomForestRegressor(
        **params["models"]["random_forest_200"],
        random_state=params["random_state"]
    ),
    "gradient_boosting": GradientBoostingRegressor(
        **params["models"]["gradient_boosting"],
        random_state=params["random_state"]
    )
}

# MLflow setup
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment(params["experiment_name"])

# Hash of original training data
with open("data/raw/train.csv", "rb") as file:
    data_sha256 = hashlib.sha256(file.read()).hexdigest()

results = []

# Train and track all three models
for model_name, model in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    with mlflow.start_run(run_name=model_name) as run:

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("data_sha256", data_sha256)
        mlflow.log_param("random_state", params["random_state"])
        mlflow.log_metric("mae", mae)

        mlflow.log_params({
            key: value
            for key, value in params["models"][model_name].items()
        })

        mlflow.sklearn.log_model(
            pipeline,
            artifact_path="model",
            serialization_format="cloudpickle"
        )

        results.append({
            "name": model_name,
            "mae": mae,
            "run_id": run.info.run_id
        })

        print(f"{model_name} MAE: {mae:.4f}")

# Select the best model
best_model = min(results, key=lambda item: item["mae"])

print("\nBest model:", best_model["name"])
print("Best MAE:", round(best_model["mae"], 4))

# Register only the best v1 model
model_name = "MP282_Solar_Energy_Model"

model_version = mlflow.register_model(
    model_uri=f"runs:/{best_model['run_id']}/model",
    name=model_name
)

# Set champion alias
client = MlflowClient()
client.set_registered_model_alias(
    model_name,
    "champion",
    model_version.version
)

print("\nModel registered successfully!")
print("Registered model:", model_name)
print("Version:", model_version.version)
print("Alias: champion")
print("MLflow database: mlflow.db")
