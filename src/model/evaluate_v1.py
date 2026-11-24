import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

LOGGED_MAE = 2.0826563392304207   # DB mein v1 (gradient_boosting) ka MAE

with open("params.yaml") as f:
    params = yaml.safe_load(f)

train = pd.read_csv("data/processed/train.csv")
later = pd.read_csv("data/processed/later.csv")

DROP = ["energy_kwh", "record_id", "date"]
X, y = train.drop(columns=DROP), train["energy_kwh"]

numeric = X.select_dtypes(include=["number"]).columns.tolist()
categorical = X.select_dtypes(exclude=["number"]).columns.tolist()

preprocessor = ColumnTransformer([
    ("numeric", SimpleImputer(strategy="median"), numeric),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ]), categorical),
])

v1 = Pipeline([
    ("preprocessor", preprocessor),
    ("model", GradientBoostingRegressor(
        **params["models"]["gradient_boosting"],
        random_state=params["random_state"])),
])

# v1 ko wahi 80/20 split pe fit karo jo train.py mein tha
X_tr, X_te, y_tr, y_te = train_test_split(
    X, y, test_size=params["test_size"], random_state=params["random_state"])
v1.fit(X_tr, y_tr)

check = mean_absolute_error(y_te, v1.predict(X_te))
print(f"Refit MAE: {check:.10f} | Logged MAE: {LOGGED_MAE:.10f}")
print("Reproduction OK" if abs(check - LOGGED_MAE) < 1e-6 else "MISMATCH - ruko, aage mat badho")

# later.csv pe v1, area ke hisaab se
X_later, y_later = later.drop(columns=DROP), later["energy_kwh"]
later = later.assign(abs_err=(y_later - v1.predict(X_later)).abs())

print("\nOverall MAE on later.csv:", round(later["abs_err"].mean(), 4))
print(later.groupby("area")["abs_err"].agg(["mean", "count"]).round(4))
worst = later.groupby("area")["abs_err"].mean().idxmax()
print("Worst served group:", worst)