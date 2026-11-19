
import pandas as pd


def check_data(train_path, later_path, holdout_path):
    train = pd.read_csv(train_path)
    later = pd.read_csv(later_path)
    holdout = pd.read_csv(holdout_path)

    print("\n1. LEAKING COLUMN")
    for col in train.columns:
        if col not in holdout.columns and col != "energy_kwh":
            corr = train[col].corr(train["energy_kwh"])
            if abs(corr) > 0.95:
                print(col, "correlation:", round(corr, 4))

    print("\n2. UNIT MIX-UP")
    for group, data in train.groupby("area"):
        print(group, data["sunshine_hours"].agg(
            ["min", "mean", "max"]
        ).to_dict())

    print("\n3. SENTINEL VALUE")
    print(train["cloud_cover_pct"].value_counts().head(10))

    print("\n4. DUPLICATE ROWS")
    print(train.duplicated().sum())

    print("\n5. STUCK SENSOR")
    data = train.sort_values("date").reset_index(drop=True)
    columns = [
        "panel_area_m2", "sunshine_hours", "temp_c",
        "dust_days", "panel_age_yr", "cloud_cover_pct"
    ]

    for col in columns:
        groups = data[col].ne(data[col].shift()).cumsum()
        runs = data.groupby(groups).agg(
            value=(col, "first"),
            rows=(col, "size"),
            start=("date", "min"),
            end=("date", "max")
        )
        print(col, runs[runs["rows"] >= 14].to_dict("records"))

    print("\n6. TARGET TYPOS")
    target = train["energy_kwh"].sort_values().reset_index(drop=True)
    print("Largest values:", target.tail(15).tolist())
    print("Largest gaps:", target.diff().nlargest(5).to_dict())

    print("\n7. SPELLING VARIANTS")
    print("Train roof types:")
    print(train["roof_type"].value_counts(dropna=False))

    print("\nClean roof types in later.csv:")
    print(later["roof_type"].value_counts(dropna=False))


if __name__ == "__main__":
    check_data(
        "data/raw/train.csv",
        "data/raw/later.csv",
        "data/raw/holdout_features.csv"
    )
