import numpy as np
import pandas as pd


def psi(expected, actual, bins=10):
    expected, actual = expected.dropna(), actual.dropna()
    # bin edges train se banate hain (har bin mein ~10% train rows)
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf   # later ke bahar ke values bhi pakadne ke liye
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)  # log(0) se bachne ke liye
    return float(np.sum((a - e) * np.log(a / e)))


if __name__ == "__main__":
    train = pd.read_csv("data/processed/train.csv")
    later = pd.read_csv("data/processed/later.csv")
    cols = ["panel_area_m2", "sunshine_hours", "temp_c",
            "dust_days", "panel_age_yr", "cloud_cover_pct"]
    result = {c: round(psi(train[c], later[c]), 4) for c in cols}
    print(result)
    print("Most drifted:", max(result, key=result.get))