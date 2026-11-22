import pandas as pd
from src.preprocessing.cleaning import clean_data

# Load original datasets
train = pd.read_csv("data/raw/train.csv")
later = pd.read_csv("data/raw/later.csv")
holdout = pd.read_csv("data/raw/holdout_features.csv")

# Clean datasets
train_clean = clean_data(train, fix_sunshine=True)
later_clean = clean_data(later, fix_sunshine=False)
holdout_clean = clean_data(holdout, fix_sunshine=False)

# Save cleaned datasets
train_clean.to_csv("data/processed/train.csv", index=False)
later_clean.to_csv("data/processed/later.csv", index=False)
holdout_clean.to_csv("data/processed/holdout_features.csv", index=False)

print("Data cleaning completed!")
print("Train:", train_clean.shape)
print("Later:", later_clean.shape)
print("Holdout:", holdout_clean.shape)
