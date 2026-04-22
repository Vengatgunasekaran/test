import pandas as pd

# Load dataset
df = pd.read_csv("data/startup_valuation_dataset.csv")

print("Original Dataset Shape:", df.shape)

# Remove columns that are not useful for ML
columns_to_drop = [
    "startup_id",
    "startup_name",
    "funding_date",
    "exit_type",
    "co_investors"
]

df = df.drop(columns=columns_to_drop)

# Convert boolean column to integer
df["exited"] = df["exited"].astype(int)

# Save cleaned dataset
df.to_csv("data/cleaned_startup_dataset.csv", index=False)

print("Cleaned Dataset Shape:", df.shape)
print("Cleaned dataset saved successfully!")