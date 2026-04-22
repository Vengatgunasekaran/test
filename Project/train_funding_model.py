import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("data/cleaned_startup_dataset.csv")

# Features
features = df[
    [
        "industry",
        "funding_round",
        "region",
        "employee_count",
        "estimated_revenue_usd",
        "founded_year"
    ]
]

# Target variable
target = df["funding_amount_usd"]

# Convert categorical features
features = pd.get_dummies(features)

# Save feature columns
joblib.dump(features.columns.tolist(), "models/funding_feature_columns.pkl")

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.2,
    random_state=42
)

# Train regression model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

score = model.score(X_test, y_test)

print("Funding Prediction Model R² Score:", score)

# Save model
joblib.dump(model, "models/funding_prediction_model.pkl")

print("Funding model saved successfully")