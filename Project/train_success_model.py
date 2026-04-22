import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

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

# Target
target = df["exited"]

# Encode categorical features
features = pd.get_dummies(features)

# Save feature columns
joblib.dump(features.columns.tolist(), "models/success_feature_columns.pkl")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    features,
    target,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)

print("Success Model Accuracy:", accuracy)

# Save model
joblib.dump(model, "models/startup_success_model.pkl")

print("Success model saved")