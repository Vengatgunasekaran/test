import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load cleaned dataset
df = pd.read_csv("data/cleaned_startup_dataset.csv")

# Select features
features = df[[
    "industry",
    "funding_round",
    "region",
    "employee_count",
    "estimated_revenue_usd",
    "founded_year"
]]

# Target variable
target = df["lead_investor"]

# Convert categorical features into numeric
features = pd.get_dummies(features)
joblib.dump(features.columns.tolist(), "models/feature_columns.pkl")

# Convert investor names into numbers
label_encoder = LabelEncoder()
target_encoded = label_encoder.fit_transform(target)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    features, target_encoded, test_size=0.2, random_state=42
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Check accuracy
accuracy = model.score(X_test, y_test)
print("Model Accuracy:", accuracy)

# Save model
joblib.dump(model, "models/investor_model.pkl")

# Save label encoder
joblib.dump(label_encoder, "models/label_encoder.pkl")

print("Model saved successfully!")