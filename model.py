import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
import joblib

# Load dataset
df = pd.read_csv("heart.csv")

# Target column
target = "Heart Disease Status"

# Encode all non-numeric columns
label_encoders = {}

for col in df.columns:
    if df[col].dtype == "object":
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

# Separate features and target
X = df.drop(target, axis=1)
y = df[target]

# IMPOER: Fill missing values (NaN) with column mean
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_imputed, y, test_size=0.2, random_state=42
)

# Train the model
model = LogisticRegression(max_iter=5000)
model.fit(X_train, y_train)

# Save model, encoders and imputer
joblib.dump((model, list(X.columns), label_encoders, imputer), "heart_model.pkl")

print("Model trained and saved as heart_model.pkl")
