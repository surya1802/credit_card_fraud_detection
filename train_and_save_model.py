# train_and_save_model.py

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pickle

# ✅ Step 1: Load your dataset
# Replace with your actual dataset path or link
data = pd.read_csv("https://huggingface.co/spaces/suryas116/credit-card-fraud-detection/resolve/main/creditcard.csv")

# ✅ Step 2: Feature engineering
features = data.drop(['Class', 'Time'], axis=1)
labels = data['Class']

# ✅ Step 3: Resample using SMOTE to handle imbalance
smote = SMOTE()
X_resampled, y_resampled = smote.fit_resample(features, labels)

# ✅ Step 4: Split and scale
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# ✅ Step 5: Train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# ✅ Step 6: Save the model and scaler
with open("fraud_detection_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("✅ Model and scaler saved!")


