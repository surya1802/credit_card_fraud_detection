import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import KFold

# Load the dataset
data = pd.read_csv('creditcard.csv')

# --- Data Exploration and Understanding ---
print("Dataset Info:")
print(data.info())
print("\nDataset Description:")
print(data.describe())
print("\nClass Distribution:")
print(data['Class'].value_counts())

# --- Feature Engineering ---
# Transaction Frequency (per time unit, e.g., per hour)
data['Transaction_Frequency'] = data.groupby(pd.cut(data['Time'], bins=24), observed=True)['Amount'].transform('count')

# Average Spending (per time unit)
data['Avg_Spending'] = data.groupby(pd.cut(data['Time'], bins=24), observed=True)['Amount'].transform('mean')

# Drop 'Time' column as it's not useful for classification
data = data.drop(columns=['Time'])

# --- Define features (X) and target variable (y) ---
X = data.drop(columns=['Class'])
y = data['Class']

# --- Normalize the features ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# --- Handle class imbalance using SMOTE ---
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

# --- Split the dataset into training and testing sets ---
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# --- Hyperparameter Tuning using GridSearchCV and Cross-Validation ---
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf_classifier = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=KFold(n_splits=5), scoring='f1', verbose=2, n_jobs=-1)
grid_search.fit(X_train, y_train)

best_rf_classifier = grid_search.best_estimator_

# --- Predict on the test set ---
y_pred = best_rf_classifier.predict(X_test)
y_prob = best_rf_classifier.predict_proba(X_test)[:, 1]

# --- Evaluate the model ---
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print("Confusion Matrix:\n", conf_matrix)
print("Classification Report:\n", class_report)

# --- Visualize the Confusion Matrix ---
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Genuine', 'Fraud'], yticklabels=['Genuine', 'Fraud'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.show()

# --- Plot ROC Curve ---
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc="lower right")
plt.show()

# --- Save the trained model ---
with open('fraud_detection_model.pkl', 'wb') as model_file:
    pickle.dump(best_rf_classifier, model_file)
