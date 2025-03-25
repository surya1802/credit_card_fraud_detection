Credit Card Fraud Detection

Project Overview:
This project aims to develop a classification model to detect fraudulent credit card transactions efficiently. The model is trained using a dataset obtained from Kaggle and employs techniques such as feature engineering, class balancing, and hyperparameter tuning to enhance accuracy while minimizing false positives.

Dataset:
The dataset is sourced from Kaggle and contains anonymized credit card transactions.
It includes features such as Transaction Amount, Time, Class (0 = Genuine, 1 = Fraudulent), and other transaction-related parameters.
The dataset is stored in the creditcard.csv file located in the credit_card folder.

Technologies Used:
Python
Pandas, NumPy (Data manipulation)
Matplotlib, Seaborn (Visualization)
Scikit-Learn (Machine Learning models and evaluation metrics)
Imbalanced-Learn (SMOTE) (Class imbalance handling)
RandomForestClassifier (Classification model)
GridSearchCV & Cross-validation (Hyperparameter tuning)

Implementation Details

1. Data Preprocessing:
Loaded the dataset from creditcard.csv.
Engineered new features:
Transaction_Frequency: Frequency of transactions per time unit.
Avg_Spending: Average spending per time unit.
Normalized features using StandardScaler.
Addressed class imbalance using SMOTE (Synthetic Minority Over-sampling Technique).

2. Model Training & Hyperparameter Tuning :
Used RandomForestClassifier for classification.
Applied GridSearchCV for hyperparameter tuning.
Implemented 5-fold cross-validation to improve performance.

3. Model Evaluation :
Evaluated model performance using:
Precision, Recall, F1-score
Confusion Matrix (Visualized using Seaborn)
ROC Curve & AUC Score

4. Model Saving :
The trained fraud detection model is saved as fraud_detection_model.pkl for future use.

How to Run the Project

Clone this repository:

git clone https://github.com/surya1802/credit-card-fraud-detection.git
cd Credit-Card-Fraud-Detection

Install dependencies:

pip install -r requirements.txt

Run the Python script:

python fraud_detection.py

Results & Observations :
The model effectively detects fraudulent transactions.
Using SMOTE improves recall, reducing false negatives.
GridSearchCV optimizes hyperparameters for better accuracy.

Author
Surya S
