import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open("fraud_detection_model.pkl", "rb"))

st.set_page_config(page_title="Credit Card Fraud Detection", layout="wide")
st.title("Credit Card Fraud Detection")

st.markdown("Enter the transaction details below to check if it's Fraudulent or Genuine.")

# Input fields for model features (V1 to V28 + 3 engineered features)
v_features = []
for i in range(1, 29):  # V1 to V28
    v = st.number_input(f"V{i}", value=0.0)
    v_features.append(v)

amount = st.number_input("Amount", value=0.0)
transaction_frequency = st.number_input("Transaction Frequency", value=0.0)
avg_spending = st.number_input("Average Spending", value=0.0)

# Create input array
input_data = np.array([v_features + [amount, transaction_frequency, avg_spending]])

# Predict
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    result = "Fraud Detected!" if prediction == 1 else "Genuine Transaction"
    st.subheader(f"Prediction: {result}")
