import gradio as gr
import numpy as np

# Dummy model that always returns "Genuine"
class DummyModel:
    def predict(self, X):
        return [0 for _ in range(len(X))]

model = DummyModel()

# Prediction function
def predict_transaction(*features):
    input_data = np.array(features).reshape(1, -1)
    prediction = model.predict(input_data)[0]
    return "Fraud Detected!" if prediction == 1 else "Genuine Transaction"

# Create Gradio input components: V1 to V28 + Amount + Transaction Frequency + Average Spending
inputs = []
for i in range(1, 29):  # V1 to V28
    inputs.append(gr.Number(label=f"V{i}"))
inputs.append(gr.Number(label="Amount"))
inputs.append(gr.Number(label="Transaction Frequency"))
inputs.append(gr.Number(label="Average Spending"))

# Gradio interface
demo = gr.Interface(
    fn=predict_transaction,
    inputs=inputs,
    outputs=gr.Textbox(label="Prediction Result"),
    title="Credit Card Fraud Detection",
    description="Enter transaction features to check if it's Fraudulent or Genuine."
)

# Launch the app
demo.launch()
