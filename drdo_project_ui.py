import streamlit as st
import joblib
import numpy as np
import os
from datetime import datetime

# Set page configuration
st.set_page_config(page_title="DRDO Fault Prediction System", page_icon=":rocket:", layout="wide")

st.markdown("""
<style>
    .main {
        background-color: white;
        padding: 20px;
    }
    .header {
        background-color: #28a745;
        padding: 10px;
        border-radius: 5px;
        text-align: center;
        color: white;
        font-size: 24px;
        font-weight: bold;
        display: flex;
        align-items: center;
        justify-content: center;
    }
    .header img {
        margin-right: 10px;
        height: 40px;
    }
    .subheader {
        color: #28a745;
        font-size: 18px;
        margin-top: 10px;
    }
    .stButton>button {
        background-color: #28a745;
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
    }
    .stButton>button:hover {
        background-color: #218838;
    }
    .model-card {
        background-color: #dee2e6;
        color: #000000;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
        margin: 10px;
        min-width: 200px;
    }
    .result-box {
        background-color: #dee2e6;
        color: #000000;
        padding: 15px;
        border-radius: 5px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin-top: 10px;
    }
    .footer {
        text-align: center;
        color: #6c757d;
        font-size: 14px;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Load predefined models at startup
model_files = {
    "SVM": "SVC.pkl",
    "Random Forest": "RandomForestClassifier.pkl",
    "Logistic Regression": "LogisticRegression.pkl",
    "KNN": "KNeighborsClassifier.pkl",
    "Naive Bayes" : "GaussianNB.pkl",
    "RUL_Model": "rul_model.pkl"

}


models = {}
for name, path in model_files.items():
    try:
        models[name] = joblib.load(path)
        print(f"Loaded {name} model from {path}")
    except Exception as e:
        st.error(f"Failed to load {name} model from {path}: {str(e)}")

# Define prediction functions
def predict_fault(model, inputs):
    try:
        input_array = np.array([inputs["DE"], inputs["FE"]]).reshape(1, -1)
        prediction = model.predict(input_array)
        return prediction[0]
    except Exception as e:
        return f"Error: {str(e)}"

def predict_rul(model, inputs):
    try:
        input_array = np.array([
            inputs["cycle"], inputs["ambient_temperature"], inputs["voltage_measured"],
            inputs["current_measured"], inputs["temperature_measured"], inputs["current_load"],
            inputs["voltage_load"], inputs["time"]
        ]).reshape(1, -1)
        prediction = model.predict(input_array)
        return prediction[0]
    except Exception as e:
        return f"Error: {str(e)}"

def predict_anomaly(model, inputs):
    try:
        input_array = np.array([inputs["DE"], inputs["FE"]]).reshape(1, -1)
        prediction = model.predict(input_array)
        return "Anomaly Detected" if prediction[0] == 1 else "No Anomaly"
    except Exception as e:
        return f"Error: {str(e)}"

# Header  title
current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")  # e.g., 02-07-2025 14:28:00
st.markdown(f'''
    <div class="header">
        Predictive maintainance system using ML algorithms
    </div>
''', unsafe_allow_html=True)

# Main content
st.markdown('<div class="subheader">Select models and input data for fault, RUL, and anomaly predictions.</div>', unsafe_allow_html=True)

# Model selection with checkboxes
st.subheader("Model Selection")
selected_fault_models = st.multiselect("Select fault prediction models:", ["SVM", "Random Forest", "Logistic Regression", "KNN", "Naive Bayes"], default=["SVM", "Random Forest"])
selected_rul_model = st.checkbox("Enable RUL Prediction (RUL_Model)")
selected_anomaly_model = st.checkbox("Enable Anomaly Detection (Anomaly_Model)")

# Input section
st.subheader("Input Data")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("**Fault & Anomaly Inputs**")
    de = st.number_input("Drive End (DE)", value=0.0, step=0.000001, format="%.6f")
    fe = st.number_input("Fan End (FE)", value=0.0, step=0.000001, format="%.6f")
    fault_inputs = {"DE": de, "FE": fe}
    anomaly_inputs = {"DE": de, "FE": fe}

with col2:
    st.markdown("**RUL Prediction Inputs**")
    rul_inputs = {}
    if selected_rul_model:
        rul_inputs["cycle"] = st.number_input("Cycle", value=0.0, step=0.000001, format="%.6f")
        rul_inputs["ambient_temperature"] = st.number_input("Ambient Temperature", value=0.0, step=0.000001, format="%.6f")
        rul_inputs["voltage_measured"] = st.number_input("Voltage Measured", value=0.0, step=0.000001, format="%.6f")
        rul_inputs["current_measured"] = st.number_input("Current Measured", value=0.0, step=0.000001, format="%.6f")
        rul_inputs["temperature_measured"] = st.number_input("Temperature Measured", value=0.0, step=0.000001, format="%.6f")
        rul_inputs["current_load"] = st.number_input("Current Load", value=0.0, step=0.000001, format="%.6f")
        rul_inputs["voltage_load"] = st.number_input("Voltage Load", value=0.0, step=0.000001, format="%.6f")
        rul_inputs["time"] = st.number_input("Time", value=0.0, step=0.000001, format="%.6f")

with col3:
    st.markdown("**Controls**")
    if selected_rul_model:
        rul_submit = st.button("Submit RUL Data")

# Prediction and comparison
if st.button("Compare Models"):
    # Fault prediction
    if selected_fault_models:
        with st.spinner("Running fault predictions..."):
            fault_predictions = {}
            for model_name in selected_fault_models:
                if model_name in models:
                    prediction = predict_fault(models[model_name], fault_inputs)
                    if isinstance(prediction, str):
                        fault_predictions[model_name] = prediction
                    else:
                        fault_predictions[model_name] = "Fault Detected" if prediction == 1 else "No Fault"
                else:
                    fault_predictions[model_name] = "Model not loaded"
            st.subheader("Fault Prediction Comparison")
            cols = st.columns(len(selected_fault_models))
            for idx, (model_name, result) in enumerate(fault_predictions.items()):
                with cols[idx]:
                    st.markdown(f'<div class="model-card">{model_name}<br>{result}</div>', unsafe_allow_html=True)

    # RUL prediction

    if selected_rul_model and rul_inputs:
        with st.spinner("Running RUL prediction..."):
            if "RUL_Model" in models:
                rul_prediction = predict_rul(models["RUL_Model"], rul_inputs)
                st.subheader("RUL Prediction")
                st.markdown(f'<div class="result-box">Remaining Useful Life (Capacity): {rul_prediction}</div>', unsafe_allow_html=True)
            else:
                st.error("RUL_Model not loaded")

    # Anomaly detection
    if selected_anomaly_model and anomaly_inputs:
        with st.spinner("Running anomaly detection..."):
            if "Anomaly_Model" in models:
                anomaly_prediction = predict_anomaly(models["Anomaly_Model"], anomaly_inputs)
                st.subheader("Anomaly Detection")
                st.markdown(f'<div class="result-box">Result: {anomaly_prediction}</div>', unsafe_allow_html=True)
            else:
                st.error("Anomaly_Model not loaded")

    if not (selected_fault_models or selected_rul_model or selected_anomaly_model):
        st.warning("Please select at least one model type for comparison.")

# Handle RUL submission separately
if selected_rul_model and rul_submit and rul_inputs:
    with st.spinner("Processing RUL data..."):
        if "RUL_Model" in models:
            rul_prediction = predict_rul(models["RUL_Model"], rul_inputs)
            st.subheader("RUL Prediction Result")
            st.markdown(f'<div class="result-box">Remaining Useful Life (Capacity): {rul_prediction}</div>', unsafe_allow_html=True)
        else:
            st.error("RUL_Model not loaded")

# Footer
st.markdown('<div class="footer">Developed for DRDO Fault Prediction Project | © 2025</div>', unsafe_allow_html=True)
