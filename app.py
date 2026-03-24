import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pennylane as qml
import os

st.title("QNose — Quantum Breath Disease Detector")
st.write("Using the UCI Machine Learning Parkinson's Dataset (via PCA Embedding)")

# Load models and data
scaler = joblib.load('scaler.pkl')
pca = joblib.load('pca.pkl')
x_mean = joblib.load('x_mean.pkl')
qsvm = joblib.load('quantum_svm_model.pkl')
X_train = np.load('X_train_qsvm.npy')

# Sliders - We are keeping the VOC feel but injecting it into the multi-dimensional feature space! 
st.header("Input VOC Levels")
acetone = st.slider("Acetone", 0.0, 3.0, 1.0)
isoprene = st.slider("Isoprene", 0.0, 5.0, 2.0)
hydrogen_cyanide = st.slider("Hydrogen Cyanide", 0.0, 1.0, 0.4)
ethane = st.slider("Ethane", 0.0, 1.5, 0.5)
pentane = st.slider("Pentane", 0.0, 2.0, 0.8)

feature_names = ["Acetone", "Isoprene", "Hydrogen Cyanide", "Ethane", "Pentane"]
features = np.array([acetone, isoprene, hydrogen_cyanide, ethane, pentane])

# Setup quantum simulator
n_qubits = 5
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def kernel_circuit(x1, x2):
    qml.AngleEmbedding(x1, wires=range(n_qubits))
    qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits))
    return qml.probs(wires=range(n_qubits))

def kernel_function(x1, x2):
    return kernel_circuit(x1, x2)[0]

if st.button("Predict"):
    # Inject 5 VOC values into the 22-dimensional original domain using pre-saved mean padding
    full_features = np.copy(x_mean)
    full_features[:5] = features  
    
    # Normalize & PCA
    X_input_scaled = scaler.transform([full_features])
    X_input_pca = pca.transform(X_input_scaled)
    
    with st.spinner("Computing Quantum Kernel against training data..."):
        K_pred = np.array([[kernel_function(X_input_pca[0], x_train) for x_train in X_train]])
        
    pred = qsvm.predict(K_pred)[0]
    
    if pred == 1:
        st.error("Parkinson's Risk Detected")
    else:
        st.success("Low Risk")
        
    st.subheader("Feature Values Chart")
    st.bar_chart(pd.DataFrame([features], columns=feature_names).T)

# SHAP Explainability Section
st.subheader("Why did QNose flag this breath sample?")
if os.path.exists("shap_explanation.png"):
    st.image("shap_explanation.png")
else:
    st.info("Run `python explainability.py` to generate the SHAP explanation diagram.")

# Quantum Circuit Section
st.subheader("How QNose thinks")
if os.path.exists("quantum_circuit.png"):
    st.image("quantum_circuit.png", caption="AngleEmbedding rotates qubits by VOC/PCA values. CNOT gates entangle qubits to capture biomarker interactions.")

# PDF Report Section
if os.path.exists("QNose_Results.pdf"):
    with open("QNose_Results.pdf", "rb") as f:
        st.download_button("Download Generated Report (PDF)", f, "QNose_Results.pdf", "application/pdf")
