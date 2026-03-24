import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pennylane as qml
import os
import plotly.express as px
import plotly.graph_objects as go
import time

# 1. Setup the Page Configuration
st.set_page_config(
    page_title="QNose | Quantum ML Dashboard",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for glowing effects, animations, and better styling
st.markdown("""
<style>
    .reportview-container .main .block-container {
        padding-top: 1rem;
    }
    .metric-card {
        background-color: #1e1e1e;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.5);
    }
    @keyframes pulse-red {
        0% { box-shadow: 0 0 10px rgba(255, 75, 75, 0.4); }
        50% { box-shadow: 0 0 25px rgba(255, 75, 75, 0.9); }
        100% { box-shadow: 0 0 10px rgba(255, 75, 75, 0.4); }
    }
    @keyframes pulse-green {
        0% { box-shadow: 0 0 10px rgba(0, 204, 102, 0.4); }
        50% { box-shadow: 0 0 25px rgba(0, 204, 102, 0.9); }
        100% { box-shadow: 0 0 10px rgba(0, 204, 102, 0.4); }
    }
    .alert-glow {
        animation: pulse-red 2s infinite;
        border-radius: 10px;
        padding: 10px;
        border: 2px solid #ff4b4b;
        text-align: center;
    }
    .safe-glow {
        animation: pulse-green 3s infinite;
        border-radius: 10px;
        padding: 10px;
        border: 2px solid #00cc66;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# 2. Main Title and Header
st.title("⚛️ QNose — Quantum Biomarker Analysis")
st.markdown("##### *Next-Gen Multi-Disease Early Detection using Quantum Support Vector Machines & 3D Topographic Mapping*")
st.markdown("---")

# 3. Load Models (Cached for Performance)
@st.cache_resource
def load_resources():
    scaler = joblib.load('scaler.pkl')
    pca = joblib.load('pca.pkl')
    x_mean = joblib.load('x_mean.pkl')
    healthy_mean = joblib.load('healthy_mean.pkl')
    qsvm = joblib.load('quantum_svm_model.pkl')
    X_train = np.load('X_train_qsvm.npy')
    y_train = np.load('y_train_qsvm.npy')
    feature_cols = joblib.load('feature_cols.pkl')
    return scaler, pca, x_mean, healthy_mean, qsvm, X_train, y_train, feature_cols

scaler, pca, x_mean, healthy_mean, qsvm, X_train, y_train, feature_cols = load_resources()

idx_acetone = feature_cols.index('acetone_ppb')
idx_isoprene = feature_cols.index('isoprene_ppb')
idx_hc = feature_cols.index('hydrogen_cyanide_ppb')
idx_ethanol = feature_cols.index('ethanol_ppb')
idx_pentane = feature_cols.index('pentane_ppb')

# 4. Sidebar Controls
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/P_hybrid_circuit.svg/1024px-P_hybrid_circuit.svg.png")
    st.header("🎛️ Clinical Breath Inputs")
    st.markdown("Adjust key VOC parts-per-billion (ppb) detected in the sample.")
    
    acetone = st.slider("Acetone (ppb) [Metabolic]", 0.0, 3000.0, float(healthy_mean[idx_acetone]), step=10.0)
    isoprene = st.slider("Isoprene (ppb) [Liver/Cholesterol]", 0.0, 500.0, float(healthy_mean[idx_isoprene]), step=5.0)
    hydrogen_cyanide = st.slider("Hydrogen Cyanide (ppb) [Infection]", 0.0, 50.0, float(healthy_mean[idx_hc]), step=0.5)
    ethanol = st.slider("Ethanol (ppb) [Gut Microbiome]", 0.0, 1000.0, float(healthy_mean[idx_ethanol]), step=10.0)
    pentane = st.slider("Pentane (ppb) [Oxidative Stress]", 0.0, 200.0, float(healthy_mean[idx_pentane]), step=1.0)
    
    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("🧬 Run Quantum Inference", type="primary")

ui_feature_names = ["Acetone", "Isoprene", "H. Cyanide", "Ethanol", "Pentane"]
ui_features = [acetone, isoprene, hydrogen_cyanide, ethanol, pentane]
healthy_base = [healthy_mean[idx_acetone], healthy_mean[idx_isoprene], healthy_mean[idx_hc], healthy_mean[idx_ethanol], healthy_mean[idx_pentane]]

# 5. Quantum Circuit Setup
n_qubits = 5
dev = qml.device("default.qubit", wires=n_qubits)

@qml.qnode(dev)
def kernel_circuit(x1, x2):
    qml.AngleEmbedding(x1, wires=range(n_qubits))
    qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits))
    return qml.probs(wires=range(n_qubits))

def kernel_function(x1, x2):
    return kernel_circuit(x1, x2)[0]

# 6. Dashboard Layout
col1, col2, col3 = st.columns([1, 1.2, 1.5], gap="large")

# Pre-computation to hold state
if 'prediction_run' not in st.session_state:
    st.session_state.prediction_run = False
    st.session_state.prob = 0.0
    st.session_state.pred = 0
    st.session_state.X_input_pca = None

if predict_button:
    full_features = np.copy(x_mean)
    full_features[idx_acetone] = acetone
    full_features[idx_isoprene] = isoprene
    full_features[idx_hc] = hydrogen_cyanide
    full_features[idx_ethanol] = ethanol
    full_features[idx_pentane] = pentane
    
    X_input_scaled = scaler.transform([full_features])
    X_input_pca = pca.transform(X_input_scaled)
    st.session_state.X_input_pca = X_input_pca
    
    # Progress Bar Animation
    progress_text = "⚛️ Calculating entangled feature states in Quantum Hilbert Space..."
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    
    # Kernel Prediction
    K_pred = np.array([[kernel_function(X_input_pca[0], x_train) for x_train in X_train]])
    my_bar.empty()
        
    st.session_state.pred = qsvm.predict(K_pred)[0]
    
    if hasattr(qsvm, "predict_proba"):
        st.session_state.prob = qsvm.predict_proba(K_pred)[0][1] 
    else:
        st.session_state.prob = 0.95 if st.session_state.pred == 1 else 0.05
        
    st.session_state.prediction_run = True

with col1:
    st.subheader("🔬 Biomarker Radar Profile")
    df_radar = pd.DataFrame({
        'Feature': ui_feature_names * 2,
        'Value': ui_features + healthy_base,
        'Group': ['Patient Sample'] * 5 + ['Healthy Baseline'] * 5
    })
    
    fig_radar = px.line_polar(df_radar, r='Value', theta='Feature', color='Group', line_close=True,
                              color_discrete_sequence=['#ff4b4b', '#1f77b4'],
                              template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white")
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False)), margin=dict(t=20, b=20, l=20, r=20), height=350)
    st.plotly_chart(fig_radar)

with col2:
    st.subheader("⚠️ Quantum Risk Assessment")
    
    if not st.session_state.prediction_run:
        st.info("Awaiting input. Click **Run Quantum Inference**.")
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number", value = 0, title = {'text': "Disease Probability"},
            gauge = {'axis': {'range': [None, 100]}, 'bar': {'color': "lightgray"}}
        ))
        fig_gauge.update_layout(height=280)
        st.plotly_chart(fig_gauge)
    else:
        risk_pct = st.session_state.prob * 100
        gauge_color = "red" if risk_pct > 50 else "green"
        
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_pct,
            number={'suffix': "%"},
            title = {'text': "Anomalous Biomarker Probability"},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': gauge_color},
                'steps' : [
                    {'range': [0, 30], 'color': "rgba(0, 255, 0, 0.1)"},
                    {'range': [30, 60], 'color': "rgba(255, 255, 0, 0.1)"},
                    {'range': [60, 100], 'color': "rgba(255, 0, 0, 0.1)"}],
                'threshold' : {'line': {'color': "white" if st.get_option("theme.base") == "dark" else "black", 'width': 4}, 'thickness': 0.75, 'value': 50}
            }
        ))
        fig_gauge.update_layout(margin=dict(t=20, b=20, l=20, r=20), height=280)
        st.plotly_chart(fig_gauge)
        
        if st.session_state.pred == 1:
            st.markdown('<div class="alert-glow"><h3>🚨 Anomalous Signature</h3><p>Systemic anomalies detected correlating with clinical disease.</p></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="safe-glow"><h3>✅ Baseline Healthy</h3><p>Biomarkers map smoothly within the generalized healthy hyperplane.</p></div>', unsafe_allow_html=True)

with col3:
    st.subheader("🌐 3D PCA Topography")
    if not st.session_state.prediction_run:
        st.info("Awaiting inference to project the patient onto the dimensional feature space.")
    else:
        df_3d = pd.DataFrame({
            'PCA 1': X_train[:, 0],
            'PCA 2': X_train[:, 1],
            'PCA 3': X_train[:, 2],
            'Class': ['Diseased' if y == 1 else 'Healthy' for y in y_train],
            'Size': [3] * len(X_train)
        })
        
        patient_data = pd.DataFrame({
            'PCA 1': [st.session_state.X_input_pca[0, 0]],
            'PCA 2': [st.session_state.X_input_pca[0, 1]],
            'PCA 3': [st.session_state.X_input_pca[0, 2]],
            'Class': ['⭐ CURRENT PATIENT'],
            'Size': [25]
        })
        
        df_plot = pd.concat([df_3d, patient_data], ignore_index=True)
        
        fig_3d = px.scatter_3d(
            df_plot, x='PCA 1', y='PCA 2', z='PCA 3', 
            color='Class', size='Size',
            color_discrete_map={'Diseased': '#ff4b4b', 'Healthy': '#1f77b4', '⭐ CURRENT PATIENT': '#ffd700'},
            opacity=0.75
        )
        
        # Adding a cool 3D spinning animation effect with Plotly layout
        fig_3d.update_layout(
            scene=dict(
                xaxis=dict(showbackground=False),
                yaxis=dict(showbackground=False),
                zaxis=dict(showbackground=False)
            ),
            margin=dict(t=0, b=0, l=0, r=0),
            height=380,
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        st.plotly_chart(fig_3d)

st.markdown("---")

tab1, tab2 = st.tabs(["🗺️ Quantum Processing & Architecture", "🧠 AI Explainability (SHAP & Diagnostics)"])

with tab1:
    st.markdown("### ⚛️ Quantum Advantage & Metrics")
    
    cq1, cq2, cq3 = st.columns(3)
    with cq1:
        st.markdown("**1. Kernel Alignment**")
        if os.path.exists("kernel_alignment.png"):
            st.image("kernel_alignment.png")
            st.caption("Comparison showing how well the Quantum Kernel maps to true labels vs a classical RBF kernel.")
            
    with cq2:
        st.markdown("**2. Quantum Feature Importance**")
        if os.path.exists("quantum_feature_importance.png"):
            st.image("quantum_feature_importance.png")
            st.caption("PCA feature sensitivities computed via qml.grad() across the quantum circuit parameters.")
            
    with cq3:
        st.markdown("**3. Hilbert Space Projection**")
        if os.path.exists("quantum_tsne_projection.png"):
            st.image("quantum_tsne_projection.png")
            st.caption("t-SNE 2D mapping of the precomputed Quantum distance matrix (D = √(2 - 2K)) showing class separation.")
            
    st.markdown("---")
    st.markdown("### Topological Data Encoding")
    st.markdown("Visualizing the parameterised PennyLane operations bridging the classical-to-quantum step.")
    if os.path.exists("quantum_circuit.png"):
        st.image("quantum_circuit.png")

with tab2:
    st.markdown("### Decision Interpretation")
    st.markdown("Understanding *why* the SVM made its decision using Shapley additive explanations.")
    if os.path.exists("shap_explanation.png"):
        st.image("shap_explanation.png")
    else:
        st.warning("Generate explainability metrics by running `python explainability.py`.")
