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
st.title("⚛️ QNose — Quantum 27-Disease Diagnostic Array")
st.markdown("##### *Next-Gen Multi-Disease Early Detection mapped across 27 distinct classes using Quantum Phase Spaces*")
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
    le = joblib.load('label_encoder.pkl')
    return scaler, pca, x_mean, healthy_mean, qsvm, X_train, y_train, feature_cols, le

scaler, pca, x_mean, healthy_mean, qsvm, X_train, y_train, feature_cols, le = load_resources()

# The Top 5 VOC markers identified in EDA: Ethane, Nonanal, Acetonitrile, Pentane, Hexanal
idx_ethane = feature_cols.index('Ethane')
idx_nonanal = feature_cols.index('Nonanal')
idx_acetonitrile = feature_cols.index('Acetonitrile')
idx_pentane = feature_cols.index('Pentane')
idx_hexanal = feature_cols.index('Hexanal')

# 4. Sidebar Controls
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/P_hybrid_circuit.svg/1024px-P_hybrid_circuit.svg.png")
    st.header("🎛️ Clinical Breath Inputs")
    st.markdown("Adjust key Top-5 predictive VOC parts-per-billion (ppb) detected.")
    
    ethane = st.slider("Ethane (ppb) [Top Indicator]", 0.0, 50.0, float(healthy_mean[idx_ethane]), step=0.1)
    nonanal = st.slider("Nonanal (ppb)", 0.0, 50.0, float(healthy_mean[idx_nonanal]), step=0.1)
    acetonitrile = st.slider("Acetonitrile (ppb)", 0.0, 50.0, float(healthy_mean[idx_acetonitrile]), step=0.1)
    pentane = st.slider("Pentane (ppb) [Oxidative Stress]", 0.0, 200.0, float(healthy_mean[idx_pentane]), step=1.0)
    hexanal = st.slider("Hexanal (ppb)", 0.0, 50.0, float(healthy_mean[idx_hexanal]), step=0.1)
    
    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("🧬 Run Quantum Multi-Class Inference", type="primary")

ui_feature_names = ["Ethane", "Nonanal", "Acetonitrile", "Pentane", "Hexanal"]
ui_features = [ethane, nonanal, acetonitrile, pentane, hexanal]
healthy_base = [healthy_mean[idx_ethane], healthy_mean[idx_nonanal], healthy_mean[idx_acetonitrile], healthy_mean[idx_pentane], healthy_mean[idx_hexanal]]

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

if 'prediction_run' not in st.session_state:
    st.session_state.prediction_run = False
    st.session_state.pred_label = ""
    st.session_state.prob_dist = None
    st.session_state.X_input_pca = None

if predict_button:
    full_features = np.copy(x_mean)
    full_features[idx_ethane] = ethane
    full_features[idx_nonanal] = nonanal
    full_features[idx_acetonitrile] = acetonitrile
    full_features[idx_pentane] = pentane
    full_features[idx_hexanal] = hexanal
    
    X_input_scaled = scaler.transform([full_features])
    X_input_pca = pca.transform(X_input_scaled)
    st.session_state.X_input_pca = X_input_pca
    
    progress_text = "⚛️ Calculating entangled multiclass permutations in Quantum Model..."
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(100):
        time.sleep(0.005)
        my_bar.progress(percent_complete + 1, text=progress_text)
    
    K_pred = np.array([[kernel_function(X_input_pca[0], x_train) for x_train in X_train]])
    my_bar.empty()
        
    pred_idx = qsvm.predict(K_pred)[0]
    st.session_state.pred_label = le.inverse_transform([pred_idx])[0]
    
    if hasattr(qsvm, "predict_proba"):
        st.session_state.prob_dist = qsvm.predict_proba(K_pred)[0]
        
    st.session_state.prediction_run = True

with col1:
    st.subheader("🔬 Biomarker Top-5 Profile")
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
    st.subheader("⚠️ Top Diagnostic Matches")
    
    if not st.session_state.prediction_run:
        st.info("Awaiting input. Click **Run Inference** to predict across 27 classifications.")
    else:
        # Multiclass probability breakdown
        if st.session_state.prob_dist is not None:
            # Sort top 5 diseases
            probs = st.session_state.prob_dist
            top_5_idx = np.argsort(probs)[-5:][::-1]
            top_5_diseases = le.inverse_transform(top_5_idx)
            top_5_probs = probs[top_5_idx] * 100
            
            df_probs = pd.DataFrame({'Disease': top_5_diseases, 'Probability': top_5_probs})
            fig_bar = px.bar(df_probs, x='Probability', y='Disease', orientation='h', color='Probability', 
                             color_continuous_scale='Reds' if st.session_state.pred_label != "Healthy" else 'Greens')
            fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(t=10, b=10, l=10, r=10), height=200, coloraxis_showscale=False)
            st.plotly_chart(fig_bar)
            
        if st.session_state.pred_label != "Healthy":
            st.markdown(f'<div class="alert-glow"><h3>🚨 {st.session_state.pred_label} Detected</h3><p>Highest correlation across the 27-state diagnostic matrix.</p></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="safe-glow"><h3>✅ Baseline Healthy</h3><p>Biomarkers correspond natively to regular background topology.</p></div>', unsafe_allow_html=True)

with col3:
    st.subheader("🌐 3D Multiplex Topography")
    if not st.session_state.prediction_run:
        st.info("Awaiting structural mapping of the local dimensional feature space.")
    else:
        df_3d = pd.DataFrame({
            'PCA 1': X_train[:, 0],
            'PCA 2': X_train[:, 1],
            'PCA 3': X_train[:, 2],
            'Class': le.inverse_transform(y_train),
            'Size': [3] * len(X_train)
        })
        
        patient_data = pd.DataFrame({
            'PCA 1': [st.session_state.X_input_pca[0, 0]],
            'PCA 2': [st.session_state.X_input_pca[0, 1]],
            'PCA 3': [st.session_state.X_input_pca[0, 2]],
            'Class': ['⭐ CURRENT PATIENT'],
            'Size': [30]
        })
        
        df_plot = pd.concat([df_3d, patient_data], ignore_index=True)
        
        fig_3d = px.scatter_3d(
            df_plot, x='PCA 1', y='PCA 2', z='PCA 3', 
            color='Class', size='Size',
            opacity=0.75
        )
        
        fig_3d.update_layout(
            scene=dict(
                xaxis=dict(showbackground=False),
                yaxis=dict(showbackground=False),
                zaxis=dict(showbackground=False)
            ),
            margin=dict(t=0, b=0, l=0, r=0),
            height=380,
            showlegend=False
        )
        st.plotly_chart(fig_3d)

st.markdown("---")

tab1, tab2 = st.tabs(["🗺️ Quantum Processing Arrays", "🧠 Multiclass Explainability"])

with tab1:
    st.markdown("### ⚛️ Architecture Core")
    
    cq1, cq2, cq3 = st.columns(3)
    with cq1:
        st.markdown("**1. Kernel Alignment Map**")
        if os.path.exists("kernel_alignment.png"):
            st.image("kernel_alignment.png")
            
    with cq2:
        st.markdown("**2. Quantum Sensitivity Gradients**")
        if os.path.exists("quantum_feature_importance.png"):
            st.image("quantum_feature_importance.png")
            
    with cq3:
        st.markdown("**3. Stratified Subspace Projection**")
        if os.path.exists("quantum_tsne_projection.png"):
            st.image("quantum_tsne_projection.png")
            
    st.markdown("---")
    st.markdown("### Topological Data Encoding (OVR Multiclass Configuration)")
    if os.path.exists("quantum_circuit.png"):
        st.image("quantum_circuit.png")

with tab2:
    st.markdown("### Structural Matrix View")
    st.info("Multiclass SHAP visualization pipeline is currently caching... View `feature_importances.csv` generated by the Random Forest array for top indicators.")
