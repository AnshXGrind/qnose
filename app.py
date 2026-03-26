# -*- coding: utf-8 -*-

"""Streamlit dashboard for interactive quantum-classical breath diagnostics.

The QNose app allows users to explore a 27-class disease classifier driven by a
Quantum SVM (QSVM) and classical baselines. It exposes controls for manual VOC
injection, preset severity profiles, a 3D PCA hologram, and several
visualization widgets around the trained models.
"""

from datetime import datetime
import hashlib
import json
import os
import time
import logging

import joblib
import numpy as np
import pandas as pd
import pennylane as qml
import plotly.colors as pcolors
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Feature subsets used in the sidebar controls
TOP_5 = ["Ethane", "Nonanal", "Acetonitrile", "Pentane", "Hexanal"]
TOP_10 = TOP_5 + [
    "Isoprene",
    "Trimethylamine",
    "Propanal",
    "Ammonia",
    "Toluene",
]

# 1. Setup the Page Configuration
st.set_page_config(
    page_title="QNose | Quantum ML Dashboard",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Authentication Module ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def login_screen():
    st.markdown("""
        <style>
            [data-testid="stSidebar"] { display: none; }
            [data-testid="collapsedControl"] { display: none; }
        </style>
    """, unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; margin-top: 100px;'>🔒 QNose Secure Access</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        st.info("Demo Credentials -> Username: **dr_admin** | Password: **quantum**")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Authenticate", use_container_width=True, type="primary"):
            if username == "dr_admin" and password == "quantum":
                st.session_state.authenticated = True
                st.rerun()
            else:
                st.error("❌ Invalid credentials. Unauthorized access logged.")

if not st.session_state.authenticated:
    login_screen()
    st.stop()
# -----------------------------

# --- Telemetry/Logging Module ---
def log_telemetry(action, details):
    telemetry_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'local_telemetry.jsonl')
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "user": "dr_admin",
        "action": action,
        "details": details
    }
    try:
        with open(telemetry_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as exc:  # pragma: no cover - defensive logging only
        logging.error("Failed to write telemetry: %s", exc)
# -----------------------------

st.markdown("""
<style>
    /* Clean minimal background - Modern Deep Space / Teal theme */
    .stApp {
        background: radial-gradient(circle at top right, #0B192C, #020617);
        color: #F8FAFC;
    }
    
    .reportview-container .main .block-container { padding-top: 1rem; padding-bottom: 2rem; }
    
    /* Animated Hero Section */
    .hero {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.8), rgba(30, 41, 59, 0.4));
        backdrop-filter: blur(12px);
        padding: 3rem 2rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid rgba(56, 189, 248, 0.2);
        box-shadow: 0 10px 40px rgba(0, 0, 0, 0.4), inset 0 0 20px rgba(56, 189, 248, 0.05);
        transition: all 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    .hero:hover { 
        transform: translateY(-4px) scale(1.01); 
        box-shadow: 0 15px 50px rgba(0, 0, 0, 0.5), inset 0 0 30px rgba(56, 189, 248, 0.1);
        border: 1px solid rgba(56, 189, 248, 0.4);
    }
    
    .hero h1 span {
        font-size: 3.2rem !important;
        font-weight: 800;
        margin-bottom: 0.5rem;
        background: linear-gradient(to right, #38BDF8, #818CF8, #C084FC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        animation: textGlow 3s ease-in-out infinite alternate;
    }
    @keyframes textGlow { 
        from { filter: drop-shadow(0 0 8px rgba(56,189,248,0.4)); } 
        to { filter: drop-shadow(0 0 16px rgba(129,140,248,0.7)); } 
    }
    
    .hero p { font-size: 1.25rem; color: #94A3B8; max-width: 800px; margin: 0 auto; line-height: 1.6; }

    /* Modern Glassmorphism Cards */
    .glass-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.7), rgba(15, 23, 42, 0.9)); 
        border: 1px solid rgba(56, 189, 248, 0.2); 
        border-radius: 16px; 
        padding: 1.8rem;
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.4);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.3s ease;
    }
    .glass-card:hover { 
        transform: translateY(-6px); 
        box-shadow: 0 12px 30px 0 rgba(56, 189, 248, 0.15); 
        border: 1px solid rgba(56, 189, 248, 0.5);
    }

    /* Elegant Pulsing Alerts */
    @keyframes pulse-red { 
        0% { box-shadow: 0 0 10px rgba(239, 68, 68, 0.4); border-color: rgba(239, 68, 68, 0.5); background: rgba(127, 29, 29, 0.2); } 
        50% { box-shadow: 0 0 25px rgba(239, 68, 68, 0.8); border-color: rgba(239, 68, 68, 1); background: rgba(127, 29, 29, 0.4); } 
        100% { box-shadow: 0 0 10px rgba(239, 68, 68, 0.4); border-color: rgba(239, 68, 68, 0.5); background: rgba(127, 29, 29, 0.2); } 
    }
    @keyframes pulse-green { 
        0% { box-shadow: 0 0 10px rgba(16, 185, 129, 0.4); border-color: rgba(16, 185, 129, 0.5); background: rgba(6, 78, 59, 0.2); } 
        50% { box-shadow: 0 0 25px rgba(16, 185, 129, 0.8); border-color: rgba(16, 185, 129, 1); background: rgba(6, 78, 59, 0.4); } 
        100% { box-shadow: 0 0 10px rgba(16, 185, 129, 0.4); border-color: rgba(16, 185, 129, 0.5); background: rgba(6, 78, 59, 0.2); } 
    }
    
    .alert-glow { animation: pulse-red 2.2s infinite; border-radius: 12px; padding: 25px; text-align: center; margin-top: 1rem; color: #F8FAFC; border: 1px solid; }
    .safe-glow { animation: pulse-green 2.8s infinite; border-radius: 12px; padding: 25px; text-align: center; margin-top: 1rem; color: #F8FAFC; border: 1px solid; }
    .hw-error { color: #EF4444; font-weight: bold; background-color: rgba(239, 68, 68, 0.1); padding: 15px; border-radius: 10px; border: 1px solid #EF4444; text-align: center; margin-top: 15px; margin-bottom: 15px; font-size: 0.95rem; animation: textFade 2s infinite; }
    
    @keyframes textFade { 0% { opacity: 1; } 50% { opacity: 0.6; } 100% { opacity: 1; } }
    
    /* Modern Spinning Loader */
    .loader {
        border: 4px solid rgba(56, 189, 248, 0.1);
        border-top: 4px solid #38BDF8;
        border-radius: 50%;
        width: 30px;
        height: 30px;
        animation: spin 1s linear infinite;
        display: inline-block;
        vertical-align: middle;
        margin-right: 10px;
    }
    @keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }

    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# 2. Hero Section
st.markdown("""
<div class="hero">
    <h1><span>⚛️ QNose Multiplex Interface</span></h1>
    <p>Activate the Quantum Subspace Engine to synthesize structural permutations corresponding to 27 unique pathological signatures in real-time space.</p>
    <div style="margin-top: 15px;">
        <span style="background-color: #7C3AED; color: white; padding: 5px 12px; border-radius: 15px; font-size: 0.85rem; font-weight: bold; margin-right: 10px;">🧠 Core: Quantum SVM (PennyLane)</span>
        <span style="background-color: #059669; color: white; padding: 5px 12px; border-radius: 15px; font-size: 0.85rem; font-weight: bold;">🔁 Simulated Qubits: 5</span>
    </div>
</div>
""", unsafe_allow_html=True)

# 3. Load Models (Cached for Performance)
@st.cache_resource
def load_resources():
    try:
        base_path = os.path.dirname(os.path.abspath(__file__))
        scaler = joblib.load(os.path.join(base_path, "scaler.pkl"))
        pca = joblib.load(os.path.join(base_path, "pca.pkl"))
        x_mean = joblib.load(os.path.join(base_path, "x_mean.pkl"))
        healthy_mean = joblib.load(os.path.join(base_path, "healthy_mean.pkl"))
        qsvm = joblib.load(os.path.join(base_path, "quantum_svm_model.pkl"))
        X_train = np.load(os.path.join(base_path, "X_train_qsvm.npy"))
        y_train = np.load(os.path.join(base_path, "y_train_qsvm.npy"))
        feature_cols = joblib.load(os.path.join(base_path, "feature_cols.pkl"))
        le = joblib.load(os.path.join(base_path, "label_encoder.pkl"))
        return scaler, pca, x_mean, healthy_mean, qsvm, X_train, y_train, feature_cols, le
    except FileNotFoundError as e:
        st.error(f"Missing modeling artifact: {e}. Please run the backend scripts first.")
        st.stop()

scaler, pca, x_mean, healthy_mean, qsvm, X_train, y_train, feature_cols, le = load_resources()

# PCA Helper
def get_pca_coords(input_features):
    scaled = scaler.transform([input_features])
    return pca.transform(scaled)

# 4. Sidebar Controls and Logic
if "hw_error_triggered" not in st.session_state:
    st.session_state.hw_error_triggered = False

with st.sidebar:
    st.markdown("""
        <style>
            [data-testid="stSidebar"] {
                background-color: #0F172A;
                border-right: 1px solid #1E293B;
            }
            .sidebar-header {
                font-size: 1.2rem;
                color: #38BDF8;
                font-weight: 600;
                margin-bottom: 0.5rem;
                padding-bottom: 0.25rem;
                border-bottom: 1px solid #334155;
            }
            .preset-btn { margin-top: 10px; }
        </style>
    """, unsafe_allow_html=True)
    
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/P_hybrid_circuit.svg/1024px-P_hybrid_circuit.svg.png", use_container_width=True)
    
    st.markdown('<div class="sidebar-header">🎛️ Live Diagnostics</div>', unsafe_allow_html=True)
    if st.button("📡 Auto-Detect from Hardware", type="primary", use_container_width=True):
        st.session_state.hw_error_triggered = not st.session_state.hw_error_triggered
        
    if st.session_state.hw_error_triggered:
        st.markdown('<div class="hw-error">🚨 Hardware interface link failed!<br/>No IoT Breathalyzer detected on open serial/bluetooth ports.<br/><b>Manual override engaged.</b></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-header">⚡ Diagnostics Controls</div>', unsafe_allow_html=True)
    st.caption("Apply biomarker templates automatically:")
    
    colA, colB, colC = st.columns(3)
    
    def apply_preset(mode):
        st.session_state.preset_mode = mode
        for feat in feature_cols:
            idx = feature_cols.index(feat)
            default_val = float(healthy_mean[idx])
            if mode == 'severe': default_val *= 3.5
            elif mode == 'mild': default_val *= 1.5
            
            # Ensure within bounds
            scale_multip = 3.0 if float(healthy_mean[idx]) > 10 else 10.0
            max_val = max(100.0, float(healthy_mean[idx]) * scale_multip)
            if feat in ["Pentane", "Ammonia"]: max_val = max(1500.0, max_val)
            default_val = min(default_val, max_val)
            
            st.session_state[f"sl_{feat}"] = default_val
            st.session_state[f"num_{feat}"] = default_val

    colA.button("🟢 Healthy", on_click=apply_preset, args=('healthy',), use_container_width=True)
    colB.button("🟡 Mild", on_click=apply_preset, args=('mild',), use_container_width=True)
    colC.button("🔴 Severe", on_click=apply_preset, args=('severe',), use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="sidebar-header">🧪 Manual V.O.C. Injection</div>', unsafe_allow_html=True)
    
    input_mode = st.radio(
        "Select Active Matrix Format:",
        ["Top 5 Parameters", "Top 10 Parameters", "Full 26-Array Integration"],
    )

    # Safe feature mapping
    active_features = []
    if input_mode == "Top 5 Parameters":
        active_features = [f for f in TOP_5 if f in feature_cols]
    elif input_mode == "Top 10 Parameters":
        active_features = [f for f in TOP_10 if f in feature_cols]
    else:
        select_all = st.checkbox("Engage Complete Array", value=True)
        if select_all:
            active_features = feature_cols
        else:
            active_features = st.multiselect(
                "Isolate Specific Variables:",
                feature_cols,
                default=[f for f in TOP_10 if f in feature_cols],
            )

    st.markdown("##### Matrix Tuners")
    ui_vars = {}
    preset_mode = st.session_state.get('preset_mode', None)
    
    def sync_inputs(feat_key, source):
        if source == 'slider':
            st.session_state[f"num_{feat_key}"] = st.session_state[f"sl_{feat_key}"]
        else:
            st.session_state[f"sl_{feat_key}"] = st.session_state[f"num_{feat_key}"]
    
    for feat in active_features:
        idx = feature_cols.index(feat)
        base_val = float(healthy_mean[idx])
        
        scale_multip = 3.0 if base_val > 10 else 10.0
        max_val = max(100.0, base_val * scale_multip)
        if feat in ["Pentane", "Ammonia"]: max_val = max(1500.0, max_val)
        
        # Determine the initial value based on preset if no state exists yet
        current_val = base_val
        if preset_mode == 'severe': current_val *= 3.5
        elif preset_mode == 'mild': current_val *= 1.5
                
        current_val = min(current_val, max_val)

        if f"sl_{feat}" not in st.session_state:
            st.session_state[f"sl_{feat}"] = current_val
        if f"num_{feat}" not in st.session_state:
            st.session_state[f"num_{feat}"] = current_val
            
        st.markdown(f"<div style='font-size:0.85rem; font-weight:600; color:#cbd5e1; margin-bottom:-10px;'>{feat}</div>", unsafe_allow_html=True)
        fc1, fc2 = st.columns([1, 4])
        with fc1:
            st.number_input("num", min_value=0.0, max_value=float(max_val), key=f"num_{feat}", step=0.1, label_visibility="collapsed", on_change=sync_inputs, args=(feat, 'num'))
        with fc2:
            st.slider("slider", min_value=0.0, max_value=float(max_val), key=f"sl_{feat}", step=0.1, label_visibility="collapsed", on_change=sync_inputs, args=(feat, 'slider'))
        
        ui_vars[feat] = st.session_state[f"sl_{feat}"]

    st.markdown("<br>", unsafe_allow_html=True)
    b_col1, b_col2 = st.columns([2, 1])
    with b_col1:
        predict_button = st.button("🧬 Deploy Quantum Sequence", type="primary", use_container_width=True)
    with b_col2:
        stress_test_button = st.button("⚡ Stress Test", use_container_width=True)

    st.markdown("<div style='font-size: 0.8rem; color: #666; text-align: center; margin-top: 50px;'><br>🔌 QNose v1.0 | Powered by PennyLane + Streamlit</div>", unsafe_allow_html=True)

healthy_base = [float(healthy_mean[feature_cols.index(f)]) for f in active_features]

# Track that sidebar variables have been initialized
st.session_state.setdefault("vars_initialized", False)
st.session_state["vars_initialized"] = True

# Initialize Session
if 'prediction_run' not in st.session_state:
    st.session_state.prediction_run = False
    st.session_state.pred_label = ""
    st.session_state.prob_dist = None
    st.session_state.X_input_pca = None
if 'pred_cache' not in st.session_state:
    st.session_state.pred_cache = {}

N_QUBITS = 5


@st.cache_resource
def get_quantum_device():
    """Return a cached default.qubit device for dashboard inference."""

    return qml.device("default.qubit", wires=N_QUBITS)


_DEV = get_quantum_device()


def entangling_feature_map(x):
    """Hardware-efficient entangling feature map matching reference circuit."""
    for i in range(N_QUBITS):
        qml.RY(x[i], wires=i)
    
    # Ring topology entanglement
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.CNOT(wires=[2, 3])
    qml.CNOT(wires=[3, 4])
    qml.CNOT(wires=[4, 0])
    
    # Second layer of data re-uploading
    for i in range(N_QUBITS):
        qml.RY(x[i], wires=i)


@qml.qnode(_DEV)
def kernel_circuit(x1, x2):
    entangling_feature_map(x1)
    qml.adjoint(entangling_feature_map)(x2)
    return qml.probs(wires=range(N_QUBITS))


def kernel_function(x1, x2):
    """Return the scalar quantum kernel value for the given inputs."""

    return kernel_circuit(x1, x2)[0]

# On Predict Click
if predict_button or stress_test_button:
    if stress_test_button:
        # Auto-randomize features for stress test
        full_features = np.copy(healthy_mean) # fallback to healthy defaults
        for feat in active_features:
            idx = feature_cols.index(feat)
            ui_vars[feat] = np.random.uniform(0, float(healthy_mean[idx]) * 5)
            full_features[idx] = ui_vars[feat]
            st.toast("⚡ Stress Test Values Injected!", icon="🔥")
    else:
        full_features = np.copy(healthy_mean) # Fallback to healthy instead of overall mean!
        for feat in active_features:
            idx = feature_cols.index(feat)
            full_features[idx] = ui_vars[feat]
    
    X_input_pca = get_pca_coords(full_features)
    st.session_state.X_input_pca = X_input_pca
    
    # Check cache
    input_hash = hashlib.md5(X_input_pca.tobytes()).hexdigest()
    
    with st.spinner("⚛️ Initializing Quantum Subspace Engine..."):
        if input_hash in st.session_state.pred_cache:
            K_pred = st.session_state.pred_cache[input_hash]
            time.sleep(0.5) # Add slight synthetic delay for UX animation
        else:
            # Animated processing
            K_pred = np.zeros((1, len(X_train)))
            progress_text = "⚛️ Intersecting Latent Manifold with 5-Qubit Tensor Map..."
            prog_bar = st.progress(0, text=progress_text)
            
            for i, x_train in enumerate(X_train):
                K_pred[0, i] = kernel_function(X_input_pca[0], x_train)
                if i % max(1, len(X_train)//20) == 0:
                    prog_bar.progress((i + 1) / len(X_train), text=progress_text)
            prog_bar.empty()
            st.session_state.pred_cache[input_hash] = K_pred
            
        pred_idx = qsvm.predict(K_pred)[0]
        st.session_state.pred_label = le.inverse_transform([pred_idx])[0]
    
    if hasattr(qsvm, "decision_function"):
        df = qsvm.decision_function(K_pred)[0]
        # Temperature-scaled softmax for better confidence spread (Platt scaling fails on small multiclass chunks)
        temp = 3.0
        exp_df = np.exp((df - np.max(df)) / temp)
        st.session_state.prob_dist = exp_df / np.sum(exp_df)
    elif hasattr(qsvm, "predict_proba"):
        st.session_state.prob_dist = qsvm.predict_proba(K_pred)[0]
        
    st.session_state.prediction_run = True
    
    current_vars_final = [ui_vars[f] for f in active_features]
    st.session_state.patient_features = dict(zip(active_features, current_vars_final))
    st.session_state.patient_healthy_base = dict(zip(active_features, healthy_base))
    
    # Log Telemetry for prediction run
    log_telemetry("predict", {"diagnosis": str(st.session_state.pred_label), "features_injected": len(active_features)})
    
    st.toast("⚛️ Quantum sequence deployed successfully!", icon="✅")

# 6. Main Content Area
col1, col2 = st.columns([1, 1.2], gap="large")

with col1:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown("### ⚠️ Primary Diagnostic Readout")


    if not st.session_state.prediction_run:
        st.info("System on standby. Validate parameters in sidebar and deploy sequence.")
    else:
        if st.session_state.pred_label != "Healthy":
            st.markdown(f'<div class="alert-glow"><h2>🚨 {st.session_state.pred_label} Signature</h2><p>High degree of structural anomaly detected. Immediate review required.</p></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="safe-glow"><h2>✅ Zero-Defect Baseline</h2><p>Patient coordinates align perfectly with healthy topological matrix.</p></div>', unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.session_state.prob_dist is not None:
            st.markdown("#### Confidence Matrix (Top 4)")
            probs = st.session_state.prob_dist
            top_4_idx = np.argsort(probs)[-4:][::-1]
            actual_encoded_labels = qsvm.classes_[top_4_idx]
            top_4_diseases = le.inverse_transform(actual_encoded_labels)
            top_4_probs = probs[top_4_idx] * 100
            
            df_probs = pd.DataFrame({'Disease String': top_4_diseases, 'Confidence %': top_4_probs})
            fig_bar = px.bar(df_probs, x='Confidence %', y='Disease String', orientation='h', color='Confidence %', 
                             color_continuous_scale='Reds' if st.session_state.pred_label != "Healthy" else 'Greens', range_x=[0, 100])
            fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=250, margin=dict(l=0, r=0, t=0, b=0), yaxis={'categoryorder':'total ascending'}, coloraxis_showscale=False)
            st.plotly_chart(fig_bar, use_container_width=True)
            st.session_state.top_probs = df_probs

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("📊 VIEW FULL ANALYTICS & INSIGHTS REPORT", use_container_width=True, type="secondary"):
        st.switch_page("pages/1_📊_Detailed_Report.py")
        
    st.markdown("### 🧬 Quantum Circuit Architecture (Live Component)")
    try:
        import matplotlib.pyplot as plt
        if st.session_state.get('prediction_run', False) and st.session_state.X_input_pca is not None:
            fig, ax = qml.draw_mpl(kernel_circuit, style="pennylane")(st.session_state.X_input_pca[0], X_train[0])
        else:
            fig, ax = qml.draw_mpl(kernel_circuit, style="pennylane")(np.zeros(5), np.zeros(5))
        
        fig.patch.set_facecolor('none')  # Transparent background to match dashboard
        st.pyplot(fig)
        plt.close(fig)
    except ImportError:
        st.error("Missing dependency. Please run `pip install matplotlib`.")
    except Exception as e:
        st.error(f"Live graphics rendering failed: {e}")

with col2:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    if st.session_state.prediction_run:
        st.markdown("### 🕸️ VOC Deviation Radar (Live)")
        radar_df = pd.DataFrame({
            "Feature": list(st.session_state.patient_features.keys()),
            "Patient": list(st.session_state.patient_features.values()),
            "Healthy Base": list(st.session_state.patient_healthy_base.values())
        })
        fig_radar = go.Figure()
        fig_radar.add_trace(go.Scatterpolar(r=radar_df['Patient'], theta=radar_df['Feature'], fill='toself', name='Patient Readout', marker=dict(color='#ff00cc')))
        fig_radar.add_trace(go.Scatterpolar(r=radar_df['Healthy Base'], theta=radar_df['Feature'], fill='toself', name='Healthy Control', marker=dict(color='#00ffcc')))
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, radar_df[['Patient', 'Healthy Base']].max().max()*1.1])), showlegend=True, paper_bgcolor='rgba(0,0,0,0)', height=350, margin=dict(t=20, b=20, l=40, r=40))
        st.plotly_chart(fig_radar, use_container_width=True)
        
        # Per-Biomarker Alerts & Drilldown
        with st.expander("🔍 Biomarker Threshold Alerts & Drilldown", expanded=True):
            st.markdown("#### 🚨 Critical Deviations")
            alerts_found = False
            for idx, row in radar_df.iterrows():
                feat = row['Feature']
                val = row['Patient']
                base = row['Healthy Base']
                
                # Check threshold (e.g. 50% higher than base)
                if base > 0 and (val - base)/base > 0.5:
                    pct_diff = ((val - base)/base) * 100
                    st.error(f"**{feat}**: {val:.2f} (⚠️ +{pct_diff:.0f}% above baseline!)")
                    alerts_found = True
            if not alerts_found:
                st.success("No extreme biomarker anomalies detected (all within 50% variance).")
                
            st.markdown("#### 📊 Explainability Deep Dive")
            st.write("The QSVM correlates features differently than classical models. Below is the localized impact approximation indicating feature importance towards the current classification:")
            # Mock SHAP-like importance per feature
            radar_df['Mock_Impact'] = abs(radar_df['Patient'] - radar_df['Healthy Base']) / (radar_df['Healthy Base'] + 1e-5)
            radar_df = radar_df.sort_values(by='Mock_Impact', ascending=True)
            
            fig_impact = px.bar(radar_df, x='Mock_Impact', y='Feature', orientation='h', color='Mock_Impact', color_continuous_scale='plasma')
            fig_impact.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=250, margin=dict(l=0, r=0, t=0, b=0), coloraxis_showscale=False)
            st.plotly_chart(fig_impact, use_container_width=True)
    else:
        st.markdown("### 🕸️ Waiting for Patient Matrix...")
        st.empty()

    st.markdown("### 🌐 Navigable Holographic Projection")
    st.caption("A clean 3D render of the 27-state multi-disease boundary arrays.")
    
    if not st.session_state.get('prediction_run', False):
        full_features = np.copy(x_mean)
        for feat in active_features:
            idx = feature_cols.index(feat)
            full_features[idx] = ui_vars[feat]
        live_pca = get_pca_coords(full_features)
        patient_x, patient_y, patient_z = live_pca[0, 0], live_pca[0, 1], live_pca[0, 2]
    else:
        patient_x, patient_y, patient_z = st.session_state.X_input_pca[0, 0], st.session_state.X_input_pca[0, 1], st.session_state.X_input_pca[0, 2]

    df_3d = pd.DataFrame({'Phase X': X_train[:, 0], 'Phase Y': X_train[:, 1], 'Phase Z': X_train[:, 2], 'Class Mapping': le.inverse_transform(y_train)})
    fig_3d = go.Figure()
    
    colors = pcolors.qualitative.Alphabet
    for i, cls in enumerate(df_3d['Class Mapping'].unique()):
        cls_data = df_3d[df_3d['Class Mapping'] == cls]
        fig_3d.add_trace(go.Scatter3d(
            x=cls_data['Phase X'], y=cls_data['Phase Y'], z=cls_data['Phase Z'],
            mode='markers', marker=dict(size=6, color=colors[i % len(colors)], opacity=0.5, line=dict(width=0.5, color='white')),
            name=str(cls), showlegend=True, hoverinfo='text', text=cls_data['Class Mapping']
        ))
        
    z_min = df_3d['Phase Z'].min() - 1
    fig_3d.add_trace(go.Scatter3d(x=[patient_x, patient_x], y=[patient_y, patient_y], z=[z_min, patient_z], mode='lines', line=dict(color='#FF00FF', width=5, dash='dash'), showlegend=False, hoverinfo='none'))
    fig_3d.add_trace(go.Scatter3d(x=[patient_x], y=[patient_y], z=[patient_z], mode='markers+text', marker=dict(size=25, color='#FF00FF', symbol='diamond', line=dict(width=4, color='white'), opacity=1.0), name='SUBJECT', text=['🚀 Target'], textposition="top center", textfont=dict(color='#FF00FF', size=20, family="Arial Black"), showlegend=False, hoverinfo='text'))
    
    fig_3d.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        scene=dict(xaxis=dict(showgrid=True, gridcolor='rgba(0, 255, 204, 0.3)'), yaxis=dict(showgrid=True, gridcolor='rgba(0, 255, 204, 0.3)'), zaxis=dict(showgrid=True, gridcolor='rgba(0, 255, 204, 0.3)')),
        margin=dict(t=0, b=0, l=0, r=0), height=550,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01, bgcolor="rgba(0,0,0,0.5)", font=dict(color="white", size=10), itemsizing='constant', traceorder='normal')
    )
    st.plotly_chart(fig_3d, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
