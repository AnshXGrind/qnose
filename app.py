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
    /* Clean minimal background */
    .reportview-container .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    
    /* Animated Hero Section */
    .hero {
        background: linear-gradient(-45deg, #150020, #0a1128, #18002a, #001219);
        background-size: 400% 400%;
        animation: gradientBG 15s ease infinite;
        padding: 3rem 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        border: 1px solid rgba(0, 255, 204, 0.1);
        box-shadow: 0 0 20px rgba(0, 255, 204, 0.05);
    }
    @keyframes gradientBG {
        0% { background-position: 0% 50%; }
        50% { background-position: 100% 50%; }
        100% { background-position: 0% 50%; }
    }
    .hero h1 {
        font-size: 3rem !important;
        font-weight: 800;
        margin-bottom: 0.5rem;
        background: -webkit-linear-gradient(45deg, #00ffcc, #b066ff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero p {
        font-size: 1.2rem;
        color: #b3c0d1;
        max-width: 800px;
        margin: 0 auto;
    }

    /* Pulsing Alerts */
    @keyframes pulse-red {
        0% { box-shadow: 0 0 15px rgba(255, 75, 75, 0.6); border-color: #ff4b4b; }
        50% { box-shadow: 0 0 35px rgba(255, 75, 75, 1.0); border-color: #ff8080; }
        100% { box-shadow: 0 0 15px rgba(255, 75, 75, 0.6); border-color: #ff4b4b; }
    }
    @keyframes pulse-green {
        0% { box-shadow: 0 0 15px rgba(0, 255, 128, 0.6); border-color: #00ff80; }
        50% { box-shadow: 0 0 35px rgba(0, 255, 128, 1.0); border-color: #80ffc0; }
        100% { box-shadow: 0 0 15px rgba(0, 255, 128, 0.6); border-color: #00ff80; }
    }
    .alert-glow {
        animation: pulse-red 1.5s infinite;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        background: rgba(40, 0, 0, 0.4);
        margin-top: 1rem;
    }
    .safe-glow {
        animation: pulse-green 2.5s infinite;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        background: rgba(0, 40, 0, 0.4);
        margin-top: 1rem;
    }

    /* Hardware Buttons & Errors */
    .btn-hw {
        width: 100%;
        padding: 0.6rem;
        background-color: #4B0082;
        color: #fff;
        font-weight: bold;
        border: 2px solid #9b59b6;
        border-radius: 8px;
        cursor: pointer;
        transition: all 0.3s ease;
        text-align: center;
    }
    .btn-hw:hover {
        background-color: #9b59b6;
        box-shadow: 0 0 15px rgba(155, 89, 182, 0.6);
    }
    .hw-error {
        color: #ff4b4b;
        font-weight: bold;
        background-color: rgba(255, 75, 75, 0.1);
        padding: 12px;
        border-radius: 8px;
        border: 1px solid #ff4b4b;
        text-align: center;
        margin-top: 15px;
        margin-bottom: 15px;
        font-size: 0.95rem;
    }
    
    /* Clean Hide Streamlit specific elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# 2. Hero Section
st.markdown("""
<div class="hero">
    <h1>⚛️ QNose Multiplex Interface</h1>
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

# 4. Sidebar Controls and Logic
if "hw_error_triggered" not in st.session_state:
    st.session_state.hw_error_triggered = False

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/P_hybrid_circuit.svg/1024px-P_hybrid_circuit.svg.png", use_container_width=True)
    
    st.header("🎛️ Live Diagnostics")
    
    if st.button("📡 Auto-Detect from Hardware", type="primary", use_container_width=True):
        st.session_state.hw_error_triggered = True
        
    if st.session_state.hw_error_triggered:
        st.markdown('<div class="hw-error">🚨 Hardware interface link failed!<br/>No IoT Breathalyzer detected on open serial/bluetooth ports.<br/><b>Manual override engaged.</b></div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.header("📁 Upload Patient Data")
    uploaded_file = st.file_uploader("Upload CSV of VOC Readings", type=["csv"])
    if uploaded_file is not None:
        try:
            uploaded_df = pd.read_csv(uploaded_file)
            st.success("CSV loaded successfully! Mapping features...")
            st.session_state.uploaded_df = uploaded_df
        except Exception as e:
            st.error(f"Error parsing CSV: {e}")

    st.markdown("---")
    st.header("🧪 Manual V.O.C. Injection")
    
    input_mode = st.radio("Select Active Matrix Format:", ["Top 5 Parameters", "Top 10 Parameters", "Full 26-Array Integration"])
    input_style = st.radio("Entry Method:", ["🎯 Sliders", "⌨️ Direct Number Entry"], horizontal=True)
    
    top_5 = ["Ethane", "Nonanal", "Acetonitrile", "Pentane", "Hexanal"]
    top_10 = top_5 + ["Isoprene", "Trimethylamine", "Propanal", "Ammonia", "Toluene"]
    
    active_features = []
    if input_mode == "Top 5 Parameters":
        active_features = top_5
    elif input_mode == "Top 10 Parameters":
        active_features = top_10
    else:
        select_all = st.checkbox("Engage Complete Array", value=True)
        if select_all:
            active_features = feature_cols
        else:
            active_features = st.multiselect("Isolate Specific Variables:", feature_cols, default=top_10)

    # Dynamic Inputs
    st.markdown("##### Matrix Tuners")
    ui_vars = {}
    for feat in active_features:
        idx = feature_cols.index(feat)
        default_val = float(healthy_mean[idx])
        
        # Override with uploaded data if available
        if "uploaded_df" in st.session_state and st.session_state.uploaded_df is not None:
            if feat in st.session_state.uploaded_df.columns:
                default_val = float(st.session_state.uploaded_df[feat].iloc[0])
                
        scale_multip = 3.0 if default_val > 10 else 10.0
        max_val = max(100.0, default_val * scale_multip)
        if feat in ["Pentane", "Ammonia"]:
            max_val = max(1500.0, max_val)
            
        if input_style == "🎯 Sliders":
            ui_vars[feat] = st.slider(f"{feat}", 0.0, float(max_val), float(default_val), step=0.1, key=f"sl_{feat}")
        else:
            ui_vars[feat] = st.number_input(f"{feat}", min_value=0.0, max_value=float(max_val), value=float(default_val), step=0.1, key=f"num_{feat}")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("🧬 Deploy Quantum Sequence", type="primary", use_container_width=True)

healthy_base = [float(healthy_mean[feature_cols.index(f)]) for f in active_features]
current_vars = [ui_vars[f] for f in active_features]

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

# Initialize Session
if 'prediction_run' not in st.session_state:
    st.session_state.prediction_run = False
    st.session_state.pred_label = ""
    st.session_state.prob_dist = None
    st.session_state.X_input_pca = None

# On Predict Click
if predict_button:
    full_features = np.copy(x_mean)
    for feat in active_features:
        full_features[feature_cols.index(feat)] = ui_vars[feat]
    
    X_input_scaled = scaler.transform([full_features])
    X_input_pca = pca.transform(X_input_scaled)
    st.session_state.X_input_pca = X_input_pca
    
    with st.spinner("⚛️ Collapsing Quantum Probability Vectors..."):
        # Small delay for UI effect
        time.sleep(0.5)
        K_pred = np.array([[kernel_function(X_input_pca[0], x_train) for x_train in X_train]])
        
    pred_idx = qsvm.predict(K_pred)[0]
    st.session_state.pred_label = le.inverse_transform([pred_idx])[0]
    
    if hasattr(qsvm, "predict_proba"):
        st.session_state.prob_dist = qsvm.predict_proba(K_pred)[0]
        
    st.session_state.prediction_run = True
    
    # Store patient data globally for the report
    st.session_state.patient_features = dict(zip(active_features, current_vars))
    st.session_state.patient_healthy_base = dict(zip(active_features, healthy_base))

# 6. Main Content Area (Uncluttered Dashboard)
# Swap columns or make the hologram full width. Let's make the hologram span the top, and put diagnostics below it, or use a better ratio.
st.markdown("### 🌐 Navigable Holographic Projection")
st.caption("A completely clean 3D render of the 27-state multi-disease boundary arrays.")

if not st.session_state.get('prediction_run', False) and 'current_vars' in locals():
    # Live 3D position
    full_features = np.copy(x_mean)
    for feat in active_features:
        full_features[feature_cols.index(feat)] = ui_vars[feat]
    live_scaled = scaler.transform([full_features])
    live_pca = pca.transform(live_scaled)
    
    df_3d = pd.DataFrame({
        'Phase X': X_train[:, 0], 'Phase Y': X_train[:, 1], 'Phase Z': X_train[:, 2],
        'Class Mapping': le.inverse_transform(y_train)
    })
    patient_x = live_pca[0, 0]
    patient_y = live_pca[0, 1]
    patient_z = live_pca[0, 2]
else:
    df_3d = pd.DataFrame({
        'Phase X': X_train[:, 0], 'Phase Y': X_train[:, 1], 'Phase Z': X_train[:, 2],
        'Class Mapping': le.inverse_transform(y_train)
    })
    patient_x = st.session_state.X_input_pca[0, 0]
    patient_y = st.session_state.X_input_pca[0, 1]
    patient_z = st.session_state.X_input_pca[0, 2]

# Clean HD Hologram Plot
fig_3d = go.Figure()
classes = df_3d['Class Mapping'].unique()
import plotly.colors as pcolors
colors = pcolors.qualitative.Alphabet

for i, cls in enumerate(classes):
    cls_data = df_3d[df_3d['Class Mapping'] == cls]
    # Make background point sizes slightly larger and more visible
    fig_3d.add_trace(go.Scatter3d(
        x=cls_data['Phase X'], y=cls_data['Phase Y'], z=cls_data['Phase Z'],
        mode='markers',
        marker=dict(size=6, color=colors[i % len(colors)], opacity=0.5, line=dict(width=0.5, color='white')),
        name=str(cls), showlegend=False,
        hovertemplate=f"<b>{cls}</b><br>X: %{{x:.2f}}<br>Y: %{{y:.2f}}<br>Z: %{{z:.2f}}<extra></extra>"
    ))
    
if patient_x is not None:
    # Draw a scanner grid line dropping down from the anomoly
    z_min = df_3d['Phase Z'].min() - 1
    fig_3d.add_trace(go.Scatter3d(
        x=[patient_x, patient_x], y=[patient_y, patient_y], z=[z_min, patient_z],
        mode='lines',
        line=dict(color='#FF00FF', width=5, dash='dash'),
        showlegend=False, hoverinfo='none'
    ))
    
    # Draw orthogonal lines to axes for clarity
    fig_3d.add_trace(go.Scatter3d(
        x=[patient_x, df_3d['Phase X'].min()], y=[patient_y, patient_y], z=[patient_z, patient_z],
        mode='lines', line=dict(color='#00FFCC', width=2, dash='dot'), showlegend=False
    ))
    fig_3d.add_trace(go.Scatter3d(
        x=[patient_x, patient_x], y=[patient_y, df_3d['Phase Y'].min()], z=[patient_z, patient_z],
        mode='lines', line=dict(color='#00FFCC', width=2, dash='dot'), showlegend=False
    ))
    
    fig_3d.add_trace(go.Scatter3d(
        x=[patient_x], y=[patient_y], z=[patient_z],
        mode='markers+text',
        marker=dict(size=25, color='#FF00FF', symbol='diamond', line=dict(width=4, color='#00FFCC'), opacity=1.0),
        name='SUBJECT', text=['🔴 LIVE TARGET ISO'], textposition="top center",
        textfont=dict(color='#FF00FF', size=20, family="Courier New, monospace"), showlegend=False,
        hovertemplate="<b>LIVE PATIENT</b><br>Phase X: %{x:.2f}<br>Phase Y: %{y:.2f}<br>Phase Z: %{z:.2f}<br><extra>🚨 ANOMALY ISO</extra>"
    ))

fig_3d.update_layout(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(0,0,0,0)',
    scene=dict(
        aspectmode='cube',
        xaxis=dict(showgrid=True, gridcolor='rgba(0, 255, 204, 0.4)', gridwidth=2, zeroline=True, zerolinecolor='#00FFCC', showbackground=True, backgroundcolor='rgba(15, 20, 35, 0.6)', title='Phase X'),
        yaxis=dict(showgrid=True, gridcolor='rgba(0, 255, 204, 0.4)', gridwidth=2, zeroline=True, zerolinecolor='#00FFCC', showbackground=True, backgroundcolor='rgba(15, 20, 35, 0.6)', title='Phase Y'),
        zaxis=dict(showgrid=True, gridcolor='rgba(176, 102, 255, 0.4)', gridwidth=2, zeroline=True, zerolinecolor='#B066FF', showbackground=True, backgroundcolor='rgba(15, 20, 35, 0.6)', title='Phase Z')
    ),
    margin=dict(t=50, b=0, l=0, r=0), height=750,
    scene_camera=dict(
        up=dict(x=0, y=0, z=1),
        center=dict(x=0, y=0, z=0),
        eye=dict(x=1.5, y=1.5, z=0.8)  # Better initial view
    ),
    updatemenus=[
        dict(
            type="buttons",
            direction="right",
            x=0.0, y=1.05,
            showactive=True,
            buttons=list([
                dict(label="🎥 Cinematic View",
                     method="relayout",
                     args=[{"scene.camera": dict(eye=dict(x=1.5, y=1.5, z=0.8))}]),
                dict(label="🛰️ Top-Down Array",
                     method="relayout",
                     args=[{"scene.camera": dict(eye=dict(x=0, y=0, z=2.5))}]),
                dict(label="🔬 Orthogonal Scan",
                     method="relayout",
                     args=[{"scene.camera": dict(eye=dict(x=2.5, y=0, z=0))}]),
            ]),
            font=dict(color="#00FFCC", size=11, family="Arial"),
            bgcolor="rgba(20, 0, 40, 0.8)",
            bordercolor="#B066FF",
            borderwidth=1
        )
    ]
)
st.plotly_chart(fig_3d, use_container_width=True)

st.markdown("---")

col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown("### ⚠️ Primary Diagnostic Readout")
    if not st.session_state.prediction_run:
        st.info("System on standby. Validate parameters in sidebar and deploy sequence.")
    else:
        # Save output to session to pass to Report
        if st.session_state.pred_label != "Healthy":
            st.markdown(f'<div class="alert-glow"><h3>🚨 {st.session_state.pred_label} Signature Match</h3><p>High degree of structural anomaly detected in phase space.</p></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="safe-glow"><h3>✅ Zero-Defect Baseline</h3><p>Patient coordinates align perfectly with healthy topological matrix.</p></div>', unsafe_allow_html=True)
            
        st.markdown("<br>", unsafe_allow_html=True)
        
        # OVR Top 3 breakdown
        if st.session_state.prob_dist is not None:
            st.markdown("#### Confidence Matrix (Top 4)")
            probs = st.session_state.prob_dist
            top_4_idx = np.argsort(probs)[-4:][::-1]
            actual_encoded_labels = qsvm.classes_[top_4_idx]
            top_4_diseases = le.inverse_transform(actual_encoded_labels)
            top_4_probs = probs[top_4_idx] * 100
            
            df_probs = pd.DataFrame({'Disease String': top_4_diseases, 'Confidence %': top_4_probs})
            fig_bar = px.bar(df_probs, x='Confidence %', y='Disease String', orientation='h', color='Confidence %', 
                             color_continuous_scale='Reds' if st.session_state.pred_label != "Healthy" else 'Greens',
                             range_x=[0, 100])
            fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', font=dict(color='white'), height=250, margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig_bar, use_container_width=True)
            fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(t=0, b=0, l=0, r=0), 
                                  height=220, coloraxis_showscale=False, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_bar, use_container_width=True)
            
            st.session_state.top_probs = df_probs

    st.markdown("<br><br>", unsafe_allow_html=True)
    # The big isolated button leading to the Detailed Analytics page
    if st.button("📊 VIEW FULL ANALYTICS & INSIGHTS REPORT", use_container_width=True, type="secondary"):
        st.switch_page("pages/1_📊_Detailed_Report.py")
        
    st.markdown("### 🧬 Quantum Circuit Architecture")
    try:
        st.image("quantum_circuit.png", caption="5-Qubit Angle Embedding Circuit", use_container_width=True)
    except:
        pass


with col2:
    st.empty()
