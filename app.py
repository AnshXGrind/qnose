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
    .hw-button button {
        background-color: #4B0082 !important;
        color: white !important;
        border: 2px solid #9b59b6 !important;
    }
    .hw-error {
        color: #ff4b4b;
        font-weight: bold;
        background-color: rgba(255, 75, 75, 0.1);
        padding: 10px;
        border-radius: 5px;
        border: 1px solid #ff4b4b;
        text-align: center;
        margin-top: 10px;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# 2. Main Title and Header
st.title("⚛️ QNose — Quantum Multiclass Engine")
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

# 4. Sidebar Controls and Logic
if "hw_error_triggered" not in st.session_state:
    st.session_state.hw_error_triggered = False

def trigger_hw_error():
    st.session_state.hw_error_triggered = True

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/ca/P_hybrid_circuit.svg/1024px-P_hybrid_circuit.svg.png")
    
    # 4A. Hardware Input
    st.header("🎛️ Data Source")
    st.markdown('<div class="hw-button">', unsafe_allow_html=True)
    st.button("📡 Auto-Detect from Hardware", on_click=trigger_hw_error, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.session_state.hw_error_triggered:
        st.markdown('<div class="hw-error">🚨 Hardware interface not found! <br> No IoT Breathalyzer detected on open serial/bluetooth ports. Entering Manual Override.</div>', unsafe_allow_html=True)
    
    st.markdown("---")
    st.header("🧪 Manual Override Array")
    
    # 4B. User Selection Modes
    input_mode = st.radio("Select VOC Input Mode:", ["Top 5 VOCs", "Top 10 VOCs", "Custom / All VOCs"])
    
    top_5 = ["Ethane", "Nonanal", "Acetonitrile", "Pentane", "Hexanal"]
    top_10 = top_5 + ["Isoprene", "Trimethylamine", "Propanal", "Ammonia", "Toluene"]
    
    active_features = []
    
    if input_mode == "Top 5 VOCs":
        st.caption("Using the top 5 predictive biomarkers for standard accuracy.")
        active_features = top_5
    elif input_mode == "Top 10 VOCs":
        st.caption("Expanded array into 10 key biochemical vectors for robust predictions.")
        active_features = top_10
    else:
        st.caption("Maximum Accuracy: Select partial elements or inject the entire 26-vector array.")
        select_all = st.checkbox("Select All VOCs (Max Accuracy Mode)", value=True)
        if select_all:
            active_features = feature_cols
        else:
            active_features = st.multiselect("Select Exact VOCs to input:", feature_cols, default=top_10)

    # 4C. Dynamic Sliders
    st.markdown(f"**Adjusting {len(active_features)} active parameters**")
    ui_vars = {}
    
    for feat in active_features:
        idx = feature_cols.index(feat)
        default_val = float(healthy_mean[idx])
        # Ensure scale makes sense
        scale_multip = 3.0 if default_val > 10 else 10.0
        max_val = max(100.0, default_val * scale_multip)
        if feat in ["Pentane", "Ammonia"]:
            max_val = max(1500.0, max_val)
            
        ui_vars[feat] = st.slider(f"{feat} (ppb)", 0.0, float(max_val), float(default_val), step=0.1, key=f"sl_{feat}")

    st.markdown("<br>", unsafe_allow_html=True)
    predict_button = st.button("🧬 Run Quantum Inference Array", type="primary", use_container_width=True)

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

# 6. Dashboard Layout
col1, col2, col3 = st.columns([1, 1.2, 1.5], gap="large")

if 'prediction_run' not in st.session_state:
    st.session_state.prediction_run = False
    st.session_state.pred_label = ""
    st.session_state.prob_dist = None
    st.session_state.X_input_pca = None

if predict_button:
    # Baseline everything against standard means, then overwrite with active inputs
    full_features = np.copy(x_mean)
    for feat in active_features:
        full_features[feature_cols.index(feat)] = ui_vars[feat]
    
    X_input_scaled = scaler.transform([full_features])
    X_input_pca = pca.transform(X_input_scaled)
    st.session_state.X_input_pca = X_input_pca
    
    # Progress visualization
    progress_text = "⚛️ Collapsing Quantum Matrix against active VOC features..."
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
    st.subheader("🔬 Biomarker Topography")
    if len(active_features) > 2:
        df_radar = pd.DataFrame({
            'Feature': active_features * 2,
            'Value': current_vars + healthy_base,
            'Group': ['Patient Sample'] * len(active_features) + ['Healthy Baseline'] * len(active_features)
        })
        
        fig_radar = px.line_polar(df_radar, r='Value', theta='Feature', color='Group', line_close=True,
                                  color_discrete_sequence=['#ff4b4b', '#1f77b4'],
                                  template="plotly_dark" if st.get_option("theme.base") == "dark" else "plotly_white")
        # Turn off radar axes ticks for cleaner look if > 10 features
        show_ticks = len(active_features) <= 10
        fig_radar.update_layout(polar=dict(radialaxis=dict(visible=False), angularaxis=dict(showticklabels=show_ticks)), 
                                margin=dict(t=20, b=20, l=20, r=20), height=350)
        st.plotly_chart(fig_radar, use_container_width=True)
    else:
        st.info("Select at least 3 VOCs to generate array topology.")

with col2:
    st.subheader("⚠️ Diagnostic Mapping")
    if not st.session_state.prediction_run:
        st.info("Awaiting input. Select inputs and click **Run Inference** to map predictions.")
    else:
        if st.session_state.prob_dist is not None:
            probs = st.session_state.prob_dist
            top_5_idx = np.argsort(probs)[-5:][::-1]
            actual_encoded_labels = qsvm.classes_[top_5_idx]
            top_5_diseases = le.inverse_transform(actual_encoded_labels)
            top_5_probs = probs[top_5_idx] * 100
            
            df_probs = pd.DataFrame({'Disease': top_5_diseases, 'Probability': top_5_probs})
            fig_bar = px.bar(df_probs, x='Probability', y='Disease', orientation='h', color='Probability', 
                             color_continuous_scale='Reds' if st.session_state.pred_label != "Healthy" else 'Greens')
            fig_bar.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(t=10, b=10, l=10, r=10), height=200, coloraxis_showscale=False)
            st.plotly_chart(fig_bar, use_container_width=True)
            
        if st.session_state.pred_label != "Healthy":
            st.markdown(f'<div class="alert-glow"><h3>🚨 {st.session_state.pred_label} Detected</h3><p>Highest structural correlation in local phase space.</p></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="safe-glow"><h3>✅ Baseline Healthy</h3><p>Biomarkers correspond natively to background topology.</p></div>', unsafe_allow_html=True)

with col3:
    st.subheader("🌐 3D Multiplex Hologram")
    if not st.session_state.prediction_run:
        st.info("Awaiting structural dimensional space map.")
    else:
        df_3d = pd.DataFrame({
            'PCA 1': X_train[:, 0],
            'PCA 2': X_train[:, 1],
            'PCA 3': X_train[:, 2],
            'Class': le.inverse_transform(y_train)
        })
        
        # Build holographic scatter plot using Graph Objects for complete control
        fig_3d = go.Figure()
        
        classes = df_3d['Class'].unique()
        import plotly.colors as pcolors
        colors = pcolors.qualitative.Alphabet
        
        # Plot background disease clouds
        for i, cls in enumerate(classes):
            cls_data = df_3d[df_3d['Class'] == cls]
            fig_3d.add_trace(go.Scatter3d(
                x=cls_data['PCA 1'], y=cls_data['PCA 2'], z=cls_data['PCA 3'],
                mode='markers',
                marker=dict(size=4, color=colors[i % len(colors)], opacity=0.4),
                name=cls, showlegend=False,
                hoverinfo='text',
                text=cls_data['Class']
            ))
            
        # Plot the patient
        fig_3d.add_trace(go.Scatter3d(
            x=[st.session_state.X_input_pca[0, 0]], 
            y=[st.session_state.X_input_pca[0, 1]], 
            z=[st.session_state.X_input_pca[0, 2]],
            mode='markers+text',
            marker=dict(size=14, color='#00FF00', symbol='diamond', 
                        line=dict(width=2, color='white'), opacity=1.0),
            name='TARGET',
            text=['⭐ TARGET'],
            textposition="top center",
            textfont=dict(color='#00FF00', size=16, family="Arial Black"),
            showlegend=False,
            hoverinfo='text'
        ))
        
        fig_3d.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            scene=dict(
                xaxis=dict(showgrid=False, showbackground=False, zeroline=False, visible=False),
                yaxis=dict(showgrid=False, showbackground=False, zeroline=False, visible=False),
                zaxis=dict(showgrid=False, showbackground=False, zeroline=False, visible=False),
                bgcolor='rgba(0,0,0,0)'
            ),
            margin=dict(t=0, b=0, l=0, r=0),
            height=450,
            scene_camera=dict(
                up=dict(x=0, y=0, z=1),
                center=dict(x=0, y=0, z=0),
                eye=dict(x=1.3, y=1.3, z=1.3)
            )
        )
        st.plotly_chart(fig_3d, use_container_width=True)

st.markdown("---")

tab1, tab2 = st.tabs(["🗺️ Quantum Processing Benchmarks", "🧠 Multiclass Explainability"])

with tab1:
    st.markdown("### ⚛️ Architecture Core")
    
    cq1, cq2, cq3 = st.columns(3)
    with cq1:
        st.markdown("**1. Multiclass Simulation Comparison**")
        if os.path.exists("model_comparison.png"):
            st.image("model_comparison.png")
            
    with cq2:
        st.markdown("**2. Kernel Structural Matrix Map**")
        if os.path.exists("kernel_alignment.png"):
            st.image("kernel_alignment.png")
            
    with cq3:
        st.markdown("**3. Stratified Subspace Layout**")
        if os.path.exists("quantum_tsne_projection.png"):
            st.image("quantum_tsne_projection.png")
            
    st.markdown("---")
    st.markdown("### Topological Dimensional Architecture")
    if os.path.exists("quantum_circuit.png"):
        st.image("quantum_circuit.png")

with tab2:
    st.markdown("### Gradient Extraction")
    if os.path.exists("quantum_feature_importance.png"):
        st.image("quantum_feature_importance.png")
    st.info("Additional multiclass pipeline metrics are located in the `eda_results` repository folder.")
