# QNose — Quantum Breath Intelligence

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PennyLane](https://img.shields.io/badge/PennyLane-Quantum%20ML-6f42c1)](https://pennylane.ai/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-ff4b4b)](https://streamlit.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Quantum ML](https://img.shields.io/badge/topic-Quantum%20ML-indigo)

<div align="center">
  <h1>⚛️ QNose — Quantum Breath Intelligence</h1>
  <p><strong>Classifying 27 disease signatures from exhaled breath using a Quantum Support Vector Machine.</strong></p>
</div>

---

## What is QNose?

QNose is a quantum–classical hybrid diagnostic system that explores whether non‑invasive analysis of exhaled breath can be used to screen for complex diseases. Each patient breath sample is converted into a panel of Volatile Organic Compound (VOC) concentrations, measured in parts‑per‑billion (ppb). These VOC fingerprints have been linked in the literature to a wide range of conditions, from cancers and neurodegenerative disorders to metabolic and infectious diseases. Instead of requiring blood draws or imaging, QNose aims to learn these signatures directly from breath.

Technically, QNose takes 26 curated VOC features, applies a `StandardScaler`, and projects them into a 5‑dimensional latent space using `PCA(n_components=5)`. Those 5 principal components map one‑to‑one onto a 5‑qubit quantum circuit via PennyLane’s `AngleEmbedding`. A quantum kernel is constructed by evaluating inner products ⟨ψ(x)|ψ(x')⟩ between embedded states, and a Support Vector Machine with a precomputed kernel is trained on this quantum similarity matrix to predict one of 27 disease labels.

To ground the quantum model, QNose also trains strong classical baselines on the same PCA representation: an RBF‑kernel SVM, a Random Forest, and an XGBoost classifier. Their accuracies, confusion matrices, and calibration behaviour are compared directly against the Quantum SVM (QSVM), providing a realistic view of where quantum methods match or improve on well‑tuned classical models.

On top of the models, QNose ships with a fully interactive Streamlit dashboard. Clinicians and researchers can manipulate VOC levels via sliders, see the resulting diagnosis and confidence, inspect a live 3D PCA “hologram” of where the patient sits within the multi‑disease manifold, and trigger explainability analyses such as SHAP waterfalls and quantum t‑SNE embeddings. The UI includes pulsing alerts for high‑risk predictions and generates PDF reports suitable for offline review.

---

## Architecture Diagram (ASCII)

```text
Raw VOC CSV (VOC_MultiDisease_Dataset.csv)
          |
          v
  Feature Selection (26 VOC columns)
          |
          v
    StandardScaler (z-score normalization)
          |
          v
        PCA (5 components)
          |
   +------+-----------------------------+
   |                                    |
   v                                    v
[ Quantum Branch ]                 [ Classical Branch ]

  5-D PCA vector x               5-D PCA vector x
          |                                |
          v                                v
  Entangling Feature Map          SVM / Random Forest / XGBoost
          |
          v
   |ψ(x)⟩ on 5-qubit device
          |
          v
   ⟨ψ(x) | ψ(x')⟩ kernel entries
          |
          v
   QSVM (SVC, kernel='precomputed')
          |
          +---------------+----------------+
                          |
                          v
                  Prediction (27 classes)
                          |
                          v
                 Streamlit Dashboard UI
   - Live 3D PCA hologram
   - Confidence matrix (Top-4)
   - SHAP & quantum t-SNE
   - PDF report generation
```

---

## Features

- **Quantum SVM classifier (QSVM)** – 5‑qubit hardware-efficient entangling kernel trained on the PCA‑compressed VOC space for 27‑class disease prediction.
- **Classical baselines (SVM / RF / XGBoost)** – Strong classical reference models on the same features for fair, side‑by‑side comparison.
- **Live 3D PCA hologram** – Interactive Plotly 3D scatter plot showing the patient’s location relative to all training samples in PCA space.
- **SHAP explainability for classical SVM** – Kernel SHAP waterfall plot explaining why a particular sample was classified as a given disease.
- **Quantum circuit diagram rendering** – PennyLane’s `draw_mpl` used to generate a visual quantum circuit diagram for the kernel circuit.
- **CSV upload / batch evaluation** – Support for loading VOC CSV data and running offline scenarios or bulk experiments through the pipeline.
- **Hardware auto‑detect stub** – Streamlit sidebar control that emulates connection to an external breathalyzer / e‑nose IoT sensor.
- **Kernel alignment heatmap** – Precomputed quantum kernel matrix saved as `kernel_matrix.npy`, ready to visualise class‑wise alignment structure.
- **Quantum t‑SNE and embeddings** – Quantum‑aware dimensionality reduction and embedding plots in the analytics pages for exploratory analysis.
- **Model comparison charts** – Side‑by‑side accuracy and confusion matrix visualisations for QSVM vs SVM/RF/XGBoost.
- **Pulsing diagnosis alerts** – Animated red/green cards in the dashboard that highlight high‑risk vs healthy predictions.
- **Confidence matrix (Top‑4)** – Horizontal bar chart of the top four predicted diseases with calibrated probability scores.
- **PDF report generation** – Automated multi‑page PDF report (via `report_generator.py`) containing diagnosis, charts, and metadata.

---

## Tech Stack

| Component          | Technology         | Version  |
|--------------------|--------------------|----------|
| Quantum Framework  | PennyLane          | 0.36     |
| Classical ML       | scikit-learn       | 1.4      |
| Boosting           | XGBoost            | 2.0      |
| Dashboard          | Streamlit          | 1.35     |
| Visualization      | Plotly             | 5.22     |
| Explainability     | SHAP               | 0.45     |
| Data               | pandas + numpy     | 2.2 / 1.26 |
| Reporting          | fpdf2              | 2.7      |

> Exact versions are recorded in `requirements.txt`. The versions above reflect the intended baseline environment.

---

## Quickstart

### 1. Clone the repository

```bash
git clone https://github.com/AnshXGrind/qnose.git
cd qnose
```

### 2. Install dependencies

It is recommended to use a virtual environment (e.g. `venv`, `conda`, or `poetry`). Then install:

```bash
pip install -r requirements.txt
```

### 3. Prepare the dataset

Place `VOC_MultiDisease_Dataset.csv` inside the `data/` directory:

```text
qnose/
  data/
    VOC_MultiDisease_Dataset.csv
```

The dataset is **not** committed to the repository; you must source it separately.

### 4. Train classical models and preprocessing artifacts

```bash
python classical_svm.py  # Step 1: train classical models + save .pkl/.npy artifacts
```

This script:

- Performs feature selection over 26 VOC columns.
- Fits `StandardScaler` and `PCA(n_components=5)` preprocessing.
- Trains an RBF SVM, Random Forest, and XGBoost classifier.
- Saves all encoders, PCA model, and evaluation predictions to disk.

### 5. Train the Quantum SVM kernel

```bash
python quantum_kernel.py  # Step 2: train Quantum SVM (takes ~10–30 min on CPU)
```

The quantum pipeline:

- Loads the preprocessed PCA features and label encoder from `classical_svm.py`.
- Uses a **5‑qubit** PennyLane `lightning.qubit` device for fast simulation.
- Subsamples to **500 training points** and **100 test points** for tractable \(O(n^2)\) kernel computation.
- Builds a precomputed kernel matrix and fits `SVC(kernel="precomputed", probability=True, break_ties=True)`.
- Saves the quantum model, kernel matrix, and down‑sampled train/test splits as `.npy` artifacts.

### 6. Launch the Streamlit dashboard

```bash
streamlit run app.py  # Step 3: launch the dashboard
```

Then open the provided local URL (typically `http://localhost:8501`) in your browser.

The app uses a lightweight `default.qubit` device for inference‑time kernel evaluation to remain compatible across environments where `lightning.qubit` may not be available.

---

## File-by-File Reference

| File                               | Purpose                                                                 | Run order |
|------------------------------------|-------------------------------------------------------------------------|-----------|
| `classical_svm.py`                 | Trains StandardScaler + PCA + SVM/RF/XGBoost, and saves all core artifacts. | 1         |
| `quantum_kernel.py`                | Builds the 5‑qubit quantum kernel and trains the Quantum SVM.          | 2         |
| `eda_analysis.py`                  | Performs exploratory data analysis and writes plots to `eda_results/`. | optional  |
| `explainability.py`                | Generates SHAP waterfall explainability for the classical SVM.         | 3         |
| `quantum_explainability.py`        | Runs quantum‑specific explainability (t‑SNE, embeddings, kernel views).| 4         |
| `compare_models.py` / `compare.py` | Compares classical vs quantum models via metrics and confusion matrices.| 5         |
| `report_generator.py`              | Builds a multi‑page PDF summary report of selected runs.               | 6         |
| `app.py`                           | Streamlit dashboard for interactive diagnostics and visualization.     | final     |
| `pages/1_📊_Detailed_Report.py`   | Additional analytics dashboard page for deeper drill‑downs.            | final     |

---

## Data

The core dataset used by QNose is `data/VOC_MultiDisease_Dataset.csv`.

- **Rows:** Individual breath samples.
- **Target column:** `Disease Label` — a categorical field with **27 disease classes**, including `Healthy`.
- **Features:** 26 VOC concentration columns in ppb, including (but not limited to):
  - `Ethane`
  - `Nonanal`
  - `Acetonitrile`
  - `Pentane`
  - `Hexanal`
  - `Isoprene`
  - `Trimethylamine`
  - `Propanal`
  - `Ammonia`
  - `Toluene`
  - `Acetone`, `Methanol`, `Ethanol`, `Acetaldehyde`, `Butanal`, `Furan`, `Limonene`, `Alpha Pinene`, and additional trace VOCs
- **Split:** The training pipeline performs a **stratified 70/30 train/test split**, ensuring that all 27 classes are represented proportionally in both sets.

The `data/` directory is intentionally **git‑ignored** to avoid committing patient or proprietary data. You must supply `VOC_MultiDisease_Dataset.csv` yourself to run the project.

---

## The Quantum Kernel — How It Works

1. **PCA → Qubits mapping**  
   The classical feature vector has 26 VOC dimensions. To keep the quantum circuit shallow and hardware‑realistic, QNose compresses this vector into **5 principal components** using PCA. Each component captures a dominant mode of variation in the VOC space. Those 5 components are mapped directly onto **5 qubits**, one scalar per qubit, which balances expressivity with simulation cost.

2. **AngleEmbedding circuit design**  
   PennyLane’s `AngleEmbedding` template encodes each PCA component into the state of its corresponding qubit via rotation gates (e.g. RX/RY/RZ). Conceptually, for a 5‑D PCA vector \(x = (x_0, …, x_4)\), the circuit applies rotations of the form
   \(R_Y(x_i)\) or similar on each qubit \(i\). This produces a quantum state \(|ψ(x)⟩\) whose amplitudes depend smoothly on the input features.

3. **Adjoint trick for kernel entries**  
   To compute the quantum kernel between two inputs \(x\) and \(x'\), QNose uses a two‑stage circuit:

   - Prepare \(|ψ(x)⟩\) using `AngleEmbedding(x)`.
   - Apply the adjoint embedding `adjoint(AngleEmbedding)(x')`.

   The resulting probability vector `kernel_circuit(x1, x2)` is a distribution over all computational basis states. The overlap ⟨ψ(x)|ψ(x')⟩ is retrieved as the probability of the all‑zeros state:

   \[
   k(x, x') = \langle ψ(x) \mid ψ(x') \rangle = \texttt{kernel\_circuit}(x, x')[0].
   \]

   This value is used as the (i, j) entry in the kernel matrix.

4. **Precomputed SVC configuration**  
   Once the kernel matrices are built, QNose trains a multi‑class SVM using scikit‑learn’s `SVC` with

   - `kernel="precomputed"`
   - `probability=True` (to enable calibrated probability estimates)
   - `break_ties=True` (more stable tie‑breaking between close classes)

   The training kernel \(K_{\text{train}}\) is used to fit the model, and the test kernel \(K_{\text{test}}\) is used for evaluation.

5. **Why `lightning.qubit` for training**  
   Kernel computation requires evaluating a quantum circuit for every pair of samples, which is \(O(n^2)\) in the number of points. Using PennyLane’s `lightning.qubit` simulator (a high‑performance C++ backend) makes this feasible on a CPU. The training script subsamples to **500 training** and **100 test** points explicitly to keep runtimes in the 10–30 minute range on a typical laptop.

6. **Why `default.qubit` in `app.py`**  
   The Streamlit dashboard (`app.py`) performs *inference* only: it computes kernel entries between the current patient vector and the fixed training subset. To maximise compatibility across environments (especially where `lightning.qubit` may not be installed), the app uses PennyLane’s `default.qubit` device wrapped in a cached resource. This avoids repeated device construction on Streamlit reruns while remaining easy to install anywhere.

---

## Artifacts Generated

| Artifact                    | Type    | Generated by         | Consumed by                          | Description |
|----------------------------|---------|----------------------|--------------------------------------|-------------|
| `label_encoder.pkl`        | `.pkl`  | `classical_svm.py`   | `classical_svm.py`, `quantum_kernel.py`, `app.py`, `compare_models.py` | Encodes string disease labels into integer classes. |
| `feature_cols.pkl`         | `.pkl`  | `classical_svm.py`   | `classical_svm.py`, `quantum_kernel.py`, `app.py`, `explainability.py` | List of selected VOC feature column names. |
| `healthy_mean.pkl`         | `.pkl`  | `classical_svm.py`   | `app.py`                             | Mean VOC profile over `Healthy` samples (dashboard baseline). |
| `x_mean.pkl`               | `.pkl`  | `classical_svm.py`   | `app.py`                             | Global mean VOC profile across all samples. |
| `scaler.pkl`               | `.pkl`  | `classical_svm.py`   | `classical_svm.py`, `quantum_kernel.py`, `app.py`, `explainability.py` | Fitted `StandardScaler` for feature normalization. |
| `pca.pkl`                  | `.pkl`  | `classical_svm.py`   | `classical_svm.py`, `quantum_kernel.py`, `app.py`, `explainability.py` | Fitted `PCA(n_components=5)` model. |
| `classical_svm_model.pkl`  | `.pkl`  | `classical_svm.py`   | `explainability.py`, `compare_models.py` | RBF‑kernel SVM trained on PCA features. |
| `classical_rf_model.pkl`   | `.pkl`  | `classical_svm.py`   | `compare_models.py`                  | Random Forest baseline classifier. |
| `classical_xgb_model.pkl`  | `.pkl`  | `classical_svm.py`   | `compare_models.py`                  | XGBoost baseline classifier. |
| `quantum_svm_model.pkl`    | `.pkl`  | `quantum_kernel.py`  | `app.py`, `compare_models.py`        | Trained Quantum SVM (precomputed kernel). |
| `kernel_matrix.npy`        | `.npy`  | `quantum_kernel.py`  | `quantum_explainability.py`, notebooks | Quantum training kernel matrix \(K_{\text{train}}\). |
| `X_train_qsvm.npy`         | `.npy`  | `quantum_kernel.py`  | `app.py`, `quantum_explainability.py` | PCA‑space training points used for QSVM. |
| `y_train_qsvm.npy`         | `.npy`  | `quantum_kernel.py`  | `app.py`, `quantum_explainability.py` | Integer labels for the QSVM training subset. |
| `y_test_classical.npy`     | `.npy`  | `classical_svm.py`   | `compare_models.py`                  | Ground‑truth labels for classical test split. |
| `y_pred_classical.npy`     | `.npy`  | `classical_svm.py`   | `compare_models.py`                  | SVM predictions on the classical test split. |
| `y_pred_rf.npy`            | `.npy`  | `classical_svm.py`   | `compare_models.py`                  | Random Forest predictions on the classical test split. |
| `y_pred_xgb.npy`           | `.npy`  | `classical_svm.py`   | `compare_models.py`                  | XGBoost predictions on the classical test split. |
| `y_test_quantum.npy`       | `.npy`  | `quantum_kernel.py`  | `compare_models.py`, `quantum_explainability.py` | Ground‑truth labels for the quantum test subset. |
| `y_pred_quantum.npy`       | `.npy`  | `quantum_kernel.py`  | `compare_models.py`, `quantum_explainability.py` | QSVM predictions on the quantum test subset. |
| `quantum_circuit.png`      | `.png`  | `quantum_kernel.py`  | Documentation, README, presentations | Rendered quantum kernel circuit diagram. |
| `shap_explanation.png`     | `.png`  | `explainability.py`  | Reports, dashboard, offline review   | SHAP waterfall diagram for a representative sample. |

---

## Known Limitations

- **Quadratic kernel cost:** Quantum kernel computation scales as \(O(n^2)\) in the number of samples, requiring subsampling to remain tractable.
- **Subsampled training set:** The QSVM is trained on **500** training samples and evaluated on **100** test samples, which may under‑represent rare diseases.
- **Simulator only:** The project currently uses PennyLane’s simulators (`lightning.qubit` and `default.qubit`); there is no direct hardware backend integration.
- **Backend availability:** `lightning.qubit` may not be installed or available on all systems; in that case only the dashboard (with `default.qubit`) will run.
- **Classical proxy for SHAP:** SHAP explainability operates on the classical SVM proxy in PCA space, not directly on the QSVM decision function.
- **IoT interface stub:** The “Auto‑Detect from Hardware” control is a UI stub; no real serial/Bluetooth breathalyzer integration is implemented yet.
- **No formal clinical validation:** QNose is a research prototype and has not been clinically validated; it must not be used for real patient diagnosis.

---

## Roadmap

| Now (done)                                              | Next (6 months)                                                   | Vision (2+ years)                                                |
|---------------------------------------------------------|-------------------------------------------------------------------|------------------------------------------------------------------|
| Multi‑disease VOC pipeline with 26 features             | Integrate real e‑nose / breathalyzer hardware via serial/BLE     | Deploy hybrid quantum–classical service in a hospital sandbox    |
| Classical SVM/RF/XGBoost baselines                      | Add automated hyperparameter search for classical + QSVM models  | Explore error‑mitigated and hardware‑native quantum kernels      |
| 5‑qubit QSVM on PCA‑compressed features                 | Extend explainability with per‑disease SHAP and counterfactuals  | Longitudinal VOC monitoring and trajectory‑based predictions     |
| Streamlit dashboard with 3D PCA hologram                | Add model versioning, A/B testing and drift monitoring           | Federated learning across multiple clinical sites                |
| SHAP waterfall and quantum circuit visualisations       | Implement richer PDF reporting and multi‑patient batch analysis  | Integration with EHR systems and clinical decision support tools |

---

## License

This project is licensed under the **MIT License**. You are free to use, modify, and distribute this software, subject to the terms and conditions of the MIT license.

See the `LICENSE` file for full details.
