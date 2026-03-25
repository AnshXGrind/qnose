# -*- coding: utf-8 -*-

"""Train the Quantum SVM kernel model on the multi-disease VOC dataset.

This module constructs a 5-qubit AngleEmbedding-based quantum kernel using
PennyLane's ``lightning.qubit`` device, trains an SVM with a precomputed
kernel over 27 disease classes, and saves all quantum-specific artifacts for
use in the Streamlit dashboard and analysis scripts.
"""

import logging
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pennylane as qml
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from tqdm import tqdm

# Number of qubits used in the quantum kernel circuit
N_QUBITS = 5

# Subsample sizes chosen to keep O(n^2) kernel computation tractable on CPU
TRAIN_SAMPLE_SIZE = 500  # training subset size for kernel matrix
TEST_SAMPLE_SIZE = 100  # test subset size for evaluation


def get_lightning_device():
    """Return a PennyLane lightning.qubit device with N_QUBITS wires."""

    return qml.device("lightning.qubit", wires=N_QUBITS)


DEV = get_lightning_device()


@qml.qnode(DEV)
def kernel_circuit(x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Quantum kernel circuit returning the full probability distribution.

    The kernel value k(x1, x2) is taken as the probability of the |0...0>
    computational basis state after preparing |ψ(x1)> and applying the adjoint
    of the embedding for x2.
    """

    qml.AngleEmbedding(x1, wires=range(N_QUBITS))
    qml.adjoint(qml.AngleEmbedding)(x2, wires=range(N_QUBITS))
    return qml.probs(wires=range(N_QUBITS))


def kernel_function(x1: np.ndarray, x2: np.ndarray) -> float:
    """Compute the quantum kernel value ⟨ψ(x1)|ψ(x2)⟩.

    Parameters
    ----------
    x1, x2:
        PCA-transformed feature vectors of length N_QUBITS.
    """

    return float(kernel_circuit(x1, x2)[0])


def main() -> None:
    """Entry point for training the Quantum SVM kernel model."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s — %(levelname)s — %(message)s",
    )

    logging.info("--- Running Quantum Kernel on Multi-Disease VOC dataset ---")

    # Load raw dataset
    try:
        df = pd.read_csv("data/VOC_MultiDisease_Dataset.csv")
    except FileNotFoundError as exc:
        logging.error(
            "Could not find data/VOC_MultiDisease_Dataset.csv. "
            "Run classical_svm.py first and ensure the dataset is available.",
        )
        raise SystemExit(1) from exc
    except Exception as exc:  # pragma: no cover - defensive logging
        logging.error("Failed to load dataset: %s", exc)
        raise SystemExit(1) from exc

    target_col = "Disease Label"

    # Load preprocessing artifacts produced by classical_svm.py
    try:
        le = joblib.load("label_encoder.pkl")
        feature_cols = joblib.load("feature_cols.pkl")
        scaler = joblib.load("scaler.pkl")
        pca = joblib.load("pca.pkl")
    except FileNotFoundError as exc:
        logging.error(
            "Required artifacts not found (label_encoder.pkl, feature_cols.pkl, "
            "scaler.pkl, pca.pkl). Please run classical_svm.py before "
            "quantum_kernel.py.",
        )
        raise SystemExit(1) from exc
    except Exception as exc:  # pragma: no cover - defensive logging
        logging.error("Failed to load preprocessing artifacts: %s", exc)
        raise SystemExit(1) from exc

    y = le.transform(df[target_col])
    X = df[feature_cols]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_scaled = scaler.transform(X)

    X_pca = pca.transform(X_scaled)

    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
        X_pca,
        y,
        test_size=0.3,  # 30% held out as a classical test set
        random_state=42,
        stratify=y,
    )

    # Render and save a reference quantum circuit diagram
    fig, _ = qml.draw_mpl(kernel_circuit)(X_train_full[0], X_train_full[0])
    fig.savefig("quantum_circuit.png")
    plt.close(fig)
    logging.info("Saved quantum_circuit.png")

    # Subsample for quantum kernel training to keep O(n^2) costs manageable
    train_size = min(TRAIN_SAMPLE_SIZE, len(X_train_full))
    test_size = min(TEST_SAMPLE_SIZE, len(X_test_full))

    X_train, _, y_train, _ = train_test_split(
        X_train_full,
        y_train_full,
        train_size=train_size,
        random_state=42,
        stratify=y_train_full,
    )

    X_test, _, y_test, _ = train_test_split(
        X_test_full,
        y_test_full,
        train_size=test_size,
        random_state=42,
        stratify=y_test_full,
    )

    logging.info(
        "Subsampled training set to %d and test set to %d (from %d / %d)",
        len(X_train),
        len(X_test),
        len(X_train_full),
        len(X_test_full),
    )

    # Compute quantum kernel matrices with progress indication
    logging.info(
        "Computing quantum kernel matrix for %d samples across %d diseases...",
        len(X_train),
        len(np.unique(y_train)),
    )

    K_train = np.zeros((len(X_train), len(X_train)), dtype=float)
    for i, a in enumerate(tqdm(X_train, desc="Computing K_train")):
        for j, b in enumerate(X_train):
            K_train[i, j] = kernel_function(a, b)

    K_test = np.zeros((len(X_test), len(X_train)), dtype=float)
    for i, a in enumerate(tqdm(X_test, desc="Computing K_test")):
        for j, b in enumerate(X_train):
            K_test[i, j] = kernel_function(a, b)

    np.save("kernel_matrix.npy", K_train)

    # Multi-class uses OVR automatically; break_ties improves close decisions
    qsvm = SVC(kernel="precomputed", probability=True, break_ties=True)
    qsvm.fit(K_train, y_train)

    y_pred = qsvm.predict(K_test)

    logging.info(
        "Quantum subsampled accuracy: %.4f", accuracy_score(y_test, y_pred)
    )

    joblib.dump(qsvm, "quantum_svm_model.pkl")

    # Save the downsampled data arrays needed for the dashboard and analysis
    np.save("X_train_qsvm.npy", X_train)
    np.save("y_train_qsvm.npy", y_train)
    np.save("y_pred_quantum.npy", y_pred)
    np.save("y_test_quantum.npy", y_test)


if __name__ == "__main__":
    main()