import pandas as pd
import joblib
import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    print("--- Running Quantum Kernel ---")
    df = pd.read_csv('data/parkinsons.csv')
    df = df.drop('name', axis=1)
    
    X = df.drop('status', axis=1).values
    y = df['status'].values

    scaler = joblib.load('scaler.pkl')
    pca = joblib.load('pca.pkl')
    
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)

    n_qubits = 5
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        qml.AngleEmbedding(x1, wires=range(n_qubits))
        qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits))
        return qml.probs(wires=range(n_qubits))

    # ---- 1. Create Quantum Circuit Visualization ----
    fig, ax = qml.draw_mpl(kernel_circuit)(X_train[0], X_train[0])
    fig.savefig('quantum_circuit.png')
    plt.close(fig)
    print("Saved quantum_circuit.png")

    # For speed, using subset (16 samples per prompt) if the dataset was too large 
    # but the whole dataset is small enough, the kernel computation should be okay. Use qml.kernels
    # We will use the built-in qml kernel mapping to speed it up.
    
    def kernel_function(x1, x2):
        return kernel_circuit(x1, x2)[0]
    
    # We can use qml.kernels.kernel_matrix function if needed, but manual array mapping is safe.
    print("Computing quantum kernel matrix... (may take a moment)")
    K_train = np.array([[kernel_function(a, b) for b in X_train] for a in X_train])
    K_test = np.array([[kernel_function(a, b) for b in X_train] for a in X_test])
    
    np.save('kernel_matrix.npy', K_train)

    qsvm = SVC(kernel='precomputed')
    qsvm.fit(K_train, y_train)

    y_pred = qsvm.predict(K_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    joblib.dump(qsvm, 'quantum_svm_model.pkl')
    np.save('X_train_qsvm.npy', X_train)
    np.save('y_pred_quantum.npy', y_pred)
    np.save('y_test_quantum.npy', y_test)

if __name__ == '__main__':
    main()