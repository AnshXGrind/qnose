import pandas as pd
import joblib
import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    print("--- Running Quantum Kernel on General Disease Dataset ---")
    df = pd.read_csv('data/qnose_synthetic_dataset.csv')
    
    y = df['is_diseased'].values
    feature_cols = [c for c in df.columns if c.endswith('_ppb') or c.endswith('_ppm')]
    X = df[feature_cols].values

    scaler = joblib.load('scaler.pkl')
    pca = joblib.load('pca.pkl')
    
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42, stratify=y)

    n_qubits = 5
    dev = qml.device("default.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        qml.AngleEmbedding(x1, wires=range(n_qubits))
        qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits))
        return qml.probs(wires=range(n_qubits))

    if not np.all(X_train[0] == X_train[0]): # just to ensure variables are valid
        pass
    
    fig, ax = qml.draw_mpl(kernel_circuit)(X_train[0], X_train[0])
    fig.savefig('quantum_circuit.png')
    plt.close(fig)
    print("Saved quantum_circuit.png")

    def kernel_function(x1, x2):
        return kernel_circuit(x1, x2)[0]
    
    # Subsample for faster execution
    if len(X_train) > 150:
        print(f"Subsampling training from {len(X_train)} to 150 to keep demo fast...")
        idx = np.random.choice(len(X_train), 150, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]
        
    if len(X_test) > 50:
        idx_test = np.random.choice(len(X_test), 50, replace=False)
        X_test = X_test[idx_test]
        y_test = y_test[idx_test]

    print(f"Computing quantum kernel matrix for {len(X_train)} samples...")
    K_train = np.array([[kernel_function(a, b) for b in X_train] for a in X_train])
    K_test = np.array([[kernel_function(a, b) for b in X_train] for a in X_test])
    
    np.save('kernel_matrix.npy', K_train)

    qsvm = SVC(kernel='precomputed', probability=True)
    qsvm.fit(K_train, y_train)

    y_pred = qsvm.predict(K_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))
    
    joblib.dump(qsvm, 'quantum_svm_model.pkl')
    np.save('X_train_qsvm.npy', X_train)
    np.save('y_train_qsvm.npy', y_train)
    np.save('y_pred_quantum.npy', y_pred)
    np.save('y_test_quantum.npy', y_test)

if __name__ == '__main__':
    main()