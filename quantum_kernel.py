import pandas as pd
import joblib
import numpy as np
import pennylane as qml
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import warnings

def main():
    print("--- Running Quantum Kernel on Multi-Disease Dataset ---")
    
    df = pd.read_csv('data/VOC_MultiDisease_Dataset.csv')
    target_col = 'Disease Label'
    
    # Needs to match classical setup
    le = joblib.load('label_encoder.pkl')
    y = le.transform(df[target_col])
    
    feature_cols = joblib.load('feature_cols.pkl')
    X = df[feature_cols]

    scaler = joblib.load('scaler.pkl')
    pca = joblib.load('pca.pkl')
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        X_scaled = scaler.transform(X)
        
    X_pca = pca.transform(X_scaled)
    
    X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X_pca, y, test_size=0.3, random_state=42, stratify=y)

    n_qubits = 5
    dev = qml.device("lightning.qubit", wires=n_qubits)

    @qml.qnode(dev)
    def kernel_circuit(x1, x2):
        qml.AngleEmbedding(x1, wires=range(n_qubits))
        qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits))
        return qml.probs(wires=range(n_qubits))

    if not np.all(X_train_full[0] == X_train_full[0]): 
        pass
    
    fig, ax = qml.draw_mpl(kernel_circuit)(X_train_full[0], X_train_full[0])
    fig.savefig('quantum_circuit.png')
    plt.close(fig)
    print("Saved quantum_circuit.png")

    def kernel_function(x1, x2):
        return kernel_circuit(x1, x2)[0]
    
    # Use larger sample size with Lightning
    sample_size = 500
    test_sample_size = 100
    print(f"Subsampling training from {len(X_train_full)} to {sample_size} with lightning.qubit...")
    
    # Using a stratified split again to ensure all 27 diseases have some examples 
    _, X_train, _, y_train = train_test_split(
        X_train_full, y_train_full, 
        test_size=sample_size/len(X_train_full), 
        random_state=42, stratify=y_train_full
    )
    
    _, X_test, _, y_test = train_test_split(
        X_test_full, y_test_full, 
        test_size=test_sample_size/len(X_test_full), 
        random_state=42, stratify=y_test_full
    )

    print(f"Computing quantum kernel matrix for {len(X_train)} samples across {len(np.unique(y_train))} unique diseases...")
    K_train = np.array([[kernel_function(a, b) for b in X_train] for a in X_train])
    K_test = np.array([[kernel_function(a, b) for b in X_train] for a in X_test])
    
    np.save('kernel_matrix.npy', K_train)

    # Multi-class uses OVR automatically, break_ties is useful here
    qsvm = SVC(kernel='precomputed', probability=True, break_ties=True)
    qsvm.fit(K_train, y_train)

    y_pred = qsvm.predict(K_test)
    
    print(f"Quantum Subsampled Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    
    joblib.dump(qsvm, 'quantum_svm_model.pkl')
    # Save the downsampled data arrays needed for explanation
    np.save('X_train_qsvm.npy', X_train)
    np.save('y_train_qsvm.npy', y_train)
    np.save('y_pred_quantum.npy', y_pred)
    np.save('y_test_quantum.npy', y_test)

if __name__ == '__main__':
    main()