import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
import pennylane as qml
import time

def main():
    print("--- Comparing Classical vs Quantum OVR Multi-Disease Models ---")
    
    # Load standardized training set and labels
    X_train_pca = np.load('X_train_qsvm.npy')
    y_train = np.load('y_train_qsvm.npy')
    le = joblib.load('label_encoder.pkl')
    
    # 1. Classical SVM (RBF) Multi-Class Evaluation
    print("Evaluating Classical RBF SVM...")
    clf_classical = SVC(kernel='rbf', probability=True, decision_function_shape='ovr')
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    start_time = time.time()
    scores_cl = cross_val_score(clf_classical, X_train_pca, y_train, cv=cv, scoring='accuracy')
    time_cl = time.time() - start_time
    
    mean_cl_acc = np.mean(scores_cl)
    print(f"Classical RBF SVM Accuracy: {mean_cl_acc:.4f} (Avg Fold Time: {time_cl/5:.2f}s)")

    # 2. Quantum Kernel Evaluation
    print("Evaluating Precomputed Quantum Kernel SVM...")
    K_matrix = np.load('kernel_matrix.npy')
    clf_quantum = SVC(kernel='precomputed', probability=True, decision_function_shape='ovr')
    
    # Note: K_matrix is precomputed for X_train_pca. For CV, we must slice the precomputed matrix:
    scores_qu = []
    
    # We will simulate time using the kernel prediction overhead instead of cross validation
    # because slicing precomputed kernels in Sklearn CV requires passing an index array.
    start_time = time.time()
    for train_idx, test_idx in cv.split(X_train_pca, y_train):
        K_train = K_matrix[train_idx][:, train_idx]
        K_test = K_matrix[test_idx][:, train_idx]
        
        clf_quantum.fit(K_train, y_train[train_idx])
        scores_qu.append(clf_quantum.score(K_test, y_train[test_idx]))
    time_qu = time.time() - start_time
    
    mean_qu_acc = np.mean(scores_qu)
    print(f"Quantum Kernel Accuracy: {mean_qu_acc:.4f} (Avg Fold Time: {time_qu/5:.2f}s [excludes kernel gen])")

    # 3. Create Comparison Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    metrics = ['Classical RBF (OVR)', 'Quantum Kernel (OVR)']
    accs = [mean_cl_acc, mean_qu_acc]
    
    bars = ax.bar(metrics, accs, color=['#3498db', '#9b59b6'])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('5-Fold Cross Validation Accuracy')
    ax.set_title('Multiplex Disease Classification Resilience (27 Classes)')
    
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.02,
                f'{h*100:.1f}%',
                ha='center', va='bottom', fontweight='bold')
                
    plt.tight_layout()
    plt.savefig('model_comparison.png')
    plt.close()
    
    print("✅ Model comparison logic complete and saved to model_comparison.png")

if __name__ == "__main__":
    main()