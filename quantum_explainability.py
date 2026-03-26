import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.manifold import TSNE
import pennylane as qml
from pennylane import numpy as pnp
import joblib

def frobenius_inner_product(A, B):
    return np.sum(A * B)

def multiclass_ideal_kernel(Y):
    # For multiclass, the ideal kernel T has T_ij = 1 if y_i == y_j else 0
    # Then we optionally center it, but standard alignment often just uses the direct outer product of equality.
    n = len(Y)
    T = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if Y[i] == Y[j]:
                T[i, j] = 1.0
            else:
                T[i, j] = -1/(len(np.unique(Y))-1) # Penalize different classes slightly
    return T

def kernel_alignment(K, Y):
    T = multiclass_ideal_kernel(Y)

    # center the kernels for centered alignment (optional but often gives cleaner results)
    n = len(Y)
    H = np.eye(n) - np.ones((n, n)) / n
    Kc = H @ K @ H
    Tc = H @ T @ H

    inner_KT = frobenius_inner_product(Kc, Tc)
    norm_K = np.sqrt(frobenius_inner_product(Kc, Kc))
    norm_T = np.sqrt(frobenius_inner_product(Tc, Tc))
    
    if norm_K * norm_T == 0:
        return 0
    return inner_KT / (norm_K * norm_T)

def main():
    print("--- Running Quantum Explainability Analysis (27-Class) ---")

    # 1. Load Data
    X_train = np.load('X_train_qsvm.npy')
    y_train = np.load('y_train_qsvm.npy')
    K_quantum = np.load('kernel_matrix.npy')
    le = joblib.load('label_encoder.pkl')

    # 2. Kernel Alignment (Quantum vs Classical)
    print("Computing Multi-Class Kernel Alignments...")
    K_classical = rbf_kernel(X_train, gamma=0.1) # Standard RBF

    align_q = kernel_alignment(K_quantum, y_train)
    align_c = kernel_alignment(K_classical, y_train)

    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(['Classical RBF', 'Quantum Kernel'], [align_c, align_q], color=['#1f77b4', '#4B0082'])
    ax.set_ylabel('Target Alignment Score')
    ax.set_title('Multiclass Kernel Alignment (27 Diseases)')
    ax.set_ylim(0, max(align_c, align_q) * 1.3)
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 0.01, f'{yval:.3f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('kernel_alignment.png')
    plt.close()

    # 3. Hilbert Space Projection (t-SNE on Precomputed Distance)
    print("Computing Hilbert Space TSNE projection...")
    # Distance in Hilbert space: D(x, y) = sqrt(k(x,x) + k(y,y) - 2k(x,y))
    # Since AngleEmbedding gives k(x,x)=1, D = sqrt(2 - 2*k(x,y))
    D_quantum = np.sqrt(np.clip(2.0 - 2.0 * K_quantum, 0, None))

    tsne = TSNE(n_components=2, metric='precomputed', random_state=42, perplexity=15, init='random')
    X_tsne = tsne.fit_transform(D_quantum)

    plt.figure(figsize=(10, 8))
    
    # We have 27 classes, let's use a large colormap
    classes = np.unique(y_train)
    cmap = plt.get_cmap('tab20', len(classes))
    
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_train, cmap=cmap, edgecolor='k', s=60, alpha=0.8)
    plt.title('Quantum Multiclass Hilbert Space t-SNE Projection')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    
    # Create legend instead of colorbar for 27 distinct classes
    import matplotlib.patches as mpatches
    handles = []
    for cls in classes:
        color = cmap(cls / (len(classes) - 1))
        label_str = le.inverse_transform([cls])[0]
        handles.append(mpatches.Patch(color=color, label=label_str))
        
    plt.legend(handles=handles, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=2)
    plt.tight_layout()
    plt.savefig('quantum_tsne_projection.png')
    plt.close()

    # 4. Quantum Feature Importance via Gradients
    print("Computing Quantum Gradients for Feature Importance...")
    n_qubits = 5
    dev = qml.device("default.qubit", wires=n_qubits)

    def entangling_feature_map(x):
        for i in range(n_qubits):
            qml.RY(x[i], wires=i)
        
        qml.CNOT(wires=[0, 1])
        qml.CNOT(wires=[1, 2])
        qml.CNOT(wires=[2, 3])
        qml.CNOT(wires=[3, 4])
        qml.CNOT(wires=[4, 0])
        
        for i in range(n_qubits):
            qml.RY(x[i], wires=i)

    @qml.qnode(dev, interface="autograd")
    def kernel_circuit(x1, x2):
        entangling_feature_map(x1)
        qml.adjoint(entangling_feature_map)(x2)
        return qml.probs(wires=range(n_qubits))

    def kernel_val(x1, x2):
        return kernel_circuit(x1, x2)[0]

    grad_fn = qml.grad(kernel_val, argnums=0)

    num_pairs = 30
    np.random.seed(42)
    indices_a = np.random.choice(len(X_train), num_pairs)
    indices_b = np.random.choice(len(X_train), num_pairs)

    gradients = []
    for i in range(num_pairs):
        a = pnp.array(X_train[indices_a[i]], requires_grad=True)
        b = pnp.array(X_train[indices_b[i]], requires_grad=False)
        g = grad_fn(a, b)
        gradients.append(np.abs(g))

    avg_gradient = np.mean(gradients, axis=0)

    # Map back to original features using PCA components
    pca = joblib.load('pca.pkl')
    # pca.components_ shape is (n_components, n_features) (5, 21)
    # The gradient is wrt the 5 PCA features.
    # By chain rule: df/dx_orig = (df/dx_pca) * (dx_pca/dx_orig) = avg_gradient @ pca.components_
    orig_gradient = np.abs(np.dot(avg_gradient, pca.components_))
    
    feature_cols = joblib.load('feature_cols.pkl')
    
    # Sort and plot top 10 original features
    sorted_idx = np.argsort(orig_gradient)[::-1][:10]
    top_features = [feature_cols[i] for i in sorted_idx]
    top_grads = orig_gradient[sorted_idx]

    plt.figure(figsize=(10, 6))
    plt.barh(top_features[::-1], top_grads[::-1], color='#4B0082', alpha=0.8)
    plt.title('Top 10 Biomarkers (Mapped from Quantum Gradients)')
    plt.xlabel('Mean Absolute Gradient Impact')
    plt.tight_layout()
    plt.savefig('quantum_feature_importance.png')
    plt.close()

    print("Quantum multiclass explainability metrics generated successfully.")

if __name__ == "__main__":
    main()