import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.manifold import TSNE
import pennylane as qml
from pennylane import numpy as pnp
import joblib

def frobenius_inner_product(A, B):
    return np.sum(A * B)

def kernel_alignment(K, Y):
    # Convert labels from {0, 1} to {-1, 1}
    y_polar = np.where(Y == 0, -1, 1)
    T = np.outer(y_polar, y_polar)
    
    # alignment
    inner_KT = frobenius_inner_product(K, T)
    norm_K = np.sqrt(frobenius_inner_product(K, K))
    norm_T = np.sqrt(frobenius_inner_product(T, T))
    return inner_KT / (norm_K * norm_T)

def main():
    print("--- Running Quantum Explainability Analysis ---")
    
    # 1. Load Data
    X_train = np.load('X_train_qsvm.npy')
    y_train = np.load('y_train_qsvm.npy')
    K_quantum = np.load('kernel_matrix.npy')
    
    # 2. Kernel Alignment (Quantum vs Classical)
    print("Computing Kernel Alignments...")
    K_classical = rbf_kernel(X_train, gamma=0.1) # Standard RBF
    
    align_q = kernel_alignment(K_quantum, y_train)
    align_c = kernel_alignment(K_classical, y_train)
    
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(['Classical RBF', 'Quantum Kernel'], [align_c, align_q], color=['#1f77b4', '#4B0082'])
    ax.set_ylabel('Target Alignment Score')
    ax.set_title('Kernel Alignment to True Labels')
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
    # We clip to 0 to prevent issues with floating point inaccuracies
    D_quantum = np.sqrt(np.clip(2.0 - 2.0 * K_quantum, 0, None))
    
    tsne = TSNE(n_components=2, metric='precomputed', random_state=42, perplexity=15, init='random')
    X_tsne = tsne.fit_transform(D_quantum)
    
    plt.figure(figsize=(7, 5))
    scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_train, cmap='coolwarm', edgecolor='k', s=60, alpha=0.8)
    plt.title('Quantum Hilbert Space t-SNE Projection')
    plt.xlabel('t-SNE 1')
    plt.ylabel('t-SNE 2')
    cbar = plt.colorbar(scatter, ticks=[0, 1])
    cbar.set_ticklabels(['Healthy', 'Diseased'])
    plt.tight_layout()
    plt.savefig('quantum_tsne_projection.png')
    plt.close()
    
    # 4. Quantum Feature Importance via Gradients
    print("Computing Quantum Gradients for Feature Importance...")
    n_qubits = 5
    dev = qml.device("default.qubit", wires=n_qubits)
    
    @qml.qnode(dev, interface="autograd")
    def kernel_circuit(x1, x2):
        qml.AngleEmbedding(x1, wires=range(n_qubits))
        qml.adjoint(qml.AngleEmbedding)(x2, wires=range(n_qubits))
        return qml.probs(wires=range(n_qubits))

    def kernel_val(x1, x2):
        return kernel_circuit(x1, x2)[0]
    
    grad_fn = qml.grad(kernel_val, argnums=0)
    
    # Compute gradients for a small random subset of pairs to estimate overall sensitivity
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
    
    plt.figure(figsize=(7, 4))
    pca_labels = [f"PCA {i+1}" for i in range(5)]
    plt.bar(pca_labels, avg_gradient, color='#4B0082', alpha=0.8)
    plt.title('Quantum Circuit Parameter Sensitivity (Feature Importance)')
    plt.ylabel('Mean Absolute Gradient')
    plt.tight_layout()
    plt.savefig('quantum_feature_importance.png')
    plt.close()
    
    print("Quantum explainability metrics generated successfully.")

if __name__ == "__main__":
    main()