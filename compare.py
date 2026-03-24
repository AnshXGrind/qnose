import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

def main():
    print("--- Running Compare ---")
    y_test_cls = np.load('y_test_classical.npy')
    y_pred_cls = np.load('y_pred_classical.npy')
    y_test_q = np.load('y_test_quantum.npy')
    y_pred_q = np.load('y_pred_quantum.npy')
    kernel_matrix = np.load('kernel_matrix.npy')

    acc_cls = accuracy_score(y_test_cls, y_pred_cls)
    acc_q = accuracy_score(y_test_q, y_pred_q)

    print("--- Comparison Summary ---")
    print(f"Classical SVM Accuracy: {acc_cls:.4f}")
    print(f"Quantum SVM Accuracy:   {acc_q:.4f}")

    # 1. Confusion Matrices Plot
    cm_cls = confusion_matrix(y_test_cls, y_pred_cls)
    cm_q = confusion_matrix(y_test_q, y_pred_q)
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.heatmap(cm_cls, annot=True, ax=axes[0], cmap='Blues', fmt='g')
    axes[0].set_title('Classical SVM')
    axes[0].set_xlabel('Predicted Label')
    axes[0].set_ylabel('True Label')
    
    sns.heatmap(cm_q, annot=True, ax=axes[1], cmap='Purples', fmt='g')
    axes[1].set_title('Quantum SVM')
    axes[1].set_xlabel('Predicted Label')
    axes[1].set_ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    print("Saved confusion_matrices.png")

    # 2. Plot kernel heatmap (Plotly)
    fig_heat = px.imshow(kernel_matrix, title="Quantum Kernel Matrix Heatmap",
                    labels=dict(x="Training Samples", y="Training Samples", color="Similarity"))
    fig_heat.write_html("kernel_heatmap.html")
    print("Saved kernel_heatmap.html")

if __name__ == '__main__':
    main()