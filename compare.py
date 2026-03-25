import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
import joblib

def main():
    print("--- Running Compare ---")
    y_test_cls = np.load('y_test_classical.npy')
    y_pred_cls = np.load('y_pred_classical.npy')
    y_pred_rf = np.load('y_pred_rf.npy')
    y_pred_xgb = np.load('y_pred_xgb.npy')
    y_test_q = np.load('y_test_quantum.npy')
    y_pred_q = np.load('y_pred_quantum.npy')
    kernel_matrix = np.load('kernel_matrix.npy')

    acc_cls = accuracy_score(y_test_cls, y_pred_cls)
    acc_rf = accuracy_score(y_test_cls, y_pred_rf)
    acc_xgb = accuracy_score(y_test_cls, y_pred_xgb)
    acc_q = accuracy_score(y_test_q, y_pred_q)

    print("--- Comparison Summary ---")
    print(f"Classical SVM Accuracy: {acc_cls:.4f}")
    print(f"Random Forest Accuracy: {acc_rf:.4f}")
    print(f"XGBoost Accuracy:       {acc_xgb:.4f}")
    print(f"Quantum SVM Accuracy:   {acc_q:.4f}")

    # 1. Confusion Matrices Plot
    cm_cls = confusion_matrix(y_test_cls, y_pred_cls)
    cm_rf = confusion_matrix(y_test_cls, y_pred_rf)
    cm_xgb = confusion_matrix(y_test_cls, y_pred_xgb)
    cm_q = confusion_matrix(y_test_q, y_pred_q)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    sns.heatmap(cm_cls, annot=True, ax=axes[0, 0], cmap='Blues', fmt='g')
    axes[0, 0].set_title('Classical SVM')
    
    sns.heatmap(cm_rf, annot=True, ax=axes[0, 1], cmap='Greens', fmt='g')
    axes[0, 1].set_title('Random Forest')
    
    sns.heatmap(cm_xgb, annot=True, ax=axes[1, 0], cmap='Oranges', fmt='g')
    axes[1, 0].set_title('XGBoost')
    
    sns.heatmap(cm_q, annot=True, ax=axes[1, 1], cmap='Purples', fmt='g')
    axes[1, 1].set_title('Quantum SVM')
    
    for ax in axes.flat:
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
    
    plt.tight_layout()
    plt.savefig('confusion_matrices.png')
    print("Saved confusion_matrices.png")

    # 2. Add precision, recall, f1
    print("\n--- Detailed Metrics ---")
    models = ['Classical SVM', 'Random Forest', 'XGBoost', 'Quantum SVM']
    y_tests = [y_test_cls, y_test_cls, y_test_cls, y_test_q]
    y_preds = [y_pred_cls, y_pred_rf, y_pred_xgb, y_pred_q]
    
    for name, y_t, y_p in zip(models, y_tests, y_preds):
        p = precision_score(y_t, y_p, average='weighted', zero_division=0)
        r = recall_score(y_t, y_p, average='weighted', zero_division=0)
        f1 = f1_score(y_t, y_p, average='weighted', zero_division=0)
        print(f"{name} -> Precision: {p:.4f} | Recall: {r:.4f} | F1: {f1:.4f}")

    # 3. Plot kernel heatmap (Plotly)
    fig_heat = px.imshow(kernel_matrix, title="Quantum Kernel Matrix Heatmap",
                    labels=dict(x="Training Samples", y="Training Samples", color="Similarity"))
    fig_heat.write_html("kernel_heatmap.html")
    print("Saved kernel_heatmap.html")

if __name__ == '__main__':
    main()