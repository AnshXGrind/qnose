import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib.pyplot as plt

def main():
    print("--- Running SHAP Explainability ---")
    df = pd.read_csv('data/parkinsons.csv')
    df = df.drop('name', axis=1)
    X = df.drop('status', axis=1)
    
    scaler = joblib.load('scaler.pkl')
    pca = joblib.load('pca.pkl')
    svm = joblib.load('classical_svm_model.pkl')
    
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    
    # 1. Take a sample Parkinson's patient for explainability
    parkinsons_idx = np.where(df['status'] == 1)[0][0]
    sample = X_pca[parkinsons_idx:parkinsons_idx+1]
    
    # 2. Setup SHAP KernelExplainer
    background = shap.kmeans(X_pca, 10)
    explainer = shap.KernelExplainer(svm.predict, background)
    
    # 3. Compute SHAP values
    shap_values = explainer.shap_values(sample)
    
    base_val = explainer.expected_value
    if isinstance(base_val, np.ndarray):
        base_val = base_val[0]
        
    exp = shap.Explanation(values=shap_values[0], 
                           base_values=base_val, 
                           data=sample[0], 
                           feature_names=["PCA-VOC 1", "PCA-VOC 2", "PCA-VOC 3", "PCA-VOC 4", "PCA-VOC 5"])
    
    # 4. Plot Waterfall Chart
    plt.figure(figsize=(8, 5))
    shap.waterfall_plot(exp, show=False)
    plt.title("Why did QNose flag this breath sample?")
    plt.tight_layout()
    plt.savefig('shap_explanation.png')
    print("Saved shap_explanation.png")

if __name__ == "__main__":
    main()