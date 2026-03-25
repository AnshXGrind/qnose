import pandas as pd
import joblib
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("--- Running Classical Multi-Disease SVM ---")
    
    # Use the new multi-disease local dataset
    df = pd.read_csv('data/VOC_MultiDisease_Dataset.csv')
    
    # Ensure there are no unname columns causing issues, just use the exact VOCs
    target_col = 'Disease Label'
    
    # Filter out text columns and IDs
    exclude_keywords = ['id', 'label']
    feature_cols = [c for c in df.columns if not any(x in c.lower() for x in exclude_keywords) and df[c].dtype in [np.float64, np.int64]]
    X = df[feature_cols]
    
    # Map the disease strings to integers
    le = LabelEncoder()
    y = le.fit_transform(df[target_col])
    joblib.dump(le, 'label_encoder.pkl')
    
    # Calculate healthy mean for the dashboard
    if 'Healthy' in le.classes_:
        healthy_val = le.transform(['Healthy'])[0]
        healthy_mean = df[df[target_col] == 'Healthy'][feature_cols].mean().values
    else:
        # fallback
        healthy_mean = X.mean().values
    
    joblib.dump(healthy_mean, 'healthy_mean.pkl')
    joblib.dump(feature_cols, 'feature_cols.pkl')
    joblib.dump(X.mean().values, 'x_mean.pkl')
    
    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'scaler.pkl')
    
    # PCA to 5 dimensions for the 5-qubit architecture
    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X_scaled)
    joblib.dump(pca, 'pca.pkl')
    
    # Stratified Split
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42, stratify=y)
    
    svm = SVC(kernel='rbf', probability=True, break_ties=True)
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)

    xgb = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)

    print(f"Classical SVM Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Random Forest Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
    print(f"XGBoost Accuracy:       {accuracy_score(y_test, y_pred_xgb):.4f}")
    print(f"Total evaluated classes: {len(le.classes_)}")

    joblib.dump(svm, 'classical_svm_model.pkl')
    joblib.dump(rf, 'classical_rf_model.pkl')
    joblib.dump(xgb, 'classical_xgb_model.pkl')
    
    np.save('y_test_classical.npy', y_test)
    np.save('y_pred_classical.npy', y_pred)
    np.save('y_pred_rf.npy', y_pred_rf)
    np.save('y_pred_xgb.npy', y_pred_xgb)

if __name__ == '__main__':
    main()