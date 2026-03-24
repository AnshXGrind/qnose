import pandas as pd
import joblib
import numpy as np
import urllib.request
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def main():
    print("--- Running Classical SVM ---")
    
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/parkinsons.data"
    os.makedirs("data", exist_ok=True)
    if not os.path.exists("data/parkinsons.csv"):
        print("Downloading UCI Parkinson's Dataset...")
        urllib.request.urlretrieve(url, "data/parkinsons.csv")

    df = pd.read_csv('data/parkinsons.csv')
    df = df.drop('name', axis=1)
    
    X = df.drop('status', axis=1)
    y = df['status']
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, 'scaler.pkl')
    
    # Save the mean values to pad Streamlit UI inputs properly later
    joblib.dump(X.mean().values, 'x_mean.pkl')
    
    # PCA down to 5 components for quantum embedding suitability
    pca = PCA(n_components=5)
    X_pca = pca.fit_transform(X_scaled)
    joblib.dump(pca, 'pca.pkl')
    
    X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.3, random_state=42)
    
    svm = SVC(kernel='rbf')
    svm.fit(X_train, y_train)
    y_pred = svm.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    joblib.dump(svm, 'classical_svm_model.pkl')
    np.save('y_test_classical.npy', y_test)
    np.save('y_pred_classical.npy', y_pred)

if __name__ == '__main__':
    main()