import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import os

def run_eda():
    excel_path = r"C:\Users\hp\OneDrive\Desktop\VOC_MultiDisease_Dataset.xlsx"
    csv_out_path = "data/VOC_MultiDisease_Dataset.csv"
    
    # 1. Load Data
    if os.path.exists(excel_path):
        print(f"Reading new Excel dataset from {excel_path}...")
        try:
            df = pd.read_excel(excel_path, header=1)
            # Save it as CSV
            df.to_csv(csv_out_path, index=False)
            print(f"✅ Successfully converted Excel to CSV at: {csv_out_path}")
        except Exception as e:
            print(f"❌ Error reading Excel. Make sure openpyxl is installed. ({e})")
            return
    else:
        print(f"❌ Excel file not found at {excel_path}. Falling back to qnose_synthetic_dataset.csv")
        df = pd.read_csv("data/qnose_synthetic_dataset.csv")
    
    print("\n--- Dataset Overview ---")
    print(f"Shape: {df.shape}")
    
    # Identify target column
    target_col = None
    if 'Disease Label' in df.columns:
        target_col = 'Disease Label'
    elif 'disease_label' in df.columns:
        target_col = 'disease_label'
    elif 'Disease' in df.columns:
        target_col = 'Disease'
    elif 'Class' in df.columns:
        target_col = 'Class'
    else:
        print("Could not find a disease label column. Available columns:", df.columns.tolist())
        return

    print(f"\n--- Disease Distribution ({target_col}) ---")
    print(df[target_col].value_counts())
    
    # 2. Extract Molecule / VOC features
    # Typical naming: '_ppb', '_ppm', or just chemical names. Let's grab all numeric features.
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Filter out obvious ID or metadata columns
    exclude_keywords = ['id', 'age', 'bmi', 'score', 'time', 'date', 'stage', 'is_diseased']
    feature_cols = [c for c in numeric_cols if not any(x in c.lower() for x in exclude_keywords)]
    
    if len(feature_cols) == 0:
        print("No numerical molecule features found.")
        return
        
    print(f"\nAnalyzed {len(feature_cols)} numeric/molecule features.")
    
    # 3. Model Feature Importance (Multi-Class)
    print("\nTraining classical Random Forest to calculate global Feature Importance for specific diseases...")
    
    # Drop rows with NaN in target or features
    df_clean = df.dropna(subset=[target_col] + feature_cols)
    X = df_clean[feature_cols]
    y = df_clean[target_col].astype(str) # ensure categorical string
    
    # Exclude "Healthy" if we want to see what distinguishes diseases from each other (optional)
    # But leaving it in is good to find disease vs healthy markers too.
    
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    
    importances = rf.feature_importances_
    imp_df = pd.DataFrame({'Molecule/Feature': feature_cols, 'Importance Rate': importances})
    imp_df = imp_df.sort_values(by='Importance Rate', ascending=False).reset_index(drop=True)
    
    print("\n=== TOP 15 MOST CRITICAL MOLECULES ===")
    print(imp_df.head(15))
    
    # Save the feature importances to a CSV for later UI reading
    imp_df.to_csv("data/feature_importances.csv", index=False)
    print("\n✅ Saved feature importances to data/feature_importances.csv")

if __name__ == "__main__":
    run_eda()