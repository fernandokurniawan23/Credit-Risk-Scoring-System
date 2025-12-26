"""
Training Module
---------------
Trains the XGBoost Credit Scoring Model.
Features:
1. Stratified Splitting (maintains imbalance ratio).
2. Imbalance handling via scale_pos_weight.
3. Model serialization (saving model + feature names).
4. Performance evaluation (ROC-AUC).
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix

# --- CONFIGURATION ---
DATA_PATH = "data/processed/train_final.parquet"
MODEL_DIR = "data/models"
MODEL_PATH = os.path.join(MODEL_DIR, "xgb_credit_risk_v1.joblib")
METADATA_PATH = os.path.join(MODEL_DIR, "model_metadata.joblib")

def load_data(path: str):
    """Loads processed data."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data not found at {path}. Run feature_engineering.py first.")
    return pd.read_parquet(path)

def train_model():
    print("--- 1. LOADING DATA ---")
    df = load_data(DATA_PATH)
    
    # Separate Features (X) and Target (y)
    X = df.drop(columns=['TARGET', 'SK_ID_CURR']) # Drop ID and Target
    y = df['TARGET']
    
    print(f"X Shape: {X.shape}, y Shape: {y.shape}")
    
    # Split Data (80% Train, 20% Test)
    # STRATIFY is crucial for imbalanced data to keep 8% default rate in both sets
    print("\n--- 2. SPLITTING DATA ---")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Calculate scale_pos_weight for Imbalance
    # Formula: sum(negative) / sum(positive)
    ratio = float(np.sum(y == 0)) / np.sum(y == 1)
    print(f"Imbalance Ratio (calculated): {ratio:.2f}")
    
    # Initialize XGBoost
    # Using specific params for stability
    print(f"\n--- 3. TRAINING XGBOOST (Ratio={ratio:.2f}) ---")
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        n_estimators=100,      # MVP: Keep it fast
        learning_rate=0.1,
        max_depth=4,           # Avoid overfitting
        scale_pos_weight=ratio, # HANDLE IMBALANCE
        n_jobs=-1,             # Use all CPU cores
        random_state=42,
        eval_metric='auc',
        early_stopping_rounds=10 # Stop if no improvement
    )
    
    # Fit Model
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=True
    )
    
    # Evaluate
    print("\n--- 4. EVALUATION ---")
    # Predict Probability (for AUC)
    y_prob = model.predict_proba(X_test)[:, 1]
    # Predict Class (for Confusion Matrix)
    y_pred = model.predict(X_test)
    
    auc_score = roc_auc_score(y_test, y_prob)
    print(f"ROC-AUC Score: {auc_score:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Save Artifacts
    print("\n--- 5. SAVING ARTIFACTS ---")
    os.makedirs(MODEL_DIR, exist_ok=True)
    
    # Save Model
    joblib.dump(model, MODEL_PATH)
    
    # Save Metadata (Feature Names) - Critical for API
    metadata = {
        "features": X.columns.tolist(),
        "imbalance_ratio": ratio,
        "model_version": "v1.0"
    }
    joblib.dump(metadata, METADATA_PATH)
    
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Metadata saved to: {METADATA_PATH}")

if __name__ == "__main__":
    train_model()