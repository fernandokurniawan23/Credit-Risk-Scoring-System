"""
Inference Module
----------------
Serves the trained model for production predictions.
Includes:
1. Data Preprocessing pipeline.
2. Model loading.
3. Probability to Score conversion.
4. SHAP Explainability (Why did the model make this decision?).
"""

import pandas as pd
import numpy as np
import joblib
import os
import sys
import shap
from typing import Dict, Any, List

# --- IMPORT HANDLING ---
try:
    from src.data_cleaning import run_cleaning_pipeline
    from src.feature_engineering import run_feature_engineering
except ModuleNotFoundError:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from src.data_cleaning import run_cleaning_pipeline
    from src.feature_engineering import run_feature_engineering

class CreditScoringModel:
    def __init__(self, model_dir: str = "data/models"):
        base_path = os.getcwd()
        self.model_path = os.path.join(base_path, model_dir, "xgb_credit_risk_v1.joblib")
        self.meta_path = os.path.join(base_path, model_dir, "model_metadata.joblib")
        
        self.model = None
        self.features = None
        self.explainer = None
        self._load_artifacts()

    def _load_artifacts(self):
        """Loads model, metadata, and initializes SHAP explainer."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        
        print(f"[INFO] Loading model from {self.model_path}...")
        self.model = joblib.load(self.model_path)
        metadata = joblib.load(self.meta_path)
        self.features = metadata["features"]
        
        # Initialize SHAP Explainer (The "Why" Engine)
        # TreeExplainer is optimized for XGBoost
        print("[INFO] Initializing SHAP Explainer...")
        self.explainer = shap.TreeExplainer(self.model)
        print("[INFO] Model & Explainer loaded successfully.")

    def _preprocess_input(self, data: Dict[str, Any]) -> pd.DataFrame:
        """Transforms input into model-ready DataFrame."""
        df = pd.DataFrame([data])
        df = run_cleaning_pipeline(df)
        df = run_feature_engineering(df)
        # Reindex ensures columns match training data exactly
        df = df.reindex(columns=self.features, fill_value=0)
        return df

    def _get_top_factors(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Calculates the top 3 factors driving the risk UP and DOWN.
        Returns a simplified list for the API.
        """
        # Calculate SHAP values for this specific instance
        shap_values = self.explainer.shap_values(df)
        
        # XGBoost binary classification returns a single array of values
        # If shap_values is a list (multiclass), take the first index
        vals = shap_values[0] if isinstance(shap_values, list) else shap_values
        
        # Map values to feature names
        feature_importance = list(zip(self.features, vals[0])) # vals is (1, n_features)
        
        # Sort by absolute impact (magnitude)
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        # Take Top 5 most influential features
        top_features = []
        for feat, impact in feature_importance[:5]:
            direction = "INCREASES Risk" if impact > 0 else "REDUCES Risk"
            top_features.append({
                "feature": feat,
                "impact": float(round(impact, 4)),
                "direction": direction
            })
            
        return top_features

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Returns prediction + explanation.
        """
        try:
            # Preprocess
            df_processed = self._preprocess_input(data)
            
            # Predict Probability
            pd_score = self.model.predict_proba(df_processed)[0][1]
            
            # Calculate Score
            credit_score = int((1 - pd_score) * 1000)
            
            # Determine Tier
            if pd_score < 0.20:
                tier, decision = "Low Risk", "APPROVE"
            elif pd_score < 0.50:
                tier, decision = "Medium Risk", "MANUAL REVIEW"
            else:
                tier, decision = "High Risk", "REJECT"
            
            # Get Explanation (SHAP)
            explanations = self._get_top_factors(df_processed)
            
            return {
                "probability_default": float(round(pd_score, 4)),
                "credit_score": credit_score,
                "risk_tier": tier,
                "decision": decision,
                "top_factors": explanations  # New Data Field
            }
            
        except Exception as e:
            return {"error": str(e)}

if __name__ == "__main__":
    # Test
    model = CreditScoringModel()
    sample = {
        "SK_ID_CURR": 100002, "NAME_CONTRACT_TYPE": "Cash loans", "CODE_GENDER": "M",
        "AMT_INCOME_TOTAL": 200000, "AMT_CREDIT": 1000000, "AMT_ANNUITY": 50000,
        "AMT_GOODS_PRICE": 900000, "DAYS_EMPLOYED": -500, "DAYS_BIRTH": -10000,
        "EXT_SOURCE_2": 0.5, "EXT_SOURCE_3": 0.5
    }
    print(model.predict(sample))