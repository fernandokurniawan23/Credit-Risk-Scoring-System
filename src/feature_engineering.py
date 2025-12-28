"""
Feature Engineering Module
--------------------------
Transforms cleaned data into machine-learning-ready features.
Focus:
1. Domain Knowledge Features (Financial Ratios).
2. Categorical Encoding (One-Hot).
3. Data Serialization for Training.
"""

import pandas as pd
import numpy as np

def create_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates strong financial ratios used in credit scoring.
    
    New Features:
    - CREDIT_INCOME_PERCENT: Ratio of credit amount to total income.
    - ANNUITY_INCOME_PERCENT: Ratio of loan annuity to total income (Debt Service Ratio).
    - CREDIT_TERM: Estimate of the loan term (Credit / Annuity).
    - GOODS_PRICE_LOAN_DIFFERENCE: Difference between goods price and loan amount.
    
    Args:
        df (pd.DataFrame): Cleaned dataframe.
        
    Returns:
        pd.DataFrame: Dataframe with new domain features.
    """
    df = df.copy()
    
    # 1. Credit to Income Ratio
    # High ratio = High Risk
    df['CREDIT_INCOME_PERCENT'] = df['AMT_CREDIT'] / df['AMT_INCOME_TOTAL']
    
    # 2. Annuity to Income Ratio (Debt Burden)
    # Critical banking metric
    df['ANNUITY_INCOME_PERCENT'] = df['AMT_ANNUITY'] / df['AMT_INCOME_TOTAL']
    
    # 3. Estimated Credit Term (in months/payments)
    # AMT_CREDIT / AMT_ANNUITY gives rough number of installments
    df['CREDIT_TERM'] = df['AMT_CREDIT'] / df['AMT_ANNUITY']
    
    # 4. Goods Price vs Credit Amount
    # Did they borrow more than the item value? (Cash out)
    df['GOODS_LOAN_RATIO'] = df['AMT_GOODS_PRICE'] / df['AMT_CREDIT']
    
    # 5. Document Age in Years (Interpretability)
    # Converts raw negative days (e.g., -291) to positive years (e.g., 0.8)
    # Easier for SHAP and humans to understand "how old is the ID".
    df['ID_AGE_YEARS'] = df['DAYS_ID_PUBLISH'].abs() / 365
    
    # 6. ID Age to Client Age Ratio (Risk Logic)
    # Logic: A 45-year-old with a 2-month-old ID is more suspicious than
    # a 20-year-old with a 2-month-old ID.
    # Formula: ID_Age / Client_Age
    # High Ratio = Normal (Long term ID holder)
    # Very Low Ratio = Suspicious (Fresh ID relative to age)
    # Note: We use max(..., 1) or abs() to avoid division by zero or negative ages
    client_age_years = df['DAYS_BIRTH'].abs() / 365
    df['ID_TO_AGE_RATIO'] = df['ID_AGE_YEARS'] / client_age_years
    
    return df

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies One-Hot Encoding to categorical columns.
    Uses pd.get_dummies with drop_first=True to reduce multicollinearity.
    
    Args:
        df (pd.DataFrame): Dataframe with object columns.
        
    Returns:
        pd.DataFrame: Fully numerical dataframe.
    """
    # Select object columns
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    print(f"[INFO] Encoding {len(categorical_cols)} categorical columns...")
    
    # Dummy encoding
    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df_encoded

def run_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executes the full Feature Engineering pipeline.
    """
    initial_shape = df.shape
    print(f"--- FE Pipeline Started | Input Shape: {initial_shape} ---")
    
    # 1. Domain Features
    df = create_domain_features(df)
    
    # 2. Encoding
    df = encode_categoricals(df)
    
    # 3. Clean Column Names (Remove spaces/special chars for XGBoost)
    import re
    df = df.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    
    final_shape = df.shape
    print(f"--- FE Pipeline Finished | Output Shape: {final_shape} ---")
    print(f"    - New Features Created: {final_shape[1] - initial_shape[1]}")
    
    return df

if __name__ == "__main__":
    # Integration Test
    from data_cleaning import run_cleaning_pipeline
    
    # Load and Clean first (Simulation of full pipeline)
    print("[TEST] Loading and Cleaning Data...")
    df_raw = pd.read_csv("data/raw/application_train.csv")
    df_clean = run_cleaning_pipeline(df_raw)
    
    # Run FE
    print("\n[TEST] Running Feature Engineering...")
    df_final = run_feature_engineering(df_clean)
    
    # Save processed data for Training Phase
    save_path = "data/processed/train_final.parquet"
    print(f"\n[INFO] Saving processed data to {save_path}...")
    df_final.to_parquet(save_path)
    print("Data pipeline complete.")