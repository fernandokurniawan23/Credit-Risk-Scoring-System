"""
Data Cleaning Module
--------------------
Handles initial data preprocessing, including:
1. Anomaly detection and correction (e.g., temporal anomalies).
2. Noise filtering (dropping features with excessive missing values).
3. Categorical cleaning (removing invalid entries).
"""

import pandas as pd
import numpy as np

def clean_employment_anomaly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles the '365243' magic number in DAYS_EMPLOYED.
    
    Operations:
    - Creates 'DAYS_EMPLOYED_ANOM' flag (1 if anomaly, 0 otherwise).
    - Replaces 365243 with NaN in 'DAYS_EMPLOYED' to preserve distribution scaling.
    
    Args:
        df (pd.DataFrame): Raw dataframe.
        
    Returns:
        pd.DataFrame: Dataframe with corrected DAYS_EMPLOYED.
    """
    df = df.copy()
    
    # Generate anomaly flag
    df['DAYS_EMPLOYED_ANOM'] = (df['DAYS_EMPLOYED'] == 365243).astype(int)
    
    # Replace anomaly with NaN
    df['DAYS_EMPLOYED'] = df['DAYS_EMPLOYED'].replace(365243, np.nan)
    
    return df

def drop_high_missing_cols(df: pd.DataFrame, threshold: float = 50.0) -> pd.DataFrame:
    """
    Drops columns exceeding the missing value threshold.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        threshold (float): Percentage limit for missing values (0-100).
        
    Returns:
        pd.DataFrame: Dataframe with sparse columns removed.
    """
    missing_percent = 100 * df.isnull().sum() / len(df)
    drop_cols = missing_percent[missing_percent > threshold].index.tolist()
    
    print(f"[INFO] Dropping {len(drop_cols)} columns (> {threshold}% missing).")
    return df.drop(columns=drop_cols)

def clean_gender_code(df: pd.DataFrame) -> pd.DataFrame:
    """
    Removes rows with invalid CODE_GENDER entries (e.g., 'XNA').
    """
    return df[df['CODE_GENDER'] != 'XNA']

def run_cleaning_pipeline(df: pd.DataFrame) -> pd.DataFrame:
    """
    Executes the full cleaning sequence.
    
    Sequence:
    1. Anomaly Correction
    2. Category Filtering
    3. Dimensionality Reduction (Missing Values)
    
    Returns:
        pd.DataFrame: Cleaned dataframe ready for feature engineering.
    """
    initial_shape = df.shape
    print(f"--- Pipeline Started | Input Shape: {initial_shape} ---")
    
    df = clean_employment_anomaly(df)
    df = clean_gender_code(df)
    df = drop_high_missing_cols(df, threshold=50.0)
    
    final_shape = df.shape
    dropped_cols = initial_shape[1] - final_shape[1]
    dropped_rows = initial_shape[0] - final_shape[0]
    
    print(f"--- Pipeline Finished | Output Shape: {final_shape} ---")
    print(f"    - Columns Removed : {dropped_cols}")
    print(f"    - Rows Removed    : {dropped_rows}")
    
    return df

if __name__ == "__main__":
    # Local testing execution
    RAW_PATH = "data/raw/application_train.csv"
    
    try:
        df_raw = pd.read_csv(RAW_PATH)
        df_clean = run_cleaning_pipeline(df_raw)
    except FileNotFoundError:
        print(f"[ERROR] File not found at {RAW_PATH}")