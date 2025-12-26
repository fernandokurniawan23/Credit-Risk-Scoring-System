import pandas as pd
import numpy as np
from typing import Dict, List

# Configuration
RAW_DATA_PATH = "data/raw/application_train.csv"

def load_data(path: str) -> pd.DataFrame:
    """Loads raw data from csv."""
    return pd.read_csv(path)

def analyze_missing_values(df: pd.DataFrame, threshold: float = 50.0) -> pd.DataFrame:
    """
    Calculates missing value percentage per column.
    Returns columns exceeding the threshold.
    """
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table = mis_val_table.rename(
        columns={0: 'Missing Values', 1: '% of Total Values'}
    )
    
    # Filter only columns with missing values and sort
    mis_val_table = mis_val_table[mis_val_table.iloc[:, 1] != 0].sort_values(
        '% of Total Values', ascending=False
    ).round(1)
    
    return mis_val_table

def check_business_anomalies(df: pd.DataFrame) -> None:
    """
    Checks for specific known anomalies in Home Credit dataset:
    1. DAYS_EMPLOYED: 365243 (magic number for anomaly/unemployed)
    2. DAYS_BIRTH: Negative values expected (convert to positive years)
    """
    # Check DAYS_EMPLOYED anomaly
    anom_employed = df[df['DAYS_EMPLOYED'] == 365243]
    print("\n[ANOMALY CHECK] DAYS_EMPLOYED")
    print(f"Magic Number (365243) Count: {len(anom_employed)}")
    print(f"Percentage: {len(anom_employed)/len(df)*100:.2f}%")
    
    # Check DAYS_BIRTH stats (should be negative)
    print("\n[ANOMALY CHECK] DAYS_BIRTH (Age)")
    years_birth = df['DAYS_BIRTH'] / -365
    print(f"Age Range: {years_birth.min():.1f} to {years_birth.max():.1f} years")

def main():
    df = load_data(RAW_DATA_PATH)
    
    # 1. Missing Values Analysis
    missing_df = analyze_missing_values(df)
    print("\n[MISSING VALUES] Top 10 Columns")
    print(missing_df.head(10))
    
    # 2. Anomaly Detection
    check_business_anomalies(df)

if __name__ == "__main__":
    main()