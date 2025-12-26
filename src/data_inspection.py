import os
import pandas as pd
from typing import Tuple


DATA_PATH: str = "data/raw/application_train.csv"


def check_file_exists(path: str) -> None:
    """
    Validate that the dataset file exists.

    Raises:
        FileNotFoundError: If the dataset file is not found.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at path: {path}. "
            "Ensure the script is executed from the project root directory."
        )


def load_dataset(path: str) -> pd.DataFrame:
    """
    Load the credit application dataset from CSV.

    Args:
        path (str): Path to the CSV dataset.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    df: pd.DataFrame = pd.read_csv(path)
    return df


def inspect_dataset_shape(df: pd.DataFrame) -> None:
    """
    Print basic dataset dimensions.
    """
    rows, cols = df.shape
    print("DATASET OVERVIEW")
    print(f"Rows    : {rows}")
    print(f"Columns : {cols}")


def inspect_target_distribution(df: pd.DataFrame) -> Tuple[int, int]:
    """
    Inspect class distribution of the target variable.

    Args:
        df (pd.DataFrame): Dataset containing TARGET column.

    Returns:
        Tuple[int, int]: Count of repayers (0) and defaulters (1).
    """
    counts = df["TARGET"].value_counts()

    repayer_count: int = counts.get(0, 0)
    defaulter_count: int = counts.get(1, 0)
    total: int = repayer_count + defaulter_count

    repayer_pct: float = (repayer_count / total) * 100
    defaulter_pct: float = (defaulter_count / total) * 100

    print("\nTARGET DISTRIBUTION")
    print(f"Repayer (0)   : {repayer_count} ({repayer_pct:.2f}%)")
    print(f"Defaulter (1) : {defaulter_count} ({defaulter_pct:.2f}%)")

    if defaulter_pct < 10:
        print(
            "Observation: Dataset is highly imbalanced. "
            "ROC-AUC and PR-AUC should be prioritized over accuracy."
        )

    return repayer_count, defaulter_count


def inspect_id_integrity(df: pd.DataFrame) -> None:
    """
    Check uniqueness of applicant IDs.
    """
    unique_ids: int = df["SK_ID_CURR"].nunique()
    total_rows: int = len(df)
    duplicate_count: int = total_rows - unique_ids

    print("\nDATA INTEGRITY CHECK")
    if duplicate_count == 0:
        print("Applicant IDs are unique.")
    else:
        print(f"Warning: {duplicate_count} duplicate applicant IDs detected.")


def run_initial_data_inspection() -> None:
    """
    Run initial inspection checks for the credit risk dataset.
    """
    print("STARTING DATA INSPECTION")

    check_file_exists(DATA_PATH)
    df: pd.DataFrame = load_dataset(DATA_PATH)

    inspect_dataset_shape(df)
    inspect_target_distribution(df)
    inspect_id_integrity(df)


if __name__ == "__main__":
    run_initial_data_inspection()
