"""
preprocessing.py
================
Data loading and sklearn-compatible preprocessing pipeline.

All transformers are fitted ONLY on training data to prevent leakage.
"""

import os
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


# ──────────────────────────────────────────────
# Constants
# ──────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "processed")

# Columns to drop (non-informative identifiers)
DROP_COLS = ["Year", "CustomerId", "Surname"]

TARGET = "Exited"

# Feature sets (after dropping identifiers and target)
NUMERIC_FEATURES = [
    "CreditScore", "Age", "Tenure", "Balance",
    "NumOfProducts", "EstimatedSalary",
    # Engineered features added later
    "Balance_to_Salary", "Products_per_Tenure",
    "Active_Products", "Age_Tenure_Interaction",
    "Tenure_Age_Ratio", "Zero_Balance_Flag",
]

CATEGORICAL_FEATURES = ["Geography", "Gender"]


# ──────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────

def load_data(filepath: str = None) -> pd.DataFrame:
    """
    Load the raw CSV, drop non-informative columns, and return a clean DataFrame.

    Parameters
    ----------
    filepath : str, optional
        Path to the CSV file. Defaults to data/raw/European_Bank.csv.

    Returns
    -------
    pd.DataFrame
        DataFrame with identifiers dropped and correct dtypes.
    """
    if filepath is None:
        filepath = os.path.join(DATA_DIR, "European_Bank.csv")

    df = pd.read_csv(filepath)

    # Drop non-informative columns (silently ignore missing)
    cols_to_drop = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # Sanity checks
    assert TARGET in df.columns, f"Target column '{TARGET}' not found."

    # Enforce correct dtypes
    binary_cols = ["HasCrCard", "IsActiveMember", TARGET]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    return df


# ──────────────────────────────────────────────
# Preprocessing Pipeline Builder
# ──────────────────────────────────────────────

def build_preprocessor(numeric_features: list = None,
                        categorical_features: list = None) -> ColumnTransformer:
    """
    Build and return a ColumnTransformer that:
      - StandardScales numeric features
      - OneHotEncodes categorical features (drop_first to avoid multicollinearity)

    Parameters
    ----------
    numeric_features : list, optional
        List of numeric column names. Defaults to NUMERIC_FEATURES.
    categorical_features : list, optional
        List of categorical column names. Defaults to CATEGORICAL_FEATURES.

    Returns
    -------
    ColumnTransformer
        Unfitted preprocessor ready to be embedded in a Pipeline.
    """
    if numeric_features is None:
        numeric_features = NUMERIC_FEATURES
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES

    numeric_transformer = Pipeline([
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("onehot", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )
    return preprocessor


def get_feature_names(preprocessor: ColumnTransformer,
                      numeric_features: list = None,
                      categorical_features: list = None) -> list:
    """
    Extract human-readable feature names from a FITTED ColumnTransformer.

    Parameters
    ----------
    preprocessor : ColumnTransformer
        A fitted ColumnTransformer.
    numeric_features : list, optional
    categorical_features : list, optional

    Returns
    -------
    list of str
        Ordered list of feature names matching the transformed array columns.
    """
    if numeric_features is None:
        numeric_features = NUMERIC_FEATURES
    if categorical_features is None:
        categorical_features = CATEGORICAL_FEATURES

    ohe = preprocessor.named_transformers_["cat"]["onehot"]
    cat_names = list(ohe.get_feature_names_out(categorical_features))
    return numeric_features + cat_names


if __name__ == "__main__":
    df = load_data()
    print(f"Data shape: {df.shape}")
    print(f"Churn rate: {df[TARGET].mean():.3f}")
    print(df.describe())
