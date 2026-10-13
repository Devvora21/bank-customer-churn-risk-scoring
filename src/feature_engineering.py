"""
feature_engineering.py
======================
Creates behaviorally meaningful features from raw bank data.

Design philosophy: features should capture engagement, product utilisation,
and relationship strength — not just demographics.
"""

import numpy as np
import pandas as pd


# ──────────────────────────────────────────────
# Age band helper
# ──────────────────────────────────────────────

AGE_BINS = [0, 30, 40, 50, 60, 120]
AGE_LABELS = ["18-30", "31-40", "41-50", "51-60", "60+"]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all engineered features to the DataFrame.

    New columns created:
    --------------------
    Balance_to_Salary : float
        Ratio of account balance to estimated salary.  Higher values can
        indicate customers who park wealth in the bank (retention indicator)
        or have zero salary estimates (data quality signal).
        Guard against divide-by-zero with epsilon.

    Products_per_Tenure : float
        Number of products normalised by tenure (+ 1 to avoid zero-division).
        Captures product adoption pace; very high values in short-tenure
        customers may signal cross-sell saturation and potential churn.

    Active_Products : int
        Interaction of IsActiveMember × NumOfProducts.
        Active customers with more products are more deeply engaged.
        Inactive customers with many products may still churn.

    Age_Tenure_Interaction : float
        Age × Tenure.  Captures long-tenured older customers who tend to be
        stickier vs young, short-tenure customers.

    Tenure_Age_Ratio : float
        Tenure / (Age + 1).  Proportion of adult life spent with the bank;
        high values indicate stronger loyalty.

    Zero_Balance_Flag : int  (0 / 1)
        Binary flag for accounts with exactly zero balance.
        ~30 % of customers have zero balance — this is a strong churn signal.

    Age_Band : str
        Ordinal age bucket for EDA segmentation (not fed directly to models).

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame (with original columns present).

    Returns
    -------
    pd.DataFrame
        Copy of df with new engineered columns appended.
    """
    df = df.copy()

    # 1. Balance-to-Salary ratio
    df["Balance_to_Salary"] = df["Balance"] / (df["EstimatedSalary"] + 1e-6)

    # 2. Product adoption pace
    df["Products_per_Tenure"] = df["NumOfProducts"] / (df["Tenure"] + 1)

    # 3. Engagement × product depth interaction
    df["Active_Products"] = df["IsActiveMember"] * df["NumOfProducts"]

    # 4. Age-tenure interaction (loyalty-age compound)
    df["Age_Tenure_Interaction"] = df["Age"] * df["Tenure"]

    # 5. Proportion of adult life with the bank
    df["Tenure_Age_Ratio"] = df["Tenure"] / (df["Age"] + 1)

    # 6. Zero-balance flag
    df["Zero_Balance_Flag"] = (df["Balance"] == 0).astype(int)

    # 7. Age band (for EDA and optional categorical use)
    df["Age_Band"] = pd.cut(
        df["Age"],
        bins=AGE_BINS,
        labels=AGE_LABELS,
        right=True,
    ).astype(str)

    return df


def report_feature_impact(
    baseline_auc: float,
    feature_aucs: dict,
) -> pd.DataFrame:
    """
    Produce a tidy DataFrame reporting delta-AUC for each engineered feature.

    Parameters
    ----------
    baseline_auc : float
        ROC-AUC of model WITHOUT engineered features.
    feature_aucs : dict
        Mapping {feature_name: auc_with_that_feature_added}.

    Returns
    -------
    pd.DataFrame
        Sorted by delta_auc descending.
    """
    rows = []
    for feat, auc in feature_aucs.items():
        rows.append({
            "Feature": feat,
            "AUC_with_feature": round(auc, 5),
            "Delta_AUC": round(auc - baseline_auc, 5),
            "Improved": auc > baseline_auc,
        })
    return pd.DataFrame(rows).sort_values("Delta_AUC", ascending=False)


if __name__ == "__main__":
    from preprocessing import load_data
    df = load_data()
    df_fe = add_engineered_features(df)
    print("Engineered features added:")
    new_cols = [
        "Balance_to_Salary", "Products_per_Tenure", "Active_Products",
        "Age_Tenure_Interaction", "Tenure_Age_Ratio", "Zero_Balance_Flag",
        "Age_Band",
    ]
    print(df_fe[new_cols].describe())
