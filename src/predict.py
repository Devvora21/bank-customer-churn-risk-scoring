"""
predict.py
==========
Batch scoring of all customers:
  - Churn probability (0–1)
  - Binary churn flag using the optimised threshold
  - Risk tier: Low / Medium / High
  - Exports scored customers to data/processed/customer_risk_scores.csv

Usage:
    python src/predict.py
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib

warnings.filterwarnings("ignore")

# ── Path bootstrap ────────────────────────────
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from preprocessing import load_data, TARGET
from feature_engineering import add_engineered_features

# ── Paths ─────────────────────────────────────
MODELS_DIR = os.path.join(ROOT_DIR, "models")
DATA_PROCESSED = os.path.join(ROOT_DIR, "data", "processed")
os.makedirs(DATA_PROCESSED, exist_ok=True)

# ── Risk tier thresholds ──────────────────────
RISK_TIER_LOW_MAX = 0.30       # < 0.30 → Low
RISK_TIER_MEDIUM_MAX = 0.60    # 0.30–0.60 → Medium
                                # > 0.60 → High


def assign_risk_tier(probability: float) -> str:
    """
    Assign a risk tier label based on churn probability.

    Thresholds:
        Low    : probability < 0.30
        Medium : 0.30 ≤ probability < 0.60
        High   : probability ≥ 0.60

    Parameters
    ----------
    probability : float
        Churn probability in [0, 1].

    Returns
    -------
    str : 'Low', 'Medium', or 'High'
    """
    if probability < RISK_TIER_LOW_MAX:
        return "Low"
    elif probability < RISK_TIER_MEDIUM_MAX:
        return "Medium"
    else:
        return "High"


def score_customers(
    df_input: pd.DataFrame = None,
    pipeline=None,
    threshold: float = None,
) -> pd.DataFrame:
    """
    Score all customers with churn probability, binary flag, and risk tier.

    Parameters
    ----------
    df_input : pd.DataFrame, optional
        Input feature DataFrame. If None, loads the full dataset.
    pipeline : sklearn Pipeline, optional
        Trained model pipeline. If None, loads from models/.
    threshold : float, optional
        Decision threshold. If None, loads from training_metadata.json.

    Returns
    -------
    pd.DataFrame
        Scored DataFrame with columns:
        [original features] + Churn_Probability + Churn_Predicted + Risk_Tier
    """
    # Load pipeline
    if pipeline is None:
        pipeline = joblib.load(os.path.join(MODELS_DIR, "best_model_pipeline.joblib"))

    # Load optimal threshold
    if threshold is None:
        with open(os.path.join(MODELS_DIR, "training_metadata.json")) as f:
            metadata = json.load(f)
        threshold = metadata.get("optimal_threshold", 0.5)

    # Load data if not provided
    if df_input is None:
        df_raw = load_data()
        df_eng = add_engineered_features(df_raw)
    else:
        df_eng = df_input.copy()

    feature_cols = [c for c in df_eng.columns if c != TARGET and c != "Age_Band"]
    X = df_eng[feature_cols]

    # Predict
    y_prob = pipeline.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    risk_tier = [assign_risk_tier(p) for p in y_prob]

    # Build output
    df_out = df_eng.copy()
    df_out["Churn_Probability"] = np.round(y_prob, 4)
    df_out["Churn_Predicted"] = y_pred
    df_out["Risk_Tier"] = risk_tier

    # Sort by churn probability descending (highest risk first)
    df_out = df_out.sort_values("Churn_Probability", ascending=False).reset_index(drop=True)

    return df_out


def main():
    print("\n" + "="*60)
    print("  BANK CHURN RISK SCORING — BATCH PREDICTION")
    print("="*60)

    print("\n[1/3] Loading model and data...")
    pipeline = joblib.load(os.path.join(MODELS_DIR, "best_model_pipeline.joblib"))

    with open(os.path.join(MODELS_DIR, "training_metadata.json")) as f:
        metadata = json.load(f)
    threshold = metadata.get("optimal_threshold", 0.5)
    best_model = metadata.get("best_model", "Unknown")

    print(f"  Model: {best_model}")
    print(f"  Decision threshold: {threshold:.2f}")

    print("\n[2/3] Scoring all customers...")
    df_scored = score_customers(pipeline=pipeline, threshold=threshold)

    print("\n[3/3] Saving results...")
    out_path = os.path.join(DATA_PROCESSED, "customer_risk_scores.csv")
    df_scored.to_csv(out_path, index=False)

    # Summary statistics
    n_total = len(df_scored)
    n_high = (df_scored["Risk_Tier"] == "High").sum()
    n_medium = (df_scored["Risk_Tier"] == "Medium").sum()
    n_low = (df_scored["Risk_Tier"] == "Low").sum()
    n_churn_pred = df_scored["Churn_Predicted"].sum()

    print(f"\n  Total customers scored: {n_total:,}")
    print(f"  Predicted churners:     {n_churn_pred:,} ({n_churn_pred/n_total:.1%})")
    print(f"\n  Risk Tier Distribution:")
    print(f"    High   (prob >= 0.60): {n_high:,}   ({n_high/n_total:.1%})")
    print(f"    Medium (0.30–0.60):   {n_medium:,}  ({n_medium/n_total:.1%})")
    print(f"    Low    (prob < 0.30): {n_low:,}    ({n_low/n_total:.1%})")
    print(f"\n  Saved to: {out_path}")
    print("\n[OK] Batch scoring complete!")

    return df_scored


if __name__ == "__main__":
    main()
