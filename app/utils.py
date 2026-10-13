"""
utils.py
========
Utility functions for the Streamlit dashboard.

Handles model loading, feature preparation, SHAP computation,
and risk tier color coding.
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import shap
import streamlit as st

warnings.filterwarnings("ignore")

# ── Path bootstrap ────────────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.insert(0, SRC_DIR)

from preprocessing import (
    load_data, build_preprocessor, get_feature_names,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET,
)
from feature_engineering import add_engineered_features
from predict import score_customers, assign_risk_tier, RISK_TIER_LOW_MAX, RISK_TIER_MEDIUM_MAX

# ── Paths ─────────────────────────────────────
MODELS_DIR = os.path.join(ROOT_DIR, "models")
DATA_PROCESSED = os.path.join(ROOT_DIR, "data", "processed")

# ── Risk tier colors ──────────────────────────
TIER_COLORS = {
    "Low": "#10B981",       # Emerald green
    "Medium": "#F59E0B",    # Amber
    "High": "#EF4444",      # Red
}

TIER_BG_COLORS = {
    "Low": "#D1FAE5",
    "Medium": "#FEF3C7",
    "High": "#FEE2E2",
}


# ──────────────────────────────────────────────
# Cached loaders
# ──────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_model_and_metadata():
    """Load the best model pipeline and training metadata."""
    pipeline = joblib.load(os.path.join(MODELS_DIR, "best_model_pipeline.joblib"))
    with open(os.path.join(MODELS_DIR, "training_metadata.json")) as f:
        metadata = json.load(f)
    return pipeline, metadata


@st.cache_resource(show_spinner=False)
def load_all_pipelines():
    """Load all model pipelines for comparison."""
    return joblib.load(os.path.join(MODELS_DIR, "all_pipelines.joblib"))


@st.cache_data(show_spinner=False)
def load_scored_data():
    """Load pre-scored customer data."""
    path = os.path.join(DATA_PROCESSED, "customer_risk_scores.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data(show_spinner=False)
def load_metrics_table():
    """Load the full model comparison metrics table."""
    path = os.path.join(DATA_PROCESSED, "model_comparison_full.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    path2 = os.path.join(DATA_PROCESSED, "model_comparison_initial.csv")
    if os.path.exists(path2):
        return pd.read_csv(path2)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_shap_importance():
    """Load pre-computed SHAP feature importance."""
    path = os.path.join(DATA_PROCESSED, "shap_importance.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


@st.cache_data(show_spinner=False)
def load_test_data():
    """Load test set for distribution plots."""
    X_test = pd.read_csv(os.path.join(DATA_PROCESSED, "X_test.csv"))
    y_test = pd.read_csv(os.path.join(DATA_PROCESSED, "y_test.csv")).squeeze()
    return X_test, y_test


# ──────────────────────────────────────────────
# Feature preparation for a single customer
# ──────────────────────────────────────────────

def prepare_customer_input(
    credit_score: int,
    geography: str,
    gender: str,
    age: int,
    tenure: int,
    balance: float,
    num_products: int,
    has_cr_card: int,
    is_active: int,
    salary: float,
) -> pd.DataFrame:
    """
    Build a single-row DataFrame for a customer, including engineered features.

    Returns
    -------
    pd.DataFrame with all feature columns (no target, no Age_Band).
    """
    raw = pd.DataFrame([{
        "CreditScore": credit_score,
        "Geography": geography,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active,
        "EstimatedSalary": salary,
    }])

    df_eng = add_engineered_features(raw)
    feature_cols = [c for c in df_eng.columns if c not in [TARGET, "Age_Band"]]
    return df_eng[feature_cols]


# ──────────────────────────────────────────────
# Prediction for a single customer
# ──────────────────────────────────────────────

def predict_single(
    customer_df: pd.DataFrame,
    pipeline,
    threshold: float,
) -> dict:
    """
    Predict churn for a single customer.

    Returns
    -------
    dict with keys: probability, predicted, risk_tier
    """
    prob = float(pipeline.predict_proba(customer_df)[0, 1])
    predicted = int(prob >= threshold)
    tier = assign_risk_tier(prob)
    return {
        "probability": prob,
        "predicted": predicted,
        "risk_tier": tier,
    }


# ──────────────────────────────────────────────
# SHAP for a single customer
# ──────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def get_shap_explainer(_pipeline, _X_background: pd.DataFrame):
    """
    Build and cache a SHAP explainer for the best model.
    Uses TreeExplainer if possible, else KernelExplainer.
    """
    preprocessor = _pipeline.named_steps["preprocessor"]
    classifier = _pipeline.named_steps["classifier"]

    numeric_features = [c for c in NUMERIC_FEATURES if c in _X_background.columns]
    feature_names = get_feature_names(preprocessor, numeric_features, CATEGORICAL_FEATURES)

    X_transformed = preprocessor.transform(_X_background)
    X_df = pd.DataFrame(X_transformed, columns=feature_names)

    try:
        explainer = shap.TreeExplainer(classifier)
        return explainer, feature_names, preprocessor
    except Exception:
        background = shap.kmeans(X_df, 50)
        explainer = shap.KernelExplainer(
            lambda x: classifier.predict_proba(x)[:, 1],
            background,
        )
        return explainer, feature_names, preprocessor


def get_individual_shap(
    explainer,
    pipeline,
    customer_df: pd.DataFrame,
    feature_names: list,
    preprocessor,
    top_n: int = 8,
) -> pd.DataFrame:
    """
    Compute SHAP values for a single customer and return top-N drivers.

    Returns
    -------
    pd.DataFrame with Feature, SHAP_Value, Feature_Value columns.
    """
    X_transformed = preprocessor.transform(customer_df)
    X_df = pd.DataFrame(X_transformed, columns=feature_names)

    shap_vals = explainer.shap_values(X_df)
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[1]

    shap_series = pd.Series(shap_vals[0], index=feature_names)
    top = shap_series.abs().sort_values(ascending=False).head(top_n)

    result = pd.DataFrame({
        "Feature": top.index,
        "SHAP_Value": shap_series[top.index].values,
        "Feature_Value": X_df.iloc[0][top.index].values,
        "Abs_SHAP": top.values,
    })
    return result


# ──────────────────────────────────────────────
# Retention recommendations
# ──────────────────────────────────────────────

def get_retention_recommendations(result: dict, customer_df: pd.DataFrame) -> list:
    """
    Generate tailored retention action recommendations based on
    customer profile and predicted risk tier.

    Returns
    -------
    list of str recommendations
    """
    recommendations = []
    prob = result["probability"]
    tier = result["risk_tier"]

    is_active = int(customer_df["IsActiveMember"].iloc[0])
    n_products = int(customer_df["NumOfProducts"].iloc[0])
    balance = float(customer_df["Balance"].iloc[0])
    age = int(customer_df["Age"].iloc[0])
    tenure = int(customer_df["Tenure"].iloc[0])

    if tier == "High":
        recommendations.append("🚨 **Priority intervention required.** Assign a personal relationship manager immediately.")

    if is_active == 0:
        recommendations.append("📱 **Re-engagement campaign:** Send personalised digital banking offer. Dormant accounts churn at 2× the rate of active ones.")

    if n_products == 1:
        recommendations.append("🎯 **Cross-sell opportunity:** Offer a relevant second product (savings account, insurance, or investment). Single-product customers are the highest churn-risk segment.")

    if balance == 0:
        recommendations.append("💳 **Zero-balance activation:** Offer an incentive to fund the account (e.g., bonus interest for 3-month minimum balance).")

    if 45 <= age <= 65:
        recommendations.append("📊 **Wealth management outreach:** Mid-to-late career customers value financial planning. Offer a complimentary financial review.")

    if tenure < 2:
        recommendations.append("🤝 **New customer nurturing:** Enrol in onboarding journey with milestone rewards at 6, 12, and 24 months.")

    if prob >= 0.8:
        recommendations.append("💰 **Retention offer:** Consider a tailored fee waiver or preferential interest rate for 6–12 months.")

    if not recommendations:
        recommendations.append("✅ **Low risk:** Continue standard engagement. Monitor quarterly.")

    return recommendations
