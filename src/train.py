"""
train.py
========
End-to-end model training pipeline.

Trains five model families, tunes hyperparameters via RandomizedSearchCV,
handles class imbalance with class_weight and SMOTE (leakage-free),
and saves the best model pipeline.

Usage:
    python src/train.py
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    RandomizedSearchCV,
    cross_val_score,
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
)
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    classification_report,
)
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

warnings.filterwarnings("ignore")

# ── Path bootstrap ────────────────────────────
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from preprocessing import (
    load_data, build_preprocessor,
    NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET,
)
from feature_engineering import add_engineered_features

# ── Paths ─────────────────────────────────────
MODELS_DIR = os.path.join(ROOT_DIR, "models")
DATA_PROCESSED = os.path.join(ROOT_DIR, "data", "processed")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(DATA_PROCESSED, exist_ok=True)

# ── Reproducibility ───────────────────────────
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ──────────────────────────────────────────────
# Helper: resolve which numeric features exist
# ──────────────────────────────────────────────

def get_available_numeric(df: pd.DataFrame) -> list:
    """Return only numeric features that are present in df."""
    return [f for f in NUMERIC_FEATURES if f in df.columns]


# ──────────────────────────────────────────────
# Model definitions
# ──────────────────────────────────────────────

def get_model_configs() -> dict:
    """
    Returns dict of {name: (estimator, param_distributions, use_smote)}.
    SMOTE is applied ONLY inside the imblearn pipeline (no leakage).
    """
    configs = {
        "Logistic Regression": (
            LogisticRegression(
                random_state=RANDOM_STATE,
                max_iter=1000,
                class_weight="balanced",
            ),
            {
                "classifier__C": [0.01, 0.1, 1, 10, 100],
                "classifier__solver": ["lbfgs", "saga"],
                "classifier__penalty": ["l2"],
            },
            False,  # No SMOTE (use class_weight instead)
        ),
        "Decision Tree": (
            DecisionTreeClassifier(
                random_state=RANDOM_STATE,
                class_weight="balanced",
            ),
            {
                "classifier__max_depth": [3, 5, 7, 10, None],
                "classifier__min_samples_split": [2, 10, 20],
                "classifier__min_samples_leaf": [1, 5, 10],
                "classifier__criterion": ["gini", "entropy"],
            },
            False,
        ),
        "Random Forest": (
            RandomForestClassifier(
                random_state=RANDOM_STATE,
                n_jobs=-1,
                class_weight="balanced",
            ),
            {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__max_depth": [5, 10, 15, None],
                "classifier__min_samples_split": [2, 5, 10],
                "classifier__min_samples_leaf": [1, 2, 4],
                "classifier__max_features": ["sqrt", "log2"],
            },
            False,
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(
                random_state=RANDOM_STATE,
            ),
            {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__learning_rate": [0.05, 0.1, 0.2],
                "classifier__max_depth": [3, 4, 5],
                "classifier__subsample": [0.7, 0.8, 1.0],
                "classifier__min_samples_leaf": [1, 5, 10],
            },
            True,  # Use SMOTE inside pipeline
        ),
        "XGBoost": (
            XGBClassifier(
                random_state=RANDOM_STATE,
                eval_metric="logloss",
                use_label_encoder=False,
                tree_method="hist",
            ),
            {
                "classifier__n_estimators": [100, 200, 300, 500],
                "classifier__learning_rate": [0.01, 0.05, 0.1, 0.2],
                "classifier__max_depth": [3, 4, 5, 6],
                "classifier__subsample": [0.7, 0.8, 0.9, 1.0],
                "classifier__colsample_bytree": [0.6, 0.8, 1.0],
                "classifier__scale_pos_weight": [1, 3, 5],
                "classifier__gamma": [0, 0.1, 0.5],
                "classifier__reg_alpha": [0, 0.1, 1.0],
            },
            False,  # XGBoost handles imbalance via scale_pos_weight
        ),
    }
    return configs


# ──────────────────────────────────────────────
# Build pipeline for a given estimator
# ──────────────────────────────────────────────

def build_pipeline(
    estimator,
    preprocessor,
    use_smote: bool = False,
) -> Pipeline:
    """
    Wrap preprocessor + optional SMOTE + estimator in a Pipeline.

    Parameters
    ----------
    estimator : sklearn estimator
    preprocessor : ColumnTransformer
    use_smote : bool
        If True, adds SMOTE step (uses imblearn Pipeline).

    Returns
    -------
    Pipeline (or ImbPipeline)
    """
    if use_smote:
        return ImbPipeline([
            ("preprocessor", preprocessor),
            ("smote", SMOTE(random_state=RANDOM_STATE)),
            ("classifier", estimator),
        ])
    else:
        return Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ])


# ──────────────────────────────────────────────
# Training function
# ──────────────────────────────────────────────

def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    numeric_features: list,
    n_iter: int = 30,
    cv_folds: int = 5,
    verbose: int = 1,
) -> dict:
    """
    Train all model configurations with RandomizedSearchCV.

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series
    numeric_features : list
        Numeric features available in X_train.
    n_iter : int
        Number of hyperparameter combinations to try.
    cv_folds : int
        Number of stratified CV folds.
    verbose : int

    Returns
    -------
    dict : {model_name: best_pipeline}
    """
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    configs = get_model_configs()
    results = {}

    for name, (estimator, param_dist, use_smote) in configs.items():
        if verbose:
            print(f"\n{'='*55}")
            print(f"  Training: {name}")
            print(f"{'='*55}")

        preprocessor = build_preprocessor(
            numeric_features=numeric_features,
            categorical_features=CATEGORICAL_FEATURES,
        )
        pipeline = build_pipeline(estimator, preprocessor, use_smote)

        search = RandomizedSearchCV(
            pipeline,
            param_distributions=param_dist,
            n_iter=n_iter,
            scoring="roc_auc",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=2,
            verbose=0,
            refit=True,
            return_train_score=True,
        )
        search.fit(X_train, y_train)

        best_pipeline = search.best_estimator_
        best_cv_auc = search.best_score_

        if verbose:
            print(f"  Best CV ROC-AUC: {best_cv_auc:.4f}")
            print(f"  Best Params: {search.best_params_}")

        results[name] = {
            "pipeline": best_pipeline,
            "best_cv_auc": best_cv_auc,
            "best_params": search.best_params_,
            "cv_results": search.cv_results_,
        }

    return results


# ──────────────────────────────────────────────
# Select best model
# ──────────────────────────────────────────────

def select_best_model(results: dict, X_test, y_test) -> tuple:
    """
    Evaluate all trained models on the test set and select the best
    by ROC-AUC (secondary: F1-score).

    Returns
    -------
    (best_name, best_pipeline, metrics_df)
    """
    rows = []
    for name, res in results.items():
        pipeline = res["pipeline"]
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        auc = roc_auc_score(y_test, y_prob)
        f1 = f1_score(y_test, y_pred)
        rows.append({
            "Model": name,
            "Test_ROC_AUC": round(auc, 4),
            "CV_ROC_AUC": round(res["best_cv_auc"], 4),
            "Test_F1": round(f1, 4),
        })

    metrics_df = pd.DataFrame(rows).sort_values("Test_ROC_AUC", ascending=False)

    best_name = metrics_df.iloc[0]["Model"]
    best_pipeline = results[best_name]["pipeline"]

    return best_name, best_pipeline, metrics_df


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  BANK CHURN RISK SCORING — MODEL TRAINING")
    print("="*60)

    # 1. Load data
    print("\n[1/6] Loading data...")
    df = load_data()
    print(f"  Rows: {len(df):,}  |  Churn rate: {df[TARGET].mean():.2%}")

    # 2. Feature engineering
    print("\n[2/6] Engineering features...")
    df = add_engineered_features(df)

    # 3. Train/test split
    print("\n[3/6] Splitting data (80/20 stratified)...")
    feature_cols = [c for c in df.columns if c != TARGET and c != "Age_Band"]
    X = df[feature_cols]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  Train: {len(X_train):,}  |  Test: {len(X_test):,}")
    print(f"  Train churn rate: {y_train.mean():.2%}  |  Test churn rate: {y_test.mean():.2%}")

    # Determine available numeric features
    numeric_features = get_available_numeric(X_train)

    # 4. Train all models
    print("\n[4/6] Training all models (this may take several minutes)...")
    results = train_all_models(
        X_train, y_train,
        numeric_features=numeric_features,
        n_iter=30,
        cv_folds=5,
        verbose=1,
    )

    # 5. Select best model
    print("\n[5/6] Selecting best model...")
    best_name, best_pipeline, metrics_df = select_best_model(results, X_test, y_test)
    print(f"\n  Best model: {best_name}")
    print("\nModel comparison (by Test ROC-AUC):")
    print(metrics_df.to_string(index=False))

    # 6. Save artifacts
    print("\n[6/6] Saving artifacts...")

    # Save all pipelines
    all_pipelines = {name: res["pipeline"] for name, res in results.items()}
    joblib.dump(all_pipelines, os.path.join(MODELS_DIR, "all_pipelines.joblib"))

    # Save best pipeline
    joblib.dump(best_pipeline, os.path.join(MODELS_DIR, "best_model_pipeline.joblib"))

    # Save split data for downstream modules
    X_train.to_csv(os.path.join(DATA_PROCESSED, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(DATA_PROCESSED, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(DATA_PROCESSED, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(DATA_PROCESSED, "y_test.csv"), index=False)

    # Save metrics summary
    metrics_df.to_csv(os.path.join(DATA_PROCESSED, "model_comparison_initial.csv"), index=False)

    # Save training metadata
    metadata = {
        "best_model": best_name,
        "numeric_features": numeric_features,
        "categorical_features": CATEGORICAL_FEATURES,
        "random_state": RANDOM_STATE,
        "train_size": len(X_train),
        "test_size": len(X_test),
        "train_churn_rate": float(y_train.mean()),
        "test_churn_rate": float(y_test.mean()),
    }
    with open(os.path.join(MODELS_DIR, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n  Best model pipeline saved to: models/best_model_pipeline.joblib")
    print(f"  All pipelines saved to:       models/all_pipelines.joblib")
    print(f"  Metadata saved to:            models/training_metadata.json")
    print("\n[OK] Training complete!")

    return best_name, best_pipeline, results, X_train, X_test, y_train, y_test, numeric_features


if __name__ == "__main__":
    main()
