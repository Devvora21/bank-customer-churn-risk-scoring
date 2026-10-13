"""
evaluate.py
===========
Comprehensive model evaluation:
  - Full metrics comparison table (Accuracy, Precision, Recall, F1, ROC-AUC)
  - Confusion matrices for all models
  - ROC curves overlay
  - Precision-Recall curves
  - Threshold analysis with business cost justification
  - Fairness check across Geography and Gender

Usage:
    python src/evaluate.py
"""

import os
import sys
import json
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    average_precision_score,
    classification_report,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

warnings.filterwarnings("ignore")

# ── Path bootstrap ────────────────────────────
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from preprocessing import load_data, TARGET, CATEGORICAL_FEATURES
from feature_engineering import add_engineered_features

# ── Paths ─────────────────────────────────────
FIGURES_DIR = os.path.join(ROOT_DIR, "reports", "figures")
MODELS_DIR = os.path.join(ROOT_DIR, "models")
DATA_PROCESSED = os.path.join(ROOT_DIR, "data", "processed")
os.makedirs(FIGURES_DIR, exist_ok=True)

RANDOM_STATE = 42

# ── Plot style ────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "font.family": "sans-serif",
    "font.size": 11,
})
PALETTE = ["#2563EB", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6"]


# ──────────────────────────────────────────────
# Metrics computation
# ──────────────────────────────────────────────

def compute_metrics(
    name: str,
    pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
    cv_folds: int = 5,
) -> dict:
    """
    Compute full evaluation metrics for one model pipeline.

    Returns a dict with Accuracy, Precision, Recall, F1, ROC-AUC
    on the test set, plus CV mean ± std.
    """
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)

    # Cross-validation on training set
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    cv_results = cross_validate(
        pipeline, X_train, y_train,
        cv=cv,
        scoring={"roc_auc": "roc_auc", "f1": "f1"},
        n_jobs=-1,
    )

    return {
        "Model": name,
        "Accuracy": round(accuracy_score(y_test, y_pred), 4),
        "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
        "F1_Score": round(f1_score(y_test, y_pred, zero_division=0), 4),
        "ROC_AUC_Test": round(roc_auc_score(y_test, y_prob), 4),
        "ROC_AUC_CV_Mean": round(cv_results["test_roc_auc"].mean(), 4),
        "ROC_AUC_CV_Std": round(cv_results["test_roc_auc"].std(), 4),
        "F1_CV_Mean": round(cv_results["test_f1"].mean(), 4),
        "F1_CV_Std": round(cv_results["test_f1"].std(), 4),
        "Avg_Precision": round(average_precision_score(y_test, y_prob), 4),
        "Threshold": threshold,
    }


def build_metrics_table(
    pipelines: dict,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """Build and return the full model comparison DataFrame."""
    rows = []
    for name, pipeline in pipelines.items():
        print(f"  Evaluating: {name}...")
        rows.append(compute_metrics(name, pipeline, X_train, y_train, X_test, y_test, threshold))
    df = pd.DataFrame(rows).sort_values("ROC_AUC_Test", ascending=False)
    return df


# ──────────────────────────────────────────────
# Plot functions
# ──────────────────────────────────────────────

def plot_confusion_matrices(pipelines: dict, X_test, y_test, threshold=0.5):
    """Plot confusion matrix for every model in a single figure."""
    n = len(pipelines)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5))
    if n == 1:
        axes = [axes]

    for ax, (name, pipeline), color in zip(axes, pipelines.items(), PALETTE):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        cm = confusion_matrix(y_test, y_pred)

        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Retained", "Churned"],
            yticklabels=["Retained", "Churned"],
        )
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_xlabel("Predicted", fontsize=10)
        ax.set_ylabel("Actual", fontsize=10)

    fig.suptitle("Confusion Matrices — All Models", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "confusion_matrices.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_roc_curves(pipelines: dict, X_test, y_test):
    """Overlay ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot([0, 1], [0, 1], "k--", lw=1.5, label="Random Classifier (AUC=0.50)")

    for (name, pipeline), color in zip(pipelines.items(), PALETTE):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, lw=2.5, color=color, label=f"{name} (AUC={auc:.4f})")

    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate (Recall)", fontsize=12)
    ax.set_title("ROC Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "roc_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


def plot_pr_curves(pipelines: dict, X_test, y_test):
    """Overlay Precision-Recall curves for all models."""
    fig, ax = plt.subplots(figsize=(9, 7))
    baseline = y_test.mean()
    ax.axhline(baseline, color="k", linestyle="--", lw=1.5, label=f"Baseline (random) = {baseline:.3f}")

    for (name, pipeline), color in zip(pipelines.items(), PALETTE):
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        prec, rec, _ = precision_recall_curve(y_test, y_prob)
        ap = average_precision_score(y_test, y_prob)
        ax.plot(rec, prec, lw=2.5, color=color, label=f"{name} (AP={ap:.4f})")

    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curves — All Models", fontsize=14, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "pr_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    return path


def threshold_analysis(
    best_pipeline,
    best_name: str,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> float:
    """
    Analyze classification threshold vs business cost.

    Business assumptions (European retail bank):
      - Cost of a false positive (unnecessary retention offer): €50
      - Cost of a false negative (losing a customer):           €500
        (conservative CLV for a mid-tier retail customer)
      - Cost ratio FN/FP = 10, so we should bias toward higher recall.

    Finds the threshold minimising total business cost.
    """
    y_prob = best_pipeline.predict_proba(X_test)[:, 1]
    thresholds = np.linspace(0.05, 0.90, 200)

    fp_costs, fn_costs, total_costs, f1s, recalls, precisions = [], [], [], [], [], []
    COST_FP = 50    # EUR
    COST_FN = 500   # EUR

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        cm = confusion_matrix(y_test, y_pred)
        tn, fp, fn, tp = cm.ravel()
        fp_cost = fp * COST_FP
        fn_cost = fn * COST_FN
        total = fp_cost + fn_cost
        fp_costs.append(fp_cost)
        fn_costs.append(fn_cost)
        total_costs.append(total)
        f1s.append(f1_score(y_test, y_pred, zero_division=0))
        recalls.append(recall_score(y_test, y_pred, zero_division=0))
        precisions.append(precision_score(y_test, y_pred, zero_division=0))

    optimal_idx = np.argmin(total_costs)
    optimal_threshold = thresholds[optimal_idx]

    # Plot
    fig, axes = plt.subplots(2, 1, figsize=(10, 10))

    # Cost curves
    ax = axes[0]
    ax.plot(thresholds, fp_costs, label=f"FP Cost (€{COST_FP}/each)", color="#3B82F6", lw=2)
    ax.plot(thresholds, fn_costs, label=f"FN Cost (€{COST_FN}/each)", color="#EF4444", lw=2)
    ax.plot(thresholds, total_costs, label="Total Cost", color="#1F2937", lw=2.5, linestyle="--")
    ax.axvline(optimal_threshold, color="#10B981", linestyle=":", lw=2.5,
               label=f"Optimal threshold = {optimal_threshold:.2f}")
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Business Cost (EUR)", fontsize=12)
    ax.set_title(f"Business Cost vs Threshold — {best_name}", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)

    # Metrics vs threshold
    ax = axes[1]
    ax.plot(thresholds, recalls, label="Recall", color="#EF4444", lw=2)
    ax.plot(thresholds, precisions, label="Precision", color="#3B82F6", lw=2)
    ax.plot(thresholds, f1s, label="F1-Score", color="#10B981", lw=2.5)
    ax.axvline(optimal_threshold, color="#8B5CF6", linestyle=":", lw=2.5,
               label=f"Optimal threshold = {optimal_threshold:.2f}")
    ax.set_xlabel("Threshold", fontsize=12)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Precision / Recall / F1 vs Threshold", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "threshold_analysis.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")
    print(f"\n  [Threshold Analysis]")
    print(f"  Optimal threshold (min business cost): {optimal_threshold:.2f}")
    print(f"  At threshold={optimal_threshold:.2f}: "
          f"Recall={recalls[optimal_idx]:.3f}, "
          f"Precision={precisions[optimal_idx]:.3f}, "
          f"F1={f1s[optimal_idx]:.3f}")

    return float(optimal_threshold)


# ──────────────────────────────────────────────
# Fairness check
# ──────────────────────────────────────────────

def fairness_check(
    best_pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    df_full: pd.DataFrame,
    threshold: float = 0.5,
) -> pd.DataFrame:
    """
    Compare model performance (ROC-AUC, Recall) across Geography and Gender subgroups.

    Returns
    -------
    pd.DataFrame with fairness metrics per subgroup.
    """
    # We need Geography and Gender from original df aligned with X_test
    # X_test preserves original index
    df_test = df_full.loc[X_test.index].copy()
    df_test["y_true"] = y_test.values
    y_prob = best_pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    df_test["y_prob"] = y_prob
    df_test["y_pred"] = y_pred

    rows = []
    for group_col in ["Geography", "Gender"]:
        for val in df_test[group_col].unique():
            mask = df_test[group_col] == val
            sub = df_test[mask]
            if sub["y_true"].nunique() < 2:
                continue
            auc = roc_auc_score(sub["y_true"], sub["y_prob"])
            rec = recall_score(sub["y_true"], sub["y_pred"], zero_division=0)
            prec = precision_score(sub["y_true"], sub["y_pred"], zero_division=0)
            f1 = f1_score(sub["y_true"], sub["y_pred"], zero_division=0)
            churn_rate = sub["y_true"].mean()
            rows.append({
                "Group": group_col,
                "Value": val,
                "N": len(sub),
                "Churn_Rate": round(churn_rate, 4),
                "ROC_AUC": round(auc, 4),
                "Recall": round(rec, 4),
                "Precision": round(prec, 4),
                "F1": round(f1, 4),
            })

    fairness_df = pd.DataFrame(rows)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, group_col in zip(axes, ["Geography", "Gender"]):
        sub = fairness_df[fairness_df["Group"] == group_col]
        x = np.arange(len(sub))
        width = 0.35
        bars1 = ax.bar(x - width/2, sub["ROC_AUC"], width, label="ROC-AUC", color="#2563EB", alpha=0.85)
        bars2 = ax.bar(x + width/2, sub["Recall"], width, label="Recall", color="#EF4444", alpha=0.85)
        ax.set_xticks(x)
        ax.set_xticklabels(sub["Value"], fontsize=11)
        ax.set_ylabel("Score", fontsize=11)
        ax.set_title(f"Fairness Check by {group_col}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=10)
        ax.set_ylim(0, 1.05)

        for bar in bars1:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)
        for bar in bars2:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                    f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "fairness_check.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    return fairness_df


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    print("\n" + "="*60)
    print("  BANK CHURN RISK SCORING — EVALUATION")
    print("="*60)

    # Load data
    print("\n[1/6] Loading data and models...")
    df = load_data()
    df = add_engineered_features(df)

    X_train = pd.read_csv(os.path.join(DATA_PROCESSED, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(DATA_PROCESSED, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(DATA_PROCESSED, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(DATA_PROCESSED, "y_test.csv")).squeeze()

    all_pipelines = joblib.load(os.path.join(MODELS_DIR, "all_pipelines.joblib"))
    best_pipeline = joblib.load(os.path.join(MODELS_DIR, "best_model_pipeline.joblib"))

    with open(os.path.join(MODELS_DIR, "training_metadata.json")) as f:
        metadata = json.load(f)
    best_name = metadata["best_model"]

    # Threshold analysis first to get optimal threshold
    print("\n[2/6] Threshold analysis (business cost optimisation)...")
    optimal_threshold = threshold_analysis(best_pipeline, best_name, X_test, y_test)

    # Full metrics table at optimal threshold
    print("\n[3/6] Computing full metrics table...")
    metrics_df = build_metrics_table(
        all_pipelines, X_train, y_train, X_test, y_test, threshold=optimal_threshold
    )
    print("\nModel Comparison Table:")
    print(metrics_df.to_string(index=False))
    metrics_df.to_csv(os.path.join(DATA_PROCESSED, "model_comparison_full.csv"), index=False)

    # Confusion matrices
    print("\n[4/6] Plotting confusion matrices...")
    plot_confusion_matrices(all_pipelines, X_test, y_test, threshold=optimal_threshold)

    # ROC curves
    print("\n[5/6] Plotting ROC curves and PR curves...")
    plot_roc_curves(all_pipelines, X_test, y_test)
    plot_pr_curves(all_pipelines, X_test, y_test)

    # Fairness check
    print("\n[6/6] Fairness check...")
    # Need original df with same index structure
    df_with_eng = add_engineered_features(load_data())
    feature_cols = [c for c in df_with_eng.columns if c != TARGET and c != "Age_Band"]
    from sklearn.model_selection import train_test_split
    X_full = df_with_eng[feature_cols]
    _, X_test_ref, _, _ = train_test_split(
        X_full, df_with_eng[TARGET], test_size=0.2, random_state=RANDOM_STATE, stratify=df_with_eng[TARGET]
    )
    fairness_df = fairness_check(best_pipeline, X_test, y_test, df_with_eng, threshold=optimal_threshold)
    print("\nFairness Check Results:")
    print(fairness_df.to_string(index=False))
    fairness_df.to_csv(os.path.join(DATA_PROCESSED, "fairness_results.csv"), index=False)

    # Save optimal threshold
    with open(os.path.join(MODELS_DIR, "training_metadata.json")) as f:
        metadata = json.load(f)
    metadata["optimal_threshold"] = optimal_threshold
    with open(os.path.join(MODELS_DIR, "training_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    print("\n[OK] Evaluation complete!")
    return metrics_df, optimal_threshold, fairness_df


if __name__ == "__main__":
    main()
