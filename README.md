# 🏦 European Bank Customer Churn Analytics & Risk Scoring Suite

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-green)](https://shap.readthedocs.io)

An enterprise-grade, end-to-end **Customer Segmentation, Churn Analytics & Predictive Risk Scoring Suite** for European retail banking. Combines exploratory customer segmentation, empirical churn pattern analytics, calibrated machine learning pipelines (XGBoost, Random Forest, Gradient Boosting), SHAP explainability, and counterfactual scenario simulation.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Findings](#key-findings)
- [Project Structure](#project-structure)
- [Customer Segmentation](#customer-segmentation)
- [Dashboard Workspaces](#dashboard-workspaces)
- [Quick Start](#quick-start)
- [Model Performance](#model-performance)
- [Research Documentation](#research-documentation)

---

## 🔍 Overview

Customer churn — the loss of retail banking clients to competitors — is one of the highest-leverage challenges in European finance. This project integrates:
1. **Exploratory Data Analysis & Segmentation**: Rigorous analysis of demographic, geographical, and balance segments across 10,000 European bank customers.
2. **Predictive Machine Learning**: 5 calibrated classification pipelines optimized for low financial misclassification cost (€500 False Negative vs. €50 False Positive).
3. **Interactive ChurnIQ Dashboard**: Real-time Streamlit suite with 9 specialized analytical workspaces.

---

## 📊 Key Findings

- **Territorial Divergence**: German accounts exhibit a **32.44% churn rate** — double that of France (16.15%) and Spain (16.67%).
- **Demographic Vulnerability**: The **45–60 age group** exhibits a **51.12% churn rate**, more than 3x higher than accounts under 30.
- **Gender Disparity**: Female clients churn at **25.07%** vs. **16.46%** for male clients.
- **High-Value Risk**: Customers with non-zero account balances churn at **24.08%** versus **13.82%** for zero-balance accounts, demonstrating that high-net-worth accounts are disproportionately vulnerable.

---

## 📁 Project Structure

```
churn-risk-scoring/
├── data/
│   ├── raw/European_Bank.csv          # Raw dataset (10,000 records)
│   └── processed/                     # Scored cohort, train/test splits, benchmark metrics
├── notebooks/
│   ├── European_Banking_EDA.ipynb     # Interactive EDA & segmentation notebook (Template)
│   ├── 01_eda.ipynb                   # Feature distribution & correlation analysis
│   └── 02_modeling.ipynb              # ML model comparison walkthrough
├── src/
│   ├── preprocessing.py               # Data cleaning & sklearn preprocessing pipeline
│   ├── feature_engineering.py         # Domain feature engineering & segment flags
│   ├── train.py                       # Multi-model training (XGBoost, RF, GBDT, DT, LR)
│   ├── evaluate.py                    # Cross-validated evaluation suite & figures
│   ├── explain.py                     # Global & individual SHAP explainability
│   └── predict.py                     # Batch scoring & risk tier categorization
├── models/
│   ├── best_model_pipeline.joblib     # Calibrated production pipeline
│   ├── all_pipelines.joblib           # Multi-model serialized checkpoints
│   └── training_metadata.json         # Metadata + optimal decision threshold (0.12)
├── reports/
│   ├── RESEARCH_PAPER.md/.pdf         # Formal academic research study (Template)
│   ├── research_paper.md/.docx        # Technical modeling audit report
│   └── figures/                       # High-resolution benchmark & ROC curves
├── app/
│   ├── app.py                         # 9-workspace Streamlit platform (ChurnIQ)
│   └── utils.py                       # Segment aggregations, caching & SHAP helpers
├── requirements.txt                   # Dependency pins
└── README.md
```

---

## 👥 Customer Segmentation

| Segment Dimension | Buckets / Classes | Primary Finding |
|---|---|---|
| **Geography** | France, Germany, Spain | Germany churn rate apex at 32.44% |
| **Gender** | Female, Male | Female churn 25.07% vs Male 16.46% |
| **Age Group** | `<30`, `30-45`, `45-60`, `60+` | 45-60 apex vulnerability at 51.12% |
| **Credit Band** | `Low` (≤580), `Medium` (581-670), `High` (>670) | Low credit band accounts churn at higher frequency |
| **Balance Segment** | `Zero` (€0), `High` (>€0) | High-balance accounts churn at 24.08% vs 13.82% |
| **Tenure Group** | `New` (0-2y), `MidTerm` (3-5y), `LongTerm` (6-10y) | New relationships require early anchoring |

---

## 🖥️ Dashboard Workspaces

1. **Customer Risk Overview**: Executive KPIs, overall churn split donut chart, and territorial breakdown.
2. **Geography & Demographics**: Segment-level empirical churn analytics by territory, age, gender, and tenure.
3. **Financial Profile**: Credit score bands, deposit balance segments, and High-Value Customer Explorer (>€100k at-risk balances).
4. **Single Customer Risk Calculator**: Real-time customer inference with localized SHAP waterfalls and prescriptive remediation playbooks.
5. **Customer Portfolio Explorer**: Multi-dimensional search, filter, and CSV data export for all 10,000 accounts.
6. **Probability Density Distribution**: Empirical cohort risk separation and decision threshold calibration.
7. **Global Feature Importance**: Global SHAP bar importance and dynamic Beeswarm summary plot.
8. **What-If Scenario Simulator**: Interactive counterfactual sandbox with parametric sensitivity curves.
9. **Model Performance & Fairness**: Cross-validation benchmarks, discrimination curves, and EBA demographic parity audit.

---

## 🚀 Quick Start

### 1. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 2. Launch the Streamlit Platform

```powershell
streamlit run app/app.py --server.port=8502
```

Navigate to `http://localhost:8502` in your browser.

---

## Dataset

**File:** `data/raw/European_Bank.csv`  
**Rows:** 10,001 customers  
**Features:** CreditScore, Geography, Gender, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary  
**Target:** `Exited` (1 = Churned, 0 = Retained)  
**Churn rate:** ~20.4%

---

## Model Performance

| Model | ROC-AUC (Test) | F1-Score |
|---|---|---|
| XGBoost ⭐ | **0.87+** | See evaluation output |
| Gradient Boosting | 0.85+ | See evaluation output |
| Random Forest | 0.84+ | See evaluation output |
| Logistic Regression | 0.76+ | See evaluation output |
| Decision Tree | 0.74+ | See evaluation output |

*Exact metrics available in `data/processed/model_comparison_full.csv` after running `evaluate.py`.*

---

## Engineered Features

| Feature | Formula | Motivation |
|---|---|---|
| `Balance_to_Salary` | Balance / (Salary + ε) | Relative wealth concentration |
| `Products_per_Tenure` | NumProducts / (Tenure + 1) | Product adoption pace |
| `Active_Products` | IsActive × NumProducts | Engagement × utilisation interaction |
| `Age_Tenure_Interaction` | Age × Tenure | Loyalty-age compound |
| `Tenure_Age_Ratio` | Tenure / (Age + 1) | Proportion of adult life with bank |
| `Zero_Balance_Flag` | 1 if Balance == 0 | Disengagement signal |

---

## Dashboard Features

The Streamlit app has 5 tabs:

1. **🎯 Risk Calculator** — Enter customer details, get churn probability gauge, risk tier, and SHAP explanations
2. **📊 Probability Distribution** — KDE/histogram of scores by actual outcome with threshold line
3. **🔍 Feature Importance** — Global SHAP bar chart, beeswarm summary, PDP selector
4. **🔮 What-If Simulator** — Adjust features with sliders, compare original vs modified probability
5. **📈 Model Performance** — Comparison table, ROC/PR curves, confusion matrices, fairness analysis

---

## Top Churn Drivers

1. **IsActiveMember** — Inactive customers churn at 2× the rate
2. **NumOfProducts** — 1-product customers: 28% churn; 2-product: 8% churn
3. **Age (45–60 cohort)** — Peak churn risk in middle-age band
4. **Balance** — Zero-balance and very-high-balance customers are high risk
5. **Geography: Germany** — ~2× higher churn rate than France/Spain

---

## Risk Tiers

| Tier | Probability | Recommended Action |
|---|---|---|
| 🟢 Low | < 30% | Standard quarterly engagement |
| 🟡 Medium | 30–60% | Targeted campaign, digital outreach |
| 🔴 High | > 60% | Priority intervention, personal RM contact |

---

## Reproducibility

All random seeds are set to `42`. The model pipeline wraps all transformations (scaling, encoding, feature engineering) ensuring identical processing at inference time.

---

## Requirements

See `requirements.txt`. Key dependencies:
- `scikit-learn` — ML pipelines and evaluation
- `xgboost` — Best model
- `shap` — Explainability
- `imbalanced-learn` — SMOTE for class imbalance
- `streamlit` — Dashboard
- `plotly` — Interactive charts
- `python-docx` — Report generation

---

## Citation

If using this project for research:

```
Unifield Mentor Analytics (2026). Predictive Modeling and Risk Scoring for 
Bank Customer Churn. European Bank Customer Dataset (N=10,001).
```
