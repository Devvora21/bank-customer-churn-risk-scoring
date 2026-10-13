# Predictive Modeling and Risk Scoring for Bank Customer Churn

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Dashboard-Streamlit-red)](https://streamlit.io)
[![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)](https://xgboost.readthedocs.io)
[![SHAP](https://img.shields.io/badge/Explainability-SHAP-green)](https://shap.readthedocs.io)

A complete, production-quality **predictive churn intelligence system** for retail banking. Predicts customer churn probability before it happens, enabling proactive retention campaigns.

---

## Project Structure

```
churn-risk-scoring/
├── data/
│   ├── raw/European_Bank.csv          # Raw dataset
│   └── processed/                     # Train/test splits, scored customers
├── notebooks/
│   ├── 01_eda.ipynb                   # Exploratory Data Analysis
│   └── 02_modeling.ipynb              # Modeling walkthrough
├── src/
│   ├── preprocessing.py               # Data loading + sklearn pipeline
│   ├── feature_engineering.py         # 7 engineered features
│   ├── train.py                       # Model training (5 models)
│   ├── evaluate.py                    # Full evaluation suite
│   ├── explain.py                     # SHAP explainability
│   └── predict.py                     # Batch scoring
├── models/
│   ├── best_model_pipeline.joblib     # Best trained pipeline
│   ├── all_pipelines.joblib           # All model pipelines
│   └── training_metadata.json         # Metadata + optimal threshold
├── reports/
│   ├── figures/                       # All generated plots
│   ├── research_paper.md/.docx        # Academic research paper
│   └── executive_summary.md/.docx     # Non-technical summary
├── app/
│   ├── app.py                         # Streamlit dashboard
│   └── utils.py                       # App utilities
├── generate_reports.py                # Markdown → .docx converter
├── requirements.txt                   # Pinned dependencies
└── README.md
```

---

## Quick Start

### 1. Install Dependencies

```powershell
cd churn-risk-scoring
pip install -r requirements.txt
```

### 2. Run the Full Pipeline

Execute in order:

```powershell
# Step 1: Train all models (~5–10 minutes)
python src/train.py

# Step 2: Evaluate all models + generate figures
python src/evaluate.py

# Step 3: SHAP explainability analysis
python src/explain.py

# Step 4: Score all customers
python src/predict.py

# Step 5: Generate .docx reports
python generate_reports.py
```

### 3. Launch the Streamlit Dashboard

```powershell
streamlit run app/app.py
```

The dashboard opens at `http://localhost:8501`.

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
