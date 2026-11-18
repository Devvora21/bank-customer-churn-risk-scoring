# Customer Segmentation & Churn Pattern Analytics in European Banking

## A Data-Driven Approach to Understanding and Mitigating Customer Attrition

---

## Abstract

Customer churn is a persistent challenge in the European banking sector, directly eroding revenue, increasing acquisition costs, and undermining long-term profitability. This study presents an exploratory data analysis (EDA) of 10,000 European bank customer records spanning three key markets — France, Germany, and Spain — to identify the demographic, financial, and behavioral factors most strongly associated with customer attrition. Through systematic data cleaning, feature engineering, and multi-dimensional segmentation, we find an overall churn rate of **20.37%**, with significant variation across customer segments. Germany exhibits a churn rate of **32.44%** — roughly double that of France (16.15%) and Spain (16.67%). Female customers churn at **25.07%** compared to **16.46%** for males. The 45–60 age group shows a strikingly high churn rate of **51.12%**, more than three times that of customers under 30. Customers with non-zero account balances churn at **24.08%** versus **13.82%** for zero-balance accounts, suggesting that high-value customers are disproportionately at risk. These findings are operationalized through an interactive Streamlit dashboard that enables real-time filtering, KPI monitoring, and data export — providing bank stakeholders with a practical tool for targeted retention strategy development.

**Keywords:** Customer Churn, European Banking, Customer Segmentation, Exploratory Data Analysis, Streamlit, Data Visualization

---

## 1. Introduction

### 1.1 Background

The European banking industry operates in an increasingly competitive landscape shaped by fintech disruption, regulatory evolution (e.g., PSD2, Open Banking), and shifting customer expectations. In this environment, customer retention has become a strategic imperative. Industry research consistently shows that acquiring a new customer costs five to seven times more than retaining an existing one (Reichheld & Sasser, 1990), making churn reduction one of the highest-ROI activities available to financial institutions.

Customer churn — defined as the discontinuation of a customer's relationship with a bank — can stem from a variety of factors: dissatisfaction with service quality, better offers from competitors, life-stage transitions, or lack of engagement with banking products. Understanding *which* customers churn and *why* is essential for designing effective, cost-efficient retention interventions.

### 1.2 Problem Statement

Despite the strategic importance of churn reduction, many banks still rely on reactive approaches — responding to churn after it occurs rather than proactively identifying at-risk customers. This project addresses this gap by performing a comprehensive, data-driven analysis of churn patterns across multiple customer dimensions to answer the following research questions:

1. **What is the overall churn rate, and how does it vary across geographies?**
2. **Which demographic segments (age, gender) exhibit the highest churn propensity?**
3. **How do financial characteristics (credit score, account balance, tenure) correlate with churn?**
4. **Can customer segmentation reveal actionable patterns for targeted retention?**

### 1.3 Objectives

- Clean and validate a dataset of 10,000 European banking customers
- Engineer meaningful customer segments from raw features
- Quantify churn rates across all segments
- Build interactive visualizations and a real-time dashboard for stakeholder use
- Derive actionable recommendations for churn mitigation

---

## 2. Literature Review

### 2.1 Customer Churn in Banking

Customer churn in the banking sector has been widely studied across both academic and industry research. Keramati et al. (2014) identified customer demographics, transaction behavior, and service usage patterns as primary churn predictors in telecommunications — findings that are directly analogous to banking. In the banking context, Xie et al. (2009) demonstrated that customer satisfaction, number of products held, and account tenure significantly influence retention.

### 2.2 Customer Segmentation Approaches

Customer segmentation — the practice of dividing a customer base into groups with shared characteristics — is a foundational technique in customer analytics. Common segmentation bases include:

- **Demographic segmentation:** Age, gender, geography
- **Behavioral segmentation:** Product usage, transaction frequency, tenure
- **Value-based segmentation:** Account balance, estimated income, profitability

RFM (Recency, Frequency, Monetary) analysis and clustering techniques (K-Means, DBSCAN) are frequently applied in the literature. This study employs rule-based segmentation using domain-informed bin boundaries, which offers the advantages of interpretability and direct business applicability.

### 2.3 Exploratory Data Analysis as a Decision Tool

While machine learning models (Logistic Regression, Random Forest, Gradient Boosting) dominate the predictive churn literature, exploratory data analysis remains a critical first step. Tukey (1977) established EDA as a disciplined approach to data investigation that reveals structure, extracts important variables, and detects anomalies. For bank management audiences, well-designed EDA with clear visualizations can often drive faster decision-making than complex model outputs.

---

## 3. Methodology

### 3.1 Data Collection

The dataset comprises **10,000 records** of European bank customers from the year **2025**, with **14 original features** covering customer demographics, financial status, product engagement, and churn outcome. The data spans three geographic markets: France, Germany, and Spain.

### 3.2 Data Exploration & Validation

The following validation steps were performed:

| Check | Result |
|-------|--------|
| Shape | 10,000 rows × 14 columns |
| Missing values | 0 across all columns |
| Data types | Correct (int64, float64, string) |
| Zero/negative values | Found in `Tenure` (413), `Balance` (3,617), `HasCrCard` (2,945), `IsActiveMember` (4,849), `Exited` (7,963) |

**Key decisions:**
- Zero values in binary columns (`HasCrCard`, `IsActiveMember`, `Exited`) are legitimate and were retained
- Zero balance is a valid financial state and was retained as a distinct segment (`BalanceSegment = "Zero"`)
- Zero/negative values in `CreditScore`, `Age`, and `EstimatedSalary` were replaced with the median of positive values

### 3.3 Data Cleaning

```python
# Median imputation for invalid values
cols_to_fix = ['CreditScore', 'Age', 'EstimatedSalary']
for col in cols_to_fix:
    median_val = df[df[col] > 0][col].median()
    df[col] = df[col].apply(lambda x: median_val if x <= 0 else x)

# Drop non-analytical columns
df.drop(columns=['Surname', 'CustomerId'], inplace=True)
```

**Imputed medians:**
| Column | Median Value |
|--------|-------------|
| CreditScore | 652.0 |
| Age | 37.0 |
| EstimatedSalary | 100,193.92 |

### 3.4 Feature Engineering

Four new categorical features were engineered to enable segment-level analysis:

| Feature | Bins | Labels |
|---------|------|--------|
| `AgeGroup` | 0–30, 30–45, 45–60, 60–100 | `<30`, `30-45`, `45-60`, `60+` |
| `CreditBand` | 0–580, 580–670, 670–850 | `Low`, `Medium`, `High` |
| `BalanceSegment` | 0, >0 | `Zero`, `High` |
| `TenureGroup` | -1–2, 2–5, 5–10 | `New`, `MidTerm`, `LongTerm` |

### 3.5 Analysis Framework

Churn rate was computed for each segment using:

```
Churn Rate (%) = (Number of Churned Customers in Segment / Total Customers in Segment) × 100
```

All analysis was performed using **Python 3.x** with **Pandas** for data manipulation and **Plotly Express** for interactive visualization. The interactive dashboard was built using **Streamlit**.

### 3.6 Tools & Technologies

| Tool | Version | Purpose |
|------|---------|---------|
| Python | 3.x | Core programming language |
| Pandas | ≥2.0 | Data manipulation & analysis |
| NumPy | ≥1.24 | Numerical computation |
| Plotly Express | ≥5.15 | Interactive visualizations |
| Matplotlib | ≥3.7 | Static plotting |
| Seaborn | ≥0.12 | Statistical visualization |
| Streamlit | ≥1.28 | Interactive dashboard |
| Jupyter Notebook | — | Exploratory analysis environment |

---

## 4. Results & Analysis

### 4.1 Overall Churn Statistics

| Metric | Value |
|--------|-------|
| Total Customers | 10,000 |
| Churned Customers | 2,037 |
| Retained Customers | 7,963 |
| **Overall Churn Rate** | **20.37%** |

The 20.37% churn rate indicates that approximately one in five customers left the bank, representing a significant retention challenge.

### 4.2 Geographic Analysis

| Geography | Churn Count | Churn Rate (%) |
|-----------|-------------|----------------|
| France | 810 | 16.15 |
| **Germany** | **814** | **32.44** |
| Spain | 413 | 16.67 |

**Key Finding:** Germany's churn rate (32.44%) is approximately **double** that of France and Spain. Despite having a smaller customer base than France, Germany contributes a nearly equal absolute number of churned customers (814 vs. 810). This suggests systemic issues in the German market — potentially related to competitive intensity, service quality, or market-specific regulatory factors.

### 4.3 Gender Analysis

| Gender | Churn Count | Churn Rate (%) |
|--------|-------------|----------------|
| **Female** | **1,139** | **25.07** |
| Male | 898 | 16.46 |

**Key Finding:** Female customers churn at a rate **52% higher** than males (25.07% vs. 16.46%). This disparity warrants investigation into gender-specific service expectations, product offerings, and customer experience touchpoints.

### 4.4 Age Group Analysis

| Age Group | Churn Count | Churn Rate (%) |
|-----------|-------------|----------------|
| <30 | 148 | 7.52 |
| 30-45 | 932 | 15.74 |
| **45-60** | **842** | **51.12** |
| 60+ | 115 | 24.78 |

**Key Finding:** The 45–60 age group exhibits a dramatically elevated churn rate of **51.12%** — more than **3× the rate of the 30–45 group** and nearly **7× the rate of the <30 group**. This is the single most powerful demographic predictor of churn in the dataset. Possible explanations include:
- Life-stage transitions (retirement planning, mortgage completion)
- Accumulated dissatisfaction over years of service
- Increased financial sophistication leading to competitive switching
- Targeted offers from competitor banks for this high-value demographic

The 60+ group shows a moderated churn rate (24.78%), possibly reflecting higher switching costs or greater inertia among older customers.

### 4.5 Credit Score Analysis

| Credit Band | Churn Count | Churn Rate (%) |
|-------------|-------------|----------------|
| Low (≤580) | 530 | 22.15 |
| Medium (581-670) | 683 | 20.39 |
| High (>670) | 824 | 19.36 |

**Key Finding:** Credit score has a relatively **modest impact** on churn. The difference between the lowest and highest credit bands is only ~3 percentage points. This suggests that churn is driven more by service/relationship factors than by creditworthiness.

### 4.6 Balance Segment Analysis

| Balance Segment | Churn Count | Churn Rate (%) |
|-----------------|-------------|----------------|
| **High (>€0)** | **1,537** | **24.08** |
| Zero (€0) | 500 | 13.82 |

**Key Finding:** Customers with non-zero balances churn at nearly **double the rate** of zero-balance customers. This is a counter-intuitive but critically important finding — it means that **higher-value customers are more likely to leave**. These are the customers whose departure carries the greatest revenue impact, making them priority targets for retention programs.

### 4.7 Tenure Group Analysis

| Tenure Group | Churn Count | Churn Rate (%) |
|--------------|-------------|----------------|
| New (0-2 years) | 528 | 21.15 |
| MidTerm (3-5 years) | 625 | 20.76 |
| LongTerm (6-10 years) | 884 | 19.67 |

**Key Finding:** Tenure shows a **slight inverse relationship** with churn — longer-tenured customers churn marginally less. However, the difference is small (~1.5 percentage points across the full tenure range), suggesting that tenure alone does not confer strong retention protection.

---

## 5. Interactive Dashboard

An interactive Streamlit dashboard (`app.py`) was developed to operationalize the analysis findings. The dashboard enables bank stakeholders to explore churn patterns in real time.

### 5.1 Architecture

- **Backend:** Pandas for data processing with `@st.cache_data` for performance optimization
- **Frontend:** Streamlit with Plotly Express for interactive charts
- **Filtering:** Sidebar-based multi-select filters for Geography, Gender, and Age Group
- **Layout:** Wide layout with 4-tab organization

### 5.2 Dashboard Components

| Component | Description |
|-----------|-------------|
| **KPI Row** | Real-time metrics: Total Customers, Churned, Retained, Churn Rate |
| **Tab 1 — Overview** | Donut chart showing overall churn/retention split |
| **Tab 2 — Geography & Demographics** | Bar charts for churn by Geography, Gender, Age Group, Tenure Group |
| **Tab 3 — Financial Profile** | Credit Band and Balance Segment analysis + High-Value Customer Explorer |
| **Tab 4 — Drill-Down Data** | Full interactive data table with CSV download |

### 5.3 Key Design Decisions

1. **Filter-driven updates:** All visualizations and KPIs update dynamically based on sidebar selections
2. **High-Value Customer Explorer:** Dedicated section for monitoring churn among high-balance customers
3. **Data export:** Built-in CSV download for filtered datasets, enabling further offline analysis
4. **Empty state handling:** Graceful warning when filter combinations yield no results

---

## 6. Discussion

### 6.1 Synthesis of Findings

The analysis reveals a **multi-dimensional churn profile** for European banking customers. The highest-risk customer persona emerges as:

> **A female customer, aged 45–60, based in Germany, with a non-zero account balance**

This persona combines the highest-risk attributes across all analyzed dimensions. While this represents the extreme case, each individual risk factor independently elevates churn probability.

### 6.2 The High-Value Churn Paradox

Perhaps the most strategically significant finding is the **positive correlation between account balance and churn** (24.08% for non-zero vs. 13.82% for zero-balance). This creates a paradoxical situation where the customers most valuable to the bank are also the most likely to leave. Possible explanations include:

- **Higher expectations:** Customers with larger balances may expect premium service and become dissatisfied more easily
- **Greater financial literacy:** These customers may be more aware of competitive alternatives
- **Attractive to competitors:** High-balance customers are likely targeted aggressively by competing institutions
- **Lower switching costs relative to balance size:** The perceived benefit of switching may outweigh the hassle for customers with significant assets

### 6.3 The Germany Question

Germany's anomalous churn rate (32.44% vs. ~16% for France and Spain) demands explanation. Potential contributing factors include:

- **Higher banking competition** in the German market (numerous Sparkassen, Volksbanken, and digital-only banks)
- **Cultural factors** around banking loyalty and willingness to switch
- **Regulatory environment** differences that may facilitate or incentivize switching
- **Service quality gaps** specific to the German operation

### 6.4 Limitations

1. **Single time point:** The dataset captures a single year (2025), limiting the ability to assess trends
2. **No causal inference:** This is a correlational analysis; observed associations do not establish causation
3. **Missing variables:** Customer satisfaction scores, complaint history, digital engagement metrics, and competitive offers are not captured
4. **Binary balance segmentation:** The balance feature was segmented into only two categories (Zero/High), which may miss nuances at intermediate balance levels
5. **No predictive modeling:** This study focuses on descriptive/diagnostic analytics; predictive models (Logistic Regression, Random Forest, XGBoost) could quantify individual churn probability

---

## 7. Recommendations

Based on the analysis, we recommend the following retention strategies:

### 7.1 Immediate Actions (0–3 months)

| # | Recommendation | Target Segment | Expected Impact |
|---|---------------|----------------|-----------------|
| 1 | Launch dedicated retention program for Germany | German customers | Address 32.44% churn rate |
| 2 | Implement proactive outreach for 45–60 age group | Age 45-60 | Reduce 51.12% churn rate |
| 3 | Design female-specific engagement programs | Female customers | Address 25.07% churn rate |

### 7.2 Medium-Term Initiatives (3–12 months)

| # | Recommendation | Rationale |
|---|---------------|-----------|
| 4 | Create high-value customer loyalty program | High-balance customers churn at 24.08% — protect revenue |
| 5 | Develop customer satisfaction surveys | Fill data gap on *why* customers churn |
| 6 | Build predictive churn model | Move from descriptive to predictive analytics |

### 7.3 Strategic Initiatives (12+ months)

| # | Recommendation | Rationale |
|---|---------------|-----------|
| 7 | Personalized product bundling | Increase products per customer to raise switching costs |
| 8 | Competitive benchmarking in German market | Understand why Germany's churn rate is 2× peer markets |
| 9 | Life-stage banking programs | Address the 45–60 age group's unique financial transition needs |

---

## 8. Future Work

1. **Predictive Modeling:** Train machine learning models (Logistic Regression, Random Forest, XGBoost) to predict individual churn probability
2. **Temporal Analysis:** Incorporate multi-year data to identify churn trends and seasonal patterns
3. **Customer Lifetime Value (CLV):** Calculate CLV to prioritize retention efforts by revenue impact
4. **Sentiment Analysis:** Integrate customer feedback/complaint data for qualitative churn driver analysis
5. **A/B Testing Framework:** Design controlled experiments to test retention intervention effectiveness
6. **Real-Time Scoring:** Deploy the churn model as an API for real-time risk scoring in CRM systems

---

## 9. Conclusion

This study demonstrates that customer churn in European banking is not uniformly distributed but is concentrated in identifiable, actionable segments. The 20.37% overall churn rate masks dramatic variation: from 7.52% among customers under 30 to 51.12% among those aged 45–60, and from 16.15% in France to 32.44% in Germany.

The finding that high-balance customers churn at disproportionately high rates (24.08%) underscores the urgency of targeted retention — the customers most likely to leave are the ones the bank can least afford to lose.

The interactive Streamlit dashboard developed as part of this project provides bank stakeholders with a practical, self-service tool for monitoring churn patterns and identifying at-risk segments in real time. By combining rigorous analysis with accessible visualization, this project bridges the gap between data science insight and business decision-making.

---

## References

1. Reichheld, F. F., & Sasser, W. E. (1990). Zero defections: Quality comes to services. *Harvard Business Review*, 68(5), 105–111.
2. Keramati, A., Jafari-Marandi, R., Aliannejadi, M., Ahmadian, I., Mozaffari, M., & Abbasi, U. (2014). Improved churn prediction in telecommunication industry using data mining techniques. *Applied Soft Computing*, 24, 994–1012.
3. Xie, Y., Li, X., Ngai, E. W. T., & Ying, W. (2009). Customer churn prediction using improved balanced random forests. *Expert Systems with Applications*, 36(3), 5445–5449.
4. Tukey, J. W. (1977). *Exploratory Data Analysis*. Addison-Wesley.
5. European Central Bank. (2024). *Report on the Payment Systems and Market Infrastructure*. ECB Publications.
6. Hadden, J., Tiwari, A., Roy, R., & Ruta, D. (2007). Computer assisted customer churn management: State-of-the-art and future trends. *Computers & Operations Research*, 34(10), 2902–2917.

---

## Appendix A: Descriptive Statistics

| Statistic | CreditScore | Age | Tenure | Balance | NumOfProducts | EstimatedSalary |
|-----------|-------------|-----|--------|---------|---------------|-----------------|
| Count | 10,000 | 10,000 | 10,000 | 10,000 | 10,000 | 10,000 |
| Mean | 650.53 | 38.92 | 5.01 | 76,485.89 | 1.53 | 100,090.24 |
| Std | 96.65 | 10.49 | 2.89 | 62,397.41 | 0.58 | 57,510.49 |
| Min | 350 | 18 | 0 | 0.00 | 1 | 11.58 |
| 25% | 584 | 32 | 3 | 0.00 | 1 | 51,002.11 |
| 50% | 652 | 37 | 5 | 97,198.54 | 1 | 100,193.92 |
| 75% | 718 | 44 | 7 | 127,644.24 | 2 | 149,388.25 |
| Max | 850 | 92 | 10 | 250,898.09 | 4 | 199,992.48 |

## Appendix B: Churn Summary Table

| Segment | Category | Churn Rate (%) | Risk Level |
|---------|----------|----------------|------------|
| Overall | — | 20.37 | Moderate |
| Geography | France | 16.15 | Low |
| Geography | Germany | 32.44 | **High** |
| Geography | Spain | 16.67 | Low |
| Gender | Female | 25.07 | Moderate-High |
| Gender | Male | 16.46 | Low |
| Age Group | <30 | 7.52 | Very Low |
| Age Group | 30-45 | 15.74 | Low |
| Age Group | 45-60 | 51.12 | **Critical** |
| Age Group | 60+ | 24.78 | Moderate-High |
| Credit Band | Low | 22.15 | Moderate |
| Credit Band | Medium | 20.39 | Moderate |
| Credit Band | High | 19.36 | Moderate |
| Balance | High (>€0) | 24.08 | Moderate-High |
| Balance | Zero (€0) | 13.82 | Low |
| Tenure | New | 21.15 | Moderate |
| Tenure | MidTerm | 20.76 | Moderate |
| Tenure | LongTerm | 19.67 | Moderate |

---

*This research paper was prepared as part of the European Banking Customer Churn Analysis project.*
*Author: Salah | Year: 2025*
