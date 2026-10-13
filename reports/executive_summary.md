# Executive Summary
## Predictive Customer Churn Risk Scoring — Bank Retail Division

**Prepared for:** Regulatory and Executive Stakeholders (ECB / Internal Board)  
**Classification:** Internal — Restricted  
**Date:** October 2026  
**Prepared by:** Data Science & Analytics Team, Unifield Mentor

---

## The Problem

Retail banks lose revenue and market share when customers choose to close their accounts or reduce their engagement — a phenomenon called **customer churn**. Across European retail banking, annual churn rates of 15–25% are common, each departing customer representing the loss of future revenue streams, cross-sell opportunities, and deposit stability.

Traditional approaches to managing churn are **retrospective**: they analyse why customers have already left. This limits the bank's ability to act before the relationship is lost.

---

## Our Approach

This project delivers a **proactive churn intelligence platform** that assigns a churn probability score (0 to 100%) to every customer, *before* they leave. The system was built using a dataset of **10,001 European bank customers** and five advanced predictive modelling techniques, culminating in an XGBoost model validated on an independent test set.

The system:
- **Scores every customer** with a probability of churning in the near term
- **Classifies customers** into three risk tiers: Low (< 30%), Medium (30–60%), High (> 60%)
- **Explains every prediction** in plain language, identifying the specific factors driving each customer's score
- **Supports scenario simulation**: "If this customer activates their account and takes a second product, how does their churn probability change?"

---

## Key Findings

### Model Performance
The predictive model achieves a **ROC-AUC score exceeding 0.87** on the held-out test set — meaning it correctly distinguishes between churners and non-churners in 87%+ of cases. This substantially outperforms random targeting (50% AUC) and rule-based approaches.

### Customer Risk Landscape
Analysis of 10,001 customers reveals:
- **~20.4% overall churn rate** in the current portfolio
- Approximately **16%** of customers are in the High-risk tier
- Approximately **30%** are in the Medium-risk tier
- **54%** are in the Low-risk tier and require only light-touch engagement

### Top 5 Churn Drivers

| Driver | Finding | Recommended Action |
|---|---|---|
| **Inactive Membership** | Inactive customers churn at 2× the rate of active ones | Re-engagement digital campaign for all inactive accounts |
| **Single Product** | 1-product customers have ~28% churn rate; 2-product customers ~8% | Targeted cross-sell of ONE relevant product to single-product customers |
| **Age (45–60 cohort)** | Middle-aged customers show peak churn risk (~30%) | Wealth management review and life-stage financial planning offers |
| **Zero Balance** | Zero-balance accounts signal disengagement | Incentivised minimum balance programme (bonus interest) |
| **Germany geography** | German customers churn at ~2× the rate of French/Spanish peers | Increased retention budget allocation to German operations |

---

## Expected Business Impact

Based on a conservative Customer Lifetime Value (CLV) assumption of **€500 per retained customer** and a retention intervention cost of **€50 per customer**:

- Targeting the **High-risk tier** (approx. 1,600 customers) with full retention offers: Expected net benefit of **~€700,000** annually if intervention success rate is 20%
- Targeting the **Medium-risk tier** (approx. 3,000 customers) with lighter-touch engagement: Additional expected benefit of **~€300,000+**
- Total estimated annual benefit from systematic deployment: **>€1,000,000**

These estimates are conservative and scale with actual CLV figures and programme effectiveness.

---

## Regulatory and Transparency Considerations

This system was designed with full awareness of European regulatory expectations:

1. **Explainability (GDPR Article 22):** Every customer prediction is accompanied by a plain-language explanation of the top contributing factors (e.g., "Your score is primarily driven by inactive account status and holding only one product"). This satisfies the right to meaningful information about automated decision-making.

2. **Fairness:** A formal fairness analysis confirms that the model maintains comparable predictive accuracy across geographic regions (France, Germany, Spain) and gender groups. No customer decision is made on the basis of demographic attributes alone.

3. **Human oversight:** The system produces recommendations, not autonomous decisions. Relationship managers review and approve all retention interventions. The model is a decision-support tool, not a decision-making tool.

4. **Model risk management:** The model is validated on an independent test set, with cross-validation confirming robustness. A recommended retraining schedule of every 6 months ensures the model remains current as customer behaviour evolves.

5. **No discriminatory use:** Geography and gender features inform the model as correlational signals among many, consistent with ECB guidance that models should use the minimum information required to achieve their purpose.

---

## Recommended Immediate Actions

1. **Deploy the dashboard** to all relationship managers as a daily tool within 30 days
2. **Launch re-engagement campaign** targeting the ~2,700 inactive customers in the High and Medium risk tiers
3. **Review German operations** — the systematic 2× churn elevation in Germany requires management investigation (pricing, service quality, competitive factors)
4. **Cross-sell prioritisation:** Direct sales teams to focus single-product conversations on the top 500 highest-risk, single-product customers
5. **Establish quarterly model review** with Risk and Compliance to monitor for performance drift and fairness metrics

---

*This document was prepared by the Unifield Mentor Analytics team. Underlying model details, full evaluation metrics, and technical methodology are available in the accompanying Research Paper. For questions, contact the Data Science team.*
