"""
app.py
======
Streamlit dashboard for Bank Customer Churn Risk Scoring.

Five tabs:
  1. Customer Churn Risk Calculator
  2. Probability Distribution
  3. Feature Importance Dashboard
  4. What-If Scenario Simulator
  5. Model Performance

Usage:
    streamlit run app/app.py
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# ── Path bootstrap ────────────────────────────
APP_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(APP_DIR)
SRC_DIR = os.path.join(ROOT_DIR, "src")
sys.path.insert(0, SRC_DIR)
sys.path.insert(0, APP_DIR)

from utils import (
    load_model_and_metadata, load_all_pipelines, load_scored_data,
    load_metrics_table, load_shap_importance, load_test_data,
    prepare_customer_input, predict_single,
    get_shap_explainer, get_individual_shap, get_retention_recommendations,
    TIER_COLORS, TIER_BG_COLORS,
)
from preprocessing import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET
from feature_engineering import add_engineered_features

FIGURES_DIR = os.path.join(ROOT_DIR, "reports", "figures")

# ──────────────────────────────────────────────
# Page configuration
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="ChurnSense — Bank Customer Churn Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Custom CSS ─────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.main-header {
    background: linear-gradient(135deg, #1E3A5F 0%, #2563EB 50%, #1E40AF 100%);
    padding: 2rem 2.5rem;
    border-radius: 16px;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 40px rgba(37, 99, 235, 0.3);
}

.main-header h1 {
    color: white;
    font-size: 2.2rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.5px;
}

.main-header p {
    color: rgba(255,255,255,0.85);
    font-size: 1.05rem;
    margin: 0.4rem 0 0;
}

.metric-card {
    background: white;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    box-shadow: 0 2px 16px rgba(0,0,0,0.08);
    border-left: 4px solid;
    margin-bottom: 0.5rem;
}

.risk-badge {
    display: inline-block;
    padding: 0.35rem 1rem;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.5px;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: #F1F5F9;
    padding: 6px;
    border-radius: 12px;
}

.stTabs [data-baseweb="tab"] {
    border-radius: 8px;
    font-weight: 500;
    padding: 0.5rem 1.2rem;
    font-size: 0.9rem;
}

.stTabs [aria-selected="true"] {
    background: #2563EB !important;
    color: white !important;
}

.insight-box {
    background: linear-gradient(135deg, #EFF6FF, #DBEAFE);
    border-radius: 10px;
    padding: 1rem 1.2rem;
    border-left: 4px solid #2563EB;
    margin: 0.5rem 0;
}

.recommendation-card {
    background: #F0FDF4;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin: 0.3rem 0;
    border-left: 3px solid #10B981;
}

div[data-testid="stMetric"] {
    background: white;
    border-radius: 10px;
    padding: 1rem;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}

.section-header {
    font-size: 1.15rem;
    font-weight: 700;
    color: #1E3A5F;
    margin-bottom: 0.5rem;
    padding-bottom: 0.3rem;
    border-bottom: 2px solid #E2E8F0;
}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Load data & models
# ──────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model pipeline...")
def load_resources():
    pipeline, metadata = load_model_and_metadata()
    return pipeline, metadata

pipeline, metadata = load_resources()
threshold = metadata.get("optimal_threshold", 0.5)
best_model_name = metadata.get("best_model", "Best Model")

scored_data = load_scored_data()
metrics_table = load_metrics_table()
shap_importance = load_shap_importance()


# ──────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────

st.markdown(f"""
<div class="main-header">
    <h1>🏦 ChurnSense — Customer Churn Intelligence Platform</h1>
    <p>Powered by {best_model_name} &nbsp;·&nbsp; Decision threshold: {threshold:.2f}
    &nbsp;·&nbsp; {len(scored_data):,} customers scored</p>
</div>
""", unsafe_allow_html=True)

# ── Top KPI strip ──────────────────────────────
if scored_data is not None:
    c1, c2, c3, c4, c5 = st.columns(5)
    total = len(scored_data)
    n_high = (scored_data["Risk_Tier"] == "High").sum()
    n_med = (scored_data["Risk_Tier"] == "Medium").sum()
    n_low = (scored_data["Risk_Tier"] == "Low").sum()
    churn_rate = scored_data["Churn_Predicted"].mean()

    c1.metric("👥 Total Customers", f"{total:,}")
    c2.metric("🔴 High Risk", f"{n_high:,}", f"{n_high/total:.1%}")
    c3.metric("🟡 Medium Risk", f"{n_med:,}", f"{n_med/total:.1%}")
    c4.metric("🟢 Low Risk", f"{n_low:,}", f"{n_low/total:.1%}")
    c5.metric("📉 Predicted Churn", f"{churn_rate:.1%}")

st.markdown("---")


# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎯 Risk Calculator",
    "📊 Probability Distribution",
    "🔍 Feature Importance",
    "🔮 What-If Simulator",
    "📈 Model Performance",
])


# ══════════════════════════════════════════════
# TAB 1: Customer Churn Risk Calculator
# ══════════════════════════════════════════════

with tab1:
    st.markdown('<div class="section-header">Customer Churn Risk Calculator</div>', unsafe_allow_html=True)
    st.caption("Enter a customer's profile below to receive an instant churn risk assessment with SHAP-based explanations.")

    with st.form("customer_form"):
        c1, c2, c3 = st.columns(3)

        with c1:
            st.markdown("**Account Details**")
            credit_score = st.slider("Credit Score", 300, 850, 650, step=1, key="cs")
            balance = st.number_input("Account Balance (€)", 0.0, 300000.0, 75000.0, step=1000.0, key="bal")
            estimated_salary = st.number_input("Estimated Salary (€)", 0.0, 250000.0, 85000.0, step=1000.0, key="sal")

        with c2:
            st.markdown("**Customer Profile**")
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"], key="geo")
            gender = st.selectbox("Gender", ["Female", "Male"], key="gen")
            age = st.slider("Age", 18, 92, 40, key="age")

        with c3:
            st.markdown("**Banking Behaviour**")
            tenure = st.slider("Tenure (years)", 0, 10, 5, key="ten")
            num_products = st.selectbox("Number of Products", [1, 2, 3, 4], key="nop")
            has_cr_card = st.radio("Has Credit Card?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True, key="hcc")
            is_active = st.radio("Active Member?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True, key="iam")

        submitted = st.form_submit_button("🔮 Predict Churn Risk", use_container_width=True, type="primary")

    if submitted:
        with st.spinner("Analysing customer profile..."):
            customer_df = prepare_customer_input(
                credit_score, geography, gender, age, tenure,
                balance, num_products, has_cr_card, is_active, estimated_salary,
            )
            result = predict_single(customer_df, pipeline, threshold)

        prob = result["probability"]
        tier = result["risk_tier"]
        tier_color = TIER_COLORS[tier]
        tier_bg = TIER_BG_COLORS[tier]

        # ── Result display ──────────────────────
        r1, r2 = st.columns([1.2, 1])

        with r1:
            st.markdown("#### 📊 Churn Risk Assessment")

            # Gauge chart
            gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=round(prob * 100, 1),
                number={"suffix": "%", "font": {"size": 48, "color": tier_color}},
                delta={"reference": threshold * 100, "increasing": {"color": "#EF4444"}, "decreasing": {"color": "#10B981"}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#334155"},
                    "bar": {"color": tier_color, "thickness": 0.35},
                    "bgcolor": "white",
                    "borderwidth": 2,
                    "bordercolor": "#E2E8F0",
                    "steps": [
                        {"range": [0, 30], "color": "#D1FAE5"},
                        {"range": [30, 60], "color": "#FEF3C7"},
                        {"range": [60, 100], "color": "#FEE2E2"},
                    ],
                    "threshold": {
                        "line": {"color": "#1E3A5F", "width": 3},
                        "thickness": 0.85,
                        "value": threshold * 100,
                    },
                },
                title={"text": "Churn Probability", "font": {"size": 16}},
            ))
            gauge.update_layout(
                height=320,
                margin=dict(l=30, r=30, t=40, b=10),
                paper_bgcolor="white",
            )
            st.plotly_chart(gauge, use_container_width=True)

            # Risk tier badge
            st.markdown(
                f"""<div style='text-align:center; margin-top:-10px;'>
                <span style='background:{tier_bg}; color:{tier_color}; border:2px solid {tier_color};
                padding:0.4rem 1.5rem; border-radius:20px; font-weight:700; font-size:1.1rem;'>
                ⚡ {tier} Risk
                </span>
                <span style='margin-left:1rem; background:{"#FEE2E2" if result["predicted"] else "#D1FAE5"};
                color:{"#EF4444" if result["predicted"] else "#10B981"};
                border:2px solid {"#EF4444" if result["predicted"] else "#10B981"};
                padding:0.4rem 1.2rem; border-radius:20px; font-weight:700; font-size:1rem;'>
                {"🚨 Likely to Churn" if result["predicted"] else "✅ Likely to Stay"}
                </span>
                </div>""",
                unsafe_allow_html=True,
            )

        with r2:
            st.markdown("#### 💡 Retention Recommendations")
            recs = get_retention_recommendations(result, customer_df)
            for rec in recs:
                st.markdown(
                    f"<div class='recommendation-card'>{rec}</div>",
                    unsafe_allow_html=True,
                )

        # ── SHAP explanation ────────────────────
        st.markdown("---")
        st.markdown("#### 🔍 Why This Prediction? (SHAP Explanation)")

        try:
            X_test, y_test = load_test_data()
            X_bg = X_test.sample(min(200, len(X_test)), random_state=42)
            explainer, feature_names, preprocessor = get_shap_explainer(pipeline, X_bg)
            shap_df = get_individual_shap(explainer, pipeline, customer_df, feature_names, preprocessor, top_n=10)

            fig_shap = go.Figure()
            colors = ["#EF4444" if v > 0 else "#10B981" for v in shap_df["SHAP_Value"]]
            fig_shap.add_trace(go.Bar(
                x=shap_df["SHAP_Value"],
                y=shap_df["Feature"],
                orientation="h",
                marker_color=colors,
                text=[f"{v:.4f}" for v in shap_df["SHAP_Value"]],
                textposition="outside",
            ))
            fig_shap.update_layout(
                title="SHAP Contributions (Red = Increases Churn Risk, Green = Decreases)",
                height=420,
                xaxis_title="SHAP Value (impact on log-odds of churn)",
                yaxis={"autorange": "reversed"},
                paper_bgcolor="white",
                plot_bgcolor="#F8FAFC",
                font=dict(family="Inter, sans-serif"),
                margin=dict(l=20, r=60, t=50, b=20),
            )
            st.plotly_chart(fig_shap, use_container_width=True)
        except Exception as e:
            st.info(f"SHAP explanation unavailable: {e}")


# ══════════════════════════════════════════════
# TAB 2: Probability Distribution
# ══════════════════════════════════════════════

with tab2:
    st.markdown('<div class="section-header">Churn Probability Distribution</div>', unsafe_allow_html=True)
    st.caption("Distribution of predicted churn probabilities across the full customer base, split by actual churn status.")

    if scored_data is not None:
        col1, col2 = st.columns([3, 1])

        with col1:
            fig = go.Figure()

            # Churned customers
            churned = scored_data[scored_data["Exited"] == 1]["Churn_Probability"]
            retained = scored_data[scored_data["Exited"] == 0]["Churn_Probability"]

            fig.add_trace(go.Histogram(
                x=retained, name="Retained (Actual)", nbinsx=50,
                marker_color="#10B981", opacity=0.65,
                histnorm="probability density",
            ))
            fig.add_trace(go.Histogram(
                x=churned, name="Churned (Actual)", nbinsx=50,
                marker_color="#EF4444", opacity=0.65,
                histnorm="probability density",
            ))

            # Threshold line
            fig.add_vline(
                x=threshold, line_dash="dash", line_color="#1E3A5F", line_width=2.5,
                annotation_text=f"Threshold = {threshold:.2f}",
                annotation_position="top right",
                annotation_font_size=13,
            )

            # Risk tier zones
            fig.add_vrect(x0=0, x1=0.30, fillcolor="#10B981", opacity=0.05, layer="below", line_width=0)
            fig.add_vrect(x0=0.30, x1=0.60, fillcolor="#F59E0B", opacity=0.08, layer="below", line_width=0)
            fig.add_vrect(x0=0.60, x1=1.0, fillcolor="#EF4444", opacity=0.07, layer="below", line_width=0)

            # Currently entered customer (from tab 1)
            if submitted and "prob" in dir():
                fig.add_vline(
                    x=prob, line_color="#8B5CF6", line_width=2,
                    annotation_text=f"This customer ({prob:.2f})",
                    annotation_position="top left",
                )

            fig.update_layout(
                barmode="overlay",
                title="Churn Probability Distribution by Actual Outcome",
                xaxis_title="Predicted Churn Probability",
                yaxis_title="Density",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                height=480,
                paper_bgcolor="white",
                plot_bgcolor="#F8FAFC",
                font=dict(family="Inter, sans-serif"),
            )
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.markdown("**Distribution Stats**")
            st.metric("Mean Probability", f"{scored_data['Churn_Probability'].mean():.3f}")
            st.metric("Median", f"{scored_data['Churn_Probability'].median():.3f}")
            st.metric("Churned Mean Prob", f"{scored_data[scored_data['Exited']==1]['Churn_Probability'].mean():.3f}")
            st.metric("Retained Mean Prob", f"{scored_data[scored_data['Exited']==0]['Churn_Probability'].mean():.3f}")

            st.markdown("---")
            st.markdown("**Tier Distribution**")
            tier_counts = scored_data["Risk_Tier"].value_counts()
            for tier_name, color in TIER_COLORS.items():
                count = tier_counts.get(tier_name, 0)
                pct = count / len(scored_data) * 100
                st.markdown(
                    f"<span style='color:{color}; font-weight:600'>●</span> **{tier_name}:** {count:,} ({pct:.1f}%)",
                    unsafe_allow_html=True,
                )

        # Churn rate by geography
        st.markdown("#### Churn Probability by Key Segments")
        seg_col1, seg_col2 = st.columns(2)

        with seg_col1:
            geo_stats = scored_data.groupby("Geography")["Churn_Probability"].mean().reset_index()
            fig_geo = px.bar(
                geo_stats, x="Geography", y="Churn_Probability",
                color="Churn_Probability", color_continuous_scale="RdYlGn_r",
                title="Mean Churn Probability by Geography",
                labels={"Churn_Probability": "Mean Prob"},
                text_auto=".3f",
            )
            fig_geo.update_layout(height=350, paper_bgcolor="white", plot_bgcolor="#F8FAFC",
                                   font=dict(family="Inter"))
            st.plotly_chart(fig_geo, use_container_width=True)

        with seg_col2:
            age_stats = scored_data.copy()
            age_stats["Age_Band"] = pd.cut(
                age_stats["Age"], bins=[0,30,40,50,60,120],
                labels=["18-30","31-40","41-50","51-60","60+"]
            )
            ab = age_stats.groupby("Age_Band", observed=True)["Churn_Probability"].mean().reset_index()
            fig_age = px.bar(
                ab, x="Age_Band", y="Churn_Probability",
                color="Churn_Probability", color_continuous_scale="RdYlGn_r",
                title="Mean Churn Probability by Age Band",
                labels={"Churn_Probability": "Mean Prob"},
                text_auto=".3f",
            )
            fig_age.update_layout(height=350, paper_bgcolor="white", plot_bgcolor="#F8FAFC",
                                   font=dict(family="Inter"))
            st.plotly_chart(fig_age, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 3: Feature Importance Dashboard
# ══════════════════════════════════════════════

with tab3:
    st.markdown('<div class="section-header">Feature Importance Dashboard</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 Global SHAP Feature Importance")
        if shap_importance is not None:
            top_n = st.slider("Show top N features", 5, 20, 12, key="shap_topn")
            top = shap_importance.head(top_n).sort_values("Mean_Abs_SHAP")
            fig_bar = go.Figure(go.Bar(
                x=top["Mean_Abs_SHAP"],
                y=top["Feature"],
                orientation="h",
                marker=dict(
                    color=top["Mean_Abs_SHAP"],
                    colorscale="Blues",
                    showscale=True,
                    colorbar=dict(title="Mean |SHAP|"),
                ),
                text=[f"{v:.4f}" for v in top["Mean_Abs_SHAP"]],
                textposition="outside",
            ))
            fig_bar.update_layout(
                height=500,
                xaxis_title="Mean |SHAP Value|",
                paper_bgcolor="white",
                plot_bgcolor="#F8FAFC",
                font=dict(family="Inter"),
                margin=dict(l=20, r=80, t=20, b=20),
            )
            st.plotly_chart(fig_bar, use_container_width=True)

    with col2:
        st.markdown("#### 🔍 SHAP Summary Plot")
        shap_summary_path = os.path.join(FIGURES_DIR, "shap_summary_beeswarm.png")
        if os.path.exists(shap_summary_path):
            st.image(shap_summary_path, use_column_width=True)
        else:
            st.info("Run `python src/explain.py` to generate SHAP plots.")

    # PDP selector
    st.markdown("---")
    st.markdown("#### 📈 Partial Dependence Plot")
    pdp_path = os.path.join(FIGURES_DIR, "partial_dependence.png")
    if os.path.exists(pdp_path):
        st.image(pdp_path, use_column_width=True)
        st.markdown(
            "<div class='insight-box'>💡 <b>Reading PDPs:</b> Each curve shows how churn probability "
            "changes as one feature varies, with all others held at their average. A rising curve "
            "means higher values increase churn risk.</div>",
            unsafe_allow_html=True,
        )
    else:
        st.info("Run `python src/explain.py` to generate Partial Dependence Plots.")

    # Permutation importance
    st.markdown("---")
    st.markdown("#### 🔄 Permutation Feature Importance")
    perm_path = os.path.join(FIGURES_DIR, "permutation_importance.png")
    if os.path.exists(perm_path):
        st.image(perm_path, use_column_width=True)
    else:
        st.info("Run `python src/explain.py` to generate permutation importance.")


# ══════════════════════════════════════════════
# TAB 4: What-If Scenario Simulator
# ══════════════════════════════════════════════

with tab4:
    st.markdown('<div class="section-header">What-If Scenario Simulator</div>', unsafe_allow_html=True)
    st.caption("Adjust customer attributes to see how interventions change churn probability in real time.")

    if not submitted:
        st.info("💡 First, enter a customer profile in the **Risk Calculator** tab and click **Predict Churn Risk**, then come back here to simulate scenarios.")
        base_vals = {
            "credit_score": 650, "geography": "France", "gender": "Female",
            "age": 42, "tenure": 5, "balance": 75000.0, "num_products": 1,
            "has_cr_card": 1, "is_active": 0, "salary": 85000.0,
        }
    else:
        base_vals = {
            "credit_score": credit_score, "geography": geography, "gender": gender,
            "age": age, "tenure": tenure, "balance": balance,
            "num_products": num_products, "has_cr_card": has_cr_card,
            "is_active": is_active, "salary": estimated_salary,
        }

    base_df = prepare_customer_input(**base_vals)
    base_result = predict_single(base_df, pipeline, threshold)
    base_prob = base_result["probability"]

    st.markdown("#### ⚙️ Scenario Adjustments")
    w1, w2, w3 = st.columns(3)

    with w1:
        new_active = st.radio(
            "Active Member Status",
            [1, 0], index=1 - base_vals["is_active"],
            format_func=lambda x: "✅ Active" if x else "❌ Inactive",
            key="wi_active",
        )
        new_products = st.selectbox(
            "Number of Products",
            [1, 2, 3, 4],
            index=base_vals["num_products"] - 1,
            key="wi_prod",
        )

    with w2:
        new_balance = st.slider(
            "Account Balance (€)", 0, 250000, int(base_vals["balance"]),
            step=5000, key="wi_bal",
        )
        new_tenure = st.slider(
            "Tenure (years)", 0, 10, base_vals["tenure"], key="wi_ten",
        )

    with w3:
        new_credit = st.slider(
            "Credit Score", 300, 850, base_vals["credit_score"], key="wi_cs",
        )
        new_age = st.slider("Age", 18, 92, base_vals["age"], key="wi_age")

    # Build modified scenario
    mod_vals = {**base_vals,
                "is_active": new_active, "num_products": new_products,
                "balance": float(new_balance), "tenure": new_tenure,
                "credit_score": new_credit, "age": new_age}
    mod_df = prepare_customer_input(**mod_vals)
    mod_result = predict_single(mod_df, pipeline, threshold)
    mod_prob = mod_result["probability"]

    delta = mod_prob - base_prob

    # Side-by-side comparison
    st.markdown("---")
    st.markdown("#### 📊 Comparison: Original vs Modified")
    cmp1, cmp2, cmp3 = st.columns(3)

    with cmp1:
        t = base_result["risk_tier"]
        st.markdown(f"""
        <div style='background:{TIER_BG_COLORS[t]}; border-radius:12px; padding:1.2rem; text-align:center; border:2px solid {TIER_COLORS[t]}'>
            <div style='font-size:0.85rem; color:#64748B; font-weight:500;'>ORIGINAL</div>
            <div style='font-size:2.5rem; font-weight:800; color:{TIER_COLORS[t]}'>{base_prob:.1%}</div>
            <div style='font-weight:600; color:{TIER_COLORS[t]}'>{t} Risk</div>
        </div>
        """, unsafe_allow_html=True)

    with cmp2:
        arrow = "🔻" if delta < -0.01 else ("🔺" if delta > 0.01 else "➡️")
        delta_color = "#10B981" if delta < 0 else ("#EF4444" if delta > 0 else "#6B7280")
        st.markdown(f"""
        <div style='background:#F8FAFC; border-radius:12px; padding:1.2rem; text-align:center; border:2px solid #E2E8F0'>
            <div style='font-size:0.85rem; color:#64748B; font-weight:500;'>CHANGE</div>
            <div style='font-size:2.5rem; font-weight:800; color:{delta_color}'>{arrow} {delta:+.1%}</div>
            <div style='font-weight:600; color:{delta_color}'>{"Risk Reduced ✓" if delta < -0.01 else ("Risk Increased ✗" if delta > 0.01 else "No Change")}</div>
        </div>
        """, unsafe_allow_html=True)

    with cmp3:
        t2 = mod_result["risk_tier"]
        st.markdown(f"""
        <div style='background:{TIER_BG_COLORS[t2]}; border-radius:12px; padding:1.2rem; text-align:center; border:2px solid {TIER_COLORS[t2]}'>
            <div style='font-size:0.85rem; color:#64748B; font-weight:500;'>MODIFIED</div>
            <div style='font-size:2.5rem; font-weight:800; color:{TIER_COLORS[t2]}'>{mod_prob:.1%}</div>
            <div style='font-weight:600; color:{TIER_COLORS[t2]}'>{t2} Risk</div>
        </div>
        """, unsafe_allow_html=True)

    # Probability across a range of one variable
    st.markdown("---")
    st.markdown("#### 📉 Probability Sensitivity Analysis")
    sweep_feature = st.selectbox(
        "Select feature to sweep:",
        ["IsActiveMember", "NumOfProducts", "Balance", "Tenure", "Age", "CreditScore"],
        key="sweep_feat",
    )

    if sweep_feature == "IsActiveMember":
        sweep_vals = [0, 1]
        sweep_labels = ["Inactive (0)", "Active (1)"]
    elif sweep_feature == "NumOfProducts":
        sweep_vals = [1, 2, 3, 4]
        sweep_labels = ["1", "2", "3", "4"]
    elif sweep_feature == "Balance":
        sweep_vals = list(range(0, 210000, 10000))
        sweep_labels = [f"€{v:,}" for v in sweep_vals]
    elif sweep_feature == "Tenure":
        sweep_vals = list(range(0, 11))
        sweep_labels = [str(v) for v in sweep_vals]
    elif sweep_feature == "Age":
        sweep_vals = list(range(18, 93, 2))
        sweep_labels = [str(v) for v in sweep_vals]
    else:
        sweep_vals = list(range(300, 860, 20))
        sweep_labels = [str(v) for v in sweep_vals]

    probs = []
    for sv in sweep_vals:
        sv_vals = {**mod_vals}
        if sweep_feature == "IsActiveMember":
            sv_vals["is_active"] = sv
        elif sweep_feature == "NumOfProducts":
            sv_vals["num_products"] = sv
        elif sweep_feature == "Balance":
            sv_vals["balance"] = float(sv)
        elif sweep_feature == "Tenure":
            sv_vals["tenure"] = sv
        elif sweep_feature == "Age":
            sv_vals["age"] = sv
        else:
            sv_vals["credit_score"] = sv
        sdf = prepare_customer_input(**sv_vals)
        probs.append(predict_single(sdf, pipeline, threshold)["probability"])

    fig_sweep = go.Figure()
    fig_sweep.add_trace(go.Scatter(
        x=sweep_labels, y=probs, mode="lines+markers",
        line=dict(color="#2563EB", width=3),
        marker=dict(size=8, color=probs, colorscale="RdYlGn_r",
                    cmin=0, cmax=1, showscale=True),
        hovertemplate="%{x}: %{y:.3f}<extra></extra>",
    ))
    fig_sweep.add_hline(
        y=threshold, line_dash="dash", line_color="#1E3A5F",
        annotation_text=f"Threshold ({threshold:.2f})", annotation_position="right",
    )
    fig_sweep.update_layout(
        title=f"Churn Probability vs {sweep_feature}",
        xaxis_title=sweep_feature,
        yaxis_title="Churn Probability",
        yaxis=dict(range=[0, 1]),
        height=380,
        paper_bgcolor="white",
        plot_bgcolor="#F8FAFC",
        font=dict(family="Inter"),
    )
    st.plotly_chart(fig_sweep, use_container_width=True)

    # Recommendations for modified scenario
    st.markdown("---")
    st.markdown("#### 💡 Recommended Retention Actions")
    mod_recs = get_retention_recommendations(mod_result, mod_df)
    for rec in mod_recs:
        st.markdown(f"<div class='recommendation-card'>{rec}</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# TAB 5: Model Performance
# ══════════════════════════════════════════════

with tab5:
    st.markdown('<div class="section-header">Model Performance & Comparison</div>', unsafe_allow_html=True)

    # Metrics table
    st.markdown("#### 📋 Model Comparison Table")
    if not metrics_table.empty:
        styled_cols = ["Model", "Accuracy", "Precision", "Recall", "F1_Score",
                       "ROC_AUC_Test", "ROC_AUC_CV_Mean", "ROC_AUC_CV_Std"]
        display_cols = [c for c in styled_cols if c in metrics_table.columns]
        st.dataframe(
            metrics_table[display_cols].style
            .highlight_max(subset=["ROC_AUC_Test", "F1_Score", "Recall"], color="#D1FAE5")
            .format({c: "{:.4f}" for c in display_cols if c != "Model"}),
            use_container_width=True,
            height=220,
        )

    c1, c2 = st.columns(2)

    with c1:
        # ROC curve
        st.markdown("#### 📈 ROC Curves")
        roc_path = os.path.join(FIGURES_DIR, "roc_curves.png")
        if os.path.exists(roc_path):
            st.image(roc_path, use_column_width=True)
        else:
            st.info("Run `python src/evaluate.py` to generate ROC curves.")

    with c2:
        # PR curve
        st.markdown("#### 📊 Precision-Recall Curves")
        pr_path = os.path.join(FIGURES_DIR, "pr_curves.png")
        if os.path.exists(pr_path):
            st.image(pr_path, use_column_width=True)
        else:
            st.info("Run `python src/evaluate.py` to generate PR curves.")

    # Confusion matrix
    st.markdown("---")
    st.markdown("#### 🔲 Confusion Matrices")
    cm_path = os.path.join(FIGURES_DIR, "confusion_matrices.png")
    if os.path.exists(cm_path):
        st.image(cm_path, use_column_width=True)
    else:
        st.info("Run `python src/evaluate.py` to generate confusion matrices.")

    # Threshold analysis
    st.markdown("---")
    st.markdown("#### ⚖️ Business Cost Threshold Analysis")
    thresh_path = os.path.join(FIGURES_DIR, "threshold_analysis.png")
    if os.path.exists(thresh_path):
        st.image(thresh_path, use_column_width=True)
    else:
        st.info("Run `python src/evaluate.py` to generate threshold analysis.")

    # Fairness check
    st.markdown("---")
    st.markdown("#### ⚖️ Fairness Analysis by Subgroup")
    fair_path = os.path.join(FIGURES_DIR, "fairness_check.png")
    if os.path.exists(fair_path):
        st.image(fair_path, use_column_width=True)

    fairness_path = os.path.join(ROOT_DIR, "data", "processed", "fairness_results.csv")
    if os.path.exists(fairness_path):
        fairness_df = pd.read_csv(fairness_path)
        st.dataframe(
            fairness_df.style.format({
                c: "{:.4f}" for c in fairness_df.columns if c not in ["Group", "Value"]
            }),
            use_container_width=True,
        )
        st.markdown(
            "<div class='insight-box'>⚠️ <b>Regulatory Note:</b> No demographic attribute "
            "(Geography or Gender) is used as the <em>sole</em> basis for any customer decision. "
            "The model is trained on behavioural signals. Subgroup performance differences should be "
            "monitored quarterly and reported to compliance teams per ECB SREP guidelines.</div>",
            unsafe_allow_html=True,
        )

# ── Footer ─────────────────────────────────────
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:#94A3B8; font-size:0.85rem; padding:0.5rem;'>"
    "ChurnSense &nbsp;·&nbsp; Predictive Modeling & Risk Scoring for Bank Customer Churn &nbsp;·&nbsp; "
    "Built with Streamlit + XGBoost + SHAP"
    "</div>",
    unsafe_allow_html=True,
)
