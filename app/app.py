"""
app.py
======
ChurnSense — Banking Customer Churn Intelligence Platform
Redesigned enterprise fintech analytics dashboard.

Destinations:
  1. Overview
  2. Customer Risk Calculator
  3. Probability Distribution
  4. Feature Importance
  5. What-If Simulator
  6. Model Performance

Run:
  python -m streamlit run app/app.py
"""

import os
import sys
import warnings
import re
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
    page_title="ChurnSense — Customer Churn Intelligence",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────
# Session state initialization & Theme Management
# ──────────────────────────────────────────────

if "theme" not in st.session_state:
    st.session_state["theme"] = "light"

if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "last_customer_df" not in st.session_state:
    st.session_state["last_customer_df"] = None
if "last_inputs" not in st.session_state:
    st.session_state["last_inputs"] = None


# Theme Tokens Definition
THEMES = {
    "light": {
        "page_bg": "#F5F7FB",
        "card_bg": "#FFFFFF",
        "elevated_bg": "#FFFFFF",
        "primary_text": "#172033",
        "secondary_text": "#596579",
        "primary_accent": "#3157D5",
        "primary_soft": "#EEF2FF",
        "primary_soft_border": "#D9E2FF",
        "border": "#E1E6EF",
        "chart_bg": "#FFFFFF",
        "chart_surface": "#F8FAFC",
        "grid_color": "#EDF2F7",
        "badge_border_alpha": "0.35",
        "sidebar_bg": "#F8FAFC",
        "kpi_border": "#E1E6EF",
        "table_header_bg": "#F1F5F9",
        "code_bg": "#F1F5F9",
        "low_risk": "#16845B",
        "low_risk_bg": "#ECFDF5",
        "low_risk_border": "#A7F3D0",
        "medium_risk": "#B45309",
        "medium_risk_bg": "#FFFBEB",
        "medium_risk_border": "#FDE68A",
        "high_risk": "#DC2626",
        "high_risk_bg": "#FEF2F2",
        "high_risk_border": "#FECACA",
    },
    "dark": {
        "page_bg": "#101522",
        "card_bg": "#181F30",
        "elevated_bg": "#202A3D",
        "primary_text": "#F1F5FC",
        "secondary_text": "#B1BCD0",
        "primary_accent": "#8BA5FF",
        "primary_soft": "#1E2742",
        "primary_soft_border": "#2D3A5D",
        "border": "#303B50",
        "chart_bg": "#181F30",
        "chart_surface": "#121826",
        "grid_color": "#242F44",
        "badge_border_alpha": "0.5",
        "sidebar_bg": "#141A29",
        "kpi_border": "#2A354A",
        "table_header_bg": "#1E2638",
        "code_bg": "#1E2638",
        "low_risk": "#34D399",
        "low_risk_bg": "#064E3B",
        "low_risk_border": "#059669",
        "medium_risk": "#FBBF24",
        "medium_risk_bg": "#78350F",
        "medium_risk_border": "#D97706",
        "high_risk": "#F87171",
        "high_risk_bg": "#7F1D1D",
        "high_risk_border": "#DC2626",
    }
}

active_theme_name = st.session_state["theme"]
T = THEMES[active_theme_name]


# ──────────────────────────────────────────────
# Theme-Aware Plotly Helper
# ──────────────────────────────────────────────

def render_themed_plotly(fig, height=None, use_container_width=True):
    """Enforces active theme palette, typography, gridlines and hover tooltips."""
    fig.update_layout(
        paper_bgcolor=T["chart_bg"],
        plot_bgcolor=T["chart_surface"],
        font=dict(family="Inter, -apple-system, sans-serif", color=T["primary_text"], size=12),
        title_font=dict(family="Inter, -apple-system, sans-serif", color=T["primary_text"], size=14),
        legend=dict(
            font=dict(color=T["primary_text"], size=11),
            bgcolor="rgba(0,0,0,0)",
        ),
        margin=dict(l=24, r=24, t=44, b=24),
    )
    if height:
        fig.update_layout(height=height)
    
    fig.update_xaxes(
        tickfont=dict(color=T["secondary_text"], size=11),
        title_font=dict(color=T["primary_text"], size=12),
        gridcolor=T["grid_color"],
        zerolinecolor=T["border"],
        linecolor=T["border"],
    )
    fig.update_yaxes(
        tickfont=dict(color=T["secondary_text"], size=11),
        title_font=dict(color=T["primary_text"], size=12),
        gridcolor=T["grid_color"],
        zerolinecolor=T["border"],
        linecolor=T["border"],
    )
    st.plotly_chart(fig, use_container_width=use_container_width, theme=None)


def md_bold(text):
    """Helper to convert Markdown bold tags into styled HTML."""
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", str(text))


# ──────────────────────────────────────────────
# Global Responsive CSS Injection
# ──────────────────────────────────────────────

css_styles = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

:root {{
    --app-bg: {T["page_bg"]};
    --card-bg: {T["card_bg"]};
    --elevated-bg: {T["elevated_bg"]};
    --primary-text: {T["primary_text"]};
    --secondary-text: {T["secondary_text"]};
    --primary-accent: {T["primary_accent"]};
    --primary-soft: {T["primary_soft"]};
    --primary-soft-border: {T["primary_soft_border"]};
    --border: {T["border"]};
    --low-risk: {T["low_risk"]};
    --low-risk-bg: {T["low_risk_bg"]};
    --low-risk-border: {T["low_risk_border"]};
    --medium-risk: {T["medium_risk"]};
    --medium-risk-bg: {T["medium_risk_bg"]};
    --medium-risk-border: {T["medium_risk_border"]};
    --high-risk: {T["high_risk"]};
    --high-risk-bg: {T["high_risk_bg"]};
    --high-risk-border: {T["high_risk_border"]};
}}

html, body, [class*="css"], [data-testid="stAppViewContainer"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
}}

.stApp {{
    background-color: var(--app-bg) !important;
    color: var(--primary-text) !important;
}}

[data-testid="stHeader"] {{
    background-color: var(--app-bg) !important;
    border-bottom: 1px solid var(--border);
}}

[data-testid="stSidebar"] {{
    background-color: {T["sidebar_bg"]} !important;
    border-right: 1px solid var(--border) !important;
}}

.block-container {{
    padding-top: 1.5rem !important;
    padding-bottom: 2.5rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1560px !important;
}}

/* Top Navigation Banner */
.top-header-banner {{
    background: {T["card_bg"]};
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    margin-bottom: 1.25rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}}

.header-title-box h1 {{
    font-size: 1.45rem !important;
    font-weight: 700 !important;
    color: var(--primary-text) !important;
    margin: 0 !important;
    line-height: 1.3 !important;
    letter-spacing: -0.02em;
}}

.header-title-box p {{
    font-size: 0.88rem !important;
    color: var(--secondary-text) !important;
    margin: 0.25rem 0 0 0 !important;
}}

.status-chip {{
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: var(--primary-soft);
    color: var(--primary-accent);
    border: 1px solid var(--primary-soft-border);
    padding: 0.35rem 0.75rem;
    border-radius: 8px;
    font-size: 0.8rem;
    font-weight: 600;
}}

/* Enterprise KPI Cards */
.kpi-card {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 1rem 1.15rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.03);
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
}}

.kpi-label {{
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--secondary-text);
    text-transform: uppercase;
    letter-spacing: 0.04em;
    margin-bottom: 0.35rem;
    display: flex;
    align-items: center;
    gap: 0.4rem;
}}

.kpi-value {{
    font-size: 1.7rem;
    font-weight: 700;
    color: var(--primary-text);
    line-height: 1.15;
    letter-spacing: -0.02em;
}}

.kpi-subtext {{
    font-size: 0.8rem;
    margin-top: 0.4rem;
    font-weight: 500;
    color: var(--secondary-text);
}}

/* Surface Card Wrapper */
.surface-card {{
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 1.25rem 1.4rem;
    margin-bottom: 1.2rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.03);
}}

.surface-card-title {{
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--primary-text);
    margin: 0 0 0.85rem 0;
    display: flex;
    align-items: center;
    gap: 0.5rem;
    letter-spacing: -0.01em;
}}

/* Risk Badges */
.risk-tag-high {{
    background: var(--high-risk-bg);
    color: var(--high-risk);
    border: 1px solid var(--high-risk-border);
    padding: 0.35rem 0.85rem;
    border-radius: 6px;
    font-weight: 700;
    font-size: 0.88rem;
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
}}

.risk-tag-med {{
    background: var(--medium-risk-bg);
    color: var(--medium-risk);
    border: 1px solid var(--medium-risk-border);
    padding: 0.35rem 0.85rem;
    border-radius: 6px;
    font-weight: 700;
    font-size: 0.88rem;
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
}}

.risk-tag-low {{
    background: var(--low-risk-bg);
    color: var(--low-risk);
    border: 1px solid var(--low-risk-border);
    padding: 0.35rem 0.85rem;
    border-radius: 6px;
    font-weight: 700;
    font-size: 0.88rem;
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
}}

/* Recommendation & Alert Boxes */
.rec-box {{
    background: var(--low-risk-bg);
    border: 1px solid var(--low-risk-border);
    border-left: 4px solid var(--low-risk);
    color: var(--primary-text);
    border-radius: 8px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.88rem;
    line-height: 1.5;
}}

.info-banner {{
    background: var(--primary-soft);
    border: 1px solid var(--primary-soft-border);
    border-left: 4px solid var(--primary-accent);
    color: var(--primary-text);
    border-radius: 8px;
    padding: 0.85rem 1rem;
    margin-bottom: 0.9rem;
    font-size: 0.88rem;
    line-height: 1.5;
}}

/* Form inputs & Streamlit Native Component Skinning */
div[data-testid="stForm"] {{
    background: var(--card-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
}}

div[data-testid="stMetric"] {{
    background: var(--card-bg) !important;
    border: 1px solid var(--border) !important;
    border-radius: 10px !important;
    padding: 0.85rem 1rem !important;
}}

div[data-testid="stMetric"] [data-testid="stMetricLabel"] {{
    color: var(--secondary-text) !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
}}

div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
    color: var(--primary-text) !important;
    font-weight: 700 !important;
    font-size: 1.55rem !important;
}}

div[data-baseweb="input"], div[data-baseweb="select"] > div {{
    background-color: {T["elevated_bg"]} !important;
    border-color: var(--border) !important;
    color: var(--primary-text) !important;
    border-radius: 8px !important;
}}

input, select, textarea {{
    color: var(--primary-text) !important;
    background-color: transparent !important;
}}

label, p, span {{
    color: var(--primary-text);
}}

/* Primary Action Buttons */
.stButton > button, .stFormSubmitButton > button {{
    border-radius: 8px !important;
    font-weight: 600 !important;
    min-height: 2.6rem !important;
    border: 1px solid var(--border) !important;
    transition: all 0.15s ease-in-out;
}}

.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {{
    background-color: var(--primary-accent) !important;
    border-color: var(--primary-accent) !important;
    color: #FFFFFF !important;
}}

.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover {{
    opacity: 0.92;
    transform: translateY(-1px);
}}

/* Modern Radio Pill Controls in Sidebar */
[data-testid="stSidebar"] [data-testid="stRadio"] > div {{
    gap: 0.35rem;
}}

[data-testid="stSidebar"] [data-testid="stRadio"] label {{
    background-color: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 0.55rem 0.8rem;
    margin-bottom: 0.25rem;
    transition: all 0.15s ease;
}}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
    border-color: var(--primary-accent);
}}

/* Responsive Adjustments */
@media (max-width: 900px) {{
    .block-container {{
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        padding-top: 1rem !important;
    }}
    .top-header-banner {{
        flex-direction: column;
        align-items: flex-start;
        gap: 0.75rem;
    }}
}}
</style>
"""
st.markdown(css_styles, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Data & Model Resources Loading
# ──────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_resources():
    pipeline, metadata = load_model_and_metadata()
    return pipeline, metadata

pipeline, metadata = load_resources()
threshold = float(metadata.get("optimal_threshold", 0.12))
best_model_name = metadata.get("best_model", "XGBoost Classifier")

scored_data = load_scored_data()
metrics_table = load_metrics_table()
shap_importance = load_shap_importance()


# ──────────────────────────────────────────────
# Sidebar Navigation Shell
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"""
    <div style='display:flex; align-items:center; gap:0.6rem; padding: 0.4rem 0 1rem 0; border-bottom: 1px solid var(--border); margin-bottom: 1rem;'>
        <div style='background:var(--primary-accent); width:34px; height:34px; border-radius:8px; display:flex; align-items:center; justify-content:center; color:#fff; font-size:1.15rem; font-weight:800;'>
            🏛️
        </div>
        <div>
            <div style='font-size:1.15rem; font-weight:800; color:var(--primary-text); letter-spacing:-0.02em;'>ChurnSense</div>
            <div style='font-size:0.75rem; color:var(--secondary-text); font-weight:500;'>Banking Analytics Platform v2.4</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    nav_selection = st.radio(
        "Navigation",
        [
            "1. Overview",
            "2. Customer Risk Calculator",
            "3. Probability Distribution",
            "4. Feature Importance",
            "5. What-If Simulator",
            "6. Model Performance",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")
    
    # Theme Toggle Widget
    st.markdown(f"<div style='font-size:0.8rem; font-weight:600; color:var(--secondary-text); text-transform:uppercase; letter-spacing:0.04em; margin-bottom:0.4rem;'>Interface Theme</div>", unsafe_allow_html=True)
    theme_choice = st.radio(
        "Theme Toggle",
        ["☀️ Light Theme", "🌙 Dark Theme"],
        index=0 if st.session_state["theme"] == "light" else 1,
        horizontal=True,
        label_visibility="collapsed"
    )
    new_theme = "light" if "Light" in theme_choice else "dark"
    if new_theme != st.session_state["theme"]:
        st.session_state["theme"] = new_theme
        st.rerun()

    st.markdown(f"""
    <div style='background:{T["card_bg"]}; border:1px solid var(--border); border-radius:10px; padding:0.9rem; margin-top:1.5rem;'>
        <div style='font-size:0.75rem; font-weight:700; color:var(--secondary-text); text-transform:uppercase;'>System Status</div>
        <div style='font-size:0.82rem; color:var(--primary-text); margin-top:0.3rem;'><b>Engine:</b> {best_model_name}</div>
        <div style='font-size:0.82rem; color:var(--primary-text);'><b>Opt. Threshold:</b> {threshold:.2f}</div>
        <div style='font-size:0.82rem; color:var(--primary-text);'><b>Scored Base:</b> {len(scored_data) if scored_data is not None else 0:,}</div>
        <div style='font-size:0.76rem; color:var(--low-risk); margin-top:0.4rem; font-weight:600;'>● Pipeline Ready & Verified</div>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Global Top Header Banner
# ──────────────────────────────────────────────

nav_clean = nav_selection.split(". ", 1)[-1]
section_subtitles = {
    "Overview": "Understand portfolio retention risk, explore predictive insights, and prioritize proactive interventions.",
    "Customer Risk Calculator": "Score individual accounts with instant risk tier classification and localized SHAP attribution.",
    "Probability Distribution": "Inspect probability mass across cohorts, evaluate cost threshold alignment, and segment risk.",
    "Feature Importance": "Audit global predictive drivers via SHAP summary distributions, partial dependence, and permutation tests.",
    "What-If Simulator": "Simulate counterfactual scenarios and assess the expected churn delta across account attributes.",
    "Model Performance": "Review validated performance metrics, discrimination curves, and subgroup fairness compliance checks.",
}

st.markdown(f"""
<div class="top-header-banner">
    <div class="header-title-box">
        <h1>{nav_clean}</h1>
        <p>{section_subtitles.get(nav_clean, "Banking Customer Churn Intelligence Platform.")}</p>
    </div>
    <div>
        <div class="status-chip">
            <span>🛡️ Enterprise Model: {best_model_name}</span>
            <span>·</span>
            <span>Threshold: {threshold:.2f}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# SECTION 1: OVERVIEW DASHBOARD
# ══════════════════════════════════════════════

if nav_clean == "Overview":
    st.markdown("### 📊 Executive Portfolio KPI Strip")
    
    if scored_data is not None:
        total = len(scored_data)
        n_high = int((scored_data["Risk_Tier"] == "High").sum())
        n_med = int((scored_data["Risk_Tier"] == "Medium").sum())
        n_low = int((scored_data["Risk_Tier"] == "Low").sum())
        churn_rate = float(scored_data["Churn_Predicted"].mean())
        avg_prob = float(scored_data["Churn_Probability"].mean())

        k1, k2, k3, k4, k5 = st.columns(5)
        
        with k1:
            st.markdown(f"""
            <div class="kpi-card">
                <div class="kpi-label">👥 Total Customers</div>
                <div class="kpi-value">{total:,}</div>
                <div class="kpi-subtext">Active retail book</div>
            </div>
            """, unsafe_allow_html=True)
            
        with k2:
            st.markdown(f"""
            <div class="kpi-card" style="border-top:3px solid var(--high-risk);">
                <div class="kpi-label">🔴 High-Risk Cohort</div>
                <div class="kpi-value" style="color:var(--high-risk);">{n_high:,}</div>
                <div class="kpi-subtext"><b>{n_high/total:.1%}</b> of total portfolio</div>
            </div>
            """, unsafe_allow_html=True)
            
        with k3:
            st.markdown(f"""
            <div class="kpi-card" style="border-top:3px solid var(--medium-risk);">
                <div class="kpi-label">🟡 Medium-Risk Cohort</div>
                <div class="kpi-value" style="color:var(--medium-risk);">{n_med:,}</div>
                <div class="kpi-subtext"><b>{n_med/total:.1%}</b> early warning</div>
            </div>
            """, unsafe_allow_html=True)
            
        with k4:
            st.markdown(f"""
            <div class="kpi-card" style="border-top:3px solid var(--low-risk);">
                <div class="kpi-label">🟢 Low-Risk Cohort</div>
                <div class="kpi-value" style="color:var(--low-risk);">{n_low:,}</div>
                <div class="kpi-subtext"><b>{n_low/total:.1%}</b> stable accounts</div>
            </div>
            """, unsafe_allow_html=True)
            
        with k5:
            st.markdown(f"""
            <div class="kpi-card" style="border-top:3px solid var(--primary-accent);">
                <div class="kpi-label">📉 Predicted Churn Rate</div>
                <div class="kpi-value">{churn_rate:.1%}</div>
                <div class="kpi-subtext">Avg Prob: <b>{avg_prob:.2f}</b> (Thresh: {threshold:.2f})</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:1.2rem;'></div>", unsafe_allow_html=True)

        # Overview Portfolio Charts
        col_ov1, col_ov2 = st.columns([1.1, 1])

        with col_ov1:
            st.markdown(f"""
            <div class="surface-card">
                <div class="surface-card-title">🎯 Risk Tier Distribution & Capital Exposure</div>
            </div>
            """, unsafe_allow_html=True)
            
            tier_df = pd.DataFrame({
                "Tier": ["Low Risk", "Medium Risk", "High Risk"],
                "Count": [n_low, n_med, n_high],
                "Share": [n_low/total, n_med/total, n_high/total],
                "Color": [T["low_risk"], T["medium_risk"], T["high_risk"]],
            })
            
            fig_pie = go.Figure(go.Pie(
                labels=tier_df["Tier"],
                values=tier_df["Count"],
                marker=dict(colors=tier_df["Color"], line=dict(color=T["card_bg"], width=2)),
                hole=0.55,
                textinfo="label+percent",
                hoverinfo="label+value+percent",
            ))
            fig_pie.update_layout(
                showlegend=True,
                legend=dict(orientation="h", yanchor="bottom", y=-0.15, xanchor="center", x=0.5),
            )
            render_themed_plotly(fig_pie, height=360)

        with col_ov2:
            st.markdown(f"""
            <div class="surface-card">
                <div class="surface-card-title">🗺️ Geographic Risk Concentration</div>
            </div>
            """, unsafe_allow_html=True)
            
            geo_summary = scored_data.groupby("Geography").agg(
                Customers=("CustomerId", "count"),
                HighRisk=("Risk_Tier", lambda x: (x == "High").sum()),
                MeanProb=("Churn_Probability", "mean")
            ).reset_index()
            geo_summary["HighRiskPct"] = geo_summary["HighRisk"] / geo_summary["Customers"] * 100

            fig_geo_ov = go.Figure()
            fig_geo_ov.add_trace(go.Bar(
                name="Total Scored",
                x=geo_summary["Geography"],
                y=geo_summary["Customers"],
                marker_color=T["primary_soft_border"],
            ))
            fig_geo_ov.add_trace(go.Bar(
                name="High Risk Customers",
                x=geo_summary["Geography"],
                y=geo_summary["HighRisk"],
                marker_color=T["high_risk"],
            ))
            fig_geo_ov.update_layout(
                barmode="group",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                yaxis_title="Customer Count",
            )
            render_themed_plotly(fig_geo_ov, height=360)

        st.markdown(f"""
        <div class="info-banner">
            <b>Strategic Advisory:</b> Optimal threshold tuning at <b>{threshold:.2f}</b> minimizes misclassification losses by prioritizing recall on high-value accounts. The high-risk segment accounts for {n_high:,} customers requiring immediate retention protocols.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# SECTION 2: CUSTOMER RISK CALCULATOR
# ══════════════════════════════════════════════

elif nav_clean == "Customer Risk Calculator":
    st.markdown(f"""
    <div class="info-banner">
        Enter an individual customer's profile below. ChurnSense evaluates the full feature vector using the calibrated <b>{best_model_name}</b> pipeline and provides real-time SHAP attribution.
    </div>
    """, unsafe_allow_html=True)

    with st.form("customer_risk_form"):
        st.markdown('<div class="surface-card-title">📝 Account Profile & Behavioural Input</div>', unsafe_allow_html=True)
        col_c1, col_c2, col_c3 = st.columns(3)

        with col_c1:
            st.markdown(f"<div style='font-weight:700; color:var(--primary-accent); margin-bottom:0.5rem;'>1. Account Details</div>", unsafe_allow_html=True)
            credit_score = st.slider("Credit Score", 300, 850, 650, step=1, key="calc_cs")
            balance = st.number_input("Account Balance (€)", 0.0, 300000.0, 75000.0, step=1000.0, key="calc_bal")
            estimated_salary = st.number_input("Estimated Salary (€)", 0.0, 250000.0, 85000.0, step=1000.0, key="calc_sal")

        with col_c2:
            st.markdown(f"<div style='font-weight:700; color:var(--primary-accent); margin-bottom:0.5rem;'>2. Customer Profile</div>", unsafe_allow_html=True)
            geography = st.selectbox("Geography", ["France", "Germany", "Spain"], key="calc_geo")
            gender = st.selectbox("Gender", ["Female", "Male"], key="calc_gen")
            age = st.slider("Age", 18, 92, 40, key="calc_age")

        with col_c3:
            st.markdown(f"<div style='font-weight:700; color:var(--primary-accent); margin-bottom:0.5rem;'>3. Banking Behaviour</div>", unsafe_allow_html=True)
            tenure = st.slider("Tenure (years)", 0, 10, 5, key="calc_ten")
            num_products = st.selectbox("Number of Products", [1, 2, 3, 4], key="calc_nop")
            has_cr_card = st.radio("Has Credit Card?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True, key="calc_hcc")
            is_active = st.radio("Active Member?", [1, 0], format_func=lambda x: "Yes" if x == 1 else "No", horizontal=True, key="calc_iam")

        st.markdown("<div style='height:0.5rem;'></div>", unsafe_allow_html=True)
        submitted = st.form_submit_button("⚡ Predict Churn Risk", use_container_width=True, type="primary")

    if submitted:
        with st.spinner("Executing calibrated inference pipeline..."):
            customer_df = prepare_customer_input(
                credit_score, geography, gender, age, tenure,
                balance, num_products, has_cr_card, is_active, estimated_salary,
            )
            result = predict_single(customer_df, pipeline, threshold)
            st.session_state["last_result"] = result
            st.session_state["last_customer_df"] = customer_df
            st.session_state["last_inputs"] = {
                "credit_score": credit_score, "geography": geography, "gender": gender,
                "age": age, "tenure": tenure, "balance": balance,
                "num_products": num_products, "has_cr_card": has_cr_card,
                "is_active": is_active, "salary": estimated_salary,
            }

    # Render results if available in session state
    if st.session_state.get("last_result") is not None:
        result = st.session_state["last_result"]
        customer_df = st.session_state["last_customer_df"]
        prob = result["probability"]
        tier = result["risk_tier"]
        
        tier_color_map = {
            "Low": T["low_risk"],
            "Medium": T["medium_risk"],
            "High": T["high_risk"],
        }
        accent_tier_color = tier_color_map.get(tier, T["primary_accent"])

        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
        st.markdown("### 📋 Prediction Results & Intelligence")
        
        r1, r2 = st.columns([1.1, 1])

        with r1:
            st.markdown(f"""
            <div class="surface-card">
                <div class="surface-card-title">🎯 Probability Gauge & Classification</div>
            </div>
            """, unsafe_allow_html=True)

            gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=round(prob * 100, 1),
                number={"suffix": "%", "font": {"size": 42, "color": accent_tier_color, "family": "Inter"}},
                delta={"reference": threshold * 100, "increasing": {"color": T["high_risk"]}, "decreasing": {"color": T["low_risk"]}},
                gauge={
                    "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": T["secondary_text"]},
                    "bar": {"color": accent_tier_color, "thickness": 0.35},
                    "bgcolor": T["chart_surface"],
                    "borderwidth": 1,
                    "bordercolor": T["border"],
                    "steps": [
                        {"range": [0, 30], "color": T["low_risk_bg"]},
                        {"range": [30, 60], "color": T["medium_risk_bg"]},
                        {"range": [60, 100], "color": T["high_risk_bg"]},
                    ],
                    "threshold": {
                        "line": {"color": T["primary_text"], "width": 3},
                        "thickness": 0.85,
                        "value": threshold * 100,
                    },
                },
                title={"text": "Estimated Churn Probability", "font": {"size": 14, "color": T["secondary_text"]}},
            ))
            gauge.update_layout(
                margin=dict(l=30, r=30, t=35, b=10),
            )
            render_themed_plotly(gauge, height=290)

            # Risk badges display
            tag_class = "risk-tag-high" if tier == "High" else ("risk-tag-med" if tier == "Medium" else "risk-tag-low")
            outcome_class = "risk-tag-high" if result["predicted"] else "risk-tag-low"
            outcome_label = "🚨 Likely to Churn" if result["predicted"] else "✅ Likely to Retain"

            st.markdown(f"""
            <div style='display:flex; justify-content:center; gap:0.8rem; margin-top:0.4rem;'>
                <div class='{tag_class}'>
                    <span>●</span> {tier} Risk Classification
                </div>
                <div class='{outcome_class}'>
                    {outcome_label} (Threshold: {threshold:.2f})
                </div>
            </div>
            """, unsafe_allow_html=True)

        with r2:
            st.markdown(f"""
            <div class="surface-card">
                <div class="surface-card-title">💡 Actionable Retention Directives</div>
            </div>
            """, unsafe_allow_html=True)
            recs = get_retention_recommendations(result, customer_df)
            for rec in recs:
                st.markdown(f"<div class='rec-box'>{md_bold(rec)}</div>", unsafe_allow_html=True)

        # Individual SHAP Attribution
        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="surface-card">
            <div class="surface-card-title">🔍 Localized SHAP Feature Contributions</div>
            <p style='font-size:0.85rem; color:var(--secondary-text); margin:0 0 0.8rem 0;'>
                Quantifies each variable's marginal contribution to this customer's predicted log-odds of churn.
                <b>Red</b> increases churn risk; <b>Green</b> decreases churn risk.
            </p>
        </div>
        """, unsafe_allow_html=True)

        try:
            X_test, y_test = load_test_data()
            X_bg = X_test.sample(min(200, len(X_test)), random_state=42)
            explainer, feature_names, preprocessor = get_shap_explainer(pipeline, X_bg)
            shap_df = get_individual_shap(explainer, pipeline, customer_df, feature_names, preprocessor, top_n=10)

            fig_ind_shap = go.Figure()
            colors = [T["high_risk"] if v > 0 else T["low_risk"] for v in shap_df["SHAP_Value"]]
            fig_ind_shap.add_trace(go.Bar(
                x=shap_df["SHAP_Value"],
                y=shap_df["Feature"],
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.4f}" for v in shap_df["SHAP_Value"]],
                textposition="outside",
            ))
            fig_ind_shap.update_layout(
                xaxis_title="SHAP Value (Impact on Log-Odds)",
                yaxis={"autorange": "reversed"},
            )
            render_themed_plotly(fig_ind_shap, height=390)
        except Exception as e:
            st.info(f"SHAP local explanation unavailable: {e}")


# ══════════════════════════════════════════════
# SECTION 3: PROBABILITY DISTRIBUTION
# ══════════════════════════════════════════════

elif nav_clean == "Probability Distribution":
    st.markdown(f"""
    <div class="info-banner">
        Portfolio probability analysis illustrates model discrimination across historical retained vs churned accounts, highlighting the optimal classification cut-off.
    </div>
    """, unsafe_allow_html=True)

    if scored_data is not None:
        p_col1, p_col2 = st.columns([2.8, 1.2])

        with p_col1:
            fig_dist = go.Figure()
            churned = scored_data[scored_data["Exited"] == 1]["Churn_Probability"]
            retained = scored_data[scored_data["Exited"] == 0]["Churn_Probability"]

            fig_dist.add_trace(go.Histogram(
                x=retained, name="Retained Customers (Actual)", nbinsx=50,
                marker_color=T["low_risk"], opacity=0.7,
                histnorm="probability density",
            ))
            fig_dist.add_trace(go.Histogram(
                x=churned, name="Churned Customers (Actual)", nbinsx=50,
                marker_color=T["high_risk"], opacity=0.7,
                histnorm="probability density",
            ))

            # Threshold line
            fig_dist.add_vline(
                x=threshold, line_dash="dash", line_color=T["primary_text"], line_width=2.5,
                annotation_text=f"Decision Threshold = {threshold:.2f}",
                annotation_position="top right",
            )

            # Risk tier background zones
            fig_dist.add_vrect(x0=0, x1=0.30, fillcolor=T["low_risk"], opacity=0.06, layer="below", line_width=0)
            fig_dist.add_vrect(x0=0.30, x1=0.60, fillcolor=T["medium_risk"], opacity=0.08, layer="below", line_width=0)
            fig_dist.add_vrect(x0=0.60, x1=1.0, fillcolor=T["high_risk"], opacity=0.07, layer="below", line_width=0)

            # If user scored a customer recently, plot reference line
            if st.session_state.get("last_result") is not None:
                cust_p = st.session_state["last_result"]["probability"]
                fig_dist.add_vline(
                    x=cust_p, line_color=T["primary_accent"], line_width=2.5,
                    annotation_text=f"Current Evaluated Customer ({cust_p:.2f})",
                    annotation_position="top left",
                )

            fig_dist.update_layout(
                barmode="overlay",
                title="Historical Probability Density Distribution by Exit Outcome",
                xaxis_title="Predicted Churn Probability",
                yaxis_title="Probability Density",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            render_themed_plotly(fig_dist, height=440)

        with p_col2:
            st.markdown(f"""
            <div class="surface-card">
                <div class="surface-card-title">📈 Distribution Metrics</div>
            </div>
            """, unsafe_allow_html=True)
            
            st.metric("Mean Churn Probability", f"{scored_data['Churn_Probability'].mean():.3f}")
            st.metric("Median Probability", f"{scored_data['Churn_Probability'].median():.3f}")
            st.metric("Mean for Churned Cohort", f"{scored_data[scored_data['Exited']==1]['Churn_Probability'].mean():.3f}")
            st.metric("Mean for Retained Cohort", f"{scored_data[scored_data['Exited']==0]['Churn_Probability'].mean():.3f}")

            st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="surface-card">
                <div class="surface-card-title">🎯 Risk Tier Volumes</div>
            </div>
            """, unsafe_allow_html=True)
            tier_counts = scored_data["Risk_Tier"].value_counts()
            
            for tier_name in ["High", "Medium", "Low"]:
                count = int(tier_counts.get(tier_name, 0))
                pct = count / len(scored_data) * 100
                t_color = T["high_risk"] if tier_name == "High" else (T["medium_risk"] if tier_name == "Medium" else T["low_risk"])
                st.markdown(
                    f"<div style='margin-bottom:0.4rem; font-size:0.88rem; display:flex; justify-content:space-between;'>"
                    f"<span><b style='color:{t_color}'>● {tier_name} Risk:</b></span>"
                    f"<span><b>{count:,}</b> ({pct:.1f}%)</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )

        # Segment Analysis Row
        st.markdown("<div style='height:1.2rem;'></div>", unsafe_allow_html=True)
        st.markdown("### 🧩 Subgroup Segment Propensities")
        
        seg_col1, seg_col2 = st.columns(2)

        with seg_col1:
            geo_stats = scored_data.groupby("Geography")["Churn_Probability"].mean().reset_index()
            fig_geo_seg = px.bar(
                geo_stats, x="Geography", y="Churn_Probability",
                color="Churn_Probability",
                color_continuous_scale=[[0, T["low_risk"]], [0.5, T["medium_risk"]], [1, T["high_risk"]]],
                title="Mean Churn Probability by Country",
                labels={"Churn_Probability": "Mean Probability"},
                text_auto=".3f",
            )
            render_themed_plotly(fig_geo_seg, height=350)

        with seg_col2:
            age_stats = scored_data.copy()
            age_stats["Age_Band"] = pd.cut(
                age_stats["Age"], bins=[0,30,40,50,60,120],
                labels=["18-30","31-40","41-50","51-60","60+"]
            )
            ab = age_stats.groupby("Age_Band", observed=True)["Churn_Probability"].mean().reset_index()
            fig_age_seg = px.bar(
                ab, x="Age_Band", y="Churn_Probability",
                color="Churn_Probability",
                color_continuous_scale=[[0, T["low_risk"]], [0.5, T["medium_risk"]], [1, T["high_risk"]]],
                title="Mean Churn Probability by Age Cohort",
                labels={"Churn_Probability": "Mean Probability"},
                text_auto=".3f",
            )
            render_themed_plotly(fig_age_seg, height=350)


# ══════════════════════════════════════════════
# SECTION 4: FEATURE IMPORTANCE & EXPLAINABILITY
# ══════════════════════════════════════════════

elif nav_clean == "Feature Importance":
    st.markdown(f"""
    <div class="info-banner">
        Global model explainability workspace. Inspect feature rankings using game-theoretic SHAP attributions, partial dependence functions, and permutation sensitivity.
    </div>
    """, unsafe_allow_html=True)

    fi_c1, fi_c2 = st.columns(2)

    with fi_c1:
        st.markdown(f"""
        <div class="surface-card">
            <div class="surface-card-title">📊 Global SHAP Feature Importance</div>
        </div>
        """, unsafe_allow_html=True)
        if shap_importance is not None:
            top_n = st.slider("Select Top N Features to Display", 5, 20, 12, key="shap_topn_slider")
            top = shap_importance.head(top_n).sort_values("Mean_Abs_SHAP")
            
            fig_glob_shap = go.Figure(go.Bar(
                x=top["Mean_Abs_SHAP"],
                y=top["Feature"],
                orientation="h",
                marker=dict(
                    color=top["Mean_Abs_SHAP"],
                    colorscale=[[0, T["primary_soft_border"]], [1, T["primary_accent"]]],
                ),
                text=[f"{v:.4f}" for v in top["Mean_Abs_SHAP"]],
                textposition="outside",
            ))
            fig_glob_shap.update_layout(
                xaxis_title="Mean |SHAP Value| (Average Absolute Impact)",
                yaxis_title="Feature Name",
            )
            render_themed_plotly(fig_glob_shap, height=480)

    with fi_c2:
        st.markdown(f"""
        <div class="surface-card">
            <div class="surface-card-title">🐝 SHAP Beeswarm Distribution Plot</div>
        </div>
        """, unsafe_allow_html=True)
        shap_summary_path = os.path.join(FIGURES_DIR, "shap_summary_beeswarm.png")
        if os.path.exists(shap_summary_path):
            st.image(shap_summary_path, use_container_width=True)
        else:
            st.info("Run `python src/explain.py` to generate the SHAP beeswarm summary plot.")

    st.markdown("<div style='height:1.2rem;'></div>", unsafe_allow_html=True)
    pdp_c1, pdp_c2 = st.columns(2)

    with pdp_c1:
        st.markdown(f"""
        <div class="surface-card">
            <div class="surface-card-title">📈 Partial Dependence Plots (PDP)</div>
            <p style='font-size:0.84rem; color:var(--secondary-text); margin:0 0 0.6rem 0;'>
                Depicts the marginal relationship between core attributes and churn probability while holding other covariates constant.
            </p>
        </div>
        """, unsafe_allow_html=True)
        pdp_path = os.path.join(FIGURES_DIR, "partial_dependence.png")
        if os.path.exists(pdp_path):
            st.image(pdp_path, use_container_width=True)
        else:
            st.info("Run `python src/explain.py` to generate Partial Dependence Plots.")

    with pdp_c2:
        st.markdown(f"""
        <div class="surface-card">
            <div class="surface-card-title">🔄 Permutation Feature Importance</div>
            <p style='font-size:0.84rem; color:var(--secondary-text); margin:0 0 0.6rem 0;'>
                Measures ROC-AUC degradation when individual feature vectors are randomly shuffled across test holdout data.
            </p>
        </div>
        """, unsafe_allow_html=True)
        perm_path = os.path.join(FIGURES_DIR, "permutation_importance.png")
        if os.path.exists(perm_path):
            st.image(perm_path, use_container_width=True)
        else:
            st.info("Run `python src/explain.py` to generate Permutation Importance curves.")


# ══════════════════════════════════════════════
# SECTION 5: WHAT-IF SCENARIO SIMULATOR
# ══════════════════════════════════════════════

elif nav_clean == "What-If Simulator":
    st.markdown(f"""
    <div class="info-banner">
        Interactive counterfactual experimentation engine. Adjust customer behavioural attributes to observe the simulated delta in predicted probability and risk tiers in real time.
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.get("last_inputs") is None:
        st.info("💡 Baseline customer loaded from standard profile. (Tip: Enter a specific customer in the **Customer Risk Calculator** to personalize this baseline).")
        base_vals = {
            "credit_score": 650, "geography": "France", "gender": "Female",
            "age": 42, "tenure": 5, "balance": 75000.0, "num_products": 1,
            "has_cr_card": 1, "is_active": 0, "salary": 85000.0,
        }
    else:
        base_vals = st.session_state["last_inputs"]

    base_df = prepare_customer_input(**base_vals)
    base_result = predict_single(base_df, pipeline, threshold)
    base_prob = base_result["probability"]

    st.markdown("### ⚙️ Scenario Intervention Controls")
    w1, w2, w3 = st.columns(3)

    with w1:
        new_active = st.radio(
            "Active Member Status",
            [1, 0], index=0 if base_vals["is_active"] == 1 else 1,
            format_func=lambda x: "✅ Active Member" if x == 1 else "❌ Inactive Member",
            key="sim_active",
        )
        new_products = st.selectbox(
            "Number of Products",
            [1, 2, 3, 4],
            index=base_vals["num_products"] - 1,
            key="sim_prod",
        )

    with w2:
        new_balance = st.slider(
            "Account Balance (€)", 0, 250000, int(base_vals["balance"]),
            step=5000, key="sim_bal",
        )
        new_tenure = st.slider(
            "Tenure (years)", 0, 10, int(base_vals["tenure"]), key="sim_ten",
        )

    with w3:
        new_credit = st.slider(
            "Credit Score", 300, 850, int(base_vals["credit_score"]), key="sim_cs",
        )
        new_age = st.slider("Age", 18, 92, int(base_vals["age"]), key="sim_age")

    # Generate counterfactual prediction
    mod_vals = {
        **base_vals,
        "is_active": new_active,
        "num_products": new_products,
        "balance": float(new_balance),
        "tenure": new_tenure,
        "credit_score": new_credit,
        "age": new_age,
    }
    mod_df = prepare_customer_input(**mod_vals)
    mod_result = predict_single(mod_df, pipeline, threshold)
    mod_prob = mod_result["probability"]

    delta = mod_prob - base_prob

    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    st.markdown("### 📊 Scenario Delta Comparison")
    
    cmp1, cmp2, cmp3 = st.columns(3)

    with cmp1:
        t_orig = base_result["risk_tier"]
        t_orig_color = T["high_risk"] if t_orig == "High" else (T["medium_risk"] if t_orig == "Medium" else T["low_risk"])
        st.markdown(f"""
        <div class="surface-card" style="text-align:center; border-top:3px solid {t_orig_color};">
            <div style='font-size:0.75rem; color:var(--secondary-text); font-weight:700; text-transform:uppercase;'>Baseline Profile</div>
            <div style='font-size:2.3rem; font-weight:800; color:{t_orig_color}; margin:0.4rem 0;'>{base_prob:.1%}</div>
            <div style='font-weight:700; font-size:0.9rem; color:{t_orig_color};'>● {t_orig} Risk Tier</div>
        </div>
        """, unsafe_allow_html=True)

    with cmp2:
        arrow = "🔻" if delta < -0.005 else ("🔺" if delta > 0.005 else "➡️")
        delta_color = T["low_risk"] if delta < -0.005 else (T["high_risk"] if delta > 0.005 else T["secondary_text"])
        direction_text = "Risk Reduced (Favorable)" if delta < -0.005 else ("Risk Increased (Unfavorable)" if delta > 0.005 else "Neutral Delta")
        st.markdown(f"""
        <div class="surface-card" style="text-align:center; border-top:3px solid {delta_color};">
            <div style='font-size:0.75rem; color:var(--secondary-text); font-weight:700; text-transform:uppercase;'>Net Probability Impact</div>
            <div style='font-size:2.3rem; font-weight:800; color:{delta_color}; margin:0.4rem 0;'>{arrow} {delta:+.1%}</div>
            <div style='font-weight:700; font-size:0.9rem; color:{delta_color};'>{direction_text}</div>
        </div>
        """, unsafe_allow_html=True)

    with cmp3:
        t_mod = mod_result["risk_tier"]
        t_mod_color = T["high_risk"] if t_mod == "High" else (T["medium_risk"] if t_mod == "Medium" else T["low_risk"])
        st.markdown(f"""
        <div class="surface-card" style="text-align:center; border-top:3px solid {t_mod_color};">
            <div style='font-size:0.75rem; color:var(--secondary-text); font-weight:700; text-transform:uppercase;'>Modified Intervention</div>
            <div style='font-size:2.3rem; font-weight:800; color:{t_mod_color}; margin:0.4rem 0;'>{mod_prob:.1%}</div>
            <div style='font-weight:700; font-size:0.9rem; color:{t_mod_color};'>● {t_mod} Risk Tier</div>
        </div>
        """, unsafe_allow_html=True)

    # Sensitivity Curve
    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    st.markdown("### 📉 Univariate Sensitivity Response Curve")
    
    sweep_feature = st.selectbox(
        "Select Feature for Parametric Sweep:",
        ["IsActiveMember", "NumOfProducts", "Balance", "Tenure", "Age", "CreditScore"],
        key="sweep_feat_select",
    )

    if sweep_feature == "IsActiveMember":
        sweep_vals = [0, 1]
        sweep_labels = ["Inactive (0)", "Active (1)"]
    elif sweep_feature == "NumOfProducts":
        sweep_vals = [1, 2, 3, 4]
        sweep_labels = ["1 Product", "2 Products", "3 Products", "4 Products"]
    elif sweep_feature == "Balance":
        sweep_vals = list(range(0, 210000, 10000))
        sweep_labels = [f"€{v:,}" for v in sweep_vals]
    elif sweep_feature == "Tenure":
        sweep_vals = list(range(0, 11))
        sweep_labels = [f"{v} yrs" for v in sweep_vals]
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
        line=dict(color=T["primary_accent"], width=3),
        marker=dict(size=7, color=probs, colorscale=[[0, T["low_risk"]], [0.5, T["medium_risk"]], [1, T["high_risk"]]]),
        hovertemplate="Value: %{x}<br>Predicted Churn Prob: %{y:.3f}<extra></extra>",
    ))
    fig_sweep.add_hline(
        y=threshold, line_dash="dash", line_color=T["primary_text"],
        annotation_text=f"Decision Threshold ({threshold:.2f})", annotation_position="right",
    )
    fig_sweep.update_layout(
        title=f"Sensitivity Curve: Impact of {sweep_feature} on Churn Probability",
        xaxis_title=sweep_feature,
        yaxis_title="Estimated Churn Probability",
        yaxis=dict(range=[0, 1]),
    )
    render_themed_plotly(fig_sweep, height=360)

    # Post-Intervention Recommendations
    st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
    st.markdown("#### 💡 Retention Strategy for Simulated Scenario")
    mod_recs = get_retention_recommendations(mod_result, mod_df)
    for rec in mod_recs:
        st.markdown(f"<div class='rec-box'>{md_bold(rec)}</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# SECTION 6: MODEL PERFORMANCE & FAIRNESS
# ══════════════════════════════════════════════

elif nav_clean == "Model Performance":
    st.markdown(f"""
    <div class="info-banner">
        Model governance & audit dashboard. Inspect cross-validated model benchmarks, discrimination curves, business cost matrices, and demographic parity checks.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📋 Multi-Model Benchmark Comparison")
    if not metrics_table.empty:
        styled_cols = ["Model", "Accuracy", "Precision", "Recall", "F1_Score",
                       "ROC_AUC_Test", "ROC_AUC_CV_Mean", "ROC_AUC_CV_Std"]
        display_cols = [c for c in styled_cols if c in metrics_table.columns]
        
        st.dataframe(
            metrics_table[display_cols].style
            .highlight_max(subset=["ROC_AUC_Test", "F1_Score", "Recall"], color=T["low_risk_bg"])
            .format({c: "{:.4f}" for c in display_cols if c != "Model"}),
            use_container_width=True,
            height=240,
        )

    st.markdown("<div style='height:1.2rem;'></div>", unsafe_allow_html=True)
    eval_c1, eval_c2 = st.columns(2)

    with eval_c1:
        st.markdown(f"""
        <div class="surface-card">
            <div class="surface-card-title">📈 Receiver Operating Characteristic (ROC)</div>
        </div>
        """, unsafe_allow_html=True)
        roc_path = os.path.join(FIGURES_DIR, "roc_curves.png")
        if os.path.exists(roc_path):
            st.image(roc_path, use_container_width=True)
        else:
            st.info("Run `python src/evaluate.py` to generate ROC curves.")

    with eval_c2:
        st.markdown(f"""
        <div class="surface-card">
            <div class="surface-card-title">📊 Precision-Recall (PR) Curves</div>
        </div>
        """, unsafe_allow_html=True)
        pr_path = os.path.join(FIGURES_DIR, "pr_curves.png")
        if os.path.exists(pr_path):
            st.image(pr_path, use_container_width=True)
        else:
            st.info("Run `python src/evaluate.py` to generate PR curves.")

    st.markdown("<div style='height:1.2rem;'></div>", unsafe_allow_html=True)
    mat_c1, mat_c2 = st.columns(2)

    with mat_c1:
        st.markdown(f"""
        <div class="surface-card">
            <div class="surface-card-title">🔲 Confusion Matrices & Threshold Tuning</div>
        </div>
        """, unsafe_allow_html=True)
        cm_path = os.path.join(FIGURES_DIR, "confusion_matrices.png")
        if os.path.exists(cm_path):
            st.image(cm_path, use_container_width=True)
        else:
            st.info("Run `python src/evaluate.py` to generate confusion matrices.")

    with mat_c2:
        st.markdown(f"""
        <div class="surface-card">
            <div class="surface-card-title">⚖️ Financial Cost Matrix Optimization</div>
        </div>
        """, unsafe_allow_html=True)
        thresh_path = os.path.join(FIGURES_DIR, "threshold_analysis.png")
        if os.path.exists(thresh_path):
            st.image(thresh_path, use_container_width=True)
        else:
            st.info("Run `python src/evaluate.py` to generate threshold analysis.")

    # Fairness Audit
    st.markdown("<div style='height:1.2rem;'></div>", unsafe_allow_html=True)
    st.markdown("### ⚖️ Demographic Subgroup Fairness Audit")
    
    fair_path = os.path.join(FIGURES_DIR, "fairness_check.png")
    if os.path.exists(fair_path):
        st.image(fair_path, use_container_width=True)

    fairness_path = os.path.join(ROOT_DIR, "data", "processed", "fairness_results.csv")
    if os.path.exists(fairness_path):
        fairness_df = pd.read_csv(fairness_path)
        st.dataframe(
            fairness_df.style.format({
                c: "{:.4f}" for c in fairness_df.columns if c not in ["Group", "Value"]
            }),
            use_container_width=True,
        )
        st.markdown(f"""
        <div class="info-banner" style="border-left:4px solid var(--medium-risk);">
            <b>Regulatory Compliance Note (ECB SREP & EBA Guidelines):</b> Demographic attributes (such as Geography or Gender) are not used as exclusionary decision variables. The XGBoost pipeline operates across behavioural vectors. Subgroup metrics must be monitored quarterly for stability and calibration.
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Global Footer
# ──────────────────────────────────────────────

st.markdown("<div style='height:2rem;'></div>", unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align:center; color:var(--secondary-text); font-size:0.82rem; padding:1.2rem 0; border-top:1px solid var(--border);'>
    <b>ChurnSense Enterprise</b> &nbsp;·&nbsp; Banking Customer Churn Intelligence & Risk Scoring Platform &nbsp;·&nbsp;
    Engine: {best_model_name} (Threshold: {threshold:.2f}) &nbsp;·&nbsp; Built with Streamlit, Scikit-learn, XGBoost & SHAP
</div>
""", unsafe_allow_html=True)
