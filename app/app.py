"""
app.py
======
ChurnIQ — Enterprise Bank Customer Churn Intelligence & Risk Platform
Precision Intelligence Design System implementation.

Workspaces:
  1. Overview (Customer Risk Overview)
  2. Risk Calculator (Single Customer Risk Calculator)
  3. Portfolio Explorer (Customer Portfolio Explorer & Filter Table)
  4. Probability Distribution (Cohort Density & Calibration)
  5. Feature Importance (SHAP Explainability & Driver Workspace)
  6. What-If Simulator (Counterfactual Scenario Simulator)
  7. Model Performance (Benchmark Auditing & Regulatory Governance)

Usage:
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
# Page Configuration
# ──────────────────────────────────────────────

st.set_page_config(
    page_title="ChurnIQ — Enterprise Risk Suite",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ──────────────────────────────────────────────
# Session State Initialization & Precision Design Tokens
# ──────────────────────────────────────────────

if "theme" not in st.session_state:
    st.session_state["theme"] = "light"

if "last_result" not in st.session_state:
    st.session_state["last_result"] = None
if "last_customer_df" not in st.session_state:
    st.session_state["last_customer_df"] = None
if "last_inputs" not in st.session_state:
    st.session_state["last_inputs"] = None


# Exact Tokens extracted from precision_intelligence/DESIGN.md & stitch code.html
DESIGN_SYSTEM = {
    "light": {
        "surface": "#f8fafc",
        "surface_dim": "#e2e8f0",
        "surface_bright": "#ffffff",
        "surface_lowest": "#ffffff",
        "surface_container_lowest": "#ffffff",
        "surface_low": "#f1f5f9",
        "surface_container_low": "#f1f5f9",
        "surface_container": "#e2e8f0",
        "surface_high": "#cbd5e1",
        "surface_container_high": "#cbd5e1",
        "surface_container_highest": "#94a3b8",
        "on_surface": "#0f172a",               # Deep Slate Navy - Maximum readability & contrast
        "on_surface_variant": "#334155",       # Slate 700 - Highly visible, sharp labels
        "outline": "#64748b",
        "outline_variant": "#cbd5e1",
        "primary": "#1d4ed8",                  # Corporate Royal Blue
        "primary_container": "#2563eb",
        "on_primary": "#ffffff",
        "on_primary_container": "#eff6ff",
        "secondary": "#4338ca",
        "secondary_container": "#6366f1",
        "tertiary": "#059669",                 # High contrast Emerald
        "tertiary_container": "#10b981",
        "tertiary_fixed": "#d1fae5",
        "on_tertiary_fixed": "#064e3b",
        "error": "#dc2626",                    # High contrast Crimson
        "error_container": "#fee2e2",
        "on_error_container": "#7f1d1d",
        "chart_bg": "#ffffff",
        "chart_surface": "#ffffff",
        "grid_color": "#e2e8f0",
        # Semantic Risk Tokens
        "low_risk_fg": "#047857",
        "low_risk_bg": "#d1fae5",
        "low_risk_border": "#6ee7b7",
        "med_risk_fg": "#b45309",
        "med_risk_bg": "#fef3c7",
        "med_risk_border": "#fcd34d",
        "high_risk_fg": "#b91c1c",
        "high_risk_bg": "#fee2e2",
        "high_risk_border": "#fca5a5",
    },
    "dark": {
        "surface": "#0b0f19",
        "surface_dim": "#070a10",
        "surface_bright": "#161f30",
        "surface_lowest": "#111827",
        "surface_container_lowest": "#111827",
        "surface_low": "#151e32",
        "surface_container_low": "#151e32",
        "surface_container": "#1e293b",
        "surface_high": "#283548",
        "surface_container_high": "#283548",
        "surface_container_highest": "#334155",
        "on_surface": "#f8fafc",
        "on_surface_variant": "#cbd5e1",
        "outline": "#64748b",
        "outline_variant": "#334155",
        "primary": "#60a5fa",
        "primary_container": "#2563eb",
        "on_primary": "#030712",
        "on_primary_container": "#eff6ff",
        "secondary": "#818cf8",
        "secondary_container": "#4f46e5",
        "tertiary": "#34d399",
        "tertiary_container": "#059669",
        "tertiary_fixed": "#064e3b",
        "on_tertiary_fixed": "#a7f3d0",
        "error": "#f87171",
        "error_container": "#7f1d1d",
        "on_error_container": "#fee2e2",
        "chart_bg": "#111827",
        "chart_surface": "#0f172a",
        "grid_color": "#1e293b",
        # Semantic Risk Tokens (Dark)
        "low_risk_fg": "#34d399",
        "low_risk_bg": "#064e3b",
        "low_risk_border": "#059669",
        "med_risk_fg": "#fbbf24",
        "med_risk_bg": "#451a03",
        "med_risk_border": "#b45309",
        "high_risk_fg": "#f87171",
        "high_risk_bg": "#450a0a",
        "high_risk_border": "#dc2626",
    }
}

active_mode = st.session_state["theme"]
TOKENS = DESIGN_SYSTEM[active_mode]


# ──────────────────────────────────────────────
# Themed Plotly Chart Helper - Data Science Master Standards
# ──────────────────────────────────────────────

def render_precision_chart(fig, height=360, use_container_width=True):
    """Enforces Master-level Data Science visualization standards on all Plotly instances,
    guaranteeing crisp, contrasting visibility for X/Y axes, tick values, legend text, and markers
    in both Dark and Light modes dynamically."""
    is_dark = st.session_state.get("theme", "light") == "dark"
    current_tokens = DESIGN_SYSTEM["dark"] if is_dark else DESIGN_SYSTEM["light"]
    
    # High-contrast accessibility tokens
    axis_color = "#FFFFFF" if is_dark else "#0F172A"
    tick_color = "#F1F5F9" if is_dark else "#1E293B"
    tick_line_color = "#64748B" if is_dark else "#94A3B8"
    grid_color = "#1E293B" if is_dark else "#E2E8F0"
    border_color = "#334155" if is_dark else "#CBD5E1"
    legend_bg = "rgba(17, 24, 39, 0.92)" if is_dark else "rgba(255, 255, 255, 0.92)"
    legend_border = "#374151" if is_dark else "#CBD5E1"
    legend_font_color = "#FFFFFF" if is_dark else "#0F172A"
    hover_bg = "#1F2937" if is_dark else "#FFFFFF"
    template_name = "plotly_dark" if is_dark else "plotly_white"

    # 1. Base Layout & Theme Template
    fig.update_layout(
        template=template_name,
        paper_bgcolor=current_tokens["chart_bg"],
        plot_bgcolor=current_tokens["chart_surface"],
        font=dict(family="Inter, -apple-system, sans-serif", color=axis_color, size=13),
        title=dict(
            font=dict(family="Inter, -apple-system, sans-serif", color=axis_color, size=15),
            x=0.01,
            xanchor="left",
        ),
        legend=dict(
            font=dict(family="Inter, -apple-system, sans-serif", color=legend_font_color, size=12),
            bgcolor=legend_bg,
            bordercolor=legend_border,
            borderwidth=1,
            itemsizing="constant",
        ),
        hoverlabel=dict(
            bgcolor=hover_bg,
            font_size=12,
            font_family="Inter, -apple-system, sans-serif",
            font_color=axis_color,
            bordercolor=border_color,
        ),
    )
    if height:
        fig.update_layout(height=height)

    # Preserve custom margins if already configured by the chart, otherwise apply default
    if not (fig.layout.margin and fig.layout.margin.t):
        fig.update_layout(margin=dict(l=32, r=32, t=50, b=32))

    # 2. Universal Axis Styling (X & Y Axes)
    fig.update_xaxes(
        color=axis_color,
        tickfont=dict(family="Inter, -apple-system, sans-serif", color=tick_color, size=12),
        title_font=dict(family="Inter, -apple-system, sans-serif", color=axis_color, size=13),
        tickcolor=tick_line_color,
        gridcolor=grid_color,
        gridwidth=1,
        zerolinecolor=border_color,
        zerolinewidth=1.5,
        linecolor=border_color,
        linewidth=1.2,
        showline=True,
        showticklabels=True,
    )
    fig.update_yaxes(
        color=axis_color,
        tickfont=dict(family="Inter, -apple-system, sans-serif", color=tick_color, size=12),
        title_font=dict(family="Inter, -apple-system, sans-serif", color=axis_color, size=13),
        tickcolor=tick_line_color,
        gridcolor=grid_color,
        gridwidth=1,
        zerolinecolor=border_color,
        zerolinewidth=1.5,
        linecolor=border_color,
        linewidth=1.2,
        showline=True,
        showticklabels=True,
    )

    # 3. Trace Elements & Outside Text Visibility
    for trace in fig.data:
        # Outside text on bars / markers
        if hasattr(trace, "textposition") and trace.textposition in ("outside", "auto"):
            if hasattr(trace, "textfont"):
                trace.textfont.color = axis_color
        # Indicator Gauge Chart visibility
        if trace.type == "indicator":
            if hasattr(trace, "number") and trace.number:
                trace.number.font.color = axis_color
            if hasattr(trace, "title") and trace.title:
                trace.title.font.color = axis_color
            if hasattr(trace, "gauge") and trace.gauge and hasattr(trace.gauge, "axis") and trace.gauge.axis:
                trace.gauge.axis.tickcolor = axis_color
                trace.gauge.axis.tickfont.color = axis_color

    # 4. Annotations Visibility
    if hasattr(fig.layout, "annotations") and fig.layout.annotations:
        for ann in fig.layout.annotations:
            if hasattr(ann, "font") and ann.font:
                ann.font.color = axis_color

    st.plotly_chart(fig, use_container_width=use_container_width, theme=None)


def md_bold(text):
    return re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", str(text))


# ──────────────────────────────────────────────
# CSS Injection - Precision Intelligence Design System
# ──────────────────────────────────────────────

css_rules = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
@import url('https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap');

:root {{
    --surface: {TOKENS["surface"]};
    --surface-lowest: {TOKENS["surface_container_lowest"]};
    --surface-low: {TOKENS["surface_container_low"]};
    --surface-container: {TOKENS["surface_container"]};
    --surface-high: {TOKENS["surface_container_high"]};
    --on-surface: {TOKENS["on_surface"]};
    --on-surface-variant: {TOKENS["on_surface_variant"]};
    --primary: {TOKENS["primary"]};
    --primary-container: {TOKENS["primary_container"]};
    --on-primary: {TOKENS["on_primary"]};
    --on-primary-container: {TOKENS["on_primary_container"]};
    --tertiary: {TOKENS["tertiary"]};
    --tertiary-fixed: {TOKENS["tertiary_fixed"]};
    --on-tertiary-fixed: {TOKENS["on_tertiary_fixed"]};
    --error: {TOKENS["error"]};
    --error-container: {TOKENS["error_container"]};
    --on-error-container: {TOKENS["on_error_container"]};
    --outline: {TOKENS["outline"]};
    --outline-variant: {TOKENS["outline_variant"]};
    
    --low-risk-fg: {TOKENS["low_risk_fg"]};
    --low-risk-bg: {TOKENS["low_risk_bg"]};
    --low-risk-border: {TOKENS["low_risk_border"]};
    --med-risk-fg: {TOKENS["med_risk_fg"]};
    --med-risk-bg: {TOKENS["med_risk_bg"]};
    --med-risk-border: {TOKENS["med_risk_border"]};
    --high-risk-fg: {TOKENS["high_risk_fg"]};
    --high-risk-bg: {TOKENS["high_risk_bg"]};
    --high-risk-border: {TOKENS["high_risk_border"]};

    /* Native Streamlit and BaseWeb CSS theme variable sync */
    --text-color: {TOKENS["on_surface"]} !important;
    --background-color: {TOKENS["surface"]} !important;
    --secondary-background-color: {TOKENS["surface_container_low"]} !important;
    --primary-color: {TOKENS["primary"]} !important;
}}

html, body, [class*="css"], [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
    color: var(--on-surface) !important;
}}

.stApp {{
    background-color: var(--surface) !important;
    color: var(--on-surface) !important;
}}

[data-testid="stHeader"] {{
    background-color: var(--surface) !important;
    border-bottom: 1px solid var(--outline-variant);
}}

[data-testid="stSidebar"] {{
    background-color: var(--surface-low) !important;
    border-right: 1px solid var(--outline-variant) !important;
}}

.block-container {{
    padding-top: 1.5rem !important;
    padding-bottom: 3rem !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    max-width: 1560px !important;
}}

/* Header Banner - Stitch ChurnIQ Structure */
.churniq-header {{
    background: var(--surface-lowest);
    border: 1px solid var(--outline-variant);
    border-radius: 12px;
    padding: 1.25rem 1.6rem;
    margin-bottom: 1.25rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.03);
}}

.churniq-header-title {{
    font-size: 1.55rem;
    font-weight: 700;
    color: var(--on-surface);
    margin: 0;
    letter-spacing: -0.02em;
    line-height: 1.25;
}}

.churniq-header-sub {{
    font-size: 0.88rem;
    color: var(--on-surface-variant);
    margin-top: 0.25rem;
}}

.model-badge-pill {{
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: var(--surface-high);
    color: var(--primary);
    border: 1px solid var(--outline-variant);
    padding: 0.35rem 0.85rem;
    border-radius: 9999px;
    font-size: 0.8rem;
    font-weight: 600;
}}

/* Precision 5-KPI Cards */
.precision-kpi-card {{
    background: var(--surface-lowest);
    border: 1px solid var(--outline-variant);
    border-radius: 12px;
    padding: 1.15rem 1.25rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02);
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    height: 100%;
}}

.kpi-top-row {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.5rem;
}}

.kpi-title-label {{
    font-size: 0.76rem;
    font-weight: 600;
    color: var(--on-surface-variant);
    text-transform: uppercase;
    letter-spacing: 0.04em;
}}

.kpi-big-num {{
    font-size: 1.75rem;
    font-weight: 700;
    color: var(--on-surface);
    line-height: 1.15;
    letter-spacing: -0.02em;
    font-feature-settings: "tnum" 1;
}}

.kpi-footer-sub {{
    font-size: 0.78rem;
    color: var(--on-surface-variant);
    margin-top: 0.45rem;
    font-weight: 500;
}}

/* Surface Card Wrapper */
.precision-card {{
    background: var(--surface-lowest);
    border: 1px solid var(--outline-variant);
    border-radius: 12px;
    padding: 1.35rem 1.5rem;
    margin-bottom: 1.25rem;
    box-shadow: 0 1px 4px rgba(0,0,0,0.02);
}}

.precision-card-header {{
    font-size: 1.05rem;
    font-weight: 700;
    color: var(--on-surface);
    margin-bottom: 0.85rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}}

/* Risk Tier Badges with Iconography & Multi-modal cues */
.badge-risk-high {{
    background: var(--high-risk-bg);
    color: var(--high-risk-fg);
    border: 1px solid var(--high-risk-border);
    padding: 0.35rem 0.85rem;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.82rem;
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    letter-spacing: 0.02em;
}}

.badge-risk-med {{
    background: var(--med-risk-bg);
    color: var(--med-risk-fg);
    border: 1px solid var(--med-risk-border);
    padding: 0.35rem 0.85rem;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.82rem;
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    letter-spacing: 0.02em;
}}

.badge-risk-low {{
    background: var(--low-risk-bg);
    color: var(--low-risk-fg);
    border: 1px solid var(--low-risk-border);
    padding: 0.35rem 0.85rem;
    border-radius: 9999px;
    font-weight: 700;
    font-size: 0.82rem;
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    letter-spacing: 0.02em;
}}

/* Callout and Remediation Cards */
.remediation-box {{
    background: var(--surface-low);
    border: 1px solid var(--outline-variant);
    border-left: 4px solid var(--primary);
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.65rem;
    font-size: 0.88rem;
    line-height: 1.5;
    color: var(--on-surface);
}}

.callout-info {{
    background: var(--surface-container);
    border: 1px solid var(--outline-variant);
    border-left: 4px solid var(--primary-container);
    border-radius: 8px;
    padding: 0.9rem 1.1rem;
    margin-bottom: 1rem;
    font-size: 0.88rem;
    line-height: 1.5;
    color: var(--on-surface);
}}

/* =========================================================================
   STREAMLIT CORE & WIDGET VISIBILITY OVERRIDES (LIGHT & DARK THEMES)
   ========================================================================= */

/* Universal Page Headings, Text & Labels */
h1, h2, h3, h4, h5, h6 {{
    color: var(--on-surface) !important;
    font-weight: 700 !important;
}}

/* Universal Page Headings, Text & Labels */
h1, h2, h3, h4, h5, h6 {{
    color: var(--on-surface) !important;
    -webkit-text-fill-color: var(--on-surface) !important;
    font-weight: 700 !important;
}}

p, span, div {{
    color: var(--on-surface);
}}

/* Universal Typography & Label Visibility Overrides */
label, 
[data-testid="stWidgetLabel"],
[data-testid="stWidgetLabel"] label,
[data-testid="stWidgetLabel"] p, 
[data-testid="stWidgetLabel"] span,
[data-testid="stWidgetLabel"] div,
[data-testid="stWidgetLabel"] *,
.stSlider label,
.stSlider label *,
.stSelectbox label,
.stSelectbox label *,
.stNumberInput label,
.stNumberInput label *,
.stTextInput label,
.stTextInput label *,
.stMultiSelect label,
.stMultiSelect label *,
.stRadio label,
.stRadio label *,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] label * {{
    color: var(--on-surface) !important;
    -webkit-text-fill-color: var(--on-surface) !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    opacity: 1 !important;
    visibility: visible !important;
    line-height: 1.4 !important;
}}

/* Slider values, ticks, min/max & labels */
[data-testid="stSliderTickBarMin"],
[data-testid="stSliderTickBarMax"],
[data-testid="stSlider"] div,
[data-testid="stSlider"] span,
[data-testid="stSliderThumbValue"] {{
    color: var(--on-surface) !important;
    -webkit-text-fill-color: var(--on-surface) !important;
    font-weight: 600 !important;
    font-size: 0.84rem !important;
}}

/* Form inputs & component styling */
div[data-testid="stForm"] {{
    background: var(--surface-lowest) !important;
    border: 1px solid var(--outline-variant) !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
}}

/* ═══════════════════════════════════════════════════
   PREMIUM DROPDOWN / SELECTBOX DESIGN SYSTEM
   Identical look in Light & Dark — polished & crisp
   ═══════════════════════════════════════════════════ */

/* ── Closed-state trigger / pill (the visible box) ── */
div[data-baseweb="select"] {{
    border-radius: 9px !important;
}}

div[data-baseweb="select"] > div {{
    background-color: var(--surface-low) !important;
    border: 1.5px solid var(--outline-variant) !important;
    border-radius: 9px !important;
    transition: border-color 0.15s ease, box-shadow 0.15s ease !important;
    min-height: 2.6rem !important;
}}

/* Focused / active trigger */
div[data-baseweb="select"] > div:focus-within,
div[data-baseweb="select"][aria-expanded="true"] > div {{
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(29, 78, 216, 0.18) !important;
    outline: none !important;
}}

/* ── Selected value text & placeholder ── */
div[data-baseweb="select"] div,
div[data-baseweb="select"] span,
div[data-baseweb="select"] p,
div[data-baseweb="select"] [aria-selected="true"],
div[data-baseweb="select"] [class*="valueContainer"] *,
div[data-baseweb="select"] * {{
    background-color: transparent !important;
    color: var(--on-surface) !important;
    -webkit-text-fill-color: var(--on-surface) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}}

/* ── Chevron / caret icon ── */
div[data-baseweb="select"] svg,
div[data-baseweb="select"] svg path {{
    fill: var(--on-surface-variant) !important;
    color: var(--on-surface-variant) !important;
    opacity: 0.85 !important;
}}

/* ══════════════════════════════════════════
   DROPDOWN LIST PANEL (the floating popup)
   ══════════════════════════════════════════ */
div[data-baseweb="popover"],
div[data-baseweb="popover"] > div,
ul[role="listbox"],
div[data-baseweb="menu"],
div[data-baseweb="menu"] > ul {{
    background-color: var(--surface-low) !important;
    border: 1.5px solid var(--outline-variant) !important;
    border-radius: 10px !important;
    box-shadow: 0 16px 40px -8px rgba(0, 0, 0, 0.42),
                0 4px 12px -4px rgba(0, 0, 0, 0.22) !important;
    padding: 6px 4px !important;
    z-index: 999999 !important;
    overflow: hidden !important;
}}

/* ── Individual dropdown option row ── */
li[role="option"],
ul[role="listbox"] li,
div[data-baseweb="menu"] li,
div[data-baseweb="menu"] [role="option"] {{
    background-color: transparent !important;
    color: var(--on-surface) !important;
    -webkit-text-fill-color: var(--on-surface) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
    padding: 0.6rem 0.9rem !important;
    border-radius: 7px !important;
    cursor: pointer !important;
    margin: 2px 2px !important;
    transition: background-color 0.1s ease, color 0.1s ease !important;
}}

/* All nested spans/divs in dropdown options — text stays crisp */
li[role="option"] *,
ul[role="listbox"] li *,
div[data-baseweb="menu"] [role="option"] *,
div[data-baseweb="menu"] li * {{
    color: var(--on-surface) !important;
    -webkit-text-fill-color: var(--on-surface) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    font-weight: 500 !important;
    background-color: transparent !important;
}}

/* ── Hover / active / selected state ── */
li[role="option"]:hover,
ul[role="listbox"] li:hover,
div[data-baseweb="menu"] [role="option"]:hover,
div[data-baseweb="menu"] li:hover {{
    background-color: var(--primary) !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}}

li[role="option"]:hover *,
ul[role="listbox"] li:hover *,
div[data-baseweb="menu"] [role="option"]:hover *,
div[data-baseweb="menu"] li:hover * {{
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}}

li[role="option"][aria-selected="true"],
ul[role="listbox"] li[aria-selected="true"] {{
    background-color: var(--primary-container) !important;
    color: var(--on-primary-container) !important;
    -webkit-text-fill-color: var(--on-primary-container) !important;
    font-weight: 700 !important;
}}

li[role="option"][aria-selected="true"] *,
ul[role="listbox"] li[aria-selected="true"] * {{
    color: var(--on-primary-container) !important;
    -webkit-text-fill-color: var(--on-primary-container) !important;
    font-weight: 700 !important;
}}

/* ══════════════════════════════════════════
   MULTISELECT TAGS / CHIPS
   ══════════════════════════════════════════ */
span[data-baseweb="tag"] {{
    background-color: var(--primary-container) !important;
    border: 1px solid var(--primary) !important;
    border-radius: 6px !important;
    padding: 3px 8px !important;
    gap: 4px !important;
}}

span[data-baseweb="tag"] span,
span[data-baseweb="tag"] div {{
    color: var(--on-primary-container) !important;
    -webkit-text-fill-color: var(--on-primary-container) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.81rem !important;
}}

span[data-baseweb="tag"] [role="presentation"] svg,
span[data-baseweb="tag"] svg {{
    fill: var(--on-primary-container) !important;
    color: var(--on-primary-container) !important;
    opacity: 0.8 !important;
}}

/* Text Input & Number Input fields */
div[data-baseweb="input"] {{
    background-color: var(--surface-low) !important;
    border-radius: 8px !important;
}}

div[data-baseweb="input"] > div {{
    background-color: var(--surface-low) !important;
    border: 1px solid var(--outline-variant) !important;
    border-radius: 8px !important;
}}

input, select, textarea {{
    color: var(--on-surface) !important;
    background-color: transparent !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
}}

input::placeholder, textarea::placeholder {{
    color: var(--on-surface-variant) !important;
    opacity: 0.8 !important;
}}

/* Radio button options */
.stRadio [role="radiogroup"] label {{
    color: var(--on-surface) !important;
}}

.stRadio [role="radiogroup"] label * {{
    color: var(--on-surface) !important;
    font-weight: 600 !important;
}}

/* Metrics styling - remove blue highlight background, keep clean precision styling */
div[data-testid="stMetric"] {{
    background: var(--surface-lowest) !important;
    border: 1px solid var(--outline-variant) !important;
    border-radius: 10px !important;
    padding: 0.95rem 1.1rem !important;
    box-shadow: 0 1px 3px rgba(0,0,0,0.02) !important;
}}

div[data-testid="stMetric"] [data-testid="stMetricLabel"],
div[data-testid="stMetric"] [data-testid="stMetricLabel"] * {{
    color: var(--on-surface-variant) !important;
    font-weight: 700 !important;
    font-size: 0.8rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.04em !important;
    background: transparent !important;
}}

div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stMetric"] [data-testid="stMetricValue"] * {{
    color: var(--on-surface) !important;
    font-weight: 800 !important;
    font-size: 1.75rem !important;
    font-feature-settings: "tnum" 1 !important;
    background: transparent !important;
}}

/* Dataframe & Tables */
[data-testid="stDataFrame"], [data-testid="stTable"] {{
    border: 1px solid var(--outline-variant) !important;
    border-radius: 8px !important;
    overflow: hidden !important;
}}

[data-testid="stDataFrame"] * {{
    color: var(--on-surface) !important;
}}

/* Primary & Secondary Buttons styling */
.stButton > button, .stFormSubmitButton > button, .stDownloadButton > button {{
    border-radius: 8px !important;
    font-weight: 600 !important;
    min-height: 2.7rem !important;
    transition: all 0.15s ease-in-out !important;
}}

.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {{
    background-color: var(--primary) !important;
    border-color: var(--primary) !important;
    color: #FFFFFF !important;
}}

.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover {{
    background-color: var(--primary-container) !important;
    transform: translateY(-1px);
}}

.stDownloadButton > button {{
    background-color: var(--primary) !important;
    border: 1px solid var(--primary) !important;
    color: #FFFFFF !important;
    font-weight: 600 !important;
}}

.stDownloadButton > button:hover {{
    background-color: var(--primary-container) !important;
    color: #FFFFFF !important;
    transform: translateY(-1px);
}}

/* Sidebar Radio Navigation Pill styling - High Contrast & Crisp */
[data-testid="stSidebar"] [data-testid="stRadio"] label {{
    background-color: var(--surface-lowest) !important;
    border: 1px solid var(--outline-variant) !important;
    border-radius: 8px !important;
    padding: 0.65rem 0.95rem !important;
    margin-bottom: 0.35rem !important;
    transition: all 0.15s ease !important;
    cursor: pointer !important;
}}

[data-testid="stSidebar"] [data-testid="stRadio"] label p,
[data-testid="stSidebar"] [data-testid="stRadio"] label span,
[data-testid="stSidebar"] [data-testid="stRadio"] label div {{
    color: var(--on-surface) !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    opacity: 1 !important;
}}

[data-testid="stSidebar"] [data-testid="stRadio"] label:hover {{
    border-color: var(--primary) !important;
    background-color: var(--surface-high) !important;
}}

[data-testid="stSidebar"] [data-testid="stRadio"] label[data-checked="true"],
[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) {{
    border-color: var(--primary) !important;
    background-color: var(--surface-container) !important;
}}

[data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p {{
    color: var(--primary) !important;
    font-weight: 700 !important;
}}

/* ──────────────────────────────────────────────
   Plotly Enterprise Chart Visibility Engine
   Ensures X/Y axis labels, tick values, legend text & markers
   remain ultra-crisp and legible in both dark & light themes
   ────────────────────────────────────────────── */
.js-plotly-plot, .plot-container {{
    border-radius: 10px !important;
}}

/* Legend Text Contrast */
.js-plotly-plot text.legendtext,
.js-plotly-plot g.legend text,
.js-plotly-plot .legend text {{
    fill: var(--on-surface) !important;
    color: var(--on-surface) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    font-weight: 500 !important;
    font-size: 12px !important;
    opacity: 1 !important;
}}

/* Axis Tick Labels (Numbers & Categories) */
.js-plotly-plot g.xtick text,
.js-plotly-plot g.ytick text,
.js-plotly-plot .xtick text,
.js-plotly-plot .ytick text {{
    fill: var(--on-surface) !important;
    color: var(--on-surface) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    font-weight: 500 !important;
    font-size: 11.5px !important;
    opacity: 0.95 !important;
}}

/* Axis Titles (X & Y Axis Labels) */
.js-plotly-plot g.xtitle text,
.js-plotly-plot g.ytitle text,
.js-plotly-plot .xtitle text,
.js-plotly-plot .ytitle text {{
    fill: var(--on-surface) !important;
    color: var(--on-surface) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    font-weight: 600 !important;
    font-size: 13px !important;
    opacity: 1 !important;
}}

/* Chart Master Titles */
.js-plotly-plot text.gtitle,
.js-plotly-plot .gtitle text {{
    fill: var(--on-surface) !important;
    color: var(--on-surface) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    font-weight: 700 !important;
}}

/* Legend Markers, Dots, & Swatches */
.js-plotly-plot g.legend .legendpoints path,
.js-plotly-plot g.legend .legendpoints rect,
.js-plotly-plot g.legend .legendpoints circle,
.js-plotly-plot .legend .legendpoints path,
.js-plotly-plot .legend .legendpoints rect,
.js-plotly-plot .legend .legendpoints circle {{
    opacity: 1 !important;
    stroke-width: 1.5px !important;
}}

/* Legend Card Background Container */
.js-plotly-plot g.legend rect.bg,
.js-plotly-plot .legend rect.bg {{
    fill-opacity: 0.92 !important;
    stroke: var(--outline-variant) !important;
    stroke-width: 1px !important;
}}

/* Outside Bar and Scatter Value Labels */
.js-plotly-plot g.text text,
.js-plotly-plot g.bar text.pointtext,
.js-plotly-plot g.trace text {{
    fill: var(--on-surface) !important;
    font-family: 'Inter', -apple-system, sans-serif !important;
    font-weight: 600 !important;
}}

/* Protect Inside Pie / Donut Slice Percentage Text (stays bright white) */
.js-plotly-plot g.slice text,
.js-plotly-plot text.insidetext {{
    fill: #ffffff !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    opacity: 1 !important;
}}
</style>
"""
st.markdown(css_rules, unsafe_allow_html=True)


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
# Sidebar - ChurnIQ Enterprise Suite Branding
# ──────────────────────────────────────────────

with st.sidebar:
    st.markdown(f"""
    <div style='display:flex; align-items:center; gap:0.65rem; padding: 0.5rem 0 1.1rem 0; border-bottom: 1px solid var(--outline-variant); margin-bottom: 1.1rem;'>
        <div style='background:var(--primary); width:36px; height:36px; border-radius:8px; display:flex; align-items:center; justify-content:center; color:#ffffff; font-size:1.2rem; font-weight:800;'>
            🏛️
        </div>
        <div style='display:flex; flex-direction:column;'>
            <span style='font-size:1.2rem; font-weight:700; color:var(--primary); letter-spacing:-0.02em; line-height:1.2;'>ChurnIQ</span>
            <span style='font-size:0.75rem; color:var(--on-surface-variant); font-weight:500;'>Enterprise Risk Suite</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    nav_selection = st.radio(
        "Navigation",
        [
            "Overview",
            "Risk Calculator",
            "Portfolio Explorer",
            "Probability Distribution",
            "Feature Importance",
            "What-If Simulator",
            "Model Performance",
        ],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")
    
    # Theme Switcher (Working Light/Dark Mode)
    st.markdown("<div style='font-size:0.75rem; font-weight:700; color:var(--on-surface-variant); text-transform:uppercase; letter-spacing:0.04em; margin-bottom:0.4rem;'>Display Theme</div>", unsafe_allow_html=True)
    theme_choice = st.radio(
        "Theme Toggle",
        ["☀️ Precision Light", "🌙 Precision Dark"],
        index=0 if st.session_state["theme"] == "light" else 1,
        horizontal=True,
        label_visibility="collapsed"
    )
    new_theme = "light" if "Light" in theme_choice else "dark"
    if new_theme != st.session_state["theme"]:
        st.session_state["theme"] = new_theme
        st.rerun()

    # System Status Box from reference UI
    st.markdown(f"""
    <div style='background:var(--surface-lowest); border:1px solid var(--outline-variant); border-radius:10px; padding:0.95rem; margin-top:1.5rem;'>
        <div style='display:flex; justify-content:space-between; align-items:center; margin-bottom:0.6rem;'>
            <span style='font-size:0.72rem; font-weight:700; color:var(--on-tertiary-fixed); background:var(--tertiary-fixed); padding:2px 8px; border-radius:9999px;'>v2.4 Production</span>
            <span style='font-size:0.72rem; color:var(--on-surface-variant); font-weight:600;'>Sync: 99.8%</span>
        </div>
        <div style='font-size:0.8rem; color:var(--on-surface);'><b>Inference:</b> {best_model_name}</div>
        <div style='font-size:0.8rem; color:var(--on-surface);'><b>Optimal Threshold:</b> {threshold:.2f}</div>
        <div style='font-size:0.8rem; color:var(--on-surface);'><b>Audited Cohort:</b> {len(scored_data) if scored_data is not None else 0:,}</div>
        <div style='display:flex; align-items:center; gap:0.4rem; margin-top:0.5rem;'>
            <span style='width:7px; height:7px; border-radius:50%; background:var(--tertiary); display:inline-block;'></span>
            <span style='font-size:0.74rem; font-weight:600; color:var(--tertiary);'>Calibrated & Online</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Global Context Header (Precision Intelligence Header)
# ──────────────────────────────────────────────

header_meta = {
    "Overview": {
        "badge": "Portfolio Health Monitor • Live Scoring Feed",
        "title": "Customer Risk Overview",
        "desc": "Portfolio-wide churn propensity monitoring, early warning signals, and retention intervention tracking",
    },
    "Risk Calculator": {
        "badge": "Inference Model v2.4r • Latency 14ms (Calibrated Pipeline)",
        "title": "Single Customer Risk Calculator",
        "desc": "Real-time inference engine evaluating customer profile attributes, account balance, and behavioral indicators",
    },
    "Portfolio Explorer": {
        "badge": "Auditing Engine • Cohort Batch #92-B",
        "title": "Customer Portfolio Explorer",
        "desc": "Search, segment, filter, and audit all 10,000 portfolio accounts by predictive churn risk, demographic traits, and account value.",
    },
    "Probability Distribution": {
        "badge": "Cohort Density & Threshold Calibration",
        "title": "Probability Density Distribution",
        "desc": "Evaluate classification cut-off calibration, cost matrix trade-offs, and empirical risk separation across cohorts.",
    },
    "Feature Importance": {
        "badge": "Game-Theoretic Explainability Workspace",
        "title": "Global Feature Importance & Drivers",
        "desc": "Audit global predictive drivers via SHAP summary distributions, partial dependence plots, and permutation feature tests.",
    },
    "What-If Simulator": {
        "badge": "Counterfactual Simulation Engine",
        "title": "What-If Scenario Simulator",
        "desc": "Simulate behavioural customer interventions and quantify the marginal shift in predicted churn likelihood and risk tiers.",
    },
    "Model Performance": {
        "badge": "Regulatory Governance & Model Validation",
        "title": "Model Performance & Fairness Audit",
        "desc": "Multi-model cross-validation benchmarks, discrimination curves, and subgroup fairness compliance checks (ECB SREP & EBA guidelines).",
    },
}

current_meta = header_meta.get(nav_selection, header_meta["Overview"])

st.markdown(f"""
<div class="churniq-header">
    <div>
        <div style="font-size:0.75rem; text-transform:uppercase; letter-spacing:0.04em; color:var(--on-surface-variant); font-weight:600; margin-bottom:0.25rem;">
            {current_meta["badge"]}
        </div>
        <h1 class="churniq-header-title">{current_meta["title"]}</h1>
        <div class="churniq-header-sub">{current_meta["desc"]}</div>
    </div>
    <div>
        <div class="model-badge-pill">
            <span>🛡️ Engine: {best_model_name}</span>
            <span>•</span>
            <span>Threshold: {threshold:.2f}</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════
# WORKSPACE 1: OVERVIEW (CUSTOMER RISK OVERVIEW)
# ══════════════════════════════════════════════

if nav_selection == "Overview":
    if scored_data is not None:
        total = len(scored_data)
        n_high = int((scored_data["Risk_Tier"] == "High").sum())
        n_med = int((scored_data["Risk_Tier"] == "Medium").sum())
        n_low = int((scored_data["Risk_Tier"] == "Low").sum())
        churn_rate = float(scored_data["Churn_Predicted"].mean())
        avg_prob = float(scored_data["Churn_Probability"].mean())

        # Exact 5-Card KPI Grid from reference screen
        k1, k2, k3, k4, k5 = st.columns(5)
        
        with k1:
            st.markdown(f"""
            <div class="precision-kpi-card">
                <div class="kpi-top-row">
                    <span class="kpi-title-label">Active Customers</span>
                    <span style="background:var(--surface-container); padding:3px 6px; border-radius:6px; font-size:0.85rem;">👥</span>
                </div>
                <div>
                    <div class="kpi-big-num">{total:,}</div>
                    <div style="color:var(--tertiary); font-size:0.78rem; font-weight:600; margin-top:2px;">↑ Verified Retail Book</div>
                </div>
                <div class="kpi-footer-sub">Total Audited: <b>100.0%</b></div>
            </div>
            """, unsafe_allow_html=True)

        with k2:
            st.markdown(f"""
            <div class="precision-kpi-card" style="border-top:3px solid var(--error);">
                <div class="kpi-top-row">
                    <span class="kpi-title-label">High-Risk Cohort</span>
                    <span class="badge-risk-high">● HIGH RISK</span>
                </div>
                <div>
                    <div class="kpi-big-num" style="color:var(--error);">{n_high:,}</div>
                    <div style="color:var(--error); font-size:0.78rem; font-weight:600; margin-top:2px;">Critical Retention Action</div>
                </div>
                <div class="kpi-footer-sub">Portfolio Share: <b>{n_high/total:.1%}</b></div>
            </div>
            """, unsafe_allow_html=True)

        with k3:
            st.markdown(f"""
            <div class="precision-kpi-card" style="border-top:3px solid var(--med-risk-fg);">
                <div class="kpi-top-row">
                    <span class="kpi-title-label">Medium-Risk Cohort</span>
                    <span class="badge-risk-med">● MODERATE</span>
                </div>
                <div>
                    <div class="kpi-big-num" style="color:var(--med-risk-fg);">{n_med:,}</div>
                    <div style="color:var(--med-risk-fg); font-size:0.78rem; font-weight:600; margin-top:2px;">Early Warning Signals</div>
                </div>
                <div class="kpi-footer-sub">Portfolio Share: <b>{n_med/total:.1%}</b></div>
            </div>
            """, unsafe_allow_html=True)

        with k4:
            st.markdown(f"""
            <div class="precision-kpi-card" style="border-top:3px solid var(--tertiary);">
                <div class="kpi-top-row">
                    <span class="kpi-title-label">Low-Risk (Safe)</span>
                    <span class="badge-risk-low">● LOW RISK</span>
                </div>
                <div>
                    <div class="kpi-big-num" style="color:var(--tertiary);">{n_low:,}</div>
                    <div style="color:var(--tertiary); font-size:0.78rem; font-weight:600; margin-top:2px;">Stable Anchored Accounts</div>
                </div>
                <div class="kpi-footer-sub">Portfolio Share: <b>{n_low/total:.1%}</b></div>
            </div>
            """, unsafe_allow_html=True)

        with k5:
            st.markdown(f"""
            <div class="precision-kpi-card" style="border-top:3px solid var(--primary);">
                <div class="kpi-top-row">
                    <span class="kpi-title-label">Predicted Churn</span>
                    <span style="background:var(--surface-container); padding:3px 6px; border-radius:6px; font-size:0.85rem;">📉</span>
                </div>
                <div>
                    <div class="kpi-big-num">{churn_rate:.1%}</div>
                    <div style="color:var(--on-surface-variant); font-size:0.78rem; margin-top:2px;">Optimal Cut-off: {threshold:.2f}</div>
                </div>
                <div class="kpi-footer-sub">Mean Probability: <b>{avg_prob:.2f}</b></div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:1.25rem;'></div>", unsafe_allow_html=True)

        # Overview Portfolio Charts
        ov_c1, ov_c2 = st.columns([1.1, 1.2])

        with ov_c1:
            st.markdown(f"""
            <div class="precision-card">
                <div class="precision-card-header">🎯 Portfolio Churn Propensity Allocation</div>
            </div>
            """, unsafe_allow_html=True)

            tier_df = pd.DataFrame({
                "Tier": ["Low Risk (Safe)", "Medium Risk (Watchlist)", "High Risk (Critical)"],
                "Count": [n_low, n_med, n_high],
                "Color": [TOKENS["tertiary"], TOKENS["med_risk_fg"], TOKENS["error"]],
            })

            fig_donut = go.Figure(go.Pie(
                labels=tier_df["Tier"],
                values=tier_df["Count"],
                marker=dict(colors=tier_df["Color"], line=dict(color=TOKENS["chart_bg"], width=2.5)),
                hole=0.62,
                textinfo="percent",
                textposition="inside",
                textfont=dict(family="Inter", size=14, color="#ffffff"),
                insidetextorientation="horizontal",
                hoverinfo="label+value+percent",
            ))
            fig_donut.update_layout(
                showlegend=True,
                legend=dict(
                    orientation="h",
                    yanchor="top",
                    y=-0.08,
                    xanchor="center",
                    x=0.5,
                    font=dict(family="Inter", size=12, color=TOKENS["on_surface"])
                ),
                annotations=[
                    dict(
                        text=f"<b>{len(scored_data):,}</b><br><span style='font-size:12px; font-weight:600; color:{TOKENS['on_surface_variant']}'>Total Accounts</span>",
                        x=0.5, y=0.5, font_size=19, font_family="Inter", font_color=TOKENS["on_surface"],
                        showarrow=False
                    )
                ],
                margin=dict(l=20, r=20, t=25, b=55),
            )
            render_precision_chart(fig_donut, height=380)

        with ov_c2:
            st.markdown(f"""
            <div class="precision-card">
                <div class="precision-card-header">🗺️ Territorial Risk Exposure (Germany / France / Spain)</div>
            </div>
            """, unsafe_allow_html=True)

            cust_col = "CustomerId" if "CustomerId" in scored_data.columns else scored_data.columns[0]
            geo_summary = scored_data.groupby("Geography").agg(
                Customers=(cust_col, "count"),
                HighRisk=("Risk_Tier", lambda x: (x == "High").sum()),
                MeanProb=("Churn_Probability", "mean")
            ).reset_index()

            max_customers = geo_summary["Customers"].max()
            fig_geo = go.Figure()
            fig_geo.add_trace(go.Bar(
                name="Total Accounts",
                x=geo_summary["Geography"],
                y=geo_summary["Customers"],
                text=geo_summary["Customers"],
                textposition="outside",
                textfont=dict(family="Inter", size=12, color=TOKENS["on_surface"]),
                marker_color=TOKENS["primary"],
            ))
            fig_geo.add_trace(go.Bar(
                name="High Risk Cohort",
                x=geo_summary["Geography"],
                y=geo_summary["HighRisk"],
                text=geo_summary["HighRisk"],
                textposition="outside",
                textfont=dict(family="Inter", size=12, color=TOKENS["on_surface"]),
                marker_color=TOKENS["error"],
            ))
            fig_geo.update_layout(
                barmode="group",
                margin=dict(l=30, r=30, t=55, b=30),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.05,
                    xanchor="right",
                    x=1,
                    font=dict(family="Inter", size=12, color=TOKENS["on_surface"])
                ),
                xaxis=dict(
                    title=dict(text="Territory / Country", font=dict(family="Inter", size=13, color=TOKENS["on_surface"])),
                    tickfont=dict(family="Inter", size=13, color=TOKENS["on_surface"])
                ),
                yaxis=dict(
                    title=dict(text="Account Count", font=dict(family="Inter", size=13, color=TOKENS["on_surface"])),
                    tickfont=dict(family="Inter", size=12, color=TOKENS["on_surface"]),
                    range=[0, max_customers * 1.22],
                ),
            )
            render_precision_chart(fig_geo, height=380)

        st.markdown(f"""
        <div class="callout-info">
            <b>Executive Summary:</b> High-risk accounts ({n_high:,} customers) represent the greatest immediate capital attrition danger. 
            The decision threshold of <b>{threshold:.2f}</b> minimizes financial misclassification cost (€500 FN vs €50 FP), yielding an 87.96% churn recall rate.
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# WORKSPACE 2: RISK CALCULATOR (SINGLE CUSTOMER)
# ══════════════════════════════════════════════

elif nav_selection == "Risk Calculator":
    st.markdown(f"""
    <div class="callout-info">
        Interactive single-customer inference. Inputs map to the calibrated pipeline preprocessor, with localized SHAP waterfalls and prescriptive remediation playbooks.
    </div>
    """, unsafe_allow_html=True)

    calc_left, calc_right = st.columns([1.15, 1])

    with calc_left:
        with st.form("precision_customer_form"):
            st.markdown(f"""
            <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:1rem; padding-bottom:0.5rem; border-bottom:1px solid var(--outline-variant);">
                <div style="font-size:1.05rem; font-weight:700; color:var(--on-surface);">📝 Customer Profile & Behavioural Parameters</div>
                <span style="font-size:0.75rem; color:var(--on-surface-variant); font-weight:600;">10 Input Dimensions</span>
            </div>
            """, unsafe_allow_html=True)

            # Section A: Personal & Demographic
            st.markdown(f"<div style='font-size:0.85rem; font-weight:700; color:var(--primary); text-transform:uppercase; margin-bottom:0.4rem;'>Section A: Personal & Demographic</div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            with c1:
                credit_score = st.slider("Credit Score (FICO Equivalent)", 300, 850, 650, step=1, key="p_cs")
                geography = st.selectbox("Geography (Territory)", ["Germany", "France", "Spain"], key="p_geo")
                age = st.slider("Customer Age", 18, 92, 42, key="p_age")
            with c2:
                gender = st.selectbox("Customer Gender", ["Female", "Male"], key="p_gen")
                tenure = st.slider("Tenure with Bank (Years)", 0, 10, 5, key="p_ten")

            st.markdown("<div style='height:0.75rem;'></div>", unsafe_allow_html=True)
            
            # Section B: Financial & Relationship Exposure
            st.markdown(f"<div style='font-size:0.85rem; font-weight:700; color:var(--primary); text-transform:uppercase; margin-bottom:0.4rem;'>Section B: Financial & Relationship Holdings</div>", unsafe_allow_html=True)
            f1, f2 = st.columns(2)
            with f1:
                balance = st.number_input("Account Balance (€)", 0.0, 300000.0, 85000.0, step=1000.0, key="p_bal")
                estimated_salary = st.number_input("Estimated Annual Salary (€)", 0.0, 250000.0, 79500.0, step=1000.0, key="p_sal")
            with f2:
                num_products = st.selectbox("Active Products Subscribed", [1, 2, 3, 4], index=0, key="p_nop")
                has_cr_card = st.radio("Has Credit Card?", [1, 0], format_func=lambda x: "Yes (Issued)" if x == 1 else "No", horizontal=True, key="p_hcc")
                is_active = st.radio("Active Digital Member?", [1, 0], index=1, format_func=lambda x: "Active (Online)" if x == 1 else "Inactive (Alert)", horizontal=True, key="p_iam")

            st.markdown("<div style='height:0.8rem;'></div>", unsafe_allow_html=True)
            submitted = st.form_submit_button("⚡ Calculate Churn Risk Propensity", use_container_width=True, type="primary")

    # If user submitted or first time landing on calculator, compute prediction so right panel is full and informative
    if submitted or st.session_state.get("last_result") is None:
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

    with calc_right:
        if st.session_state.get("last_result") is not None:
            res = st.session_state["last_result"]
            c_df = st.session_state["last_customer_df"]
            prob = res["probability"]
            tier = res["risk_tier"]

            badge_html = (
                '<span class="badge-risk-high">● HIGH RISK</span>' if tier == "High"
                else ('<span class="badge-risk-med">● MODERATE RISK</span>' if tier == "Medium"
                      else '<span class="badge-risk-low">● LOW RISK</span>')
            )

            st.markdown(f"""
            <div class="precision-card">
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;">
                    <div>
                        <div style="font-size:0.75rem; text-transform:uppercase; color:var(--on-surface-variant); font-weight:700;">Forecast Window</div>
                        <div style="font-size:0.95rem; font-weight:700; color:var(--on-surface);">Next 90-Day Retention Risk</div>
                    </div>
                    {badge_html}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Precision Radial Gauge
            gauge_fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=round(prob * 100, 1),
                number={"suffix": "%", "font": {"size": 44, "color": TOKENS["on_surface"], "family": "Inter"}},
                delta={
                    "reference": threshold * 100,
                    "increasing": {"color": TOKENS["error"]},
                    "decreasing": {"color": TOKENS["tertiary"]},
                    "font": {"size": 13, "family": "Inter"}
                },
                gauge={
                    "axis": {
                        "range": [0, 100],
                        "tickwidth": 1.5,
                        "tickcolor": TOKENS["on_surface"],
                        "tickfont": {"size": 12, "color": TOKENS["on_surface"], "family": "Inter"},
                        "dtick": 20,
                    },
                    "bar": {"color": TOKENS["error"] if tier=="High" else (TOKENS["med_risk_fg"] if tier=="Medium" else TOKENS["tertiary"]), "thickness": 0.35},
                    "bgcolor": TOKENS["surface_container_lowest"],
                    "borderwidth": 1.5,
                    "bordercolor": TOKENS["outline_variant"],
                    "steps": [
                        {"range": [0, 30], "color": TOKENS["low_risk_bg"]},
                        {"range": [30, 60], "color": TOKENS["med_risk_bg"]},
                        {"range": [60, 100], "color": TOKENS["high_risk_bg"]},
                    ],
                    "threshold": {
                        "line": {"color": TOKENS["primary"], "width": 3},
                        "thickness": 0.85,
                        "value": threshold * 100,
                    },
                },
                title={"text": "<b>Estimated Churn Probability</b>", "font": {"size": 15, "color": TOKENS["on_surface"], "family": "Inter"}},
            ))
            gauge_fig.update_layout(margin=dict(l=25, r=25, t=40, b=25))
            render_precision_chart(gauge_fig, height=310)

            # Tailored Retention Playbook (from stitch code.html)
            st.markdown(f"""
            <div class="precision-card">
                <div class="precision-card-header">💡 Prescriptive Remediation Playbook</div>
            </div>
            """, unsafe_allow_html=True)
            recs = get_retention_recommendations(res, c_df)
            for rec in recs:
                st.markdown(f"<div class='remediation-box'>{md_bold(rec)}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="precision-card" style="text-align:center; padding:3.5rem 1.5rem;">
                <div style="font-size:2.5rem; margin-bottom:0.75rem;">⚡</div>
                <div style="font-size:1.15rem; font-weight:700; color:var(--on-surface);">Awaiting Parameter Input</div>
                <div style="font-size:0.86rem; color:var(--on-surface-variant); max-width:320px; margin:0.5rem auto 0 auto;">
                    Adjust the 10 account parameters on the left and click <b>Calculate Churn Risk Propensity</b> to generate instant predictions.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Localized SHAP waterfall attribution
    if st.session_state.get("last_result") is not None:
        st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
        st.markdown(f"""
        <div class="precision-card">
            <div class="precision-card-header">🔍 Top Contributing Risk Drivers (SHAP Value Attribution)</div>
            <p style='font-size:0.84rem; color:var(--on-surface-variant); margin:0 0 0.85rem 0;'>
                Quantitative factor attribution indicating whether attributes pushed churn likelihood higher (red) or shielded the account (green).
            </p>
        </div>
        """, unsafe_allow_html=True)
        try:
            X_test, y_test = load_test_data()
            X_bg = X_test.sample(min(200, len(X_test)), random_state=42)
            explainer, feature_names, preprocessor = get_shap_explainer(pipeline, X_bg)
            shap_df = get_individual_shap(explainer, pipeline, st.session_state["last_customer_df"], feature_names, preprocessor, top_n=8)

            fig_ind = go.Figure()
            colors = [TOKENS["error"] if v > 0 else TOKENS["tertiary"] for v in shap_df["SHAP_Value"]]
            fig_ind.add_trace(go.Bar(
                x=shap_df["SHAP_Value"],
                y=shap_df["Feature"],
                orientation="h",
                marker_color=colors,
                text=[f"{v:+.4f}" for v in shap_df["SHAP_Value"]],
                textposition="outside",
                textfont=dict(family="Inter", size=12, color=TOKENS["on_surface"]),
            ))
            max_v = max(abs(shap_df["SHAP_Value"].min()), abs(shap_df["SHAP_Value"].max())) * 1.35
            fig_ind.update_layout(
                margin=dict(l=35, r=35, t=35, b=35),
                xaxis=dict(
                    range=[-max_v, max_v],
                    title=dict(text="SHAP Impact on Model Log-Odds", font=dict(family="Inter", size=13, color=TOKENS["on_surface"])),
                    tickfont=dict(family="Inter", size=12, color=TOKENS["on_surface"])
                ),
                yaxis=dict(
                    autorange="reversed",
                    tickfont=dict(family="Inter", size=13, color=TOKENS["on_surface"])
                ),
            )
            render_precision_chart(fig_ind, height=360)
        except Exception as e:
            st.info(f"Local SHAP visualization unavailable: {e}")


# ══════════════════════════════════════════════
# WORKSPACE 3: PORTFOLIO EXPLORER (SEARCH & FILTER)
# ══════════════════════════════════════════════

elif nav_selection == "Portfolio Explorer":
    st.markdown(f"""
    <div class="callout-info">
        Interactive portfolio audit workspace. Search, segment, and filter the entire 10,000-customer portfolio by territory, risk tier, holding size, and churn propensity.
    </div>
    """, unsafe_allow_html=True)

    if scored_data is not None:
        # Top 3 Metric Summary Ribbon
        pf1, pf2, pf3 = st.columns(3)
        with pf1:
            total_balance = scored_data["Balance"].sum()
            st.markdown(f"""
            <div class="precision-kpi-card">
                <span class="kpi-title-label">Total Balances in View</span>
                <div class="kpi-big-num">€{total_balance:,.0f}</div>
                <div class="kpi-footer-sub" style="color:var(--tertiary);">Verified European Retail Deposits</div>
            </div>
            """, unsafe_allow_html=True)
        with pf2:
            mean_prop = scored_data["Churn_Probability"].mean()
            st.markdown(f"""
            <div class="precision-kpi-card" style="border-top:3px solid var(--primary);">
                <span class="kpi-title-label">Portfolio Mean Propensity</span>
                <div class="kpi-big-num">{mean_prop:.1%}</div>
                <div class="kpi-footer-sub">Calibrated Threshold: <b>{threshold:.2f}</b></div>
            </div>
            """, unsafe_allow_html=True)
        with pf3:
            high_conc = (scored_data["Risk_Tier"] == "High").mean()
            st.markdown(f"""
            <div class="precision-kpi-card" style="border-top:3px solid var(--error);">
                <span class="kpi-title-label">High-Risk Concentration</span>
                <div class="kpi-big-num" style="color:var(--error);">{high_conc:.1%}</div>
                <div class="kpi-footer-sub">Critical Remediation Needed</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height:1.25rem;'></div>", unsafe_allow_html=True)

        # Filter Ribbon
        st.markdown('<div class="precision-card-header">🔍 Filter & Segment Portfolio</div>', unsafe_allow_html=True)
        flt1, flt2, flt3, flt4 = st.columns(4)

        with flt1:
            geo_filter = st.multiselect("Territory Filter", ["All", "France", "Germany", "Spain"], default=["All"])
        with flt2:
            tier_filter = st.multiselect("Risk Tier Filter", ["All", "High", "Medium", "Low"], default=["All"])
        with flt3:
            active_filter = st.selectbox("Digital Member Status", ["All", "Active Only", "Inactive Only"])
        with flt4:
            search_query = st.text_input("Search Customer ID / Surname", "")

        # Apply filtering
        filtered_df = scored_data.copy()
        if "All" not in geo_filter and len(geo_filter) > 0:
            filtered_df = filtered_df[filtered_df["Geography"].isin(geo_filter)]
        if "All" not in tier_filter and len(tier_filter) > 0:
            filtered_df = filtered_df[filtered_df["Risk_Tier"].isin(tier_filter)]
        if active_filter == "Active Only":
            filtered_df = filtered_df[filtered_df["IsActiveMember"] == 1]
        elif active_filter == "Inactive Only":
            filtered_df = filtered_df[filtered_df["IsActiveMember"] == 0]
        if search_query.strip():
            q = search_query.strip().lower()
            cond = pd.Series([False] * len(filtered_df), index=filtered_df.index)
            if "CustomerId" in filtered_df.columns:
                cond = cond | filtered_df["CustomerId"].astype(str).str.contains(q)
            if "Surname" in filtered_df.columns:
                cond = cond | filtered_df["Surname"].astype(str).str.lower().str.contains(q)
            if not cond.any() and "CustomerId" not in filtered_df.columns and "Surname" not in filtered_df.columns:
                cond = filtered_df["Geography"].astype(str).str.lower().str.contains(q)
            filtered_df = filtered_df[cond]

        st.markdown(f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin: 0.5rem 0 1rem 0;">
            <div style="font-size:0.86rem; color:var(--on-surface-variant);">
                Displaying <b>{len(filtered_df):,}</b> of <b>{len(scored_data):,}</b> accounts (<b>{len(filtered_df)/len(scored_data):.1%}</b> of portfolio)
            </div>
            <div>
                <span style="font-size:0.8rem; font-weight:600; color:var(--primary);">● Live Filter Active</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Display Table with Streamlit styled columns
        table_cols = [
            "CustomerId", "Surname", "Geography", "Gender", "Age",
            "CreditScore", "Balance", "NumOfProducts", "IsActiveMember",
            "Churn_Probability", "Risk_Tier"
        ]
        available_cols = [c for c in table_cols if c in filtered_df.columns]

        st.dataframe(
            filtered_df[available_cols].head(500).style
            .format({
                "Balance": "€{:,.2f}",
                "Churn_Probability": "{:.3f}",
                "CreditScore": "{:.0f}",
                "Age": "{:.0f}",
                "NumOfProducts": "{:.0f}",
            }),
            use_container_width=True,
            height=420,
        )

        # CSV Download
        csv_data = filtered_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Export Filtered Accounts CSV",
            data=csv_data,
            file_name="churniq_filtered_portfolio.csv",
            mime="text/csv",
        )


# ══════════════════════════════════════════════
# WORKSPACE 4: PROBABILITY DISTRIBUTION
# ══════════════════════════════════════════════

elif nav_selection == "Probability Distribution":
    st.markdown(f"""
    <div class="callout-info">
        Cohort probability density inspection. Visualizes clear empirical separation between retained accounts and exited accounts.
    </div>
    """, unsafe_allow_html=True)

    if scored_data is not None:
        pd_c1, pd_c2 = st.columns([2.8, 1.2])

        with pd_c1:
            fig_dist = go.Figure()
            churned = scored_data[scored_data["Exited"] == 1]["Churn_Probability"]
            retained = scored_data[scored_data["Exited"] == 0]["Churn_Probability"]

            fig_dist.add_trace(go.Histogram(
                x=retained, name="Retained Customers (Actual)", nbinsx=50,
                marker_color=TOKENS["tertiary"], opacity=0.75,
                histnorm="probability density",
            ))
            fig_dist.add_trace(go.Histogram(
                x=churned, name="Churned Customers (Actual)", nbinsx=50,
                marker_color=TOKENS["error"], opacity=0.75,
                histnorm="probability density",
            ))

            fig_dist.add_vline(
                x=threshold, line_dash="dash", line_color=TOKENS["primary"], line_width=2.5,
                annotation_text=f"<b>Decision Cut-off ({threshold:.2f})</b>",
                annotation_position="top right",
                annotation_font=dict(family="Inter", size=12, color=TOKENS["on_surface"]),
                annotation_bgcolor=TOKENS["surface_container"],
                annotation_bordercolor=TOKENS["outline"],
                annotation_borderpad=5,
            )

            # Zone rectangles
            fig_dist.add_vrect(x0=0, x1=0.30, fillcolor=TOKENS["tertiary"], opacity=0.06, layer="below", line_width=0)
            fig_dist.add_vrect(x0=0.30, x1=0.60, fillcolor=TOKENS["med_risk_fg"], opacity=0.08, layer="below", line_width=0)
            fig_dist.add_vrect(x0=0.60, x1=1.0, fillcolor=TOKENS["error"], opacity=0.07, layer="below", line_width=0)

            fig_dist.update_layout(
                barmode="overlay",
                title=dict(
                    text="<b>Empirical Churn Probability Density Across Exited vs Retained Cohorts</b>",
                    font=dict(family="Inter", size=15, color=TOKENS["on_surface"]),
                    x=0.01,
                    y=0.98,
                ),
                margin=dict(l=35, r=35, t=75, b=35),
                xaxis=dict(
                    title=dict(text="Predicted Churn Probability", font=dict(family="Inter", size=13, color=TOKENS["on_surface"])),
                    tickfont=dict(family="Inter", size=12, color=TOKENS["on_surface"])
                ),
                yaxis=dict(
                    title=dict(text="Probability Density", font=dict(family="Inter", size=13, color=TOKENS["on_surface"])),
                    tickfont=dict(family="Inter", size=12, color=TOKENS["on_surface"])
                ),
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.04,
                    xanchor="right",
                    x=1,
                    font=dict(family="Inter", size=12, color=TOKENS["on_surface"])
                ),
            )
            render_precision_chart(fig_dist, height=430)

        with pd_c2:
            st.markdown(f"""
            <div class="precision-card">
                <div class="precision-card-header">📊 Statistical Moments</div>
            </div>
            """, unsafe_allow_html=True)
            st.metric("Mean Churn Probability", f"{scored_data['Churn_Probability'].mean():.3f}")
            st.metric("Median Churn Probability", f"{scored_data['Churn_Probability'].median():.3f}")
            st.metric("Exited Cohort Mean", f"{scored_data[scored_data['Exited']==1]['Churn_Probability'].mean():.3f}")
            st.metric("Retained Cohort Mean", f"{scored_data[scored_data['Exited']==0]['Churn_Probability'].mean():.3f}")

        st.markdown("<div style='height:1.2rem;'></div>", unsafe_allow_html=True)
        seg1, seg2 = st.columns(2)

        with seg1:
            geo_stats = scored_data.groupby("Geography")["Churn_Probability"].mean().reset_index()
            max_geo = geo_stats["Churn_Probability"].max()
            fig_geo_b = px.bar(
                geo_stats, x="Geography", y="Churn_Probability",
                color="Churn_Probability",
                color_continuous_scale=[[0, TOKENS["tertiary"]], [0.5, TOKENS["med_risk_fg"]], [1, TOKENS["error"]]],
                title="<b>Mean Churn Probability by Country</b>",
                labels={"Churn_Probability": "Mean Prob"},
                text_auto=".3f",
            )
            fig_geo_b.update_traces(
                textposition="outside",
                textfont=dict(family="Inter", size=12, color=TOKENS["on_surface"])
            )
            fig_geo_b.update_layout(
                coloraxis_showscale=False,
                margin=dict(l=30, r=30, t=55, b=30),
                xaxis=dict(
                    title=dict(font=dict(family="Inter", size=13, color=TOKENS["on_surface"])),
                    tickfont=dict(family="Inter", size=12, color=TOKENS["on_surface"])
                ),
                yaxis=dict(
                    range=[0, max_geo * 1.25],
                    title=dict(font=dict(family="Inter", size=13, color=TOKENS["on_surface"])),
                    tickfont=dict(family="Inter", size=12, color=TOKENS["on_surface"])
                ),
            )
            render_precision_chart(fig_geo_b, height=360)

        with seg2:
            age_stats = scored_data.copy()
            age_stats["Age_Band"] = pd.cut(
                age_stats["Age"], bins=[0,30,40,50,60,120],
                labels=["18-30","31-40","41-50","51-60","60+"]
            )
            ab = age_stats.groupby("Age_Band", observed=True)["Churn_Probability"].mean().reset_index()
            max_age = ab["Churn_Probability"].max()
            fig_age_b = px.bar(
                ab, x="Age_Band", y="Churn_Probability",
                color="Churn_Probability",
                color_continuous_scale=[[0, TOKENS["tertiary"]], [0.5, TOKENS["med_risk_fg"]], [1, TOKENS["error"]]],
                title="<b>Mean Churn Probability by Age Cohort</b>",
                labels={"Churn_Probability": "Mean Prob"},
                text_auto=".3f",
            )
            fig_age_b.update_traces(
                textposition="outside",
                textfont=dict(family="Inter", size=12, color=TOKENS["on_surface"])
            )
            fig_age_b.update_layout(
                coloraxis_showscale=False,
                margin=dict(l=30, r=30, t=55, b=30),
                xaxis=dict(
                    title=dict(font=dict(family="Inter", size=13, color=TOKENS["on_surface"])),
                    tickfont=dict(family="Inter", size=12, color=TOKENS["on_surface"])
                ),
                yaxis=dict(
                    range=[0, max_age * 1.25],
                    title=dict(font=dict(family="Inter", size=13, color=TOKENS["on_surface"])),
                    tickfont=dict(family="Inter", size=12, color=TOKENS["on_surface"])
                ),
            )
            render_precision_chart(fig_age_b, height=360)


# ══════════════════════════════════════════════
# WORKSPACE 5: FEATURE IMPORTANCE & EXPLAINABILITY
# ══════════════════════════════════════════════

elif nav_selection == "Feature Importance":
    st.markdown(f"""
    <div class="callout-info">
        Game-theoretic model explainability. Inspect SHAP global feature importances, beeswarm distribution plots, and partial dependence functions.
    </div>
    """, unsafe_allow_html=True)

    fi1, fi2 = st.columns(2)

    with fi1:
        st.markdown(f"""
        <div class="precision-card">
            <div class="precision-card-header">📊 Global SHAP Feature Importance</div>
        </div>
        """, unsafe_allow_html=True)
        if shap_importance is not None:
            top_n = st.slider("Select Top N Attributes to Audit", 5, 20, 12, key="p_shap_slider")
            top = shap_importance.head(top_n).sort_values("Mean_Abs_SHAP")

            max_shap = top["Mean_Abs_SHAP"].max()
            fig_bar = go.Figure(go.Bar(
                x=top["Mean_Abs_SHAP"],
                y=top["Feature"],
                orientation="h",
                marker=dict(
                    color=top["Mean_Abs_SHAP"],
                    colorscale=[[0, TOKENS["primary_container"]], [1, TOKENS["primary"]]],
                ),
                text=[f"{v:.4f}" for v in top["Mean_Abs_SHAP"]],
                textposition="outside",
                textfont=dict(family="Inter", size=12, color=TOKENS["on_surface"]),
            ))
            fig_bar.update_layout(
                margin=dict(l=35, r=50, t=40, b=35),
                xaxis=dict(
                    range=[0, max_shap * 1.25],
                    title=dict(text="Mean |SHAP Value| (Impact on Model Confidence)", font=dict(family="Inter", size=13, color=TOKENS["on_surface"])),
                    tickfont=dict(family="Inter", size=12, color=TOKENS["on_surface"])
                ),
                yaxis=dict(
                    title=dict(text="Feature Dimension", font=dict(family="Inter", size=13, color=TOKENS["on_surface"])),
                    tickfont=dict(family="Inter", size=13, color=TOKENS["on_surface"])
                ),
            )
            render_precision_chart(fig_bar, height=480)

    with fi2:
        st.markdown(f"""
        <div class="precision-card">
            <div class="precision-card-header">🐝 SHAP Beeswarm Distribution Plot</div>
            <p style='font-size:0.84rem; color:var(--on-surface-variant); margin:0 0 0.6rem 0;'>
                Each dot is one customer. Horizontal position = model impact. Color = feature value (red = high, blue = low).
                Updates live with the Top N slider on the left.
            </p>
        </div>
        """, unsafe_allow_html=True)

        # ── Dynamic beeswarm: regenerated whenever slider changes ──
        try:
            is_dark = st.session_state.get("theme", "light") == "dark"
            bg_color  = "#111827" if is_dark else "#ffffff"
            text_color = "#f8fafc" if is_dark else "#0f172a"
            grid_color = "#1e293b" if is_dark else "#e2e8f0"

            X_test_sw, _ = load_test_data()
            X_bg_sw = X_test_sw.sample(min(300, len(X_test_sw)), random_state=42)
            explainer_sw, feature_names_sw, preprocessor_sw = get_shap_explainer(pipeline, X_bg_sw)

            # Transform a sample of test data
            X_sample_sw = X_test_sw.sample(min(500, len(X_test_sw)), random_state=7)
            X_trans_sw  = preprocessor_sw.transform(X_sample_sw)
            import pandas as _pd
            X_df_sw = _pd.DataFrame(X_trans_sw, columns=feature_names_sw)

            # Compute SHAP values
            shap_vals_sw = explainer_sw.shap_values(X_df_sw)
            if isinstance(shap_vals_sw, list):
                shap_vals_sw = shap_vals_sw[1]

            # Filter to top_n features (from the left-column slider)
            slider_val = st.session_state.get("p_shap_slider", 12)
            mean_abs   = abs(shap_vals_sw).mean(axis=0)
            top_idx    = mean_abs.argsort()[::-1][:slider_val]
            shap_top   = shap_vals_sw[:, top_idx]
            X_top      = X_df_sw.iloc[:, top_idx]

            # Plot with matplotlib — themed
            matplotlib.rcParams.update({
                "text.color":   text_color,
                "axes.labelcolor": text_color,
                "xtick.color":  text_color,
                "ytick.color":  text_color,
                "axes.facecolor": bg_color,
                "figure.facecolor": bg_color,
            })

            fig_bee, ax_bee = plt.subplots(figsize=(7, max(4, slider_val * 0.42)))
            fig_bee.patch.set_facecolor(bg_color)
            ax_bee.set_facecolor(bg_color)

            shap.summary_plot(
                shap_top,
                X_top,
                plot_type="dot",
                show=False,
                color_bar=True,
                plot_size=None,
                max_display=slider_val,
            )

            # Re-theme all text elements after shap draws them
            for item in ([ax_bee.title, ax_bee.xaxis.label, ax_bee.yaxis.label] +
                         ax_bee.get_xticklabels() + ax_bee.get_yticklabels()):
                item.set_color(text_color)
            ax_bee.tick_params(colors=text_color)
            ax_bee.spines["bottom"].set_color(grid_color)
            ax_bee.spines["left"].set_color(grid_color)
            ax_bee.spines["top"].set_visible(False)
            ax_bee.spines["right"].set_visible(False)
            ax_bee.axvline(0, color=grid_color, linewidth=0.8, linestyle="--")

            # Colorbar theming
            for cb_ax in fig_bee.axes[1:]:
                cb_ax.set_facecolor(bg_color)
                cb_ax.tick_params(colors=text_color)
                cb_ax.yaxis.label.set_color(text_color)

            plt.tight_layout(pad=1.2)
            st.pyplot(fig_bee, use_container_width=True)
            plt.close(fig_bee)

        except Exception as _e:
            # Graceful fallback to static PNG if live generation fails
            beeswarm_path = os.path.join(FIGURES_DIR, "shap_summary_beeswarm.png")
            if os.path.exists(beeswarm_path):
                st.image(beeswarm_path, use_container_width=True)
                st.caption(f"⚠️ Live render failed ({_e}) — showing cached plot.")
            else:
                st.info(f"Could not generate beeswarm: {_e}")

    st.markdown("<div style='height:1.25rem;'></div>", unsafe_allow_html=True)
    p1, p2 = st.columns(2)

    with p1:
        st.markdown(f"""
        <div class="precision-card">
            <div class="precision-card-header">📈 Partial Dependence Plots (PDP)</div>
            <p style='font-size:0.84rem; color:var(--on-surface-variant); margin:0 0 0.6rem 0;'>
                Depicts the marginal relationship between core variables and churn likelihood while holding other features at their mean.
            </p>
        </div>
        """, unsafe_allow_html=True)
        pdp_path = os.path.join(FIGURES_DIR, "partial_dependence.png")
        if os.path.exists(pdp_path):
            st.image(pdp_path, use_container_width=True)
        else:
            st.info("Run `python src/explain.py` to generate PDPs.")

    with p2:
        st.markdown(f"""
        <div class="precision-card">
            <div class="precision-card-header">🔄 Permutation Feature Importance</div>
            <p style='font-size:0.84rem; color:var(--on-surface-variant); margin:0 0 0.6rem 0;'>
                Quantifies degradation in test holdout ROC-AUC when feature vectors are randomly permuted.
            </p>
        </div>
        """, unsafe_allow_html=True)
        perm_path = os.path.join(FIGURES_DIR, "permutation_importance.png")
        if os.path.exists(perm_path):
            st.image(perm_path, use_container_width=True)
        else:
            st.info("Run `python src/explain.py` to generate Permutation Importance curves.")


# ══════════════════════════════════════════════
# WORKSPACE 6: WHAT-IF SCENARIO SIMULATOR
# ══════════════════════════════════════════════

elif nav_selection == "What-If Simulator":
    st.markdown(f"""
    <div class="callout-info">
        Counterfactual intervention sandbox. Adjust customer attributes to simulate the net reduction in churn likelihood before applying CRM campaigns.
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.get("last_inputs") is None:
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

    st.markdown('<div class="precision-card-header">⚙️ Scenario Intervention Parameters</div>', unsafe_allow_html=True)
    w1, w2, w3 = st.columns(3)

    with w1:
        new_active = st.radio(
            "Active Member Status",
            [1, 0], index=0 if base_vals["is_active"] == 1 else 1,
            format_func=lambda x: "✅ Active Member" if x == 1 else "❌ Inactive (Alert)",
            key="p_sim_act",
        )
        new_products = st.selectbox(
            "Number of Products",
            [1, 2, 3, 4], index=base_vals["num_products"] - 1,
            key="p_sim_prod",
        )

    with w2:
        new_balance = st.slider(
            "Account Balance (€)", 0, 250000, int(base_vals["balance"]),
            step=5000, key="p_sim_bal",
        )
        new_tenure = st.slider(
            "Tenure (Years)", 0, 10, int(base_vals["tenure"]), key="p_sim_ten",
        )

    with w3:
        new_credit = st.slider(
            "Credit Score", 300, 850, int(base_vals["credit_score"]), key="p_sim_cs",
        )
        new_age = st.slider("Customer Age", 18, 92, int(base_vals["age"]), key="p_sim_age")

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

    st.markdown("<div style='height:1.2rem;'></div>", unsafe_allow_html=True)
    st.markdown('<div class="precision-card-header">📊 Scenario Delta Impact Analysis</div>', unsafe_allow_html=True)

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        t_orig = base_result["risk_tier"]
        color_orig = TOKENS["error"] if t_orig=="High" else (TOKENS["med_risk_fg"] if t_orig=="Medium" else TOKENS["tertiary"])
        st.markdown(f"""
        <div class="precision-card" style="text-align:center; border-top:3px solid {color_orig};">
            <span class="kpi-title-label">Baseline Customer Profile</span>
            <div class="kpi-big-num" style="color:{color_orig}; margin:0.4rem 0;">{base_prob:.1%}</div>
            <div style="font-weight:700; color:{color_orig}; font-size:0.88rem;">● {t_orig} Risk Tier</div>
        </div>
        """, unsafe_allow_html=True)

    with sc2:
        arrow = "🔻" if delta < -0.005 else ("🔺" if delta > 0.005 else "➡️")
        delta_color = TOKENS["tertiary"] if delta < -0.005 else (TOKENS["error"] if delta > 0.005 else TOKENS["on_surface_variant"])
        direction_text = "Favorable Risk Reduction" if delta < -0.005 else ("Risk Elevation (Caution)" if delta > 0.005 else "Neutral")
        st.markdown(f"""
        <div class="precision-card" style="text-align:center; border-top:3px solid {delta_color};">
            <span class="kpi-title-label">Simulated Net Propensity Shift</span>
            <div class="kpi-big-num" style="color:{delta_color}; margin:0.4rem 0;">{arrow} {delta:+.1%}</div>
            <div style="font-weight:700; color:{delta_color}; font-size:0.88rem;">{direction_text}</div>
        </div>
        """, unsafe_allow_html=True)

    with sc3:
        t_mod = mod_result["risk_tier"]
        color_mod = TOKENS["error"] if t_mod=="High" else (TOKENS["med_risk_fg"] if t_mod=="Medium" else TOKENS["tertiary"])
        st.markdown(f"""
        <div class="precision-card" style="text-align:center; border-top:3px solid {color_mod};">
            <span class="kpi-title-label">Post-Intervention Profile</span>
            <div class="kpi-big-num" style="color:{color_mod}; margin:0.4rem 0;">{mod_prob:.1%}</div>
            <div style="font-weight:700; color:{color_mod}; font-size:0.88rem;">● {t_mod} Risk Tier</div>
        </div>
        """, unsafe_allow_html=True)

    # Parametric sweep
    st.markdown("<div style='height:1rem;'></div>", unsafe_allow_html=True)
    sweep_feature = st.selectbox(
        "Parametric Sweep Attribute:",
        ["IsActiveMember", "NumOfProducts", "Balance", "Tenure", "Age", "CreditScore"],
        key="p_sweep_select",
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

    fig_sw = go.Figure()
    fig_sw.add_trace(go.Scatter(
        x=sweep_labels, y=probs, mode="lines+markers",
        line=dict(color=TOKENS["primary"], width=3),
        marker=dict(size=8, color=probs, colorscale=[[0, TOKENS["tertiary"]], [0.5, TOKENS["med_risk_fg"]], [1, TOKENS["error"]]]),
    ))
    fig_sw.add_hline(
        y=threshold, line_dash="dash", line_color=TOKENS["primary"], line_width=2.5,
        annotation_text=f"<b>Decision Cut-off ({threshold:.2f})</b>",
        annotation_position="top left",
        annotation_font=dict(family="Inter", size=12, color=TOKENS["on_surface"]),
        annotation_bgcolor=TOKENS["surface_container"],
        annotation_bordercolor=TOKENS["outline"],
        annotation_borderpad=5,
    )
    fig_sw.update_layout(
        title=f"<b>Sensitivity Curve: Impact of {sweep_feature} on Predicted Churn Risk</b>",
        margin=dict(l=35, r=35, t=55, b=35),
        xaxis=dict(
            title=dict(text=sweep_feature, font=dict(family="Inter", size=13, color=TOKENS["on_surface"])),
            tickfont=dict(family="Inter", size=12, color=TOKENS["on_surface"])
        ),
        yaxis=dict(
            title=dict(text="Estimated Churn Probability", font=dict(family="Inter", size=13, color=TOKENS["on_surface"])),
            tickfont=dict(family="Inter", size=12, color=TOKENS["on_surface"]),
            range=[0, 1.05]
        ),
    )
    render_precision_chart(fig_sw, height=360)


# ══════════════════════════════════════════════
# WORKSPACE 7: MODEL PERFORMANCE & FAIRNESS
# ══════════════════════════════════════════════

elif nav_selection == "Model Performance":
    st.markdown(f"""
    <div class="callout-info">
        Model governance & audit dashboard. Inspect cross-validated model benchmarks, discrimination curves, business cost matrices, and demographic parity checks.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="precision-card-header">📋 Multi-Model Benchmark Comparison</div>', unsafe_allow_html=True)
    if not metrics_table.empty:
        styled_cols = ["Model", "Accuracy", "Precision", "Recall", "F1_Score",
                       "ROC_AUC_Test", "ROC_AUC_CV_Mean", "ROC_AUC_CV_Std"]
        display_cols = [c for c in styled_cols if c in metrics_table.columns]
        
        st.dataframe(
            metrics_table[display_cols].style
            .highlight_max(subset=["ROC_AUC_Test", "F1_Score", "Recall"], color=TOKENS["low_risk_bg"])
            .format({c: "{:.4f}" for c in display_cols if c != "Model"}),
            use_container_width=True,
            height=230,
        )

    st.markdown("<div style='height:1.25rem;'></div>", unsafe_allow_html=True)
    m1, m2 = st.columns(2)

    with m1:
        st.markdown(f"""
        <div class="precision-card">
            <div class="precision-card-header">📈 ROC Discrimination Curves</div>
        </div>
        """, unsafe_allow_html=True)
        roc_path = os.path.join(FIGURES_DIR, "roc_curves.png")
        if os.path.exists(roc_path):
            st.image(roc_path, use_container_width=True)
        else:
            st.info("Run `python src/evaluate.py` to generate ROC curves.")

    with m2:
        st.markdown(f"""
        <div class="precision-card">
            <div class="precision-card-header">📊 Precision-Recall (PR) Curves</div>
        </div>
        """, unsafe_allow_html=True)
        pr_path = os.path.join(FIGURES_DIR, "pr_curves.png")
        if os.path.exists(pr_path):
            st.image(pr_path, use_container_width=True)
        else:
            st.info("Run `python src/evaluate.py` to generate PR curves.")

    # Confusion Matrices Full Width Card
    st.markdown("<div style='height:1.25rem;'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="precision-card">
        <div class="precision-card-header">🔲 Confusion Matrices Across All Evaluated Models (Default vs Calibrated Threshold)</div>
        <p style='font-size:0.84rem; color:var(--on-surface-variant); margin:0 0 0.85rem 0;'>
            Compares true positives, true negatives, false positives, and false negatives across all 5 benchmarked algorithms on the holdout test set.
        </p>
    </div>
    """, unsafe_allow_html=True)
    cm_path = os.path.join(FIGURES_DIR, "confusion_matrices.png")
    if os.path.exists(cm_path):
        st.image(cm_path, use_container_width=True)
    else:
        st.info("Run `python src/evaluate.py` to generate confusion matrices.")

    # Threshold & Financial Cost Analysis
    st.markdown("<div style='height:1.25rem;'></div>", unsafe_allow_html=True)
    th_col1, th_col2 = st.columns([1.1, 1])

    with th_col1:
        st.markdown(f"""
        <div class="precision-card">
            <div class="precision-card-header">⚖️ Business Cost Matrix Optimization Curve</div>
            <p style='font-size:0.84rem; color:var(--on-surface-variant); margin:0 0 0.6rem 0;'>
                Financial trade-off analysis evaluating False Negative cost (€500 lost customer) vs False Positive cost (€50 retention campaign).
            </p>
        </div>
        """, unsafe_allow_html=True)
        thresh_path = os.path.join(FIGURES_DIR, "threshold_analysis.png")
        if os.path.exists(thresh_path):
            st.image(thresh_path, use_container_width=True)
        else:
            st.info("Run `python src/evaluate.py` to generate threshold analysis.")

    with th_col2:
        st.markdown(f"""
        <div class="precision-card">
            <div class="precision-card-header">💰 Financial Optimization Summary</div>
            <div style="display:flex; flex-direction:column; gap:0.75rem; margin-top:0.5rem;">
                <div style="background:var(--surface-low); border:1px solid var(--outline-variant); border-radius:8px; padding:0.9rem;">
                    <div style="font-size:0.75rem; font-weight:700; color:var(--on-surface-variant); text-transform:uppercase;">Optimal Operating Threshold</div>
                    <div style="font-size:1.8rem; font-weight:800; color:var(--primary); margin:0.2rem 0;">0.12</div>
                    <div style="font-size:0.8rem; color:var(--on-surface-variant);">Calibrated from standard 0.50 cutoff to capture <b>87.96%</b> of at-risk accounts.</div>
                </div>
                <div style="background:var(--surface-low); border:1px solid var(--outline-variant); border-radius:8px; padding:0.9rem;">
                    <div style="font-size:0.75rem; font-weight:700; color:var(--on-surface-variant); text-transform:uppercase;">Portfolio Cost Reduction</div>
                    <div style="font-size:1.8rem; font-weight:800; color:var(--tertiary); margin:0.2rem 0;">-53.2%</div>
                    <div style="font-size:0.8rem; color:var(--on-surface-variant);">Reduces expected customer attrition losses by over <b>€94,000</b> across the verified book.</div>
                </div>
                <div style="background:var(--surface-low); border:1px solid var(--outline-variant); border-radius:8px; padding:0.9rem;">
                    <div style="font-size:0.75rem; font-weight:700; color:var(--on-surface-variant); text-transform:uppercase;">Cost Assumptions</div>
                    <div style="font-size:0.82rem; color:var(--on-surface); margin-top:0.35rem;">
                        • <b>False Negative (FN):</b> €500 (Loss of lifetime customer value)<br>
                        • <b>False Positive (FP):</b> €50 (Cost of retention outreach or fee incentive)
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Fairness Section
    st.markdown("<div style='height:1.25rem;'></div>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="precision-card">
        <div class="precision-card-header">⚖️ Demographic Subgroup Fairness Audit (Demographic Parity)</div>
        <p style='font-size:0.84rem; color:var(--on-surface-variant); margin:0 0 0.85rem 0;'>
            Empirical evaluation of true positive rates, false positive rates, and selection parity across Geography and Gender subgroups.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
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
        <div class="callout-info" style="border-left:4px solid var(--med-risk-fg); margin-top:0.75rem;">
            <b>Regulatory Compliance Mandate (ECB SREP & EBA Guidelines):</b> Demographic attributes (Territory & Gender) are validated to ensure no systemic exclusionary bias. The pipeline runs across behavioural vectors. Quarterly monitoring reports are retained for risk compliance.
        </div>
        """, unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Global Footer
# ──────────────────────────────────────────────

st.markdown(f"""
<div style='text-align:center; color:var(--on-surface-variant); font-size:0.8rem; padding:1.2rem 0 0.5rem 0; margin-top:1.5rem; border-top:1px solid var(--outline-variant);'>
    <b>ChurnIQ Enterprise Risk Suite</b> &nbsp;•&nbsp; Precision Intelligence Architecture &nbsp;•&nbsp;
    Inference Engine: {best_model_name} (Threshold: {threshold:.2f}) &nbsp;•&nbsp; Powered by XGBoost, Scikit-Learn & SHAP
</div>
""", unsafe_allow_html=True)
