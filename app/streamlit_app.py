"""FloodSense: Regional Flood Risk Assessment & Prediction Dashboard.

Built with Streamlit for regional planning authorities and disaster management teams.
"""

import os
import sys
import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st

# Configure page layout and style
st.set_page_config(
    page_title="FloodSense | Regional Flood Risk Analysis",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Ensure project root is on Python path so joblib resolves src.preprocess
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "final_model.joblib")
CUTOFFS_PATH = os.path.join(PROJECT_ROOT, "models", "tier_cutoffs.json")
METRICS_PATH = os.path.join(PROJECT_ROOT, "models", "metrics.json")
RESULTS_PATH = os.path.join(PROJECT_ROOT, "reports", "results", "model_comparison.csv")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "reports", "figures")

# Define feature metadata
ENV_FEATURES = [
    ("MonsoonIntensity", "Monsoon Intensity", 0, 16, 5, "Severity and duration of seasonal monsoon rainfall"),
    ("TopographyDrainage", "Topography Drainage", 0, 17, 5, "Natural slope and gradient efficiency for rainwater runoff"),
    ("Deforestation", "Deforestation", 0, 16, 5, "Extent of forest cover cleared in the surrounding catchment"),
    ("ClimateChange", "Climate Change", 0, 16, 5, "Observed frequency of extreme weather anomalies"),
    ("CoastalVulnerability", "Coastal Vulnerability", 0, 16, 5, "Exposure to sea-level surges, tides, and coastal erosion"),
    ("Landslides", "Landslides Risk", 0, 16, 5, "Susceptibility to slope failures and debris blocking rivers"),
    ("Watersheds", "Watershed Degradation", 0, 16, 5, "Loss of natural retention and water catchment capacity"),
    ("WetlandLoss", "Wetland Loss", 0, 16, 5, "Destruction of natural flood buffering marshes and ponds")
]

INFRA_FEATURES = [
    ("RiverManagement", "River Management Deficit", 0, 16, 5, "Poor river basin regulation, lack of dredging, or weak embankments (higher = greater risk)"),
    ("Urbanization", "Urbanization", 0, 16, 5, "Proportion of land covered with impermeable concrete (higher = more runoff)"),
    ("DamsQuality", "Dams Vulnerability (Quality Defect)", 0, 16, 5, "Structural deterioration, age, or failure hazard of dams (higher = greater risk)"),
    ("Siltation", "Siltation Level", 0, 16, 5, "Sediment accumulation reducing river and canal depth"),
    ("AgriculturalPractices", "Agricultural Practices", 0, 16, 5, "Runoff-inducing land cultivation and lack of soil contouring"),
    ("Encroachments", "Encroachments", 0, 16, 5, "Illegal structures and human settlements on natural drainage pathways"),
    ("IneffectiveDisasterPreparedness", "Disaster Preparedness Gap", 0, 16, 5, "Deficiency in early warning systems and evacuation infrastructure"),
    ("DrainageSystems", "Drainage System Deficiency", 0, 17, 5, "Inadequacy, blockages, or absence of stormwater drainage networks (higher = greater risk)"),
    ("DeterioratingInfrastructure", "Deteriorating Infrastructure", 0, 16, 5, "Age and failure risk of culverts, bridges, and sluices"),
    ("PopulationScore", "Population Density Exposure", 0, 16, 5, "Concentration of communities in low-lying hazard zones"),
    ("InadequatePlanning", "Inadequate Planning", 0, 16, 5, "Poor zoning laws and uncontrolled floodplain development"),
    ("PoliticalFactors", "Governance & Policy Factors", 0, 16, 5, "Policy roadblocks, resource delays, and inter-agency gaps")
]

# Canonical training feature order as fitted by scikit-learn
TRAINING_FEATURE_ORDER = [
    "MonsoonIntensity",
    "TopographyDrainage",
    "RiverManagement",
    "Deforestation",
    "Urbanization",
    "ClimateChange",
    "DamsQuality",
    "Siltation",
    "AgriculturalPractices",
    "Encroachments",
    "IneffectiveDisasterPreparedness",
    "DrainageSystems",
    "CoastalVulnerability",
    "Landslides",
    "Watersheds",
    "DeterioratingInfrastructure",
    "PopulationScore",
    "WetlandLoss",
    "InadequatePlanning",
    "PoliticalFactors"
]


@st.cache_resource
def load_model_and_cutoffs():
    """Load model pipeline and risk cutoffs safely with error checks."""
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CUTOFFS_PATH):
        return None, None
    try:
        model = joblib.load(MODEL_PATH)
        with open(CUTOFFS_PATH, "r", encoding="utf-8") as f:
            cutoffs = json.load(f)
        return model, cutoffs
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        return None, None


# App Header
st.title("🌊 FloodSense: Regional Flood Risk Assessment")
st.markdown(
    "**Decision Support System for Regional Authorities** | Predicts continuous regional flood probability and classifies actionable operational risk tiers (**Low**, **Medium**, **High**) based on 20 hydro-environmental and governance indicators."
)
st.divider()

# Check file availability
model, cutoffs = load_model_and_cutoffs()

if model is None or cutoffs is None:
    st.error(
        "⚠️ **Model Artifacts Not Found!**\n\n"
        "The required files `models/final_model.joblib` or `models/tier_cutoffs.json` are missing.\n\n"
        "**To resolve this:** Please execute the training script in your terminal:\n"
        "```bash\npython flood-risk/src/train.py\n```"
    )
    st.stop()

# Main Application Tabs
tab1, tab2 = st.tabs(["🎯 Regional Risk Assessment", "📊 Model Performance & Audit"])

# ==============================================================================
# TAB 1: REGIONAL RISK ASSESSMENT
# ==============================================================================
with tab1:
    st.markdown("### 1. Set Regional Indicator Scores")
    st.caption("Adjust the survey scores for the regional area (scores range from 0 to 16/17, with 5 being the baseline median).")

    col_env, col_infra = st.columns(2, gap="large")

    input_data = {}

    with col_env:
        st.subheader("🌿 Environmental & Hydrological Indicators")
        for col_id, label, min_val, max_val, default_val, help_text in ENV_FEATURES:
            input_data[col_id] = st.slider(
                label=f"{label} ({min_val}–{max_val})",
                min_value=min_val,
                max_value=max_val,
                value=default_val,
                step=1,
                help=help_text,
                key=f"input_{col_id}"
            )

    with col_infra:
        st.subheader("🏗️ Infrastructure, Demographic & Governance")
        for col_id, label, min_val, max_val, default_val, help_text in INFRA_FEATURES:
            input_data[col_id] = st.slider(
                label=f"{label} ({min_val}–{max_val})",
                min_value=min_val,
                max_value=max_val,
                value=default_val,
                step=1,
                help=help_text,
                key=f"input_{col_id}"
            )

    st.divider()

    # Create input DataFrame with exact canonical training column order
    cols_to_use = list(getattr(model, "feature_names_in_", TRAINING_FEATURE_ORDER))
    input_df = pd.DataFrame([input_data])[cols_to_use]

    # Predict
    try:
        pred_prob = float(model.predict(input_df)[0])
    except Exception as e:
        st.error(f"Inference error: {e}")
        st.stop()

    # Determine Tier
    q33 = cutoffs["q33"]
    q67 = cutoffs["q67"]

    if pred_prob <= q33:
        tier_label = "LOW RISK"
        tier_color = "#28a745"  # Green
        tier_bg = "#d4edda"
        tier_border = "#c3e6cb"
        recommendation = "✅ **Routine Surveillance:** The region exhibits low composite flood vulnerability. Standard drainage maintenance and routine seasonal monitoring are recommended."
    elif pred_prob <= q67:
        tier_label = "MEDIUM RISK"
        tier_color = "#d39e00"  # Orange/Amber
        tier_bg = "#fff3cd"
        tier_border = "#ffeeba"
        recommendation = "⚠️ **Elevated Caution:** The region exhibits moderate vulnerability. Local authorities should verify sluice gate functionality, clear stormwater canals, and alert regional first responders."
    else:
        tier_label = "HIGH RISK"
        tier_color = "#dc3545"  # Red
        tier_bg = "#f8d7da"
        tier_border = "#f5c6cb"
        recommendation = "🚨 **Priority Emergency Intervention:** High composite flood vulnerability detected. Immediate mobilization of flood defense assets, pre-positioning of evacuation personnel, and high-priority infrastructure budget allocation required."

    # Prediction Output Banner
    st.markdown("### 2. Assessment Results")

    res_col1, res_col2 = st.columns([1, 2], gap="medium")

    with res_col1:
        st.metric(
            label="Predicted Flood Probability",
            value=f"{pred_prob * 100:.2f}%",
            delta=f"{(pred_prob - 0.5043) * 100:+.2f}% vs regional mean"
        )
        st.caption(f"Valid Probability Range: 28.5% – 72.5% | Median Baseline: 50.5%")

    with res_col2:
        st.markdown(
            f"""
            <div style="background-color: {tier_bg}; border: 2px solid {tier_border}; border-left: 8px solid {tier_color}; border-radius: 8px; padding: 18px 20px; margin-top: 5px;">
                <span style="font-size: 13px; font-weight: 700; text-transform: uppercase; color: {tier_color}; letter-spacing: 1px;">Assigned Risk Tier</span>
                <h2 style="margin: 4px 0 10px 0; color: {tier_color}; font-size: 28px; font-weight: 800;">{tier_label}</h2>
                <p style="margin: 0; font-size: 15px; color: #333333; line-height: 1.5;">{recommendation}</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    # Contextual Guidance Expandable Box
    with st.expander("ℹ️ How are risk tiers determined?"):
        st.markdown(
            f"""
            - **Low Risk ($\\\\le {q33*100:.1f}\\%$):** Bottom one-third of regional vulnerability scores.
            - **Medium Risk (${q33*100:.1f}\\% - {q67*100:.1f}\\%$):** Middle one-third of regional vulnerability scores.
            - **High Risk ($> {q67*100:.1f}\\%$):** Top one-third of highest-risk regional territories.
            
            *Note:* Thresholds are calculated strictly using percentiles from the 80,000-sample training dataset to ensure fair and balanced resource distribution across all jurisdictions.
            """
        )

# ==============================================================================
# TAB 2: MODEL PERFORMANCE & AUDIT
# ==============================================================================
with tab2:
    st.markdown("### Model Architecture & Performance Validation")
    st.caption("Detailed statistical evaluation on the independent 20,000-sample holdout test set.")

    # High-level metrics cards
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    m_col1.metric("Test R² Score", "0.8709", help="Explains 87.1% of total variance in flood risk")
    m_col2.metric("Test RMSE", "0.0185", help="Root Mean Squared Error on unseen test data")
    m_col3.metric("Test MAE", "0.0144", help="Mean Absolute Error (sub-1.5% accuracy)")
    m_col4.metric("High-Risk Recall", "90.51%", help="0 true high-risk regions missed as Low Risk")

    st.divider()

    # Section A: Full Comparison Table
    st.subheader("1. Experimental Model Comparison (5-Fold CV + Test)")
    if os.path.exists(RESULTS_PATH):
        df_comp = pd.read_csv(RESULTS_PATH)
        st.dataframe(df_comp, use_container_width=True, hide_index=True)
    else:
        st.warning("Model comparison table not found at reports/results/model_comparison.csv.")

    st.divider()

    # Section B: Diagnostic Plots
    st.subheader("2. Diagnostic Visualizations")
    img_col1, img_col2 = st.columns(2)

    with img_col1:
        cm_fig_path = os.path.join(FIGURES_DIR, "confusion_matrix.png")
        if os.path.exists(cm_fig_path):
            st.image(cm_fig_path, caption="Confusion Matrix: 3-Class Risk Tier Classification (20k Test Samples)", use_container_width=True)

    with img_col2:
        perm_fig_path = os.path.join(FIGURES_DIR, "permutation_importance.png")
        if os.path.exists(perm_fig_path):
            st.image(perm_fig_path, caption="Permutation Feature Importance (Decrease in Test R²)", use_container_width=True)

    img_col3, img_col4 = st.columns(2)
    with img_col3:
        pred_act_fig = os.path.join(FIGURES_DIR, "predicted_vs_actual.png")
        if os.path.exists(pred_act_fig):
            st.image(pred_act_fig, caption="Predicted vs. Actual Probability Scatter (Test Set)", use_container_width=True)

    with img_col4:
        res_hist_fig = os.path.join(FIGURES_DIR, "residual_histogram.png")
        if os.path.exists(res_hist_fig):
            st.image(res_hist_fig, caption="Residual Error Distribution (Centered at Zero)", use_container_width=True)

    st.divider()

    # Section C: Governance Limitations
    st.subheader("3. Academic Limitations & Governance Notice")
    st.markdown(
        """
        1. **Regional Screening Scope:** FloodSense is an administrative risk-screening tool based on standardized survey scores, not a real-time physical hydrodynamic forecasting system.
        2. **Synthetic Data Characteristics:** Features are integer indicators without physical dimensions (like river discharge in $\\text{m}^3/\\text{s}$ or precipitation in millimeters).
        3. **No Spatial Topology or Time Series:** Models do not incorporate elevation digital elevation models (DEM) or temporal sequence lags (rainfall accumulated over preceding days).
        4. **Relative Quantile Thresholds:** The risk tiers reflect relative percentile rankings across surveyed territories rather than absolute physical flood inundation depths.
        """
    )
