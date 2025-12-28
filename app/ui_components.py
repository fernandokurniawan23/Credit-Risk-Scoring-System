"""
UI Components Module (Professional SVG Version)
-----------------------------------------------
Handles frontend rendering with rich visualizations using Plotly.
Replaces Emojis with inline SVG for a cleaner, enterprise look.
"""

import streamlit as st
import plotly.express as px
import pandas as pd
from typing import Dict, Any

def inject_custom_css() -> None:
    """
    Injects professional CSS variables and layout adjustments.
    """
    st.markdown(
        """
        <style>
            div[data-testid="metric-container"] {
                background-color: #1E1E1E;
                border: 1px solid #333;
                padding: 20px;
                border-radius: 8px;
                color: white;
            }
            
            div[data-testid="metric-container"] label {
                color: #A0A0A0;
            }
            
            /* Typography */
            h1, h2, h3 { font-family: 'Inter', sans-serif; }
            
            /* SVG Alignment Helper */
            .decision-box {
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 10px;
                padding: 15px;
                border-radius: 8px;
            }
        </style>
        """,
        unsafe_allow_html=True
    )

def render_sidebar_form() -> Dict[str, Any]:
    """
    Renders the sidebar input form.
    """
    st.sidebar.header("Applicant Profile")
    st.sidebar.caption("Input customer details below")

    # --- 1. Contract Info ---
    with st.sidebar.expander("Application ID & Type", expanded=True):
        contract_type = st.selectbox("Contract Type", ["Cash loans", "Revolving loans"])
        gender = st.selectbox("Gender", ["M", "F"])
    
    # --- 2. Financials ---
    with st.sidebar.expander("Financials", expanded=True):
        income = st.number_input("Total Income (Yearly)", min_value=0.0, value=200000.0, step=10000.0)
        credit_amount = st.number_input("Credit Amount", min_value=0.0, value=500000.0, step=10000.0)
        annuity = st.number_input("Annuity", min_value=0.0, value=25000.0, step=1000.0)
        goods_price = st.number_input("Goods Price", min_value=0.0, value=500000.0, step=10000.0)

    # --- 3. Demographics ---
    with st.sidebar.expander("Demographics", expanded=True):
        age_years = st.slider("Age", 20, 70, 35)
        employed_years = st.slider("Years Employed", 0, 40, 8)
        
        st.caption("Document Validity")
        id_age_years = st.slider(
            "ID Document Age (Years)", 
            min_value=0.0, 
            max_value=50.0, 
            value=1.0, 
            step=0.5,
            help="0 means new ID."
        )

    # --- 4. External Data ---
    with st.sidebar.expander("External Data", expanded=True):
        ext_source_2 = st.slider("External Source 2", 0.0, 1.0, 0.5)
        ext_source_3 = st.slider("External Source 3", 0.0, 1.0, 0.49)

    # Payload Construction
    payload = {
        "SK_ID_CURR": 100003, 
        "NAME_CONTRACT_TYPE": contract_type,
        "CODE_GENDER": gender,
        "AMT_INCOME_TOTAL": float(income),
        "AMT_CREDIT": float(credit_amount),
        "AMT_ANNUITY": float(annuity),
        "AMT_GOODS_PRICE": float(goods_price),
        "DAYS_BIRTH": int(age_years * -365),
        "DAYS_EMPLOYED": int(employed_years * -365),
        "DAYS_ID_PUBLISH": int(id_age_years * -365),
        "EXT_SOURCE_2": float(ext_source_2),
        "EXT_SOURCE_3": float(ext_source_3),
        "EXT_SOURCE_1": 0.5,
        "NAME_EDUCATION_TYPE": "Secondary / secondary special",
        "NAME_FAMILY_STATUS": "Married"
    }
    
    return payload

def render_results(result: Dict[str, Any]) -> None:
    """
    Renders the prediction results using SVG Icons instead of Emojis.
    """
    
    risk_tier = result.get('risk_tier', 'Unknown')
    decision = result.get('decision', 'N/A')
    
    # --- DEFINISI SVG ICON (Format: string HTML) ---
    # Check Icon (Green)
    svg_check = """
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <polyline points="20 6 9 17 4 12"></polyline>
    </svg>
    """
    
    # X/Cross Icon (Red)
    svg_x = """
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <circle cx="12" cy="12" r="10"></circle>
        <line x1="15" y1="9" x2="9" y2="15"></line>
        <line x1="9" y1="9" x2="15" y2="15"></line>
    </svg>
    """
    
    # Alert Icon (Orange)
    svg_alert = """
    <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
        <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path>
        <line x1="12" y1="9" x2="12" y2="13"></line>
        <line x1="12" y1="17" x2="12.01" y2="17"></line>
    </svg>
    """

    # --- Logic color & Icon ---
    if risk_tier == "High Risk":
        bg_color = "#B91C1C"  # Dark Red
        icon_svg = svg_x
    elif risk_tier == "Medium Risk":
        bg_color = "#D97706"  # Dark Orange
        icon_svg = svg_alert
    else:
        bg_color = "#047857"  # Dark Green
        icon_svg = svg_check

    st.subheader("Credit Decision Engine")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            label="CREDIT SCORE", 
            value=result.get('credit_score', 0),
            delta="FICO Scaled"
        )
        
    with col2:
        pd_val = result.get('probability_default', 0)
        st.metric(
            label="RISK PROBABILITY", 
            value=f"{pd_val:.1%}",
            delta="Likelihood of Default",
            delta_color="inverse"
        )
        
    with col3:
        st.markdown(f"**FINAL DECISION**")
        # Render HTML with svg
        st.markdown(
            f"""
            <div class="decision-box" style="background-color: {bg_color};">
                <div style="width: 24px; height: 24px; display:flex;">{icon_svg}</div>
                <div>
                    <h3 style="color: white; margin:0; font-size: 1.1em;">{decision}</h3>
                    <p style="color: rgba(255,255,255,0.8); margin:0; font-size: 0.8em;">Tier: {risk_tier}</p>
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

def render_shap_explanation(factors: list) -> None:
    """
    Renders the SHAP values using Plotly and Tables.
    Clean text only, no emojis.
    """
    st.markdown("---")
    st.subheader("Decision Factors")
    
    if not factors:
        st.warning("No explanation factors available.")
        return

    # 1. Prepare Data
    df = pd.DataFrame(factors)
    df = df.sort_values(by="impact", ascending=True)

    # 2. Colors: Red for Risk Increase, Blue for Risk Decrease
    df['Color'] = df['impact'].apply(lambda x: '#FF4B4B' if x > 0 else '#0068C9')
    
    fig = px.bar(
        df, 
        x="impact", 
        y="feature", 
        orientation='h',
        text_auto='.4f',
        title="Impact on Risk Probability"
    )
    
    fig.update_traces(marker_color=df['Color'], textposition='outside')
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="white"),
        xaxis=dict(showgrid=True, gridcolor='#333'),
        margin=dict(l=0, r=0, t=30, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)

    # 3. Clean Table (No Emojis)
    with st.expander("Detailed Breakdown (Table View)", expanded=False):
        display_df = df.sort_values(by="impact", ascending=False).copy()
        display_df = display_df[['feature', 'impact', 'direction']]
        
        st.dataframe(
            display_df, 
            use_container_width=True,
            column_config={
                "impact": st.column_config.NumberColumn(format="%.4f"),
            }
        )