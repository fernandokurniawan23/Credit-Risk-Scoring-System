"""
Credit Scoring Dashboard
------------------------
Main controller for the Credit Risk Application.

Architecture:
    - View Layer: ui_components.py (Handles rendering & CSS)
    - Service Layer: services.py (Handles API communication)
    - Controller: dashboard.py (Orchestrates user interaction)
"""

import streamlit as st
from services import CreditRiskService
from ui_components import (
    inject_custom_css, 
    render_sidebar_form, 
    render_results, 
    render_shap_explanation
)

# --- 1. APPLICATION CONFIGURATION ---
st.set_page_config(
    page_title="Credit Scoring System", 
    layout="wide"
)

# --- 2. INITIALIZATION ---
# Instantiate service layer for API communication
service = CreditRiskService()

# Inject global CSS styles
inject_custom_css()

# --- 3. MAIN LAYOUT ---
st.title("Credit Scoring System")
st.markdown("Internal Decision Support Tool")

# Render Sidebar (Input Form) & Capture Payload
payload = render_sidebar_form()

# --- 4. INTERACTION LOGIC ---
if st.sidebar.button("Calculate Risk Score", type="primary"):
    try:
        with st.spinner("Connecting to Scoring Engine..."):
            # Execute Prediction (Service Layer)
            result = service.get_prediction(payload)
            
            # Render Main Metrics (View Layer)
            render_results(result)
            
            # Render Explainability / SHAP (View Layer)
            st.markdown("---")
            if 'top_factors' in result:
                render_shap_explanation(result['top_factors'])
            
    except Exception as e:
        # Graceful Error Handling
        st.error(f"System Error: {str(e)}")
else:
    # Default State Prompt
    st.info("Please configure the loan application in the sidebar to proceed.")