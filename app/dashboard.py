"""
Credit Scoring Dashboard
------------------------
Main controller for the Credit Risk Application.
Orchestrates the flow between User Input (View) and Prediction Logic (Service).
"""

import streamlit as st
from services import CreditRiskService
from ui_components import (
    inject_custom_css, 
    render_sidebar_form, 
    render_results, 
    render_shap_explanation
)

# --- 1. CONFIGURATION ---
st.set_page_config(
    page_title="Credit Scoring System",
    layout="wide"
)

# --- 2. INITIALIZATION ---
# Dependency Injection: Service Layer
service = CreditRiskService()

# Global UI Setup
inject_custom_css()

# --- 3. LAYOUT ORCHESTRATION ---
st.title("Credit Scoring System")
st.markdown("### Internal Decision Support Tool")
st.markdown("---")

# Render View: Input Form
# The payload is fully constructed in the UI component to ensure separation of concerns.
payload = render_sidebar_form()

# --- 4. CONTROLLER LOGIC ---
if st.sidebar.button("Calculate Risk Score", type="primary"):
    try:
        with st.spinner("Analyzing Credit Profile..."):
            # A. Service Call
            result = service.get_prediction(payload)
            
            # B. Check for API-level errors gracefully
            if "error" in result:
                st.error(f"Scoring Engine Error: {result['error']}")
            else:
                # C. Render View: Results
                render_results(result)
                
                # D. Render View: Explainability
                if 'top_factors' in result:
                    st.markdown("---")
                    render_shap_explanation(result['top_factors'])
                    
    except Exception as e:
        # Catch-all for unexpected frontend/connection failures
        st.error(f"Application Error: {str(e)}")
        st.toast("Please check API connection.")
else:
    # Idle State
    st.info("Please configure the loan application parameters in the sidebar.")