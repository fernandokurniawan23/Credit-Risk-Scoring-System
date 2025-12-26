"""
app/ui_components.py
View Layer: Handles UI rendering, CSS styling, and visual components.
"""
import streamlit as st
from typing import Dict, Any
from assets import ICON_CREDIT_SCORE, ICON_RISK_PROB, ICON_DECISION

def inject_custom_css():
    """Injects professional CSS for cards and typography."""
    st.markdown("""
        <style>
        .metric-container {
            background-color: #ffffff;
            border: 1px solid #e0e0e0;
            border-radius: 8px;
            padding: 20px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
            margin-bottom: 20px;
            display: flex;
            align-items: center;
        }
        .icon-box {
            margin-right: 15px;
            display: flex;
            align-items: center;
            justify-content: center;
            width: 40px; 
            height: 40px;
        }
        .text-box h3 {
            margin: 0;
            font-size: 14px;
            color: #888;
            font-weight: 500;
            text-transform: uppercase;
            letter-spacing: 1px;
        }
        .text-box h1 {
            margin: 0;
            font-size: 28px;
            color: #333;
            font-weight: 700;
        }
        .text-box p {
            margin: 0;
            font-size: 12px;
            color: #666;
        }
        .status-approve { color: #1cc88a !important; }
        .status-reject { color: #e74a3b !important; }
        .status-review { color: #f6c23e !important; }
        </style>
    """, unsafe_allow_html=True)

def render_sidebar_form() -> Dict[str, Any]:
    """Renders the input form in the sidebar and returns valid payload."""
    
    st.sidebar.title("Applicant Profile")
    st.sidebar.caption("Input customer details below")
    st.sidebar.markdown("---")
    
    sk_id = st.sidebar.number_input("Application ID", value=100001)
    contract_type = st.sidebar.selectbox("Contract Type", ["Cash loans", "Revolving loans"])
    gender = st.sidebar.selectbox("Gender", ["M", "F"])
    
    st.sidebar.subheader("Financials")
    income = st.sidebar.number_input("Total Income (Yearly)", value=500000.0, step=10000.0)
    credit_amount = st.sidebar.number_input("Credit Amount", value=200000.0, step=50000.0)
    annuity = st.sidebar.number_input("Annuity", value=10000.0)
    goods_price = st.sidebar.number_input("Goods Price", value=900000.0)
    
    st.sidebar.subheader("Demographics")
    age = st.sidebar.slider("Age", 20, 70, 45)
    employed = st.sidebar.slider("Years Employed", 0, 40, 15)
    
    st.sidebar.subheader("External Data")
    ext2 = st.sidebar.slider("External Source 2", 0.0, 1.0, 0.8)
    ext3 = st.sidebar.slider("External Source 3", 0.0, 1.0, 0.8)

    return {
        "SK_ID_CURR": sk_id,
        "NAME_CONTRACT_TYPE": contract_type,
        "CODE_GENDER": gender,
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit_amount,
        "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": goods_price,
        "DAYS_EMPLOYED": int(employed * -365),
        "DAYS_BIRTH": int(age * -365),
        "EXT_SOURCE_2": ext2,
        "EXT_SOURCE_3": ext3
    }

def render_metric_card(title: str, value: str, subtext: str, icon_svg: str, color_class: str = ""):
    """Renders a single metric card using HTML/CSS/SVG."""
    
    # FIX: Menghilangkan newline agar tidak dianggap code block oleh Markdown
    html = f"""
    <div class="metric-container">
        <div class="icon-box">{icon_svg}</div>
        <div class="text-box">
            <h3>{title}</h3>
            <h1 class="{color_class}">{value}</h1>
            <p>{subtext}</p>
        </div>
    </div>
    """
    # Critical Fix: Strip newlines
    st.markdown(html.replace('\n', ''), unsafe_allow_html=True)

def render_shap_explanation(factors: list):
    """Renders SHAP chart."""
    import pandas as pd
    st.markdown("### 🔍 Decision Factors")
    st.markdown("Top drivers affecting this score:")
    
    data = []
    for f in factors:
        data.append({
            "Feature": f['feature'],
            "Impact": f['impact'],
            "Direction": "Increases Risk" if f['impact'] > 0 else "Reduces Risk"
        })
        
    df_shap = pd.DataFrame(data)
    
    # Visual Chart
    st.bar_chart(df_shap.set_index("Feature")['Impact'])
    
    # Data Table
    with st.expander("Detailed Breakdown"):
        st.dataframe(df_shap)

def render_results(result: Dict[str, Any]):
    """Orchestrates the display of results."""
    col1, col2, col3 = st.columns(3)
    
    decision = result['decision']
    color_class = "status-review"
    if decision == "APPROVE": color_class = "status-approve"
    elif decision == "REJECT": color_class = "status-reject"

    with col1:
        render_metric_card(
            "Credit Score", 
            str(result['credit_score']), 
            "FICO Scaled (0-1000)", 
            ICON_CREDIT_SCORE
        )
    
    with col2:
        render_metric_card(
            "Risk Probability", 
            f"{result['probability_default']:.1%}", 
            "Likelihood of Default", 
            ICON_RISK_PROB
        )
        
    with col3:
        render_metric_card(
            "Final Decision", 
            decision, 
            f"Tier: {result['risk_tier']}", 
            ICON_DECISION,
            color_class
        )