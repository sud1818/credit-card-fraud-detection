import streamlit as st
import pandas as pd
import os

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💳",
    layout="centered"
)

# -------------------------------------------------
# Page Header
# -------------------------------------------------
st.markdown(
    """
    <h1 style='text-align:center; color:#2A6AB0;'>💳 Credit Card Fraud Detection</h1>
    <p style='text-align:center; font-size:18px;'>
        Enter transaction details below and the machine learning model will 
        detect whether the transaction is <b>Legitimate</b> or <b>Fraudulent</b>.
    </p>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------
# Safely Import joblib
# -------------------------------------------------
try:
    import joblib
except ModuleNotFoundError as e:
    st.error(
        f"❌ <b>joblib is not installed</b><br><br>"
        f"Install it using:<br><code>pip install joblib</code><br><br>"
        f"<b>Error:</b> {e}",
        unsafe_allow_html=True
    )
    st.stop()

# -------------------------------------------------
# Load Model Safely
# -------------------------------------------------
MODEL_PATH = "decision_tree_gini_model.pkl"

if not os.path.exists(MODEL_PATH):
    st.error(
        f"❌ <b>Model not found!</b><br>"
        f"Place <code>{MODEL_PATH}</code> in the same folder as the app.",
        unsafe_allow_html=True
    )
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(
        f"❌ Failed to load model.<br>"
        f"Possible reasons:<br>"
        f"- Different scikit-learn version<br>"
        f"- Corrupted model file<br><br>"
        f"<b>Error:</b> {e}",
        unsafe_allow_html=True
    )
    st.stop()

# -------------------------------------------------
# Input Section
# -------------------------------------------------
st.markdown(
    """
    <h3 style='margin-top:30px; color:#2A6AB0;'>📌 Enter Transaction Details</h3>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns(2)

with col1:
    transaction_amount = st.number_input(
        "💰 Transaction Amount",
        min_value=0.01,
        value=100.0,
        help="Total amount spent in this transaction"
    )

    account_age_days = st.number_input(
        "📅 Account Age (days)",
        min_value=1.0,
        value=120.0,
        help="Number of days since account was opened"
    )

with col2:
    transaction_time = st.number_input(
        "⏱ Transaction Time (sec since midnight)",
        min_value=0.0,
        max_value=86400.0,
        value=35000.0,
        help="Time of day the transaction occurred"
    )

    transaction_velocity = st.number_input(
        "📈 Transaction Velocity (transactions/hour)",
        min_value=0.0,
        max_value=20.0,
        value=3.0,
        help="How many transactions happen in last hour"
    )

merchant_risk_score = st.slider(
    "🏪 Merchant Risk Score",
    0.0, 1.0, 0.5,
    help="Risk rating of merchant (0 = Safe, 1 = Very Risky)"
)

# -------------------------------------------------
# Convert to DataFrame
# -------------------------------------------------
input_data = pd.DataFrame({
    "Transaction_Amount": [transaction_amount],
    "Transaction_Time": [transaction_time],
    "Account_Age_Days": [account_age_days],
    "Merchant_Risk_Score": [merchant_risk_score],
    "Transaction_Velocity": [transaction_velocity]
})

# Preview
st.markdown(
    "<h4 style='color:#2A6AB0;'>📊 Input Summary</h4>",
    unsafe_allow_html=True
)
st.dataframe(input_data, use_container_width=True)

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if st.button("🔍 Run Fraud Detection"):
    try:
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            st.success(
                "🟢 **Legitimate Transaction**\n\n"
                "No suspicious behavior detected."
            )
        else:
            st.error(
                "🔴 **Fraudulent Transaction Detected!**\n\n"
                "Model determined that this transaction is high-risk."
            )
    except Exception as e:
        st.error(f"❌ Prediction failed.\n\nError: {e}")
