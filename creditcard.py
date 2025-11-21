# ---------------------------------------------------
# app.py  (Complete Working Version)
# ---------------------------------------------------

import streamlit as st
import pandas as pd

# -------------------------------
# SAFE IMPORT OF JOBLIB
# -------------------------------
try:
    import joblib
except ModuleNotFoundError:
    st.title("❌ Missing Dependency: joblib")
    st.error(
        "Please install joblib using:\n"
        "```bash\npip install joblib\n```"
    )
    st.stop()

# -------------------------------
# SAFE MODEL LOAD
# -------------------------------
try:
    model = joblib.load("rf_fraud_model.pkl")
except FileNotFoundError:
    st.title("⚠ Model File Not Found")
    st.error(
        "The model file `rf_fraud_model.pkl` is missing.\n"
        "Train your model and place it in the same folder as app.py."
    )
    st.stop()
except Exception as e:
    st.title("⚠ Error Loading Model")
    st.error(f"Error details:\n\n{str(e)}")
    st.stop()

# ---------------------------------------------------
# STREAMLIT UI
# ---------------------------------------------------
st.set_page_config(page_title="Fraud Detection", page_icon="💳")
st.title("💳 Credit Card Fraud Detection System")
st.write("Enter transaction details to predict fraud probability.")

# ---------------------------------------------------
# USER INPUTS (YOUR REQUIRED FIELDS)
# ---------------------------------------------------

transaction_amount = st.number_input(
    "Transaction Amount (₹)",
    min_value=1.0,
    step=1.0
)

transaction_time = st.number_input(
    "Transaction Time – seconds since midnight",
    min_value=0.0,
    max_value=86400.0,
    step=1.0
)

account_age_days = st.number_input(
    "Account Age (Days)",
    min_value=1.0,
    step=1.0
)

merchant_risk_score = st.slider(
    "Merchant Risk Score (0 to 1)",
    0.0,
    1.0,
    0.5
)

transaction_velocity = st.number_input(
    "Transaction Velocity – transactions/hour",
    min_value=0.0,
    max_value=50.0,
    step=0.1
)

# ---------------------------------------------------
# CREATE INPUT DATAFRAME (input_data)
# ---------------------------------------------------

input_data = pd.DataFrame({
    "Transaction_Amount": [transaction_amount],
    "Transaction_Time": [transaction_time],
    "Account_Age_Days": [account_age_days],
    "Merchant_Risk_Score": [merchant_risk_score],
    "Transaction_Velocity": [transaction_velocity]
})

st.subheader("🔎 Input Data")
st.write(input_data)

# ---------------------------------------------------
# PREDICTION
# ---------------------------------------------------
if st.button("Predict"):
    try:
        fraud_prob = model.predict_proba(input_data)[0][1]
        threshold = 0.35    # Fraud threshold

        if fraud_prob > threshold:
            st.error(
                f"🚨 Fraudulent Transaction Detected!\n"
                f"Risk Score: {fraud_prob:.2f}"
            )
        else:
            st.success(
                f"✅ Transaction is Legitimate\n"
                f"Risk Score: {fraud_prob:.2f}"
            )

    except Exception as e:
        st.error(
            "Prediction failed. Model and input features may not match.\n"
            f"Error: {str(e)}"
        )
