# -------------------------------------
# app.py  (Error-Free, Final Version)
# -------------------------------------

import streamlit as st
import pandas as pd

# Try importing joblib safely
try:
    import joblib
except ModuleNotFoundError:
    st.title("❌ Missing Library - joblib")
    st.error(
        "joblib is not installed.\n\n"
        "Please run the following command:\n\n"
        "```bash\npip install joblib\n```"
    )
    st.stop()

# Try loading the model safely
try:
    model = joblib.load("rf_fraud_model.pkl")
except FileNotFoundError:
    st.title("⚠ Model File Not Found")
    st.error(
        "The model file `rf_fraud_model.pkl` is missing.\n"
        "Train the model and place it in the same folder as `app.py`."
    )
    st.stop()
except Exception as e:
    st.title("⚠ Error Loading Model")
    st.error(f"Error details:\n\n{str(e)}")
    st.stop()

# ------------------------------------------------------
# STREAMLIT UI
# ------------------------------------------------------
st.set_page_config(page_title="Fraud Detection", page_icon="💳")
st.title("💳 Credit Card Fraud Detection App")
st.write("Enter transaction details below:")

transaction_amount = st.number_input(
    "Transaction Amount (₹)", min_value=1.0
)

transaction_time = st.number_input(
    "Transaction Time (seconds since midnight)",
    min_value=0.0,
    max_value=86400.0,
    step=1.0
)

account_age_days = st.number_input(
    "Account Age (Days)",
    min_value=1.0
)

merchant_risk_score = st.slider(
    "Merchant Risk Score (0 to 1)", 0.0, 1.0, 0.5
)

transaction_velocity = st.number_input(
    "Transaction Velocity (transactions/hour)",
    min_value=0.0,
    max_value=50.0
)

# Create input dataframe
input_data = pd.DataFrame({
    "Transaction_Amount": [transaction_amount],
    "Transaction_Time": [transaction_time],
    "Account_Age_Days": [account_age_days],
    "Merchant_Risk_Score": [merchant_risk_score],
    "Transaction_Velocity": [transaction_velocity]
})

# ------------------------------------------------------
# PREDICT BUTTON
# ------------------------------------------------------
if st.button("Predict"):
    try:
        probability = model.predict_proba(input_data)[0][1]
        threshold = 0.35  # custom fraud threshold

        if probability > threshold:
            st.error(f"🚨 Fraud Detected! (Risk Score: {probability:.2f})")
        else:
            st.success(f"✅ Legitimate Transaction (Risk Score: {probability:.2f})")

    except Exception as e:
        st.error(
            "Prediction failed.\n"
            "Possible reason: the model and input features do not match.\n"
            f"Error: {str(e)}"
        )
        pip install joblib
        


