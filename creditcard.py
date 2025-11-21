# ------------------------------------
# app.py  (Crash-proof version)
# ------------------------------------
import streamlit as st

# Try importing modules safely
missing_modules = []
try:
    import pandas as pd
except:
    missing_modules.append("pandas")

try:
    import joblib
except:
    missing_modules.append("joblib")

# Check for missing dependencies
if missing_modules:
    st.title("❌ Missing Dependencies")
    st.error(
        "The following required modules are missing:\n\n"
        + "\n".join(f"- {m}" for m in missing_modules)
        + "\n\nInstall them using:\n"
        "```bash\npip install -r requirements.txt\n```"
    )
    st.stop()

# Try loading model safely
model = None
try:
    model = joblib.load("rf_fraud_model.pkl")
except FileNotFoundError:
    st.title("⚠ Model Not Found")
    st.error(
        "The model file `rf_fraud_model.pkl` is missing.\n"
        "Please train and save the model first."
    )
    st.stop()
except Exception as e:
    st.title("⚠ Model Load Error")
    st.error(
        "Something went wrong while loading the model:\n\n"
        f"{str(e)}"
    )
    st.stop()


# ------------------------------------
# UI
# ------------------------------------
st.set_page_config(page_title="Fraud Detection", page_icon="💳")
st.title("💳 Credit Card Fraud Detection")
st.write("Enter transaction details to predict fraud risk.")

transaction_amount = st.number_input(
    "Transaction Amount (₹)", min_value=1.0
)

transaction_time = st.number_input(
    "Transaction Time (seconds since midnight)",
    min_value=0.0,
    max_value=86400.0
)

account_age_days = st.number_input(
    "Account Age (days)", min_value=1.0
)

merchant_risk_score = st.slider(
    "Merchant Risk Score", 0.0, 1.0, 0.5
)

transaction_velocity = st.number_input(
    "Transaction Velocity (transactions/hour)",
    min_value=0.0,
    max_value=50.0
)

# Create input DataFrame
import pandas as pd    # Now safe to assume exists
input_data = pd.DataFrame({
    "Transaction_Amount": [transaction_amount],
    "Transaction_Time": [transaction_time],
    "Account_Age_Days": [account_age_days],
    "Merchant_Risk_Score": [merchant_risk_score],
    "Transaction_Velocity": [transaction_velocity]
})


# ------------------------------------
# Prediction
# ------------------------------------
if st.button("Predict"):
    try:
        probability = model.predict_proba(input_data)[0][1]
        threshold = 0.35  # tuned threshold

        if probability > threshold:
            st.error(f"🚨 Fraud Detected! (Risk Score: {probability:.2f})")
        else:
            st.success(f"✅ Legitimate Transaction (Risk Score: {probability:.2f})")

    except Exception as e:
        st.error(
            "Prediction failed. Possible cause: model and inputs are not aligned.\n"
            f"Error: {str(e)}"
        )
