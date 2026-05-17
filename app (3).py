
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load the trained model
model = pickle.load(open("fraud_detection.pkl", "rb"))

st.title("Invoice Fraud Detection")

st.header("Enter Invoice Details:")

# User-requested simplified input fields
invoice_amount = st.number_input("Invoice Amount", min_value=0.0, value=1000.0)
tax_amount = st.number_input("Tax Amount", min_value=0.0, value=100.0)
vendor_reliability_score = st.slider("Vendor Reliability Score", min_value=0.0, max_value=10.0, value=5.0, step=0.1)

# Default values for other features (not shown in UI but needed by model)
vendor_id = 100
processing_days = 10
approval_level = 1
approved_by = 50
invoice_frequency = 1
avg_invoice_amount = 950.0
deviation_from_avg = 50.0
credit_period_days = 30
is_late_payment = 0 # Default to no late payment
payment_status = 'Approved' # Default payment status
transaction_type = 'Invoice' # Default transaction type

# Calculate total_amount based on invoice_amount and tax_amount
total_amount = invoice_amount + tax_amount

# Define the expected feature columns from training (including one-hot encoded columns)
expected_columns = [
    'Vendor_ID', 'Invoice_Amount', 'Tax_Amount', 'Total_Amount',
    'Processing_Days', 'Approval_Level', 'Approved_By',
    'Vendor_Reliability_Score', 'Invoice_Frequency', 'Avg_Invoice_Amount',
    'Deviation_From_Avg', 'Credit_Period_Days', 'Is_Late_Payment',
    'Payment_Status_Approved', 'Payment_Status_Paid',
    'Payment_Status_Pending', 'Payment_Status_Rejected',
    'Transaction_Type_Adjustment', 'Transaction_Type_Credit',
    'Transaction_Type_Credit Note', 'Transaction_Type_Debit Note',
    'Transaction_Type_Invoice'
]

if st.button("Predict Fraud Risk"):
    # Create a dictionary for the input features using user inputs and defaults
    input_data = {
        'Vendor_ID': vendor_id,
        'Invoice_Amount': invoice_amount,
        'Tax_Amount': tax_amount,
        'Total_Amount': total_amount,
        'Processing_Days': processing_days,
        'Approval_Level': approval_level,
        'Approved_By': approved_by,
        'Vendor_Reliability_Score': vendor_reliability_score,
        'Invoice_Frequency': invoice_frequency,
        'Avg_Invoice_Amount': avg_invoice_amount,
        'Deviation_From_Avg': deviation_from_avg,
        'Credit_Period_Days': credit_period_days,
        'Is_Late_Payment': is_late_payment,
        # One-hot encoded Payment_Status (using default)
        'Payment_Status_Approved': 1 if payment_status == 'Approved' else 0,
        'Payment_Status_Paid': 1 if payment_status == 'Paid' else 0,
        'Payment_Status_Pending': 1 if payment_status == 'Pending' else 0,
        'Payment_Status_Rejected': 1 if payment_status == 'Rejected' else 0,
        # One-hot encoded Transaction_Type (using default)
        'Transaction_Type_Adjustment': 1 if transaction_type == 'Adjustment' else 0,
        'Transaction_Type_Credit': 1 if transaction_type == 'Credit' else 0,
        'Transaction_Type_Credit Note': 1 if transaction_type == 'Credit Note' else 0,
        'Transaction_Type_Debit Note': 1 if transaction_type == 'Debit Note' else 0,
        'Transaction_Type_Invoice': 1 if transaction_type == 'Invoice' else 0,
    }

    # Convert to DataFrame, ensuring all expected columns are present and in order
    features_df = pd.DataFrame([input_data])
    features_df = features_df.reindex(columns=expected_columns, fill_value=0)

    prediction_proba = model.predict_proba(features_df)[:, 1]

    st.subheader("Prediction Results:")
    # Simplify output to 'High Risk' or 'Low Risk' based on a probability threshold
    if prediction_proba[0] > 0.5: # If probability indicates Medium, High, or Critical risk
        st.error(f"**High Risk Invoice** (Fraud Probability: {prediction_proba[0]:.2f})")
        st.write("This invoice is flagged as potentially high risk based on the model's assessment.")
    else: # If probability indicates Low risk
        st.success(f"**Low Risk Invoice** (Fraud Probability: {prediction_proba[0]:.2f})")
        st.write("This invoice is categorized as lower risk.")

    st.markdown("--- ")
    st.write("**Note**: This is a simplified interface for demonstration. A production application would include more robust input validation and user feedback.")
