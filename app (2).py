
import streamlit as st
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open("fraud_detection.pkl", "rb"))

st.title("Invoice Fraud Detection")

# Original features used for training:
# 'Vendor_ID', 'Invoice_Amount', 'Tax_Amount', 'Total_Amount',
# 'Processing_Days', 'Approval_Level', 'Approved_By', 'Vendor_Reliability_Score',
# 'Invoice_Frequency', 'Avg_Invoice_Amount', 'Deviation_From_Avg',
# 'Credit_Period_Days', 'Is_Late_Payment', 'Payment_Status_Paid',
# 'Payment_Status_Pending', 'Payment_Status_Rejected', 'Transaction_Type_Credit',
# 'Transaction_Type_Credit Note', 'Transaction_Type_Debit Note', 'Transaction_Type_Invoice'

# Create input fields for all features
# For simplicity, using st.number_input for numerical features and placeholders for others.
# In a real app, you'd want appropriate widgets for each feature type (e.g., st.selectbox for categorical).

st.header("Enter Invoice Details:")

invoice_amount = st.number_input("Invoice Amount", min_value=0.0, value=1000.0)
tax_amount = st.number_input("Tax Amount", min_value=0.0, value=100.0)
total_amount = st.number_input("Total Amount", min_value=0.0, value=1100.0)
vendor_id = st.number_input("Vendor ID", min_value=0, value=100)
processing_days = st.number_input("Processing Days", min_value=0, value=10)
approval_level = st.number_input("Approval Level", min_value=0, value=1)
approved_by = st.number_input("Approved By ID", min_value=0, value=50)
vendor_reliability_score = st.slider("Vendor Reliability Score", min_value=0.0, max_value=10.0, value=5.0, step=0.1)
invoice_frequency = st.number_input("Invoice Frequency", min_value=0, value=1)
avg_invoice_amount = st.number_input("Average Invoice Amount", min_value=0.0, value=950.0)
deviation_from_avg = st.number_input("Deviation From Average", value=50.0)
credit_period_days = st.number_input("Credit Period Days", value=30)
is_late_payment = st.selectbox("Is Late Payment?", options=[0, 1])

# Payment Status (one-hot encoded)
payment_status_paid = st.selectbox("Payment Status: Paid", options=[0, 1])
payment_status_pending = st.selectbox("Payment Status: Pending", options=[0, 1])
payment_status_rejected = st.selectbox("Payment Status: Rejected", options=[0, 1])

# Transaction Type (one-hot encoded)
transaction_type_credit = st.selectbox("Transaction Type: Credit", options=[0, 1])
transaction_type_credit_note = st.selectbox("Transaction Type: Credit Note", options=[0, 1])
transaction_type_debit_note = st.selectbox("Transaction Type: Debit Note", options=[0, 1])
transaction_type_invoice = st.selectbox("Transaction Type: Invoice", options=[0, 1])


if st.button("Predict Fraud Risk"):
    # Collect all features into a DataFrame in the correct order
    features = pd.DataFrame([[vendor_id, invoice_amount, tax_amount, total_amount,
                              processing_days, approval_level, approved_by, vendor_reliability_score,
                              invoice_frequency, avg_invoice_amount, deviation_from_avg,
                              credit_period_days, is_late_payment, payment_status_paid,
                              payment_status_pending, payment_status_rejected,
                              transaction_type_credit, transaction_type_credit_note,
                              transaction_type_debit_note, transaction_type_invoice]],
                            columns=[
                                'Vendor_ID', 'Invoice_Amount', 'Tax_Amount', 'Total_Amount',
                                'Processing_Days', 'Approval_Level', 'Approved_By', 'Vendor_Reliability_Score',
                                'Invoice_Frequency', 'Avg_Invoice_Amount', 'Deviation_From_Avg',
                                'Credit_Period_Days', 'Is_Late_Payment', 'Payment_Status_Paid',
                                'Payment_Status_Pending', 'Payment_Status_Rejected',
                                'Transaction_Type_Credit', 'Transaction_Type_Credit Note',
                                'Transaction_Type_Debit Note', 'Transaction_Type_Invoice'
                            ])

    prediction = model.predict(features)
    prediction_proba = model.predict_proba(features)[:, 1] # Probability of being class 1 (Higher Risk)

    st.subheader("Prediction Results:")
    if prediction[0] == 1:
        st.error(f"**Higher Risk Invoice** (Fraud Probability: {prediction_proba[0]:.2f})")
        st.write("This invoice is flagged as potentially higher risk based on the model's assessment.")
    else:
        st.success(f"**Lower Risk Invoice** (Fraud Probability: {prediction_proba[0]:.2f})")
        st.write("This invoice is categorized as lower risk.")

    st.markdown("--- ")
    st.write("**Note**: This is a simplified interface for demonstration. A production application would include more robust input validation and user feedback.")
