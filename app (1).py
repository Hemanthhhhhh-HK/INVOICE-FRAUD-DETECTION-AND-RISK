
import streamlit as st
import pickle
import numpy as np

model = pickle.load(open("fraud_detection.pkl", "rb"))

st.title("Invoice Fraud Detection")

amount = st.number_input("Enter Amount")
tax = st.number_input("Enter Tax")

if st.button("Predict"):

    features = np.array([[amount, tax]])

    prediction = model.predict(features)

    if prediction[0] == 1:
        st.error("Fraudulent Invoice")
    else:
        st.success("Legitimate Invoice")
