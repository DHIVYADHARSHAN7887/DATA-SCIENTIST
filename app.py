import streamlit as st
import pandas as pd
import joblib

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(
    page_title="Customer Segmentation App",
    page_icon="📊",
    layout="centered"
)

# -------------------------------
# Load model & scaler
# -------------------------------
model = joblib.load("kmeans_model.pkl")
scaler = joblib.load("Scaler.pkl")

# -------------------------------
# Title
# -------------------------------
st.title("📊 Customer Segmentation using K-Means")
st.write("This app predicts the customer group based on input values.")

# -------------------------------
# Sidebar inputs
# -------------------------------
st.sidebar.header("Enter Customer Details")

age = st.sidebar.number_input("Age", min_value=1, max_value=100, value=25)
income = st.sidebar.number_input("Annual Income (k$)", min_value=1, max_value=200, value=50)
spending = st.sidebar.number_input("Spending Score (1-100)", min_value=1, max_value=100, value=50)

# -------------------------------
# Predict button
# -------------------------------
if st.sidebar.button("Predict Cluster"):
    input_data = pd.DataFrame([[income, spending]],
                              columns=["Annual Income (k$)", "Spending Score (1-100)"])

    scaled_data = scaler.transform(input_data)
    cluster = model.predict(scaled_data)

    st.success(f"🧠 Predicted Customer Cluster: {cluster[0]}")

# -------------------------------
# Dataset preview
# -------------------------------
st.subheader("📁 Dataset Preview")
df = pd.read_csv("Mall_Customers.csv")
st.dataframe(df.head())