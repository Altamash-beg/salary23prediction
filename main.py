import streamlit as st
import joblib
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

# ----------------------------
# Load Model
# ----------------------------
lr = joblib.load("lgghdr.pkl")
pf = PolynomialFeatures()
# Assuming model was trained with single feature "Level" up to default degree
# If your saved PolynomialFeatures had specific degree, replace accordingly

# ----------------------------
# Page Config
# ----------------------------
st.set_page_config(page_title="Salary Predictor", page_icon="💼", layout="centered")

# ----------------------------
# Title
# ----------------------------
st.title("💼 Polynomial Salary Predictor")
st.markdown("Predict employee salary based on their level using Polynomial Regression.")

# ----------------------------
# Input
# ----------------------------
level = st.number_input("Enter Level", min_value=0, step=1)

# ----------------------------
# Predict Button
# ----------------------------
if st.button("Predict Salary"):
    # Transform input for polynomial regression
    level_transformed = pf.fit_transform(np.array([[level]]))
    prediction = lr.predict(level_transformed)[0]
    st.success(f"✅ Predicted Salary: **{prediction:.2f}**")
