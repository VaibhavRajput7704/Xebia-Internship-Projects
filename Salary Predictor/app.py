import streamlit as st
import joblib
import numpy as np
import pandas as pd

#load the model and scaler
model = joblib.load("predict_salary.pkl")
scaler= joblib.load("scaler.pkl")

#design the layout of our basic app
st.set_page_config(page_title="Salary Predictor", layout="centered")
st.title("Salary Predictor App")
st.subheader("Predict the Salary based on Years of Experience")
st.write("Select the years of experience to see the estimated salary")

# create a dropdown for the years of experience
years= [x for x in range(0, 20)]
years_exp = st.selectbox("Years of Experience", years)

#Predict the salary 
if st.button("Predict the salary"):
    input_data=np.array([[years_exp]])
    input_scaled=scaler.transform(input_data)
    predicted_salary=model.predict(input_scaled)
    st.success(f"Estimated salary is: ₹ {predicted_salary[0]:,.2f}")

    