#Importing libraries
import numpy as np
import pandas as pd 
import joblib
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans  
import seaborn as sns
import streamlit as st

# Load the pre-trained model
kmeans = joblib.load("Customer_Segmentation_Model.pkl")

df=pd.read_csv("Mall_Customers.csv")
X=df[["Annual Income (k$)","Spending Score (1-100)"]]
X_array=X.values

# Streamlit Application Page
st.set_page_config(page_title="Customer Cluster Prediction", layout="centered")
st.title("Customer Cluster Prediction")
st.write("Enter the customer annual income and spending score to predict the cluster.")

# Inputs
income = st.number_input("Annual Income of a customer",min_value=0,max_value=400,value=50)
spending_score = st.slider("Spending Score of a customer",1,100,20)

# Predicting the cluster
if st.button("Predict Cluster"):
    input_data = np.array([[income, spending_score]])
    cluster = kmeans.predict(input_data)[0]
    st.success(f"Predicted Cluster is:{cluster}")
    
