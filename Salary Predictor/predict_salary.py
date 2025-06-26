# Joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
import joblib 

#load the dataset
df = pd.read_csv('salary_data.csv')
#print(df.info())

#split the data into target variable and independent variables
x= df[['YearsExperience']]
y= df['Salary']

#split the data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)

#Scaling down the data
scaler = StandardScaler()# this is we are creating an object of StandardScaler class
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

#creating the model
model = LinearRegression()
model.fit(x_train, y_train)

# save the model and scaler
joblib.dump(model,"predict_salary.pkl")
joblib.dump(scaler,"scaler.pkl")
print("Model and scaler are saved")