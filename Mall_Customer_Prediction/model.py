# Prepare a cluster of customer to predict the purchase power based on their income and spend
# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.cluster import KMeans

#Loading the dataset into dataframe
df= pd.read_csv("Mall_Customers.csv")

# print(df.info())

X=df[["Annual Income (k$)","Spending Score (1-100)"]]
wcss_list=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init="k-means++",random_state=1)
    kmeans.fit(X)
    wcss_list.append(kmeans.inertia_)

#visualize the clusters
# plt.plot(range(1, 11), wcss_list, marker='o')
# plt.title("Elbow Method")
# plt.xlabel("Number of clusters")
# plt.ylabel("WCSS")
# plt.show()

# Training the model on our dataset
model=KMeans(n_clusters=6,init="k-means++",random_state=1)
y_predict=model.fit_predict(X)

print(y_predict)

# converting the dataframe x into a numpy array
X_array=X.values

#Plotting the graph of clusters
plt.scatter(X_array[y_predict==0,0],X_array[y_predict==0,1],s=100,color="Green")
plt.scatter(X_array[y_predict==1,0],X_array[y_predict==1,1],s=100,color="Red")
plt.scatter(X_array[y_predict==2,0],X_array[y_predict==2,1],s=100,color="Yellow")
plt.scatter(X_array[y_predict==3,0],X_array[y_predict==3,1],s=100,color="Blue")
plt.scatter(X_array[y_predict==4,0],X_array[y_predict==4,1],s=100,color="Pink")
plt.scatter(X_array[y_predict==5,0],X_array[y_predict==5,1],s=100,color="Black")



plt.title("Customer Segmentation Graph")
plt.xlabel("Annual Income")
plt.ylabel("Spennding Score")
plt.show()

joblib.dump(model,"Customer_Segmentation_Model.pkl")
print("Model FIle created Successfully!!")