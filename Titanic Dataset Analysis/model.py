# Importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
df=pd.read_csv('Titanic-Dataset.csv')
df.head()

# How many rows and columns are there
df.shape

# Basic Information about the datatypes of each column
df.info()

# checking the missing values in our dataset
df.isnull().sum(axis=0)

# divide the dataframes into dependent and independent variables
x= df[["Pclass","Age","Sex","SibSp","Parch","Fare"]]
y= df["Survived"]

# Handling missing values in the age column
x['Age']=x['Age'].fillna(x['Age'].mean())

# encoding the sex column
from sklearn.preprocessing import LabelEncoder
encoder=LabelEncoder()
x['Sex']=encoder.fit_transform(x['Sex'])

# dividing the dataset into training and test dataset
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=1)


# Dictionary of model for model setup
models={
    'LinearRegression':LinearRegression(),
    'SVC':SVC(),
    'GaussianNB':GaussianNB(),
    'KNeighborsClassifier':KNeighborsClassifier(),
    'DecisionTreeClassifier':DecisionTreeClassifier(),
    'RandomForestClassifier':RandomForestClassifier()
}

from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

results = []

for name, model in models.items():
    model.fit(X_train, y_train) # train the model
    y_raw_pred = model.predict(X_test) # Predict
    if y_raw_pred.ndim == 1 and y_raw_pred.dtype in ['float32', 'float64']:
        y_pred = (y_raw_pred >= 0.5).astype(int)
    elif len(y_raw_pred.shape) > 1 and y_raw_pred.shape[1] == 2:
        y_pred = (y_raw_pred[:, 1] >= 0.5).astype(int)
    else:
        y_pred = y_raw_pred

    cm=confusion_matrix(y_test, y_pred)
    print(f"\nModel: {name}")
    print("Confusion Matrix:")
    print(cm)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    results.append({
        "Model": name,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1_score": f1
    })

    # Summary Table
    results_df = pd.DataFrame(results)
    print("\nSummary of all models:")
    print(results_df)

    #visualize the confusion matrix
    plt.figure(figsize=(5,4))
    sns.heatmap(cm,annot=True)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # visualize the comparision
    plt.figure(figsize=(12,8))
    results_df.set_index('Model')[["Accuracy","Precision","Recall"]].plot(kind='bar',cmap="magma")
    plt.title("visualize the performance")
    plt.show()

