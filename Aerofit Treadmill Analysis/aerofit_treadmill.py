import pandas as pd
import numpy as np
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

# Setting the page configuration of streamlit dashboard
st.set_page_config(page_title="Aerofit Treadmill Analysis", layout="wide")
st.title("Aerofit Treadmill Data Analysis Dashboard")

#Upload the dataset
uploaded_file=st.file_uploader("Please upload your dataset",type=["csv"])
if uploaded_file is not None:
    df=pd.read_csv(uploaded_file)

    # Basic Data Analysis
    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    # Shape of the dataset
    st.subheader("Shape of the dataset")
    st.write("Number of rows and columns in the datset are:",df.shape)
    st.write("Columns names of my dataset are:", df.columns.tolist())

    # Create few checkboxes
    st.subheader("Statistics of the Dataset")
    data_type=st.checkbox("Show the data type")
    missing_value= st.checkbox("Show Missing values")
    statistics=st.checkbox("Show the statistical summary of the dataset")

    if data_type:
        st.write("The Datatype in this dataset are: ",df.info())
    if missing_value:
        st.write("Missing Values of the dataset are: ",df.isna().sum(axis=0))
    if statistics:
        st.write("Dataset Statistics are: ",df.describe())

    # Radio button 
    option = st.radio("Choose what to display:", ("Data Types", "Missing Values", "Statistics"))
    
    if option == "Data Types":
        st.write("The Datatype in this dataset are: ",df.info())
    elif option == "Missing Values":
        st.write("Missing Values of the dataset are: ",df.isna().sum(axis=0))
    elif option == "Statistics":
        st.write("Dataset Statistics are: ",df.describe())

    #Visual Analysis of our Dataset
    #Column Selector
    numeric_cols=df.select_dtypes(include=["int64","float64"]).columns.tolist()
    categorical_cols=df.select_dtypes(include=["object"]).columns.tolist()
    st.write(numeric_cols)
    st.write(categorical_cols)

    #Uni Variate Analysis
    #count plot
    st.subheader("Count Plot")
    selected_cols=st.selectbox("select a numeric columns: ",categorical_cols)
    fig,ax=plt.subplots()
    sns.countplot(x=df[selected_cols],ax=ax)
    st.pyplot(fig)

    #count plot
    st.subheader("Count Plot")
    selected_cols=st.selectbox("select a numeric columns: ",numeric_cols)
    fig,ax=plt.subplots()
    sns.countplot(x=df[selected_cols],ax=ax)
    st.pyplot(fig)

    #Box plot for numerical columns
    st.subheader("Box plots for checking in the outliers")
    box_cols=st.selectbox("Select a numeric column: ",numeric_cols)
    fig,ax=plt.subplots()
    sns.boxplot(x=df[box_cols],ax=ax)
    st.pyplot(fig)

    # Hist PLot
    st.subheader("Histogram Plot")
    hist_col = st.selectbox("Select a numeric column for histogram:", numeric_cols)
    fig,ax= plt.subplots()
    sns.histplot(data=df, x=hist_col, kde=True, ax=ax)
    st.pyplot(fig)

    # Bi variate Analysis
    st.subheader("Bi Variate Analysis of our dataset: Categorical vs Numerical")
    num_cols= st.selectbox("Select a numeric column:",numeric_cols,key="num1")
    category_cols=st.selectbox("Select a categorical column: ",categorical_cols,key="cat1")
    fig,ax=plt.subplots()
    sns.boxplot(x=df[num_cols],y=df[category_cols],ax=ax)
    st.pyplot(fig)

    # # Scatter plot Analysis
    # st.subheader("Scatter Plot Analysis of our dataset: Categorical vs Numerical")
    # x_cols= st.selectbox("Select a numeric column:",numeric_cols,key="num1")
    # y_cols=st.selectbox("Select a categorical column: ",categorical_cols,key="cat1")
    # fig,ax=plt.subplots()
    # sns.scatterplot(x=df[x_cols],y=df[y_cols],ax=ax)
    # st.pyplot(fig)

    #Multi Variate Analysis
    #Heatmap of our dataset to check the corelation
    st.subheader("Co-relation heatmap")
    fig,ax=plt.subplots(figsize=(10,6))
    sns.heatmap(df[numeric_cols].corr(),annot=True,cmap="magma",ax=ax)
    st.pyplot(fig)

    #Pair plot
    st.subheader("Pair-plot of our dataset")
    fig=sns.pairplot(df[numeric_cols])
    st.pyplot(fig)


else:
    st.write("Please upload the dataset first for the exploratory data analysis")
