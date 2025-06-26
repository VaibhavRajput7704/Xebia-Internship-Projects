import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load the model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("lung_cancer_classifier.h5")

model = load_model()

# Define image size expected by the model
IMAGE_SIZE = (254, 254)  # change this if your model used different size

# Define class names
CLASS_NAMES = ['Benign', 'Malignant', 'Normal']  # adjust as per your dataset

# Title
st.title("Lung Cancer Detection ")
st.write("Upload image to predict the type of lung cancer.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption='Uploaded Image', use_column_width=True)

    # Preprocess image
    img = image.resize(IMAGE_SIZE)
    img_array = np.array(img) / 255.0  # Normalize
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

    # Predict
    prediction = model.predict(img_array)
    predicted_class = CLASS_NAMES[np.argmax(prediction)]

    st.subheader("Prediction:")
    st.write(f"The model predicts: **{predicted_class}**")
