import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Page config
st.set_page_config(page_title="Brain Tumor Prediction", layout="centered")

st.title("🧠 Brain Tumor Prediction")
st.write("Upload an MRI image to predict the tumor type.")

# Load model
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "model/brain_tumor_efficientnetb3.h5",
        compile=False
    )

model = load_model()

# Class labels (MUST match training folder names order)
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# Image preprocessing
def preprocess_image(image):
    image = image.resize((300, 300))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# File upload
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if st.button("Predict"):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        st.success(f"Prediction: **{predicted_class.upper()}**")
        st.info(f"Confidence: **{confidence:.2f}%**")
