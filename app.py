import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import base64
import pandas as pd
import plotly.express as px

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="Brain Tumor Prediction",
    layout="centered"
)

# --------------------------------------------------
# Background image function
# --------------------------------------------------
def set_bg(image_file):
    with open(image_file, "rb") as f:
        encoded = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <style>
        html, body {{
            height: 100%;
            margin: 0;
        }}

        .stApp {{
            background:
                linear-gradient(rgba(0,0,0,0.35), rgba(0,0,0,0.35)),
                url("data:image/jpg;base64,{encoded}") center / cover no-repeat fixed;
        }}

        .block-container {{
            max-width: 900px;
            padding-top: 3rem;
        }}

        h1, h2, h3, p, label {{
            color: white !important;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# --------------------------------------------------
# Set background image
# --------------------------------------------------
set_bg("medical_bg.jpg")

# --------------------------------------------------
# Load trained model
# --------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "model/brain_tumor_efficientnetb3.h5",
        compile=False
    )

model = load_model()

# --------------------------------------------------
# Class labels (must match training order)
# --------------------------------------------------
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# --------------------------------------------------
# Image preprocessing
# --------------------------------------------------
def preprocess_image(image):
    image = image.resize((300, 300))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# --------------------------------------------------
# UI - Title
# --------------------------------------------------
st.markdown(
    "<h1 style='text-align:center;'>🧠 Brain Tumor Prediction</h1>",
    unsafe_allow_html=True
)

st.markdown(
    "<p style='text-align:center;'>Upload an MRI image to predict the tumor type</p>",
    unsafe_allow_html=True
)

# --------------------------------------------------
# File upload
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# Prediction + Visualization
# --------------------------------------------------
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", width=450)

    if st.button("🔍 Predict"):
        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)

        probabilities = prediction[0]
        predicted_class = class_names[np.argmax(probabilities)]
        confidence = np.max(probabilities) * 100

        # ---------------- Prediction Result ----------------
        st.success(f"Prediction: **{predicted_class.upper()}**")
        st.info(f"Confidence: **{confidence:.2f}%**")

        # ---------------- Professional Confidence Graph ----------------
        st.subheader("🧪 Model Confidence Distribution")

        df = pd.DataFrame({
            "Tumor Type": class_names,
            "Confidence (%)": probabilities * 100
        })

        df = df.sort_values("Confidence (%)", ascending=True)

        fig = px.bar(
            df,
            x="Confidence (%)",
            y="Tumor Type",
            orientation="h",
            text="Confidence (%)",
            color="Confidence (%)",
            color_continuous_scale="Blues"
        )

        fig.update_traces(
            texttemplate="%{text:.2f}%",
            textposition="outside"
        )

        fig.update_layout(
            xaxis_title="Confidence (%)",
            yaxis_title="Tumor Type",
            xaxis_range=[0, 100],
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            coloraxis_showscale=False
        )

        st.plotly_chart(fig, use_container_width=True)

        # ---------------- Progress Bar ----------------
        st.subheader("🎯 Prediction Confidence Level")
        st.progress(int(confidence))
        st.caption(f"{confidence:.2f}% confidence for **{predicted_class.upper()}**")

        # ---------------- Disclaimer ----------------
        st.caption(
            "⚠️ Confidence represents model probability, not medical certainty. "
            "Predictions should be reviewed by qualified medical professionals."
        )
