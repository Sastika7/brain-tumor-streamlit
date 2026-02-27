import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import io
import base64
from datetime import datetime

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4

st.set_page_config(page_title="Brain Tumor Report System", layout="centered")

# --------------------------------------------------
# BACKGROUND
# --------------------------------------------------
def set_bg():
    with open("medical_bg.jpg", "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                linear-gradient(rgba(0,0,0,0.7), rgba(0,0,0,0.7)),
                url("data:image/jpg;base64,{encoded}");
            background-size: cover;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

set_bg()

# --------------------------------------------------
# DISCLAIMER
# --------------------------------------------------
st.title("⚠️ Medical Prototype")
agree = st.checkbox("This system is for educational purposes only.")
if not agree:
    st.stop()

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model(
        "model/brain_tumor_efficientnetb3.h5",
        compile=False
    )

model = load_model()
class_names = ['glioma', 'meningioma', 'notumor', 'pituitary']

# --------------------------------------------------
# RISK LEVEL FUNCTION
# --------------------------------------------------
def calculate_risk(predicted_class, confidence):

    if predicted_class == "notumor":
        if confidence >= 85:
            return "Low Risk", "green"
        else:
            return "Uncertain", "orange"

    if confidence >= 85:
        return "High Confidence Detection", "red"
    elif confidence >= 60:
        return "Moderate Confidence Detection", "orange"
    else:
        return "Low Confidence Detection", "yellow"


# --------------------------------------------------
# IMAGE PREPROCESS
# --------------------------------------------------
def preprocess_image(image):
    image = image.resize((300, 300))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


# --------------------------------------------------
# PDF GENERATOR
# --------------------------------------------------
def generate_pdf(patient_name, prediction, confidence, risk_level):

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("MRI BRAIN AI REPORT", styles["Title"]))
    elements.append(Spacer(1, 20))

    elements.append(Paragraph(f"Patient: {patient_name}", styles["Normal"]))
    elements.append(Paragraph(f"Prediction: {prediction.upper()}", styles["Normal"]))
    elements.append(Paragraph(f"Confidence: {confidence:.2f}%", styles["Normal"]))
    elements.append(Paragraph(f"Risk Level: {risk_level}", styles["Normal"]))

    elements.append(Spacer(1, 20))
    elements.append(Paragraph(
        "Disclaimer: AI-based classification only. Not for final medical diagnosis.",
        styles["Normal"]
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer


# --------------------------------------------------
# UI
# --------------------------------------------------
st.markdown("## 🧠 Brain Tumor Prediction Dashboard")

patient_name = st.text_input("Patient Name")
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, width=400)

    if st.button("Generate Report"):

        if not patient_name:
            st.warning("Enter patient name.")
            st.stop()

        processed = preprocess_image(image)
        prediction = model.predict(processed)

        probabilities = prediction[0]
        predicted_class = class_names[np.argmax(probabilities)]
        confidence = np.max(probabilities) * 100

        # Risk Calculation
        risk_level, color = calculate_risk(predicted_class, confidence)

        # Display Results
        st.success(f"Prediction: {predicted_class.upper()}")
        st.info(f"Confidence: {confidence:.2f}%")

        # Colored Status Badge
        st.markdown(
            f"""
            <div style='padding:10px;
                        border-radius:10px;
                        background-color:{color};
                        color:black;
                        font-weight:bold;
                        text-align:center;'>
                {risk_level}
            </div>
            """,
            unsafe_allow_html=True
        )

        # Risk Meter
        st.subheader("Risk Confidence Meter")
        st.progress(int(confidence))

        # Probability Chart
        df = pd.DataFrame({
            "Tumor Type": class_names,
            "Confidence (%)": probabilities * 100
        })

        fig = px.bar(df, x="Confidence (%)", y="Tumor Type",
                     orientation="h", text="Confidence (%)")

        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

        # PDF
        pdf_file = generate_pdf(
            patient_name,
            predicted_class,
            confidence,
            risk_level
        )

        st.download_button(
            "Download Report (PDF)",
            pdf_file,
            file_name="MRI_Report.pdf",
            mime="application/pdf"
        )
