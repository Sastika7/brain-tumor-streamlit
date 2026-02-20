import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import pandas as pd
import plotly.express as px
import base64
import io
from datetime import datetime

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.pagesizes import A4

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Brain Tumor AI Report System", layout="centered")

# --------------------------------------------------
# LEGAL DISCLAIMER WALL
# --------------------------------------------------
st.title("⚠️ AI Medical Prototype Disclaimer")

agree = st.checkbox(
    "I understand this tool is NOT a certified medical device and is for educational/demo purposes only."
)

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
# RATE LIMITING (Basic)
# --------------------------------------------------
if "prediction_count" not in st.session_state:
    st.session_state.prediction_count = 0

if st.session_state.prediction_count >= 5:
    st.error("Prediction limit reached. Please refresh the page.")
    st.stop()

# --------------------------------------------------
# IMAGE PREPROCESS
# --------------------------------------------------
def preprocess_image(image):
    image = image.resize((300, 300))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# --------------------------------------------------
# PDF GENERATION
# --------------------------------------------------
def generate_pdf(patient_name, age, gender, patient_id,
                 prediction, confidence, image):

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []

    styles = getSampleStyleSheet()

    elements.append(Paragraph("MRI BRAIN RADIOLOGY REPORT", styles["Title"]))
    elements.append(Spacer(1, 20))

    # Patient Info Table
    data = [
        ["Patient Name:", patient_name],
        ["Patient ID:", patient_id],
        ["Age / Gender:", f"{age} / {gender}"],
        ["Study Date:", datetime.now().strftime("%d-%m-%Y")],
        ["Modality:", "MRI Brain"]
    ]

    table = Table(data, colWidths=[150, 300])
    table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (-1, 0), colors.whitesmoke),
    ]))

    elements.append(table)
    elements.append(Spacer(1, 25))

    # Findings
    elements.append(Paragraph("<b>Clinical Indication:</b>", styles["Heading2"]))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(
        "Patient referred for AI-assisted MRI brain evaluation.",
        styles["Normal"]
    ))

    elements.append(Spacer(1, 15))

    elements.append(Paragraph("<b>Findings:</b>", styles["Heading2"]))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(
        f"AI model predicts presence of <b>{prediction.upper()}</b>.",
        styles["Normal"]
    ))
    elements.append(Paragraph(
        f"Model Confidence: {confidence:.2f}%",
        styles["Normal"]
    ))

    elements.append(Spacer(1, 20))

    elements.append(Paragraph("<b>Impression:</b>", styles["Heading2"]))
    elements.append(Spacer(1, 10))
    elements.append(Paragraph(
        f"Findings are suggestive of {prediction.upper()}. "
        "Clinical correlation with certified radiologist is mandatory.",
        styles["Normal"]
    ))

    elements.append(Spacer(1, 25))

    # MRI Image
    img_buffer = io.BytesIO()
    image.save(img_buffer, format="PNG")
    img_buffer.seek(0)

    pdf_image = RLImage(img_buffer, width=4*inch, height=4*inch)
    elements.append(pdf_image)

    elements.append(Spacer(1, 30))

    elements.append(Paragraph(
        "<b>LEGAL DISCLAIMER:</b> This AI system is an experimental prototype. "
        "It is NOT FDA/CE approved and must NOT be used for clinical decision making. "
        "Always consult a certified radiologist.",
        styles["Normal"]
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer

# --------------------------------------------------
# UI
# --------------------------------------------------
st.markdown("## 🧠 Brain Tumor AI Prediction & Report System")

# Patient Form
st.subheader("🧾 Patient Information")

col1, col2 = st.columns(2)

with col1:
    patient_name = st.text_input("Patient Name")
    age = st.number_input("Age", min_value=1, max_value=120)

with col2:
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    patient_id = st.text_input("Patient ID", value=f"PID-{np.random.randint(1000,9999)}")

# Upload
uploaded_file = st.file_uploader(
    "Upload MRI Image",
    type=["jpg", "jpeg", "png"]
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", width=400)

    if st.button("🔍 Generate Report"):

        # Validation
        if not patient_name:
            st.warning("Please enter patient name.")
            st.stop()

        st.session_state.prediction_count += 1

        processed = preprocess_image(image)
        prediction = model.predict(processed)

        probabilities = prediction[0]
        predicted_class = class_names[np.argmax(probabilities)]
        confidence = np.max(probabilities) * 100

        # Result
        st.success(f"Prediction: {predicted_class.upper()}")
        st.info(f"Confidence: {confidence:.2f}%")

        # Graph
        df = pd.DataFrame({
            "Tumor Type": class_names,
            "Confidence (%)": probabilities * 100
        })

        fig = px.bar(
            df,
            x="Confidence (%)",
            y="Tumor Type",
            orientation="h",
            text="Confidence (%)",
            color="Confidence (%)",
            color_continuous_scale="Blues"
        )

        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        fig.update_layout(coloraxis_showscale=False)

        st.plotly_chart(fig, use_container_width=True)

        # Progress
        st.progress(int(confidence))

        # Generate PDF
        pdf_file = generate_pdf(
            patient_name,
            age,
            gender,
            patient_id,
            predicted_class,
            confidence,
            image
        )

        st.download_button(
            label="📥 Download Official MRI Report (PDF)",
            data=pdf_file,
            file_name=f"{patient_name}_MRI_Report.pdf",
            mime="application/pdf"
        )

# --------------------------------------------------
# MODEL METRICS SECTION
# --------------------------------------------------
st.markdown("---")
st.subheader("📊 Model Performance")

st.write("""
- Architecture: EfficientNetB3
- Classes: Glioma, Meningioma, Pituitary, No Tumor
- Validation Accuracy: ~96%
- Dataset: Public Brain MRI Dataset
""")
