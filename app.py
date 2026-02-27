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


# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Brain Tumor Report System", layout="centered")


# --------------------------------------------------
# BACKGROUND IMAGE
# --------------------------------------------------
def set_bg():
    with open("medical_bg.jpg", "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background:
                linear-gradient(rgba(0,0,0,0.6), rgba(0,0,0,0.6)),
                url("data:image/jpg;base64,{encoded}");
            background-size: cover;
            background-position: center;
            background-attachment: fixed;
        }}
        h1, h2, h3, h4, p, label {{
            color: white !important;
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

agree = st.checkbox(
    "I understand this system is NOT a certified medical device and is for educational/demo purposes only."
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
# MEDICAL INFORMATION
# --------------------------------------------------
tumor_info = {
    "glioma": {
        "definition": "Glioma is a tumor arising from glial cells in the brain or spinal cord. It may be low-grade or high-grade depending on aggressiveness.",
        "symptoms": "Headache, seizures, nausea, vomiting, memory problems, personality changes, weakness in limbs.",
        "recommendation": "Consult a neurologist or neurosurgeon. Further MRI evaluation and possible biopsy may be required."
    },
    "meningioma": {
        "definition": "Meningioma develops from the meninges, the protective membranes covering the brain. It is usually slow-growing and often benign.",
        "symptoms": "Vision problems, hearing loss, memory issues, headaches, weakness in arms or legs.",
        "recommendation": "Neurological assessment is recommended. Regular imaging follow-up may be required depending on tumor size."
    },
    "pituitary": {
        "definition": "Pituitary tumor forms in the pituitary gland and may affect hormone production.",
        "symptoms": "Hormonal imbalance, vision disturbances, unexplained weight changes, fatigue, menstrual irregularities.",
        "recommendation": "Endocrinology consultation is advised. Hormonal blood tests and MRI follow-up may be necessary."
    },
    "notumor": {
        "definition": "No tumor detected based on AI prediction.",
        "symptoms": "No tumor-related symptoms detected from imaging.",
        "recommendation": "Continue routine health monitoring. Consult a doctor if symptoms persist."
    }
}


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
def generate_pdf(patient_name, age, gender, patient_id,
                 prediction, confidence, image):

    buffer = io.BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=A4)
    elements = []
    styles = getSampleStyleSheet()

    elements.append(Paragraph("MRI BRAIN AI REPORT", styles["Title"]))
    elements.append(Spacer(1, 20))

    data = [
        ["Patient Name:", patient_name],
        ["Patient ID:", patient_id],
        ["Age / Gender:", f"{age} / {gender}"],
        ["Study Date:", datetime.now().strftime("%d-%m-%Y")],
        ["Modality:", "MRI Brain"]
    ]

    table = Table(data, colWidths=[150, 300])
    table.setStyle(TableStyle([('GRID', (0, 0), (-1, -1), 0.5, colors.grey)]))

    elements.append(table)
    elements.append(Spacer(1, 20))

    elements.append(Paragraph(f"Prediction: <b>{prediction.upper()}</b>", styles["Normal"]))
    elements.append(Paragraph(f"Confidence: {confidence:.2f}%", styles["Normal"]))
    elements.append(Spacer(1, 10))

    elements.append(Paragraph(f"<b>Definition:</b> {tumor_info[prediction]['definition']}", styles["Normal"]))
    elements.append(Spacer(1, 5))
    elements.append(Paragraph(f"<b>Common Symptoms:</b> {tumor_info[prediction]['symptoms']}", styles["Normal"]))
    elements.append(Spacer(1, 5))
    elements.append(Paragraph(f"<b>Recommendation:</b> {tumor_info[prediction]['recommendation']}", styles["Normal"]))

    elements.append(Spacer(1, 20))

    img_buffer = io.BytesIO()
    image.save(img_buffer, format="PNG")
    img_buffer.seek(0)
    pdf_image = RLImage(img_buffer, width=4*inch, height=4*inch)
    elements.append(pdf_image)

    elements.append(Spacer(1, 20))
    elements.append(Paragraph(
        "<b>DISCLAIMER:</b> This AI system is experimental and not approved for clinical diagnosis.",
        styles["Normal"]
    ))

    doc.build(elements)
    buffer.seek(0)
    return buffer


# --------------------------------------------------
# UI
# --------------------------------------------------
st.markdown("## 🧠 Brain Tumor Prediction & Report System")

col1, col2 = st.columns(2)

with col1:
    patient_name = st.text_input("Patient Name")
    age = st.number_input("Age", min_value=1, max_value=120)

with col2:
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
    patient_id = st.text_input("Patient ID", value=f"PID-{np.random.randint(1000,9999)}")

uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg", "jpeg", "png"])


# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded MRI Image", width=400)

    if st.button("🔍 Generate Report"):

        if not patient_name:
            st.warning("Please enter patient name.")
            st.stop()

        processed = preprocess_image(image)
        prediction = model.predict(processed)

        probabilities = prediction[0]
        predicted_class = class_names[np.argmax(probabilities)]
        confidence = np.max(probabilities) * 100

        st.success(f"Prediction: {predicted_class.upper()}")
        st.info(f"Confidence: {confidence:.2f}%")

        st.subheader("🧠 Tumor Details")
        st.write("**Definition:**", tumor_info[predicted_class]["definition"])
        st.write("**Common Symptoms:**", tumor_info[predicted_class]["symptoms"])
        st.write("**Recommendation:**", tumor_info[predicted_class]["recommendation"])

        df = pd.DataFrame({
            "Tumor Type": class_names,
            "Confidence (%)": probabilities * 100
        })

        fig = px.bar(df, x="Confidence (%)", y="Tumor Type",
                     orientation="h", text="Confidence (%)")

        fig.update_traces(texttemplate="%{text:.2f}%", textposition="outside")
        st.plotly_chart(fig, use_container_width=True)

        pdf_file = generate_pdf(
            patient_name, age, gender, patient_id,
            predicted_class, confidence, image
        )

        st.download_button(
            label="📥 Download MRI Report (PDF)",
            data=pdf_file,
            file_name=f"{patient_name}_MRI_Report.pdf",
            mime="application/pdf"
        )
