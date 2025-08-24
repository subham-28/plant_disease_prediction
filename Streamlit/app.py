import streamlit as st
import numpy as np
import json
from tensorflow.keras.models import load_model
from PIL import Image
import os

os.environ["STREAMLIT_SERVER_RUN_ON_SAVE"] = "false"

# Load model
model = load_model("Models/plant_model.keras")

with open("Models/class_indices.json", "r") as f:
    class_indices = json.load(f)
class_indices = {int(k): v for k, v in class_indices.items()}

# ------------ Functions ------------
def load_and_preprocess_image(image_path, target_size=(224, 224)):
    img = Image.open(image_path).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

def predict_image_class(model, image_path, class_indices):
    preprocessed_img = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_img)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = float(np.max(predictions) * 100)
    predicted_class_name = class_indices[predicted_class_index]
    return predicted_class_name, confidence

st.set_page_config(page_title="Plant Disease Classification", layout="wide")

# Title
st.markdown("<h1 style='text-align: center; color: #00FF00;'>🌱 Plant Disease Classification</h1>", unsafe_allow_html=True)

st.markdown(
    """
    <div style="text-align: center; font-size:18px; margin-top:20px; line-height:1.6; color:#E0E0E0;">
        🌿 This app uses a <span style="color:#00FF00; font-weight:bold;">Deep Learning Convolutional Neural Network (CNN)</span></b>  
        to classify <b>plant leaf diseases</b>.  
        <br>
        📸 Simply <b>upload a leaf image</b> below, and the model will predict its  
        <span style="color:#00FF00; font-weight:bold;">disease category</span>.
    </div>
    """,
    unsafe_allow_html=True
)


uploaded_image = st.file_uploader("📂 Upload a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)

    col1, col2 = st.columns([1,1.5])

    with col1:
        st.image(image, caption="Uploaded Leaf", use_container_width=True)

    with col2:
        if st.button("🔍 Classify", use_container_width=True):
            predicted_class, confidence = predict_image_class(model, uploaded_image, class_indices)

            st.markdown(
                f"""
                <div style='background-color:#eafbea;padding:20px;border-radius:10px;'>
                <h3 style='color:#1b5e20;'>✅ Prediction: {predicted_class}</h3>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.progress(int(confidence))
            st.info(f"Confidence: {confidence:.2f}%")


# ---------- 🔗 Footer ----------
st.markdown(
    """
    <style>
    footer {
        visibility: hidden;
    }
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        text-align: center;
        padding: 10px;
        font-size: 14px;
        color: white;
        background: rgba(0,0,0,0.8);
    }
    </style>
    <div class="footer">
        Made by Subham Mohanty | Powered by Streamlit
    </div>
    """,
    unsafe_allow_html=True
)
