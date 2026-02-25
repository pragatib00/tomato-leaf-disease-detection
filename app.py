import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

model = load_model("models/plant_disease_model.keras")

st.set_page_config(page_title="Tomato Disease Detection", layout="wide")

st.title(" Tomato Plant Disease Detection")

uploaded_file = st.file_uploader("Upload a tomato leaf image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = image.load_img(uploaded_file, target_size=(224, 224))

    # Create two columns
    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Uploaded Image", width=300)

    if st.button(" Check Disease"):

        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = preprocess_input(img_array)

        with st.spinner("Analyzing image..."):
            prediction = model.predict(img_array)
        

        class_names = [
            "Early Blight",
            "Late Blight",
            "Septoria",
            "Healthy"
        ]

        predicted_index = np.argmax(prediction)
        predicted_class = class_names[predicted_index]
        confidence = float(np.max(prediction)) * 100

        with col2:
            st.success(f"Prediction: {predicted_class}")
            st.info(f"Confidence: {confidence:.2f}%")

            st.subheader("Class Probabilities")

            for i, prob in enumerate(prediction[0]):
                percentage = float(prob) * 100
                st.write(f"{class_names[i]} — {percentage:.2f}%")
                st.progress(float(prob))

st.caption("Image Classification Model | Developed by Pragati Basnet")
