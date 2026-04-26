import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

st.title("🧠 Skin Disease Prediction App")

# Load model
model = tf.keras.models.load_model("skin_model.keras")
st.success("Model loaded successfully!")

# Upload image
file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if file:
    image = Image.open(file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess (same as training)
    img = image.resize((224, 224))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    # Predict
    pred = model.predict(img)
    st.write("Prediction values:", pred)

    # ✅ YOUR CORRECT CLASS ORDER
    classes = ["Eczema", "Vitiligo", "Acne"]

    predicted_class = classes[np.argmax(pred)]
    confidence = np.max(pred) * 100

    st.success(f"Prediction: {predicted_class}")
    st.info(f"Confidence: {confidence:.2f}%")