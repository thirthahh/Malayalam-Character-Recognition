import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="Malayalam Character Recognition",
    page_icon="🔤",
    layout="centered"
)

# -----------------------------
# Load trained model
# -----------------------------
@st.cache_resource
def load_trained_model():
    return load_model("malayalam_character_model.keras")


model = load_trained_model()

# -----------------------------
# Title
# -----------------------------
st.title("🔤 Malayalam Handwritten Character Recognition")

st.write(
    "Upload a handwritten Malayalam character image "
    "and the trained CNN model will predict its class."
)

# -----------------------------
# Upload image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload a character image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# Prediction
# -----------------------------
if uploaded_file is not None:

    image = Image.open(uploaded_file)

    st.subheader("Uploaded Image")
    st.image(image, width=250)

    # Convert to grayscale
    img = image.convert("L")

    # Resize to model input size
    img = img.resize((32, 32))

    # Convert to numpy array
    img = np.array(img)

    # Normalize
    img = img.astype("float32") / 255.0

    # Add channel dimension
    img = np.expand_dims(img, axis=-1)

    # Add batch dimension
    img = np.expand_dims(img, axis=0)

    # Predict
    prediction = model.predict(img, verbose=0)

    predicted_class = int(np.argmax(prediction[0]))
    original_class = predicted_class + 1
    confidence = float(np.max(prediction[0])) * 100
    # Get top 3 predictions
    top_3 = np.argsort(prediction[0])[-3:][::-1]

    st.subheader("Top 3 Predictions")

    for i in top_3:
        class_number = int(i) + 1
        class_confidence = float(prediction[0][i]) * 100

        st.write(
            f"Class {class_number}: {class_confidence:.2f}%"
        )

    # Display result
    st.subheader("Prediction")
    st.success(f"Predicted Class: {original_class}")
    st.info(f"Confidence: {confidence:.2f}%")
    st.caption(
        "Class numbers correspond to the 85 classes in the Amrita MalCharDb dataset."
    )