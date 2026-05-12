# =============================================================================
# app.py
# Streamlit web dashboard for the Face Mask Detector.
# Designed for deployment on HuggingFace Spaces.
#
# Since HuggingFace is a remote server (no webcam), this app works with
# uploaded images instead of a live webcam feed.
#
# Run locally: streamlit run app.py
# Deploy:      push this file + model + cascade to a HuggingFace Space
# =============================================================================

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from src.utils import preprocess_face, draw_prediction, draw_stats
from src.config import CASCADE_PATH, CONFIDENCE_THRESHOLD


# =============================================================================
# PAGE CONFIG — must be the first Streamlit call
# =============================================================================
st.set_page_config(
    page_title="Face Mask Detector",
    page_icon="😷",
    layout="centered"
)


# =============================================================================
# MODEL LOADING — cached so it only loads once, not on every interaction
# @st.cache_resource: Streamlit's caching decorator for heavy resources.
# The model loads once when the app starts, then stays in memory.
# =============================================================================
@st.cache_resource
def load_mask_model(model_path):
    return load_model(model_path)


@st.cache_resource
def load_face_cascade():
    cascade = cv2.CascadeClassifier(CASCADE_PATH)
    return cascade


# =============================================================================
# UI
# =============================================================================

# --- Header ---
st.title("😷 Face Mask Detector")
st.markdown(
    "Upload an image to detect whether people in it are wearing face masks. "
    "The model will draw a **green box** for masked faces and a **red box** for unmasked faces."
)

st.divider()

# --- Sidebar ---
with st.sidebar:
    st.header("⚙️ Settings")

    model_choice = st.selectbox(
        "Select Model",
        options=["Custom CNN", "MobileNetV2", "VGG16"],
        help="Choose which trained model to use for detection."
    )

    model_path_map = {
        "Custom CNN":  "models/mymodel_custom.h5",
        "MobileNetV2": "models/mymodel_mobilenet.h5",
        "VGG16":       "models/mymodel_vgg16.h5"
    }
    model_path = model_path_map[model_choice]

    st.divider()
    st.header("📊 About")
    st.markdown("""
    **Model options:**
    - **Custom CNN**: 3 conv blocks, trained from scratch
    - **MobileNetV2**: Transfer learning, lightweight
    - **VGG16**: Transfer learning, heavier baseline

    **Dataset:** ~1,500 images (with/without mask)

    **Face Detector:** Haar Cascade (OpenCV)
    """)

    st.divider()
    st.markdown("Built with TensorFlow, OpenCV & Streamlit")


# --- Load model and cascade ---
try:
    model = load_mask_model(model_path)
    face_cascade = load_face_cascade()
    st.success(f"✅ Model loaded: **{model_choice}**")
except Exception as e:
    st.error(f"❌ Could not load model: {e}")
    st.stop()

# --- File uploader ---
uploaded_file = st.file_uploader(
    "Upload an image",
    type=["jpg", "jpeg", "png"],
    help="Upload a photo with one or more faces."
)

if uploaded_file is not None:

    # Convert uploaded file to numpy array (OpenCV format)
    # PIL reads as RGB, we convert to BGR for OpenCV compatibility
    pil_image = Image.open(uploaded_file).convert("RGB")
    img_rgb = np.array(pil_image)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

    # --- Detect faces ---
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)

    if len(faces) == 0:
        st.warning("⚠️ No faces detected in this image. Try a clearer frontal photo.")
    else:
        masked_count = 0
        unmasked_count = 0

        # Process each detected face
        for (x, y, w, h) in faces:
            face_crop = img_bgr[y:y + h, x:x + w]
            processed = preprocess_face(face_crop)
            pred = model.predict(processed, verbose=0)[0][0]
            draw_prediction(img_bgr, x, y, w, h, pred)

            if pred >= CONFIDENCE_THRESHOLD:
                unmasked_count += 1
            else:
                masked_count += 1

        # Draw stats on image
        draw_stats(img_bgr, len(faces), masked_count, unmasked_count)

        # Convert back to RGB for display in Streamlit
        result_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # --- Display result image ---
        st.image(result_rgb, caption="Detection Result", use_column_width=True)

        st.divider()

        # --- Metrics row ---
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Faces Detected", len(faces))
        with col2:
            st.metric("✅ Masked", masked_count)
        with col3:
            st.metric("❌ No Mask", unmasked_count)
        with col4:
            compliance = (masked_count / len(faces) * 100) if len(faces) > 0 else 0
            st.metric("Compliance", f"{compliance:.0f}%")

        # --- Result summary ---
        st.divider()
        if unmasked_count == 0:
            st.success("✅ All detected faces are wearing masks!")
        elif masked_count == 0:
            st.error("❌ No masks detected — all faces are unmasked.")
        else:
            st.warning(
                f"⚠️ Mixed: {masked_count} masked, {unmasked_count} unmasked out of {len(faces)} faces."
            )
