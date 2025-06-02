import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np

# Load model
model = YOLO("best.pt")

# Set Streamlit page config
st.set_page_config(page_title="Weapon Detection", page_icon="🔫", layout="centered")

# Initialize session state
if "webcam_active" not in st.session_state:
    st.session_state.webcam_active = False

# Sidebar settings
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2949/2949927.png", width=100)
    st.title("🔧 Settings")
    st.markdown("Use the buttons below to start or stop real-time weapon detection using your webcam.")

    if st.button("▶️ Start Webcam", key="start"):
        st.session_state.webcam_active = True

    if st.button("⛔ Stop Webcam", key="stop"):
        st.session_state.webcam_active = False

    st.markdown("---")
    st.markdown("📦 **Model**: YOLOv8n")
    st.markdown("🎯 **Use Case**: Real-time Weapon Detection")
    st.markdown("📁 **Model file**: `best.pt`")

# Main layout
st.markdown("<h1 style='text-align: center;'>🔫 Real-Time Weapon Detection</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Streamlit app powered by YOLOv8n to detect weapons via webcam.</p>", unsafe_allow_html=True)

status_placeholder = st.empty()
frame_placeholder = st.empty()

# Webcam logic
if st.session_state.webcam_active:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        status_placeholder.error("❌ Could not access the webcam.")
        st.session_state.webcam_active = False
    else:
        status_placeholder.success("✅ Webcam active. Detecting weapons...")

        while st.session_state.webcam_active:
            ret, frame = cap.read()
            if not ret:
                status_placeholder.warning("⚠️ Frame capture failed.")
                break

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = model.predict(img_rgb, conf=0.3, verbose=False)
            annotated_frame = results[0].plot()

            frame_placeholder.image(annotated_frame, channels="RGB", use_column_width=True)

            # Refresh UI
            if not st.session_state.webcam_active:
                break

        cap.release()
        status_placeholder.info("🛑 Webcam has been stopped.")