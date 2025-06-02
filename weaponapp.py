import streamlit as st
import cv2
import tempfile
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Page config
st.set_page_config(page_title="🔫 Weapon Detection", layout="centered", page_icon="🛡️")

# Custom CSS for styling
st.markdown("""
    <style>
    .main { background-color: #f4f4f4; }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
        background-color: white;
        border-radius: 15px;
        box-shadow: 0px 0px 10px rgba(0,0,0,0.1);
    }
    .stButton > button {
        background-color: #6c63ff;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #5146d4;
    }
    </style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.image("https://img.icons8.com/external-flat-icons-inmotus-design/512/external-weapon-military-flat-icons-inmotus-design-2.png", width=100)
    st.title("🛡️ Weapon Detector")
    st.markdown("""
    Upload an image or video to detect **weapons** using a custom-trained **YOLOv8n** model.
    
    - 💡 **Upload image/video**
    - 🧠 **AI model**: YOLOv8n
    - 🚀 **Fast & Accurate**
    """)

# Load model
model = YOLO("best.pt")

st.markdown("## 📸 Upload Image or Video")
uploaded_file = st.file_uploader("Choose a media file", type=["jpg", "jpeg", "png", "mp4"])

if uploaded_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())

    if uploaded_file.type.startswith("image"):
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

        image = cv2.imread(tfile.name)
        results = model.predict(image)
        for r in results:
            annotated = r.plot()

        st.markdown("### 🎯 Detection Result")
        st.image(annotated, caption="Detected Objects", channels="BGR", use_column_width=True)

    elif uploaded_file.type.startswith("video"):
        st.video(uploaded_file)
        st.warning("Video preview only. For real-time detection, run locally.")
