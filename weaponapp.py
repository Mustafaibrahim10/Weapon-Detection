import streamlit as st
from ultralytics import YOLO
import tempfile
import numpy as np
from PIL import Image
import cv2

st.set_page_config(page_title="Weapon Detection", layout="centered")

st.title("🛡️ Weapon Detection using YOLOv8n")

model = YOLO("best.pt")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert PIL to OpenCV format
    img = np.array(image)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    results = model.predict(img_bgr)
    for r in results:
        annotated = r.plot()

    st.image(annotated, caption="Detection Result", channels="BGR", use_column_width=True)
