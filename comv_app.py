import streamlit as st
from PIL import Image
import numpy as np
import os
import requests
from ultralytics import YOLO

# Set Streamlit config
st.set_page_config(page_title="Weapon Detection", layout="centered")

MODEL_PATH = "best.pt"
FILE_ID = "https://drive.google.com/file/d/1TZ9TvSFIhhC0nnNuqDcD7E56jg83OOzc/view?usp=sharing"  # <-- Replace with your actual file ID

@st.cache_resource
def download_model():
    if not os.path.exists(MODEL_PATH):
        gdown_url = f"https://drive.google.com/uc?id={FILE_ID}"
        with st.spinner("Downloading model..."):
            response = requests.get(gdown_url)
            with open(MODEL_PATH, "wb") as f:
                f.write(response.content)
    return YOLO(MODEL_PATH)

model = download_model()

st.title("Weapon Detection App")
st.write("Upload an image to detect weapons")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    image_np = np.array(image)

    st.subheader("Running Object Detection...")
    results = model.predict(source=image_np, conf=0.3, save=False)

    result_img = results[0].plot()
    st.image(result_img, caption="Detected Objects", use_column_width=True)

    st.subheader("Detected Classes")
    class_names = model.names
    detected_classes = set()
    for box in results[0].boxes.data.tolist():
        class_id = int(box[5])
        detected_classes.add(class_names[class_id])

    if detected_classes:
        st.success(", ".join(detected_classes))
    else:
        st.warning("No objects detected.")
