import streamlit as st
from PIL import Image
import numpy as np
from ultralytics import YOLO

# ✅ MUST be the first Streamlit command
st.set_page_config(page_title="YOLOv8 Object Detection", layout="centered")

# Load the YOLO model
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

# UI layout
st.title("📦 YOLO Object Detection App")
st.write("Upload an image to detect objects using your trained YOLOv8 model.")

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to NumPy array
    image_np = np.array(image)

    # Run model prediction
    st.subheader("Running Object Detection...")
    results = model.predict(source=image_np, conf=0.3, save=False)

    # Plot results
    result_img = results[0].plot()
    st.image(result_img, caption="Detected Objects", use_column_width=True)

    # Show detected class names
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
