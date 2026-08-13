"""
streamlit_app.py

Web application for Colorectal Cancer Histopathology
Image Classification
"""

import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

from model import create_model

# -----------------------------------------
# Configuration
# -----------------------------------------

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

MODEL_PATH = "saved_models/best_model.pth"

CLASS_NAMES = [
    "ADI",
    "BACK",
    "DEB",
    "LYM",
    "MUC",
    "MUS",
    "NORM",
    "STR",
    "TUM"
]

# -----------------------------------------
# Load Model
# -----------------------------------------

@st.cache_resource
def load_model():

    model = create_model(num_classes=9)

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=DEVICE
        )
    )

    model.to(DEVICE)

    model.eval()

    return model


model = load_model()

# -----------------------------------------
# Image Transform
# -----------------------------------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

# -----------------------------------------
# Streamlit UI
# -----------------------------------------

st.title("Colorectal Cancer Histopathology Classification")

st.write(
    "Upload a histopathology image to classify tissue type."
)

uploaded_file = st.file_uploader(
    "Choose an Image",
    type=["png","jpg","jpeg","tif","tiff"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    st.image(
        image,
        caption="Uploaded Image",
        use_container_width=True
    )

    x = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(x)

        probabilities = torch.softmax(outputs,1)

        confidence, prediction = torch.max(
            probabilities,
            1
        )

    st.success(
        f"Prediction: {CLASS_NAMES[prediction.item()]}"
    )

    st.write(
        f"Confidence: {confidence.item()*100:.2f}%"
    )

    st.subheader("Class Probabilities")

    for i, name in enumerate(CLASS_NAMES):

        st.write(
            f"{name}: "
            f"{probabilities[0][i].item()*100:.2f}%"
        )
