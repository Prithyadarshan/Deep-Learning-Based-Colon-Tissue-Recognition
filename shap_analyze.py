"""
shap_analysis.py

SHAP Explainability for Colorectal Cancer
Histopathology Image Classification
"""

import os
import torch
import shap
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

from model import create_model

# --------------------------------------------------
# Configuration
# --------------------------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_PATH = "saved_models/best_model.pth"

IMAGE_PATH = "sample_image.tif"

OUTPUT_DIR = "results"

os.makedirs(OUTPUT_DIR, exist_ok=True)

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

# --------------------------------------------------
# Image Transform
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])

# --------------------------------------------------
# Load Model
# --------------------------------------------------

model = create_model(num_classes=9)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.to(DEVICE)
model.eval()

# --------------------------------------------------
# Load Image
# --------------------------------------------------

image = Image.open(IMAGE_PATH).convert("RGB")

image_tensor = transform(image).unsqueeze(0).to(DEVICE)

# --------------------------------------------------
# Background Images
# --------------------------------------------------

background = image_tensor.repeat(10,1,1,1)

# --------------------------------------------------
# SHAP Explainer
# --------------------------------------------------

explainer = shap.GradientExplainer(
    model,
    background
)

# --------------------------------------------------
# Compute SHAP Values
# --------------------------------------------------

shap_values = explainer.shap_values(image_tensor)

# --------------------------------------------------
# Prediction
# --------------------------------------------------

with torch.no_grad():

    output = model(image_tensor)

    prediction = torch.argmax(output,1).item()

print("Prediction :", CLASS_NAMES[prediction])

# --------------------------------------------------
# Plot SHAP
# --------------------------------------------------

image_np = image_tensor.squeeze().cpu().numpy()

image_np = np.transpose(image_np,(1,2,0))

image_np = (
    image_np *
    np.array([0.229,0.224,0.225])
) + np.array([0.485,0.456,0.406])

image_np = np.clip(image_np,0,1)

shap.image_plot(
    shap_values,
    image_np[np.newaxis,...],
    show=False
)

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "shap_explanation.png"
    ),
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("SHAP explanation saved.")
