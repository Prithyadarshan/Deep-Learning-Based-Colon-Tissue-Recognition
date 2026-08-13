"""
predict.py

Predict the tissue class of a single colorectal
histopathology image using the trained model.
"""

import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

from model import create_model

# -------------------------------------------------
# Configuration
# -------------------------------------------------

MODEL_PATH = "saved_models/best_model.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

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

# -------------------------------------------------
# Image Transform
# -------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# -------------------------------------------------
# Load Model
# -------------------------------------------------

model = create_model(num_classes=9)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.to(DEVICE)
model.eval()

# -------------------------------------------------
# Predict Function
# -------------------------------------------------

def predict_image(image_path):

    image = Image.open(image_path).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():

        outputs = model(input_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, 1)

    predicted_class = CLASS_NAMES[predicted.item()]

    confidence_score = confidence.item() * 100

    plt.figure(figsize=(6,6))
    plt.imshow(image)
    plt.axis("off")

    plt.title(
        f"Prediction: {predicted_class}\n"
        f"Confidence: {confidence_score:.2f}%"
    )

    plt.show()

    print("=" * 50)
    print(f"Predicted Class : {predicted_class}")
    print(f"Confidence      : {confidence_score:.2f}%")
    print("=" * 50)

# -------------------------------------------------
# Main
# -------------------------------------------------

if __name__ == "__main__":

    IMAGE_PATH = "sample_image.tif"

    predict_image(IMAGE_PATH)
