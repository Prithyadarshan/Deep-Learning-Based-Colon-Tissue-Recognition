"""
gradcam.py

Generate Grad-CAM heatmaps for EfficientNet-B3
trained on the CRC Histopathology dataset.
"""

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms

from model import create_model


# -----------------------------------------------------
# Configuration
# -----------------------------------------------------

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


# -----------------------------------------------------
# Load Model
# -----------------------------------------------------

model = create_model(9)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.to(DEVICE)
model.eval()


# -----------------------------------------------------
# Target Layer
# -----------------------------------------------------

target_layer = model.model.features[-1]


activations = None
gradients = None


def forward_hook(module, input, output):
    global activations
    activations = output


def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]


target_layer.register_forward_hook(forward_hook)
target_layer.register_full_backward_hook(backward_hook)


# -----------------------------------------------------
# Transform
# -----------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])


# -----------------------------------------------------
# GradCAM
# -----------------------------------------------------

def generate_gradcam(image_path):

    image = Image.open(image_path).convert("RGB")

    original = np.array(image)

    x = transform(image).unsqueeze(0).to(DEVICE)

    output = model(x)

    pred = output.argmax(dim=1)

    model.zero_grad()

    output[0,pred].backward()

    pooled_gradients = gradients.mean(dim=[0,2,3])

    feature_maps = activations.squeeze(0)

    for i in range(feature_maps.shape[0]):
        feature_maps[i] *= pooled_gradients[i]

    heatmap = feature_maps.mean(dim=0).cpu().detach().numpy()

    heatmap = np.maximum(heatmap,0)

    heatmap /= np.max(heatmap)

    heatmap = cv2.resize(
        heatmap,
        (original.shape[1], original.shape[0])
    )

    heatmap = np.uint8(255*heatmap)

    heatmap = cv2.applyColorMap(
        heatmap,
        cv2.COLORMAP_JET
    )

    overlay = cv2.addWeighted(
        original,
        0.6,
        heatmap,
        0.4,
        0
    )

    plt.figure(figsize=(12,5))

    plt.subplot(1,2,1)
    plt.imshow(original)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1,2,2)
    plt.imshow(cv2.cvtColor(overlay,cv2.COLOR_BGR2RGB))
    plt.title(
        f"Grad-CAM\nPrediction: {CLASS_NAMES[pred.item()]}"
    )
    plt.axis("off")

    plt.tight_layout()

    plt.savefig("results/gradcam.png")

    plt.show()

    print("Grad-CAM saved to results/gradcam.png")


if __name__ == "__main__":

    IMAGE = "sample_image.tif"

    generate_gradcam(IMAGE)
