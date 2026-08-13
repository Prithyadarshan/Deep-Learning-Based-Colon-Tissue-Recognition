"""
test.py

Evaluate the trained EfficientNet-B3 model on the
CRC Histopathology Test Dataset.
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    ConfusionMatrixDisplay
)

from dataset import get_dataloaders
from model import create_model

# -------------------------------------------------
# Configuration
# -------------------------------------------------

DATASET_PATH = "dataset"
MODEL_PATH = "saved_models/best_model.pth"

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

RESULT_DIR = "results"
os.makedirs(RESULT_DIR, exist_ok=True)

# -------------------------------------------------
# Load Dataset
# -------------------------------------------------

train_loader, val_loader, test_loader, classes = get_dataloaders(DATASET_PATH)

# -------------------------------------------------
# Load Model
# -------------------------------------------------

model = create_model(num_classes=9)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model = model.to(DEVICE)
model.eval()

# -------------------------------------------------
# Prediction
# -------------------------------------------------

true_labels = []
pred_labels = []

with torch.no_grad():

    for images, labels in test_loader:

        images = images.to(DEVICE)

        outputs = model(images)

        _, preds = torch.max(outputs, 1)

        true_labels.extend(labels.numpy())

        pred_labels.extend(preds.cpu().numpy())

true_labels = np.array(true_labels)
pred_labels = np.array(pred_labels)

# -------------------------------------------------
# Metrics
# -------------------------------------------------

accuracy = accuracy_score(true_labels, pred_labels)

precision = precision_score(
    true_labels,
    pred_labels,
    average="weighted"
)

recall = recall_score(
    true_labels,
    pred_labels,
    average="weighted"
)

f1 = f1_score(
    true_labels,
    pred_labels,
    average="weighted"
)

print("\n===============================")
print("TEST RESULTS")
print("===============================")

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# -------------------------------------------------
# Classification Report
# -------------------------------------------------

print("\nClassification Report\n")

print(
    classification_report(
        true_labels,
        pred_labels,
        target_names=classes
    )
)

# -------------------------------------------------
# Confusion Matrix
# -------------------------------------------------

cm = confusion_matrix(true_labels, pred_labels)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=classes
)

fig, ax = plt.subplots(figsize=(10,10))

disp.plot(
    cmap="Blues",
    xticks_rotation=45,
    ax=ax
)

plt.title("Confusion Matrix")

plt.tight_layout()

plt.savefig(
    os.path.join(
        RESULT_DIR,
        "confusion_matrix.png"
    )
)

plt.show()

print("\nConfusion Matrix saved.")
