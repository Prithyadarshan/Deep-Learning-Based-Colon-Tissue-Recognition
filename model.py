"""
model.py

EfficientNet-B3 Transfer Learning Model
for Colorectal Cancer Histopathology Classification
"""

import torch
import torch.nn as nn
from torchvision import models


class CRCClassifier(nn.Module):
    def __init__(self, num_classes=9, dropout=0.4):
        super(CRCClassifier, self).__init__()

        # Load pretrained EfficientNet-B3
        self.model = models.efficientnet_b3(
            weights=models.EfficientNet_B3_Weights.DEFAULT
        )

        # Freeze feature extractor initially
        for param in self.model.features.parameters():
            param.requires_grad = False

        # Get input features of classifier
        in_features = self.model.classifier[1].in_features

        # Replace classifier
        self.model.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1024),
            nn.ReLU(inplace=True),

            nn.BatchNorm1d(1024),

            nn.Dropout(dropout),

            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),

            nn.BatchNorm1d(512),

            nn.Dropout(dropout),

            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.model(x)


def create_model(num_classes=9):
    return CRCClassifier(num_classes=num_classes)


# ----------------------------------------------------
# Unfreeze feature extractor for fine-tuning
# ----------------------------------------------------

def unfreeze_layers(model):
    """
    Enables training of all EfficientNet layers.
    Call this after a few epochs.
    """

    for param in model.model.features.parameters():
        param.requires_grad = True

    return model


# ----------------------------------------------------
# Count Parameters
# ----------------------------------------------------

def count_parameters(model):

    total = sum(p.numel() for p in model.parameters())

    trainable = sum(
        p.numel() for p in model.parameters()
        if p.requires_grad
    )

    return total, trainable


# ----------------------------------------------------
# Test
# ----------------------------------------------------

if __name__ == "__main__":

    model = create_model()

    print(model)

    total, trainable = count_parameters(model)

    print(f"\nTotal Parameters : {total:,}")
    print(f"Trainable Parameters : {trainable:,}")

    x = torch.randn(2, 3, 224, 224)

    y = model(x)

    print("\nOutput Shape :", y.shape)
