"""
train.py

Training script for Colorectal Cancer Histopathology
Classification using EfficientNet-B3.
"""

import os
import time
import copy

import torch
import torch.nn as nn

from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau

from torch.cuda.amp import autocast, GradScaler

from dataset import get_dataloaders
from model import create_model, unfreeze_layers

# ----------------------------
# Configuration
# ----------------------------

DATASET_PATH = "dataset"

NUM_CLASSES = 9

EPOCHS = 30

LEARNING_RATE = 1e-4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

SAVE_DIR = "saved_models"

os.makedirs(SAVE_DIR, exist_ok=True)

# ----------------------------
# Load Dataset
# ----------------------------

train_loader, val_loader, test_loader, classes = get_dataloaders(
    DATASET_PATH
)

print("Classes:", classes)

# ----------------------------
# Create Model
# ----------------------------

model = create_model(NUM_CLASSES)

model = model.to(DEVICE)

criterion = nn.CrossEntropyLoss()

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=1e-4
)

scheduler = ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=3
)

scaler = GradScaler()

# ----------------------------
# Training Function
# ----------------------------

def train_one_epoch():

    model.train()

    running_loss = 0.0

    running_correct = 0

    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        with autocast():

            outputs = model(images)

            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        scaler.step(optimizer)

        scaler.update()

        running_loss += loss.item()

        _, predicted = outputs.max(1)

        total += labels.size(0)

        running_correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_loader)

    epoch_acc = 100 * running_correct / total

    return epoch_loss, epoch_acc

# ----------------------------
# Validation Function
# ----------------------------

def validate():

    model.eval()

    running_loss = 0

    running_correct = 0

    total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, predicted = outputs.max(1)

            total += labels.size(0)

            running_correct += predicted.eq(labels).sum().item()

    val_loss = running_loss / len(val_loader)

    val_acc = 100 * running_correct / total

    return val_loss, val_acc
# ----------------------------
# Early Stopping
# ----------------------------

best_accuracy = 0.0
best_weights = copy.deepcopy(model.state_dict())

early_stop_counter = 0
patience = 7

history = {
    "train_loss": [],
    "train_acc": [],
    "val_loss": [],
    "val_acc": []
}

print("\nStarting Training...\n")

start_time = time.time()

# ----------------------------
# Training Loop
# ----------------------------

for epoch in range(EPOCHS):

    print("=" * 60)
    print(f"Epoch [{epoch+1}/{EPOCHS}]")

    # Fine-tune entire network after 5 epochs
    if epoch == 5:
        print("Unfreezing EfficientNet feature extractor...")
        model = unfreeze_layers(model)

    train_loss, train_acc = train_one_epoch()

    val_loss, val_acc = validate()

    scheduler.step(val_acc)

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"Train Loss : {train_loss:.4f}")
    print(f"Train Acc  : {train_acc:.2f}%")
    print(f"Val Loss   : {val_loss:.4f}")
    print(f"Val Acc    : {val_acc:.2f}%")

    # Save best model
    if val_acc > best_accuracy:

        best_accuracy = val_acc

        best_weights = copy.deepcopy(model.state_dict())

        torch.save(
            model.state_dict(),
            os.path.join(SAVE_DIR, "best_model.pth")
        )

        print("Best model saved.")

        early_stop_counter = 0

    else:

        early_stop_counter += 1

    # Early stopping
    if early_stop_counter >= patience:

        print("\nEarly stopping triggered.")
        break

print("=" * 60)

elapsed = time.time() - start_time

print(f"\nTraining Completed in {elapsed/60:.2f} minutes")

print(f"Best Validation Accuracy : {best_accuracy:.2f}%")

model.load_state_dict(best_weights)

torch.save(
    model.state_dict(),
    os.path.join(SAVE_DIR, "final_model.pth")
)

print("Final model saved.")
