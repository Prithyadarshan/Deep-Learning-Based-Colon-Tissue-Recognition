"""
train_kfold.py

Research Grade 5-Fold Cross Validation
EfficientNet-B3

Dataset:
NCT-CRC-HE-100K
"""

import os
import copy
import random
import numpy as np

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from torch.utils.data import Subset

from sklearn.model_selection import StratifiedKFold

from torchvision import datasets
from torchvision import transforms

from model import create_model


# ----------------------------------------------------
# Configuration
# ----------------------------------------------------

DATASET_PATH = "dataset/train"

NUM_CLASSES = 9

IMAGE_SIZE = 224

BATCH_SIZE = 32

EPOCHS = 15

LR = 1e-4

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)

SAVE_DIR = "saved_models"

os.makedirs(SAVE_DIR, exist_ok=True)


# ----------------------------------------------------
# Seed
# ----------------------------------------------------

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True

    torch.backends.cudnn.benchmark = False


set_seed()


# ----------------------------------------------------
# Transform
# ----------------------------------------------------

transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomVerticalFlip(),

    transforms.RandomRotation(20),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )

])


dataset = datasets.ImageFolder(
    DATASET_PATH,
    transform=transform
)

labels = dataset.targets


# ----------------------------------------------------
# KFold
# ----------------------------------------------------

kfold = StratifiedKFold(

    n_splits=5,

    shuffle=True,

    random_state=42

)


# ----------------------------------------------------
# Train Function
# ----------------------------------------------------

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer
):

    model.train()

    running_loss = 0

    running_correct = 0

    total = 0

    for images, labels in loader:

        images = images.to(DEVICE)

        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()

        _, preds = outputs.max(1)

        total += labels.size(0)

        running_correct += preds.eq(labels).sum().item()

    loss = running_loss / len(loader)

    acc = 100 * running_correct / total

    return loss, acc


# ----------------------------------------------------
# Validation Function
# ----------------------------------------------------

def validate(
    model,
    loader,
    criterion
):

    model.eval()

    running_loss = 0

    running_correct = 0

    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)

            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()

            _, preds = outputs.max(1)

            total += labels.size(0)

            running_correct += preds.eq(labels).sum().item()

    loss = running_loss / len(loader)

    acc = 100 * running_correct / total

    return loss, acc
from torch.optim import AdamW
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

# ----------------------------------------------------
# Start Cross Validation
# ----------------------------------------------------

fold_results = []

best_overall_accuracy = 0.0

print("=" * 70)
print("5-FOLD CROSS VALIDATION TRAINING")
print("=" * 70)

for fold, (train_idx, val_idx) in enumerate(
    kfold.split(dataset.samples, labels)
):

    print(f"\n{'='*60}")
    print(f"Fold {fold + 1}/5")
    print(f"{'='*60}")

    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=4
    )

    model = create_model(NUM_CLASSES)
    model.to(DEVICE)

    criterion = nn.CrossEntropyLoss()

    optimizer = AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=1e-4
    )

    best_fold_acc = 0

    best_weights = copy.deepcopy(model.state_dict())

    for epoch in range(EPOCHS):

        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer
        )

        val_loss, val_acc = validate(
            model,
            val_loader,
            criterion
        )

        print(
            f"Epoch {epoch+1:02d}/{EPOCHS} | "
            f"Train Acc: {train_acc:.2f}% | "
            f"Val Acc: {val_acc:.2f}%"
        )

        if val_acc > best_fold_acc:

            best_fold_acc = val_acc

            best_weights = copy.deepcopy(
                model.state_dict()
            )

    model.load_state_dict(best_weights)

    torch.save(
        model.state_dict(),
        os.path.join(
            SAVE_DIR,
            f"fold_{fold+1}_best.pth"
        )
    )

    # -----------------------------
    # Final Prediction
    # -----------------------------

    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():

        for images, labels_batch in val_loader:

            images = images.to(DEVICE)

            outputs = model(images)

            preds = outputs.argmax(1).cpu().numpy()

            y_pred.extend(preds)

            y_true.extend(labels_batch.numpy())

    acc = accuracy_score(y_true, y_pred)

    precision = precision_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    fold_results.append({
        "Fold": fold + 1,
        "Accuracy": acc,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    })

    if acc > best_overall_accuracy:

        best_overall_accuracy = acc

        torch.save(
            model.state_dict(),
            os.path.join(
                SAVE_DIR,
                "best_kfold_model.pth"
            )
        )

# ----------------------------------------------------
# Final Results
# ----------------------------------------------------

print("\n")
print("=" * 70)
print("FINAL CROSS VALIDATION RESULTS")
print("=" * 70)

accuracies = []
precisions = []
recalls = []
f1_scores = []

for result in fold_results:

    accuracies.append(result["Accuracy"])
    precisions.append(result["Precision"])
    recalls.append(result["Recall"])
    f1_scores.append(result["F1"])

    print(
        f"Fold {result['Fold']} | "
        f"Accuracy={result['Accuracy']:.4f} | "
        f"Precision={result['Precision']:.4f} | "
        f"Recall={result['Recall']:.4f} | "
        f"F1={result['F1']:.4f}"
    )

print("\nAverage Results")

print(f"Accuracy : {np.mean(accuracies):.4f}")
print(f"Precision: {np.mean(precisions):.4f}")
print(f"Recall   : {np.mean(recalls):.4f}")
print(f"F1 Score : {np.mean(f1_scores):.4f}")

print("=" * 70)
print("Cross Validation Completed")
print("=" * 70)
