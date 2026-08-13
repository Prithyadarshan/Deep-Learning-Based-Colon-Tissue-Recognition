"""
dataset.py

Loads the colorectal histopathology dataset using ImageFolder.

Folder Structure

dataset/
    train/
        ADI/
        BACK/
        DEB/
        LYM/
        MUC/
        MUS/
        NORM/
        STR/
        TUM/

    val/
        ...

    test/
        ...
"""

import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 4


# -----------------------------
# Data Augmentation
# -----------------------------

train_transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.RandomHorizontalFlip(),

    transforms.RandomVerticalFlip(),

    transforms.RandomRotation(20),

    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),

    transforms.RandomAffine(
        degrees=15,
        translate=(0.05,0.05)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


# -----------------------------
# Validation Transform
# -----------------------------

val_transform = transforms.Compose([

    transforms.Resize((224,224)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485,0.456,0.406],
        std=[0.229,0.224,0.225]
    )
])


# -----------------------------
# Test Transform
# -----------------------------

test_transform = val_transform


# -----------------------------
# Dataset Loader
# -----------------------------

def get_datasets(dataset_root):

    train_dataset = datasets.ImageFolder(

        os.path.join(dataset_root, "train"),
        transform=train_transform

    )

    val_dataset = datasets.ImageFolder(

        os.path.join(dataset_root, "val"),
        transform=val_transform

    )

    test_dataset = datasets.ImageFolder(

        os.path.join(dataset_root, "test"),
        transform=test_transform

    )

    return train_dataset, val_dataset, test_dataset


# -----------------------------
# DataLoader
# -----------------------------

def get_dataloaders(dataset_root):

    train_dataset, val_dataset, test_dataset = get_datasets(dataset_root)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        train_dataset.classes
    )


# -----------------------------
# Example
# -----------------------------

if __name__ == "__main__":

    root = "dataset"

    train_loader, val_loader, test_loader, classes = get_dataloaders(root)

    print("Classes:", classes)

    print("Number of Classes:", len(classes))

    print("Training Images:", len(train_loader.dataset))

    print("Validation Images:", len(val_loader.dataset))

    print("Testing Images:", len(test_loader.dataset))
