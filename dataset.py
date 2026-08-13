import os
import random

from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset


IMAGE_SIZE = 224
BATCH_SIZE = 32
NUM_WORKERS = 0

DATASET_PATH = r"database\NCT-CRC-HE-100K"


train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


def get_dataloaders(dataset_root=None):

    dataset_path = DATASET_PATH

    base_dataset = datasets.ImageFolder(dataset_path)

    total_size = len(base_dataset)

    indices = list(range(total_size))

    random.seed(42)
    random.shuffle(indices)

    train_size = int(0.8 * total_size)
    val_size = int(0.1 * total_size)

    train_indices = indices[:train_size]

    val_indices = indices[
        train_size:train_size + val_size
    ]

    test_indices = indices[
        train_size + val_size:
    ]

    train_dataset = datasets.ImageFolder(
        dataset_path,
        transform=train_transform
    )

    val_dataset = datasets.ImageFolder(
        dataset_path,
        transform=val_transform
    )

    test_dataset = datasets.ImageFolder(
        dataset_path,
        transform=val_transform
    )

    train_subset = Subset(
        train_dataset,
        train_indices
    )

    val_subset = Subset(
        val_dataset,
        val_indices
    )

    test_subset = Subset(
        test_dataset,
        test_indices
    )

    train_loader = DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    test_loader = DataLoader(
        test_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    print("=" * 50)

    print("DATASET INFORMATION")

    print("=" * 50)

    print("Classes:", base_dataset.classes)

    print("Total Images:", total_size)

    print("Training Images:", len(train_subset))

    print("Validation Images:", len(val_subset))

    print("Testing Images:", len(test_subset))

    print("=" * 50)

    return (
        train_loader,
        val_loader,
        test_loader,
        base_dataset.classes
    )


if __name__ == "__main__":

    train_loader, val_loader, test_loader, classes = (
        get_dataloaders()
    )
