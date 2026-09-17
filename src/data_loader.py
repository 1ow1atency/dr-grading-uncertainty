"""APTOS 2019 DR dataset: stratified splits, augmentations, and DataLoaders."""

import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

NUM_CLASSES = 5
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class APTOSDataset(Dataset):
    """Wraps an APTOS 2019 (id_code, diagnosis) dataframe over a directory of images."""

    def __init__(self, dataframe, img_dir, transform=None, img_ext=".png"):
        self.dataframe = dataframe.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform
        self.img_ext = img_ext

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = os.path.join(self.img_dir, f"{row['id_code']}{self.img_ext}")
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        label = int(row["diagnosis"])
        return image, label


def get_transforms(split: str, img_size: int = 224):
    """Fundus images are rotationally symmetric (no canonical "up"), so unlike
    natural-image pipelines, vertical flips and free rotation are valid augmentations
    here rather than label-corrupting distortions."""
    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize((img_size, img_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5),
                transforms.RandomRotation(20),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def stratified_split(df: pd.DataFrame, val_size: float = 0.15, test_size: float = 0.15, seed: int = 42):
    train_df, temp_df = train_test_split(
        df, test_size=val_size + test_size, stratify=df["diagnosis"], random_state=seed
    )
    relative_test_size = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df, test_size=relative_test_size, stratify=temp_df["diagnosis"], random_state=seed
    )
    return train_df, val_df, test_df


def compute_class_weights(df: pd.DataFrame, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    """Inverse-frequency class weights for use with CrossEntropyLoss(weight=...)."""
    weights = compute_class_weight(
        class_weight="balanced", classes=np.arange(num_classes), y=df["diagnosis"].values
    )
    return torch.tensor(weights, dtype=torch.float32)


def get_dataloaders(
    csv_path: str,
    img_dir: str,
    img_size: int = 224,
    batch_size: int = 32,
    val_size: float = 0.15,
    test_size: float = 0.15,
    num_workers: int = 4,
    seed: int = 42,
    img_ext: str = ".png",
):
    """Builds train/val/test DataLoaders from the APTOS 2019 train.csv layout
    (columns: id_code, diagnosis) plus class weights computed on the train split."""
    df = pd.read_csv(csv_path)
    train_df, val_df, test_df = stratified_split(df, val_size, test_size, seed)

    train_dataset = APTOSDataset(train_df, img_dir, get_transforms("train", img_size), img_ext)
    val_dataset = APTOSDataset(val_df, img_dir, get_transforms("val", img_size), img_ext)
    test_dataset = APTOSDataset(test_df, img_dir, get_transforms("test", img_size), img_ext)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    class_weights = compute_class_weights(train_df)

    return train_loader, val_loader, test_loader, class_weights
