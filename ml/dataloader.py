"""
dataloader.py — PyTorch Dataset and DataLoader factory.
"""

import functools
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

from .config import (
    IMG_SIZE, BATCH_SIZE, NUM_WORKERS, PIN_MEMORY, SPLITS_CSV,
    NORMALIZE_MEAN, NORMALIZE_STD,
    AUG_HFLIP_PROB, AUG_ROTATION_DEG, AUG_BRIGHTNESS, AUG_CONTRAST,
    CROP_SCALE_MIN, EVAL_RESIZE_RATIO,
    USE_MIXUP, MIXUP_ALPHA, EXCLUDE_PATHS_FILE,
)


def get_train_transforms() -> transforms.Compose:
    """
    Augmentation pipeline for the training split.

    Design notes for X-ray images:
    - RandomResizedCrop: zooms into a random 60–100% area region before
      resizing, forcing the model to generalise to different scales/offsets.
      Much stronger than a plain Resize which always sees the same field-of-view.
    - HorizontalFlip: valid — left/right forearm views are symmetric
    - NO vertical flip — anatomical orientation (proximal/distal) matters clinically
    - Rotation ±15°: mimics real positioning variation in the X-ray room
    - Brightness/contrast jitter: simulates exposure variation across machines
    - NO elastic deformation — fracture lines are subtle; aggressive spatial
      distortion risks removing or fabricating pathology
    - Grayscale → 3-channel: pretrained models expect 3-channel input;
      repeating the single channel 3× preserves intensity while satisfying the API
    """
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomResizedCrop(IMG_SIZE, scale=(CROP_SCALE_MIN, 1.0)),
        transforms.RandomHorizontalFlip(p=AUG_HFLIP_PROB),
        transforms.RandomRotation(degrees=AUG_ROTATION_DEG),
        transforms.ColorJitter(
            brightness=AUG_BRIGHTNESS,
            contrast=AUG_CONTRAST,
        ),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])


def get_val_transforms() -> transforms.Compose:
    """
    Deterministic pipeline for val/test — no augmentation.

    Resize to slightly larger than IMG_SIZE then centre-crop — the standard
    ImageNet evaluation protocol. Avoids the minor distortion of squeezing
    a non-square image directly to IMG_SIZE × IMG_SIZE.
    """
    eval_size = int(IMG_SIZE * EVAL_RESIZE_RATIO)
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(eval_size),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])


def get_tta_transforms(img_size: int = IMG_SIZE, n_views: int = 5) -> list:
    """
    Return a list of N deterministic TTA (Test-Time Augmentation) transform pipelines.

    Safe augmentations for X-rays (anatomical orientation preserved):
        0: original (centre crop, no flip)
        1: horizontal flip  (left/right forearm symmetry)
        2: rotation +10°    (positioning variation)
        3: rotation -10°
        4: slight zoom-in   (95% resize → centre crop)

    All views share Grayscale → Resize → (view-specific) → ToTensor → Normalize.
    """
    eval_size  = int(img_size * EVAL_RESIZE_RATIO)
    base_pre   = [
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(eval_size),
        transforms.CenterCrop(img_size),
    ]
    base_post  = [
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ]
    view_augmentations = [
        [],                                                           # 0: original
        [transforms.RandomHorizontalFlip(p=1.0)],                    # 1: hflip
        [transforms.RandomRotation(degrees=(10, 10))],               # 2: +10°
        [transforms.RandomRotation(degrees=(-10, -10))],             # 3: -10°
        [transforms.Resize(int(img_size * 0.95)),
         transforms.CenterCrop(img_size)],                           # 4: slight zoom
    ]
    return [
        transforms.Compose(base_pre + aug + base_post)
        for aug in view_augmentations[:n_views]
    ]


def mixup_collate_fn(batch, alpha: float = 0.2):
    """
    MixUp collate function for the training DataLoader.

    Randomly interpolates image pairs within a batch:
        mixed_image = λ * img_a + (1 − λ) * img_b  where λ ~ Beta(alpha, alpha)

    Labels remain integer class indices (primary label kept). CrossEntropyLoss
    with label_smoothing handles soft-label regularisation implicitly.

    Args:
        batch: list of (tensor, label) from FractureDataset.__getitem__
        alpha: Beta distribution parameter. 0.2 = mild mixing (recommended).
               Increase to 0.4 for stronger regularisation.
    """
    images, labels = zip(*batch)
    images = torch.stack(images)         # (B, C, H, W)
    labels = torch.tensor(labels)        # (B,)
    lam    = np.random.beta(alpha, alpha)
    idx    = torch.randperm(images.size(0))
    mixed  = lam * images + (1 - lam) * images[idx]
    return mixed, labels


class FractureDataset(Dataset):
    """
    PyTorch Dataset for binary fracture classification.

    Args:
        splits_csv: Path to splits.csv generated by dataset.py
        split:      'train', 'val', or 'test'
        transform:  torchvision Compose pipeline to apply
    """

    def __init__(
        self,
        splits_csv: Path,
        split: Literal["train", "val", "test"],
        transform: transforms.Compose,
        exclude_paths: Path | None = None,
    ):
        df = pd.read_csv(splits_csv)
        df = df[df["split"] == split].reset_index(drop=True)

        # Exclude quality-flagged images (only meaningful for train split)
        if exclude_paths is not None and Path(exclude_paths).exists():
            excluded = set(Path(exclude_paths).read_text().splitlines())
            before   = len(df)
            df       = df[~df["path"].isin(excluded)].reset_index(drop=True)
            print(f"[dataset] {split}: excluded {before - len(df)} paths via {exclude_paths}")
        elif "quality_flag" in df.columns and split == "train":
            before = len(df)
            df     = df[df["quality_flag"].fillna("") == ""].reset_index(drop=True)
            if before > len(df):
                print(f"[dataset] {split}: excluded {before - len(df)} quality-flagged images")

        assert len(df) > 0, f"No records found for split='{split}' in {splits_csv}"
        # Pre-extract to plain lists — avoids per-sample pandas iloc overhead in workers
        self.paths  = df["path"].tolist()
        self.labels = df["label_id"].tolist()
        self._label_series = df["label"]  # kept only for label_counts()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        # convert("L") normalises to 8-bit grayscale regardless of source bit depth
        image = Image.open(self.paths[idx]).convert("L")
        label = int(self.labels[idx])  # 0=normal, 1=fracture
        if self.transform:
            image = self.transform(image)
        return image, label

    def label_counts(self) -> dict:
        return self._label_series.value_counts().to_dict()


def get_dataloaders(
    splits_csv: Path = SPLITS_CSV,
    exclude_paths: Path | None = None,
) -> dict[str, DataLoader]:
    """
    Build and return train / val / test DataLoaders.

    Args:
        exclude_paths: Optional path to a text file (one image path per line)
                       listing images to exclude from the train split.
                       Defaults to EXCLUDE_PATHS_FILE from config if it exists.

    Returns:
        dict with keys 'train', 'val', 'test'
    """
    _exclude = exclude_paths if exclude_paths is not None else EXCLUDE_PATHS_FILE

    train_ds = FractureDataset(splits_csv, "train", get_train_transforms(), exclude_paths=_exclude)
    val_ds   = FractureDataset(splits_csv, "val",   get_val_transforms())
    test_ds  = FractureDataset(splits_csv, "test",  get_val_transforms())

    print(f"Train : {len(train_ds):,} images  {train_ds.label_counts()}")
    print(f"Val   : {len(val_ds):,} images  {val_ds.label_counts()}")
    print(f"Test  : {len(test_ds):,} images  {test_ds.label_counts()}")

    persistent  = NUM_WORKERS > 0
    train_collate = functools.partial(mixup_collate_fn, alpha=MIXUP_ALPHA) if USE_MIXUP else None

    train_dl = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=persistent,
        drop_last=True,   # avoids unstable BatchNorm on tiny last batch
        collate_fn=train_collate,
    )
    val_dl = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=persistent,
    )
    test_dl = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        persistent_workers=persistent,
    )

    return {"train": train_dl, "val": val_dl, "test": test_dl}
