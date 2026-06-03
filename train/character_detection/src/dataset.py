import random
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset, Subset, DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

from src.model import CLASSES, CLASS_TO_IDX


class AddGaussianNoise:
    """Additive Gaussian noise applied on a normalized tensor."""

    def __init__(self, std: float = 0.05):
        self.std = std

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + torch.randn_like(tensor) * self.std

    def __repr__(self) -> str:
        return f"AddGaussianNoise(std={self.std})"


class TransformSubset(Dataset):
    """
    Wraps a Subset and applies a transform independently of the parent dataset's
    transform.  This lets train/val/test share the same underlying ImageFolder
    while each split gets its own augmentation pipeline.
    """

    def __init__(self, subset: Subset, transform):
        self.subset = subset
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.subset[idx]
        return self.transform(img), label

    def __len__(self) -> int:
        return len(self.subset)


def get_transforms(cfg: dict, augment: bool = False) -> transforms.Compose:
    size = cfg["data"]["img_size"]
    aug = cfg["augmentation"]

    pipeline = [
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((size, size)),
    ]

    if augment:
        pipeline += [
            transforms.RandomRotation(degrees=aug["rotation_degrees"]),
            transforms.RandomAffine(
                degrees=0,
                translate=tuple(aug["translate"]),
            ),
            transforms.ColorJitter(
                brightness=aug["brightness"],
                contrast=aug["contrast"],
            ),
            transforms.RandomAutocontrast(p=0.2),
        ]

    pipeline += [
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ]

    if augment:
        pipeline.append(AddGaussianNoise(std=aug["noise_std"]))

    return transforms.Compose(pipeline)


def build_dataloaders(cfg: dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    root = Path(cfg["data"]["root"])
    if not root.exists():
        raise FileNotFoundError(
            f"Dataset root not found: {root}\n"
            "Run:  python scripts/generate_synthetic.py\n"
            "Then: python scripts/prepare_chars74k.py  (optional, if you have Chars74K)"
        )

    # Load without transforms so PIL images are returned
    dataset = ImageFolder(root=str(root))
    dataset = _remap_classes(dataset)

    if len(dataset) == 0:
        raise RuntimeError(
            f"No valid samples found in {root}.\n"
            "Ensure subdirectories are named with single chars: A-Z or 0-9."
        )

    train_idx, val_idx, test_idx = _stratified_split(
        dataset,
        cfg["data"]["train_split"],
        cfg["data"]["val_split"],
    )

    train_tf = get_transforms(cfg, augment=True)
    eval_tf = get_transforms(cfg, augment=False)

    train_ds = TransformSubset(Subset(dataset, train_idx), train_tf)
    val_ds = TransformSubset(Subset(dataset, val_idx), eval_tf)
    test_ds = TransformSubset(Subset(dataset, test_idx), eval_tf)

    num_workers = cfg["training"].get("num_workers", 2)
    batch = cfg["training"]["batch_size"]

    train_loader = DataLoader(
        train_ds, batch_size=batch, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    print(
        f"Dataset  total={len(dataset):,}  "
        f"train={len(train_ds):,}  val={len(val_ds):,}  test={len(test_ds):,}"
    )
    return train_loader, val_loader, test_loader


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _remap_classes(dataset: ImageFolder) -> ImageFolder:
    """
    Rebuilds dataset.samples so labels match our canonical CLASSES order
    (A=0 … Z=25, 0=26 … 9=35).  Filters out any folder not in CLASSES.
    Handles lowercase folder names by uppercasing them.
    """
    new_samples = []
    for path, label in dataset.samples:
        name = dataset.classes[label].upper()
        if name in CLASS_TO_IDX:
            new_samples.append((path, CLASS_TO_IDX[name]))

    dataset.samples = new_samples
    dataset.targets = [s[1] for s in new_samples]
    dataset.class_to_idx = CLASS_TO_IDX
    dataset.classes = CLASSES
    return dataset


def _stratified_split(dataset: ImageFolder, train_frac: float, val_frac: float):
    """Stratified split preserving per-class distribution."""
    buckets: dict[int, list[int]] = {i: [] for i in range(len(CLASSES))}
    for idx, (_, label) in enumerate(dataset.samples):
        buckets[label].append(idx)

    train_idx, val_idx, test_idx = [], [], []
    for indices in buckets.values():
        random.shuffle(indices)
        n = len(indices)
        n_train = int(n * train_frac)
        n_val = int(n * val_frac)
        train_idx.extend(indices[:n_train])
        val_idx.extend(indices[n_train : n_train + n_val])
        test_idx.extend(indices[n_train + n_val :])

    return train_idx, val_idx, test_idx
