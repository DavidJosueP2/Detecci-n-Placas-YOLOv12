import torch
import torch.nn as nn

CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
NUM_CLASSES = len(CLASSES)  # 36
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}


class CharCNN(nn.Module):
    """
    LeNet-5 extended for 36-class character recognition.

    Architecture:
        3x [Conv(3x3) -> BN -> ReLU -> MaxPool(2x2)]
        Flatten -> Dropout -> Dense(256) -> ReLU -> Dropout -> Dense(36)

    Input:  (N, 1, 32, 32)  — grayscale, normalized to [-1, 1]
    Output: (N, 36)          — raw logits (use CrossEntropyLoss)
    """

    def __init__(
        self,
        num_classes: int = 36,
        dropout1: float = 0.5,
        dropout2: float = 0.3,
    ):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 32x32 -> 16x16
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 16x16 -> 8x8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 8x8 -> 4x4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # 128 channels × 4×4 spatial = 2048
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(p=dropout1),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


class CharCNNv2(nn.Module):
    """
    4-block CNN for 36-class character recognition.

    Differences from CharCNN:
      - Block 4: Conv(128→256) without pooling, deepening RF to 38px (full
        coverage of the 32px input) without further spatial collapse.
      - Global Average Pooling replaces Flatten+FC(2048→256).  GAP reduces
        parameters from ~627K to ~398K and makes the classifier insensitive
        to small translations within the character crop — better suited to
        the 40K-sample regime than Dropout(0.5) alone.
      - Single Linear(256→36) after GAP; no second hidden layer needed.

    Input:  (N, 1, 32, 32)  — grayscale, normalized to [-1, 1]
    Output: (N, 36)          — raw logits (use CrossEntropyLoss)
    """

    def __init__(self, num_classes: int = 36, dropout: float = 0.35):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 32×32 → 16×16
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: 16×16 → 8×8
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: 8×8 → 4×4
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4: 4×4 → 4×4 (no pooling — deepen RF without more spatial collapse)
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),   # 256×4×4 → 256×1×1
            nn.Flatten(),
            nn.Dropout(p=dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(mcfg: dict) -> nn.Module:
    """
    Instantiate the correct model class from a model config dict.
    Compatible with checkpoints saved by trainer.py (cfg['model'] key).
    """
    arch = mcfg.get("arch", "CharCNN")
    num_classes = mcfg.get("num_classes", 36)
    if arch == "CharCNNv2":
        return CharCNNv2(
            num_classes=num_classes,
            dropout=mcfg.get("dropout", 0.35),
        )
    return CharCNN(
        num_classes=num_classes,
        dropout1=mcfg.get("dropout1", 0.5),
        dropout2=mcfg.get("dropout2", 0.3),
    )
