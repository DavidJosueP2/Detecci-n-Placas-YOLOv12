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
