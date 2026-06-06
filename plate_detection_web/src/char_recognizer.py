"""
Character recognition pipeline: classical CV segmentation + CNN classifier.

Drop-in replacement for PlateReader with the same read(crop) interface:
    (text, confidence, characters) = recognizer.read(crop_bgr)

Design:
  - segment(crop) from train/character_segmentation: locates 7 character
    bounding boxes and returns 32×32 grayscale crops for each.
  - CharCNN from train/character_detection: classifies each crop in a
    single batched forward pass.
  - Position constraint: positions 0-2 → letters (A-Z), 3-6 → digits (0-9),
    matching Ecuadorian plate format ABC-1234.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Add train/ to sys.path so both packages are importable without conflicts.
# Using parent (train/) avoids the ambiguous `src` namespace that appears when
# character_detection/ and character_segmentation/ are both in sys.path.
_TRAIN_DIR = str(Path(__file__).resolve().parent.parent.parent / "train")
if _TRAIN_DIR not in sys.path:
    sys.path.insert(0, _TRAIN_DIR)

from character_detection.src.model import CLASSES, CharCNN  # noqa: E402
from character_segmentation.src.segmentation import segment as _segment  # noqa: E402

# Identical preprocessing as inference.py — copied here to avoid that
# module's os.chdir() call which breaks when imported inside a web server.
_IMG_SIZE = 32
_PREPROCESS = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((_IMG_SIZE, _IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def _load_cnn(checkpoint_path: str, device: torch.device) -> CharCNN:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    mcfg = ckpt.get("cfg", {}).get("model", {})
    model = CharCNN(
        num_classes=mcfg.get("num_classes", 36),
        dropout1=mcfg.get("dropout1", 0.5),
        dropout2=mcfg.get("dropout2", 0.3),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model.to(device)


class CharRecognizer:
    """
    Segmentation + CNN OCR for Ecuadorian license plates.

    Interface matches PlateReader.read() so VideoStream needs no changes:
        text, confidence, characters = recognizer.read(crop_bgr_ndarray)
    """

    def __init__(self, model_path: str, device: str = "auto"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = _load_cnn(model_path, self.device)

    def read(self, crop: np.ndarray) -> tuple[str, float, list]:
        """
        Parameters
        ----------
        crop : BGR numpy array (output of plate detector)

        Returns
        -------
        (text, confidence, characters)
          text        : plate string, e.g. "ABC1234"
          confidence  : mean per-character confidence [0, 1]
          characters  : list of dicts compatible with postprocess_ecuador_plate()
                        keys: value, confidence, center_x, center_y, height, width,
                              x1, y1, x2, y2
        """
        if crop is None or crop.size == 0:
            return "", 0.0, []

        try:
            seg = _segment(crop, expected_chars=7)
        except Exception:
            return "", 0.0, []

        images = seg.get("images", [])
        bboxes = seg.get("chars", [])
        if not images or not bboxes:
            return "", 0.0, []

        # Convert grayscale numpy arrays to PIL, apply preprocessing, stack batch
        pil_imgs = [Image.fromarray(img) for img in images]
        batch = torch.stack([_PREPROCESS(img) for img in pil_imgs]).to(self.device)

        with torch.no_grad():
            logits = self.model(batch)
            probs = F.softmax(logits, dim=1).cpu()  # (N, 36)

        characters = []
        total_conf = 0.0

        for i, (prob_vec, bbox) in enumerate(zip(probs, bboxes)):
            # Plate format ABC-1234: positions 0-2 = letters, 3-6 = digits
            if i < 3:
                prob_vec[26:] = 0.0    # mask digit logits
            else:
                prob_vec[:26] = 0.0    # mask letter logits

            s = prob_vec.sum()
            if s > 0:
                prob_vec = prob_vec / s

            best_idx = int(prob_vec.argmax())
            conf = float(prob_vec[best_idx])
            value = CLASSES[best_idx]

            total_conf += conf
            characters.append({
                "value": value,
                "confidence": conf,
                "center_x": float(bbox["x"] + bbox["w"] / 2.0),
                "center_y": float(bbox["y"] + bbox["h"] / 2.0),
                "height": float(bbox["h"]),
                "width": float(bbox["w"]),
                "x1": bbox["x"],
                "y1": bbox["y"],
                "x2": bbox["x"] + bbox["w"],
                "y2": bbox["y"] + bbox["h"],
            })

        text = "".join(c["value"] for c in characters)
        avg_conf = total_conf / len(characters) if characters else 0.0
        return text, avg_conf, characters
