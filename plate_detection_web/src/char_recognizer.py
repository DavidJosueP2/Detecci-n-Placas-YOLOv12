"""
Character recognition pipeline: classical CV segmentation + CNN classifier.

OCR pipeline with the same read(crop) interface used by VideoStream:
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

import cv2
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

# Must match character_segmentation/src/preprocessing.py CANONICAL_WIDTH
_CANONICAL_W = 440


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


def _draw_char_bboxes(
    crop: np.ndarray,
    bboxes: list,
    characters: list,
) -> np.ndarray:
    """
    Returns a BGR image (canonical 440 px wide) with character bounding boxes
    drawn over the plate crop.  Bboxes use the coordinate space that
    _segment() produces, which operates at _CANONICAL_W px width — so the
    crop is resized to that width before drawing.
    """
    if crop is None or crop.size == 0:
        return np.zeros((64, _CANONICAL_W, 3), dtype=np.uint8)

    h, w = crop.shape[:2]
    scale = _CANONICAL_W / max(w, 1)
    canvas_h = max(1, int(round(h * scale)))

    if len(crop.shape) == 2:
        resized = cv2.cvtColor(
            cv2.resize(crop, (_CANONICAL_W, canvas_h)), cv2.COLOR_GRAY2BGR
        )
    else:
        resized = cv2.resize(crop, (_CANONICAL_W, canvas_h))

    canvas = resized.copy()

    for i, (bbox, char) in enumerate(zip(bboxes, characters)):
        x1 = int(bbox["x"])
        y1 = int(bbox["y"])
        x2 = x1 + int(bbox["w"])
        y2 = y1 + int(bbox["h"])
        conf = float(char["confidence"])

        g = int(255 * conf)
        r = int(255 * (1.0 - conf))
        color = (0, g, r)

        cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 2)

        label = f"{i}:{char['value']} {conf:.0%}"
        fs = 0.45
        th = 1
        (tw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, th)
        tx = max(0, x1)
        ty = max(lh + 3, y1 - 3)
        cv2.rectangle(canvas, (tx, ty - lh - 2), (tx + tw + 4, ty + 2), color, -1)
        cv2.putText(
            canvas, label, (tx + 2, ty),
            cv2.FONT_HERSHEY_SIMPLEX, fs, (0, 0, 0), th,
        )

    return canvas


class CharRecognizer:
    """
    Segmentation + CNN OCR for Ecuadorian license plates.

    Interface used by VideoStream:
        text, confidence, characters = recognizer.read(crop_bgr_ndarray)
    """

    def __init__(self, model_path: str, device: str = "auto", crop_source: str = "gray"):
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = _load_cnn(model_path, self.device)
        self._crop_source = crop_source

    # ── Public interface ──────────────────────────────────────────────────────

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
        text, confidence, characters, _images, _bboxes = self._run_inference(crop)
        return text, confidence, characters

    def read_debug(self, crop: np.ndarray, debug_dir=None) -> dict:
        """
        Same OCR as read() but also returns segmentation debug artefacts.

        Parameters
        ----------
        crop      : BGR numpy array
        debug_dir : optional Path or str; when set, all intermediate segmentation
                    stage images are written there by _segment() (gray, CLAHE,
                    denoised, binary, bboxes, chars strip, mosaic).

        Returns
        -------
        dict with keys:
          text, confidence, characters  — identical to read()
          char_images : list of 32×32 uint8 grayscale arrays, one per character,
                        exactly the arrays that enter the CNN (before tensor conversion)
          bbox_image  : BGR ndarray — plate crop resized to canonical width with
                        character bboxes, indices, labels and confidences drawn on it
        """
        text, confidence, characters, images, bboxes = self._run_inference(
            crop, debug_dir=debug_dir
        )
        bbox_image = _draw_char_bboxes(crop, bboxes, characters)
        return {
            "text": text,
            "confidence": confidence,
            "characters": characters,
            "char_images": images,
            "bbox_image": bbox_image,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run_inference(
        self, crop: np.ndarray, debug_dir=None
    ) -> tuple[str, float, list, list, list]:
        """
        Shared core: segment → CNN → decode.

        Returns (text, confidence, characters, char_images, bboxes).
        char_images and bboxes are empty lists when segmentation fails.
        """
        if crop is None or crop.size == 0:
            return "", 0.0, [], [], []

        try:
            seg = _segment(
                crop,
                expected_chars=7,
                crop_source=self._crop_source,
                debug_dir=debug_dir,
            )
        except Exception:
            return "", 0.0, [], [], []

        images = seg.get("images", [])
        bboxes = seg.get("chars", [])
        if not images or not bboxes:
            return "", 0.0, [], images, bboxes

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
        return text, avg_conf, characters, images, bboxes
