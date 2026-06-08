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
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Add train/ to sys.path so both packages are importable without conflicts.
# Using parent (train/) avoids the ambiguous `src` namespace that appears when
# character_detection/ and character_segmentation/ are both in sys.path.
_TRAIN_DIR = str(Path(__file__).resolve().parent.parent.parent / "train")
if _TRAIN_DIR not in sys.path:
    sys.path.insert(0, _TRAIN_DIR)

from character_detection.src.model import CLASSES, build_model  # noqa: E402
from character_segmentation.src import binarization as _binarization  # noqa: E402
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
CNN_MODEL_CLASSES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def current_binarization_config() -> dict:
    return {
        "method": _binarization.get_method(),
        "options": ["otsu", "adaptive"],
    }


def update_binarization_config(method: str) -> dict:
    return {
        "method": _binarization.set_method(method),
        "options": ["otsu", "adaptive"],
    }


class ProjectCharCNN(nn.Module):
    """
    CNN used by CNN_MODEL_PATH=models/char_classifier.pth.
    Contract from its training script: input 1x28x28 in [0, 1], binary
    white character over black background, classes 0-9 then A-Z.
    """

    def __init__(self, num_classes: int = 36):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout1 = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(p=0.3)
        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout1(F.relu(self.bn_fc1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn_fc2(self.fc2(x))))
        return self.fc3(x)


def _load_cnn(checkpoint_path: str, device: torch.device) -> tuple[torch.nn.Module, list[str]]:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
        state_dict = ckpt["model_state_dict"]
        mcfg = ckpt.get("cfg", {}).get("model", {})
        model = build_model(mcfg)
        classes = CLASSES
    elif isinstance(ckpt, dict) and "conv1.weight" in ckpt:
        state_dict = ckpt
        model = ProjectCharCNN(num_classes=len(CLASSES))
        classes = CNN_MODEL_CLASSES
    elif isinstance(ckpt, dict) and "features.0.weight" in ckpt:
        state_dict = ckpt
        model = build_model({})
        classes = CLASSES
    else:
        keys = list(ckpt.keys())[:8] if isinstance(ckpt, dict) else []
        raise ValueError(
            "Checkpoint CNN no compatible. Se esperaba 'model_state_dict' "
            f"o un state_dict directo. Claves iniciales: {keys}"
        )

    model.load_state_dict(state_dict)
    model.eval()
    return model.to(device), classes


def _draw_char_bboxes(
    base_image: np.ndarray,
    bboxes: list,
    characters: list,
) -> np.ndarray:
    """
    Returns a BGR image with character bounding boxes drawn over the same
    normalized image used by the segmenter. Bboxes are already in this
    coordinate space, so this function must not redraw them over the raw crop.
    """
    if base_image is None or base_image.size == 0:
        return np.zeros((64, _CANONICAL_W, 3), dtype=np.uint8)

    if len(base_image.shape) == 2:
        canvas = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
    else:
        canvas = base_image.copy()

    img_h, img_w = canvas.shape[:2]

    for i, (bbox, char) in enumerate(zip(bboxes, characters)):
        x1 = max(0, min(img_w - 1, int(bbox["x"])))
        y1 = max(0, min(img_h - 1, int(bbox["y"])))
        x2 = max(0, min(img_w - 1, x1 + int(bbox["w"])))
        y2 = max(0, min(img_h - 1, y1 + int(bbox["h"])))
        if x2 <= x1 or y2 <= y1:
            continue
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
        self.model, self.classes = _load_cnn(model_path, self.device)
        self._uses_project_cnn = isinstance(self.model, ProjectCharCNN)
        self._crop_source = "binary" if self._uses_project_cnn else crop_source

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
        text, confidence, characters, _images, _bboxes, _stages, _seg_method = self._run_inference(crop)
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
          bbox_image  : BGR ndarray — normalized segmenter image with character
                        bboxes, indices, labels and confidences drawn on it
        """
        text, confidence, characters, images, bboxes, stages, seg_method = self._run_inference(
            crop, debug_dir=debug_dir
        )
        base_image = stages.get("1_original") if stages else None
        bbox_image = _draw_char_bboxes(base_image, bboxes, characters)
        return {
            "text": text,
            "confidence": confidence,
            "characters": characters,
            "char_images": images,
            "bboxes": bboxes,
            "stages": stages,
            "binarization_method": seg_method,
            "bbox_image": bbox_image,
        }

    # ── Internal ──────────────────────────────────────────────────────────────

    def _run_inference(
        self, crop: np.ndarray, debug_dir=None
    ) -> tuple[str, float, list, list, list, dict, str]:
        """
        Shared core: segment → CNN → decode.

        Returns (text, confidence, characters, char_images, bboxes, stages, binarization_method).
        char_images and bboxes are empty lists when segmentation fails.
        """
        if crop is None or crop.size == 0:
            return "", 0.0, [], [], [], {}, _binarization.get_method()

        try:
            seg = _segment(
                crop,
                expected_chars=7,
                crop_source=self._crop_source,
                debug_dir=debug_dir,
            )
        except Exception:
            return "", 0.0, [], [], [], {}, _binarization.get_method()

        images = seg.get("images", [])
        bboxes = seg.get("chars", [])
        stages = seg.get("_stages", {})
        seg_method = seg.get("binarization_method", _binarization.get_method())
        if not images or not bboxes:
            return "", 0.0, [], images, bboxes, stages, seg_method

        if self._uses_project_cnn:
            batch = torch.stack([self._cnn_preprocess(img) for img in images]).to(self.device)
        else:
            pil_imgs = [Image.fromarray(img) for img in images]
            batch = torch.stack([_PREPROCESS(img) for img in pil_imgs]).to(self.device)

        with torch.no_grad():
            logits = self.model(batch)
            probs = F.softmax(logits, dim=1).cpu()  # (N, 36)

        selected = self._select_plate_sequence(probs, bboxes)
        if not selected:
            return "", 0.0, [], images, bboxes, stages, seg_method

        characters = []
        total_conf = 0.0

        for value, conf, bbox in selected:
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
        return text, avg_conf, characters, images, bboxes, stages, seg_method

    @staticmethod
    def _cnn_preprocess(image: np.ndarray) -> torch.Tensor:
        if image is None or image.size == 0:
            image = np.zeros((28, 28), dtype=np.float32)

        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()

        resized = cv2.resize(gray, (28, 28), interpolation=cv2.INTER_AREA)
        if resized.dtype in {np.float32, np.float64} or (
            image.dtype in {np.float32, np.float64} and float(np.max(image)) <= 1.0
        ):
            normalized = resized.astype(np.float32)
        else:
            normalized = resized.astype(np.float32) / 255.0

        return torch.from_numpy(normalized).unsqueeze(0).float()

    def _select_plate_sequence(self, probs: torch.Tensor, bboxes: list) -> list[tuple[str, float, dict]]:
        count = min(len(probs), len(bboxes))
        if count < 6:
            return []

        target_lengths = [7] if count >= 7 else [6]
        if count >= 7:
            target_lengths.append(6)

        best = None
        for target_len in target_lengths:
            if count < target_len:
                continue
            index_sets = combinations(range(count), target_len) if count <= 12 else (
                range(start, start + target_len)
                for start in range(0, count - target_len + 1)
            )

            for indices_iter in index_sets:
                indices = tuple(indices_iter)
                selected = []
                score = 0.0
                for position, source_index in enumerate(indices):
                    kind = "letter" if position < 3 else "digit"
                    prediction = self._masked_prediction(probs[source_index], kind)
                    if prediction is None:
                        break
                    value, confidence = prediction
                    selected.append((value, confidence, bboxes[source_index]))
                    score += confidence
                else:
                    skipped_inside = (indices[-1] - indices[0] + 1) - target_len
                    edge_skip = indices[0] + (count - 1 - indices[-1])
                    avg_score = score / target_len
                    candidate_score = avg_score - skipped_inside * 0.08 - edge_skip * 0.025
                    if target_len == 7:
                        candidate_score += 0.05
                    candidate = (candidate_score, selected)
                    if best is None or candidate[0] > best[0]:
                        best = candidate

        return best[1] if best is not None else []

    def _masked_prediction(self, prob_vec: torch.Tensor, kind: str) -> tuple[str, float] | None:
        masked = prob_vec.clone()
        for class_index, class_name in enumerate(self.classes):
            if kind == "letter" and not class_name.isalpha():
                masked[class_index] = 0.0
            elif kind == "digit" and not class_name.isdigit():
                masked[class_index] = 0.0

        total = masked.sum()
        if float(total) <= 0.0:
            return None

        masked = masked / total
        best_idx = int(masked.argmax())
        return self.classes[best_idx], float(masked[best_idx])
