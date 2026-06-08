"""
Debug visualization: annotated images and mosaic grid.
"""

import math
from pathlib import Path

import cv2
import numpy as np


# ── Annotation helpers ────────────────────────────────────────────────────────

def draw_bboxes(
    img: np.ndarray,
    bboxes: list[dict],
    color: tuple = (0, 255, 0),
    thickness: int = 2,
    label_idx: bool = True,
) -> np.ndarray:
    """Returns a copy of img with bounding boxes drawn."""
    out = img.copy() if img.ndim == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for i, b in enumerate(bboxes):
        x, y, w, h = b["x"], b["y"], b["w"], b["h"]
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
        if label_idx:
            cv2.putText(
                out, str(i), (x, max(0, y - 4)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA,
            )
    return out


# ── Per-character strips ──────────────────────────────────────────────────────

def chars_strip(char_images: list[np.ndarray], size: int = 32) -> np.ndarray:
    """Returns a horizontal strip of all character images (for quick preview)."""
    if not char_images:
        return np.zeros((size, size), dtype=np.uint8)
    row = []
    for img in char_images:
        c = cv2.resize(img, (size, size))
        if c.ndim == 2:
            c = cv2.cvtColor(c, cv2.COLOR_GRAY2BGR)
        # Add a thin vertical separator
        sep = np.zeros((size, 2, 3), dtype=np.uint8)
        row.extend([c, sep])
    return np.hstack(row[:-1])  # remove trailing separator


# ── Debug mosaic ──────────────────────────────────────────────────────────────

def _to_bgr(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def _pad_to_width(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    if w >= width:
        return img
    pad = np.zeros((h, width - w, 3), dtype=np.uint8)
    return np.hstack([img, pad])


def make_mosaic(stages: dict[str, np.ndarray], target_width: int = 440) -> np.ndarray:
    """
    Creates a vertical mosaic of all stage images, each labeled.
    stages: OrderedDict of {label: image}
    """
    rows = []
    label_h = 20
    font = cv2.FONT_HERSHEY_SIMPLEX

    for label, img in stages.items():
        bgr = _to_bgr(img)
        # Resize to target_width preserving aspect ratio
        h, w = bgr.shape[:2]
        scale = target_width / w if w > 0 else 1
        new_h = max(1, int(h * scale))
        bgr = cv2.resize(bgr, (target_width, new_h))

        # Label bar
        bar = np.zeros((label_h, target_width, 3), dtype=np.uint8)
        bar[:] = (40, 40, 40)
        cv2.putText(bar, label, (4, 14), font, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        rows.extend([bar, bgr])

    return np.vstack(rows) if rows else np.zeros((100, target_width, 3), dtype=np.uint8)


# ── File I/O ──────────────────────────────────────────────────────────────────

def save_debug(debug_dir: str | Path, stages: dict[str, np.ndarray]) -> None:
    """Saves every stage image to debug_dir/<label>.png and a combined mosaic."""
    debug_dir = Path(debug_dir)
    debug_dir.mkdir(parents=True, exist_ok=True)

    for label, img in stages.items():
        fname = label.replace(" ", "_").replace("/", "-") + ".png"
        cv2.imwrite(str(debug_dir / fname), img)

    mosaic = make_mosaic(stages)
    cv2.imwrite(str(debug_dir / "_mosaic.png"), mosaic)
    print(f"Debug images saved to: {debug_dir.resolve()}")


def save_chars(output_dir: str | Path, char_images: list[np.ndarray]) -> None:
    """Saves individual character images as char_00.png, char_01.png, …"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(char_images):
        cv2.imwrite(str(output_dir / f"char_{i:02d}.png"), img)
    print(f"Saved {len(char_images)} character images to: {output_dir.resolve()}")
