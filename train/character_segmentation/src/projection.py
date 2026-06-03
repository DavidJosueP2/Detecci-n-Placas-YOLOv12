"""
Vertical projection segmentation.
Computes a histogram of foreground pixels per column and finds character
regions between the valleys.
"""

import cv2
import numpy as np


def compute(binary: np.ndarray) -> np.ndarray:
    """Returns number of foreground pixels per column (1-D array, length = image width)."""
    return np.sum(binary > 0, axis=0).astype(np.float32)


def smooth(proj: np.ndarray, kernel: int = 5) -> np.ndarray:
    """Moving-average smoothing to avoid splitting chars at thin internal gaps."""
    k = np.ones(kernel) / kernel
    return np.convolve(proj, k, mode="same")


def find_char_regions(
    proj: np.ndarray,
    threshold_ratio: float = 0.07,
    min_width: int = 4,
) -> list[tuple[int, int]]:
    """
    Returns (x_start, x_end) pairs for each character region.
    A column is part of a character if proj[col] >= threshold_ratio * max(proj).
    """
    if proj.max() == 0:
        return []

    threshold = proj.max() * threshold_ratio
    in_char = proj >= threshold
    regions = []
    start = None

    for i, active in enumerate(in_char):
        if active and start is None:
            start = i
        elif not active and start is not None:
            if i - start >= min_width:
                regions.append((start, i - 1))
            start = None

    if start is not None and len(proj) - start >= min_width:
        regions.append((start, len(proj) - 1))

    return regions


def bboxes_from_regions(
    binary: np.ndarray,
    regions: list[tuple[int, int]],
) -> list[dict]:
    """
    Converts (x_start, x_end) regions into tight bounding boxes using
    the actual foreground pixel extent per region.
    """
    h = binary.shape[0]
    bboxes = []
    for x1, x2 in regions:
        col_slice = binary[:, x1 : x2 + 1]
        rows = np.any(col_slice > 0, axis=1)
        if not rows.any():
            continue
        y1 = int(np.argmax(rows))
        y2 = int(h - 1 - np.argmax(rows[::-1]))
        bboxes.append({"x": x1, "y": y1, "w": x2 - x1 + 1, "h": y2 - y1 + 1})
    return bboxes


def segment(binary: np.ndarray) -> list[dict]:
    """Full projection pipeline: binary → list of {x, y, w, h}."""
    proj = smooth(compute(binary))
    regions = find_char_regions(proj)
    return bboxes_from_regions(binary, regions)
