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


def smooth(proj: np.ndarray, kernel: int = 3) -> np.ndarray:
    """
    Moving-average smoothing to bridge thin intra-character gaps (e.g. 'I').
    Kernel reduced from 5 to 3: a kernel of 5 spreads ±2 columns into gaps,
    which causes threshold to classify gap columns as 'in character'.
    Kernel 3 only fills single-column holes, leaving real inter-char gaps intact.
    """
    k = np.ones(kernel) / kernel
    return np.convolve(proj, k, mode="same")


def find_char_regions(
    proj: np.ndarray,
    threshold_ratio: float = 0.10,
    min_width: int = 4,
) -> list[tuple[int, int]]:
    """
    Returns (x_start, x_end) pairs for each character region.
    A column is part of a character if proj[col] >= threshold_ratio * max(proj).

    threshold_ratio raised from 0.04 to 0.10:
    After band isolation the binary is clean, so a column must contribute at
    least 10% of the tallest column's pixel count to be "active". Combined with
    the smaller smooth kernel this keeps inter-character gaps open even when
    there are 2-3 residual noise pixels per gap column.
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


def remove_small_blobs(binary: np.ndarray, min_area_ratio: float = 0.005) -> np.ndarray:
    """
    Removes connected components whose area is below min_area_ratio × image area.

    After band isolation, stray noise pixels remain in inter-character gap columns.
    Even 3-5 isolated pixels per column push projection above threshold_ratio=0.10
    and collapse the gap. Removing them ensures gap columns are truly zero.

    min_area_ratio=0.005 (0.5% of image):
      - 440×140 image → min_area ≈ 308px
      - ECUADOR residual char (~15×20px = 300px): removed ✓
      - Smallest perspective-warped '1'/'I' (~10×40px = 400px): kept ✓
      - Noise dot (< 50px): removed ✓
      - Main char (~50×80px = 4000px): kept ✓
    """
    h, w = binary.shape
    min_area = max(10, int(h * w * min_area_ratio))
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    out = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            out[labels == i] = 255
    return out


def isolate_text_band(binary: np.ndarray, pad_ratio: float = 0.08) -> np.ndarray:
    """
    Finds the tallest contiguous horizontal band of foreground activity and
    returns a copy of binary with rows outside that band zeroed out.

    Eliminates ECUADOR text, plate border rows, screws, and floor/background
    that survive binarization — all of which are outside the main char band.

    Strategy: horizontal projection (row sums) → smooth → threshold at 15% of
    max → find contiguous bands → keep the TALLEST one (main characters always
    form the tallest band because they are the largest glyphs on the plate).
    """
    row_proj = np.sum(binary > 0, axis=1).astype(np.float32)
    # Smooth over 7 rows to bridge intra-character gaps
    row_proj = np.convolve(row_proj, np.ones(7) / 7, mode="same")

    if row_proj.max() == 0:
        return binary

    threshold = row_proj.max() * 0.15
    active = row_proj >= threshold

    # Collect contiguous bands
    bands: list[tuple[int, int]] = []
    in_band = False
    start = 0
    for i, is_active in enumerate(active):
        if is_active and not in_band:
            start = i
            in_band = True
        elif not is_active and in_band:
            bands.append((start, i - 1))
            in_band = False
    if in_band:
        bands.append((start, len(active) - 1))

    if not bands:
        return binary

    # The main character band is always the tallest
    y1, y2 = max(bands, key=lambda b: b[1] - b[0])

    # Small vertical padding so we don't clip ascenders/descenders
    pad = max(2, int(binary.shape[0] * pad_ratio))
    y1 = max(0, y1 - pad)
    y2 = min(binary.shape[0] - 1, y2 + pad)

    masked = binary.copy()
    masked[:y1, :] = 0
    masked[y2 + 1:, :] = 0
    return masked
