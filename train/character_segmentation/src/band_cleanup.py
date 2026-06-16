"""
Band cleanup helpers used before connected-components segmentation.
"""

import cv2
import numpy as np


def remove_small_blobs(binary: np.ndarray, height_ratio: float = 0.45) -> np.ndarray:
    """
    Keeps components tall enough to belong to the main plate text row.
    Small top text such as ECUADOR, dots, screws, and thin border fragments
    are removed before character candidate extraction.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if num_labels < 2:
        return binary

    areas = sorted(
        [(i, int(stats[i, cv2.CC_STAT_AREA])) for i in range(1, num_labels)],
        key=lambda x: x[1],
        reverse=True,
    )
    ref_idx = areas[0][0]
    ref_h = int(stats[ref_idx, cv2.CC_STAT_HEIGHT])
    min_h = max(4, int(ref_h * height_ratio))

    out = np.zeros_like(binary)
    for i, _ in areas:
        if int(stats[i, cv2.CC_STAT_HEIGHT]) >= min_h:
            out[labels == i] = 255
    return out


def isolate_text_band(binary: np.ndarray, pad_ratio: float = 0.08) -> np.ndarray:
    """
    Keeps the densest horizontal text band and clears rows above/below it.
    This focuses connected-components segmentation on the main plate characters.
    """
    row_density = np.sum(binary > 0, axis=1).astype(np.float32)
    row_density = np.convolve(row_density, np.ones(7) / 7, mode="same")

    if row_density.max() == 0:
        return binary

    threshold = row_density.max() * 0.15
    active = row_density >= threshold

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

    y1, y2 = max(bands, key=lambda b: int(np.sum(binary[b[0] : b[1] + 1] > 0)))

    pad = max(2, int(binary.shape[0] * pad_ratio))
    y1 = max(0, y1 - pad)
    y2 = min(binary.shape[0] - 1, y2 + pad)

    masked = binary.copy()
    masked[:y1, :] = 0
    masked[y2 + 1 :, :] = 0
    return masked
