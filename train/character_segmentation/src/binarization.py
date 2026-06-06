"""
Binarization: denoised grayscale → binary mask where characters = 255, background = 0.
Auto-detects polarity (dark chars on light bg vs light chars on dark bg).
"""

import cv2
import numpy as np


def _chars_are_dark(gray: np.ndarray) -> bool:
    """True when characters are darker than background (standard plate case)."""
    return float(gray.mean()) > 100


def adaptive(gray: np.ndarray, block_size: int = 31, C: int = 12) -> np.ndarray:
    """
    Adaptive Gaussian thresholding.
    THRESH_BINARY_INV makes dark regions white (foreground=255 for standard plates).

    block_size=31: at 440px width each char is ~60px wide; 31px windows adapt to
    illumination at the character scale, not the stroke scale (old 15px caused
    intra-stroke fragmentation that generated dozens of false components).
    C=12: slightly higher constant reduces noise sensitivity.
    """
    binary = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        block_size, C,
    )
    if not _chars_are_dark(gray):
        binary = cv2.bitwise_not(binary)
    return binary


def otsu(gray: np.ndarray) -> np.ndarray:
    """Otsu's global threshold — useful when illumination is uniform."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if not _chars_are_dark(gray):
        binary = cv2.bitwise_not(binary)
    return binary


def cleanup(binary: np.ndarray, open_k: int = 2, close_k: int = 3) -> np.ndarray:
    """
    Morphological opening removes isolated noise dots.
    Morphological closing fills small gaps inside character strokes.
    """
    open_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (open_k, open_k))
    close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (close_k, close_k))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, open_kernel)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, close_kernel)
    return binary


def run(gray: np.ndarray) -> dict:
    """
    Tries adaptive threshold first; falls back to Otsu if the result is poor
    (fewer than 1% or more than 40% foreground pixels).
    Returns dict with 'adaptive', 'otsu', and 'best' keys.
    """
    adp = cleanup(adaptive(gray))
    ots = cleanup(otsu(gray))

    def _foreground_ratio(b):
        return (b > 0).mean()

    adp_ratio = _foreground_ratio(adp)
    ots_ratio = _foreground_ratio(ots)

    def _score(ratio):
        # Ideal: 10-30% foreground (characters fill roughly that fraction of a plate)
        if 0.05 <= ratio <= 0.40:
            return 1.0 - abs(ratio - 0.18) * 3
        return 0.0

    best = adp if _score(adp_ratio) >= _score(ots_ratio) else ots
    return {"adaptive": adp, "otsu": ots, "best": best}
