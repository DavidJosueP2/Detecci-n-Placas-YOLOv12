"""
Binarization: denoised grayscale → binary mask where characters = 255, background = 0.
Auto-detects polarity (dark chars on light bg vs light chars on dark bg).
"""

import cv2
import numpy as np

VALID_METHODS = {"otsu", "adaptive"}
_current_method = "otsu"


def normalize_method(method: str | None) -> str:
    value = str(method or "otsu").strip().lower()
    if value not in VALID_METHODS:
        raise ValueError(f"Metodo de binarizacion no valido: {method}")
    return value


def get_method() -> str:
    return _current_method


def set_method(method: str) -> str:
    global _current_method
    _current_method = normalize_method(method)
    return _current_method


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


def run(gray: np.ndarray, method: str | None = None) -> dict:
    """
    Computes only the configured production thresholding method.
    Returns dict with 'method' and 'best' keys.
    """
    selected = normalize_method(method or _current_method)
    if selected == "adaptive":
        best = cleanup(adaptive(gray))
    else:
        best = cleanup(otsu(gray))

    return {"method": selected, "best": best}
