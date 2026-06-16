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


def _character_polarity_score(binary: np.ndarray) -> float:
    """
    Scores how much the white foreground looks like plate characters.

    The selected mask must end as white characters over black background. A
    plain brightness check is fragile because it confuses dark plates, shadows,
    or glare with character polarity.
    """
    h, w = binary.shape[:2]
    if h <= 0 or w <= 0:
        return 0.0

    num_labels, _, stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    score = 0.0
    for label in range(1, num_labels):
        x, y, bw, bh, area = stats[label]
        if area <= 0:
            continue

        area_ratio = area / float(h * w)
        height_ratio = bh / float(h)
        width_ratio = bw / float(w)
        fill_ratio = area / float(max(1, bw * bh))
        center_y = centroids[label][1] / float(h)

        if area_ratio < 0.0008 or area_ratio > 0.12:
            continue
        if height_ratio < 0.22 or height_ratio > 0.90:
            continue
        if width_ratio < 0.01 or width_ratio > 0.28:
            continue
        if fill_ratio < 0.12 or fill_ratio > 0.88:
            continue

        band_bonus = 1.4 if 0.30 <= center_y <= 0.82 else 0.75
        score += area * height_ratio * band_bonus

    return score


def _ensure_character_foreground(binary: np.ndarray) -> np.ndarray:
    """Return a mask where characters are white and background is black."""
    inverted = cv2.bitwise_not(binary)
    if _character_polarity_score(inverted) > _character_polarity_score(binary):
        return inverted
    return binary


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
    return _ensure_character_foreground(binary)


def otsu(gray: np.ndarray) -> np.ndarray:
    """Otsu's global threshold — useful when illumination is uniform."""
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return _ensure_character_foreground(binary)


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
