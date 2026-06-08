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
    Scores whether foreground=255 looks like plate characters.

    A global brightness heuristic breaks when the crop includes dark car body
    around a white plate. Component shape is more reliable: the chosen polarity
    should produce several medium-height, narrow-ish connected components,
    not one large plate/background blob.
    """
    h, w = binary.shape[:2]
    img_area = max(1, h * w)
    foreground_ratio = float(np.count_nonzero(binary)) / img_area
    if foreground_ratio <= 0.005 or foreground_ratio >= 0.72:
        return -10.0

    num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    score = 0.0
    character_like = 0
    huge_components = 0

    for label in range(1, num_labels):
        x = int(stats[label, cv2.CC_STAT_LEFT])
        y = int(stats[label, cv2.CC_STAT_TOP])
        bw = int(stats[label, cv2.CC_STAT_WIDTH])
        bh = int(stats[label, cv2.CC_STAT_HEIGHT])
        area = int(stats[label, cv2.CC_STAT_AREA])
        area_ratio = area / img_area

        if area_ratio > 0.22:
            huge_components += 1
            continue
        if area_ratio < 0.0004:
            continue

        height_ratio = bh / max(1, h)
        width_ratio = bw / max(1, w)
        aspect = bw / max(1, bh)
        touches_border = x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1

        if 0.16 <= height_ratio <= 0.82 and 0.006 <= width_ratio <= 0.24 and 0.08 <= aspect <= 1.45:
            character_like += 1
            score += 1.0
            score += min(0.7, height_ratio)
            if touches_border:
                score -= 0.25

    if character_like == 0:
        score -= 4.0

    target_count_bonus = max(0.0, 1.0 - abs(character_like - 7) / 7.0)
    score += target_count_bonus * 2.0
    score -= huge_components * 2.5
    score -= max(0.0, foreground_ratio - 0.45) * 5.0
    return score


def _ensure_character_foreground(binary: np.ndarray) -> np.ndarray:
    """
    Returns a mask where characters are white (255) and background is black.
    """
    inverse = cv2.bitwise_not(binary)
    return binary if _character_polarity_score(binary) >= _character_polarity_score(inverse) else inverse


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
