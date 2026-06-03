"""
Preprocessing pipeline: resize → grayscale → CLAHE → denoise.
All downstream modules expect the dict returned by `run()`.
"""

import cv2
import numpy as np

CANONICAL_WIDTH = 440   # px — all processing runs at this width


def load(path: str) -> np.ndarray:
    img = cv2.imread(str(path))
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def resize_canonical(img: np.ndarray, width: int = CANONICAL_WIDTH) -> np.ndarray:
    h, w = img.shape[:2]
    if w == width:
        return img.copy()
    scale = width / w
    return cv2.resize(img, (width, max(1, int(h * scale))), interpolation=cv2.INTER_LANCZOS4)


def to_grayscale(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def apply_clahe(gray: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return clahe.apply(gray)


def denoise(gray: np.ndarray) -> np.ndarray:
    return cv2.GaussianBlur(gray, (3, 3), 0.8)


def run(img: np.ndarray) -> dict:
    """
    Returns dict with every intermediate image keyed by stage name.
    The 'denoised' entry is what downstream modules consume for binarization.
    The 'gray' entry is used for final character crops (natural appearance).
    """
    resized = resize_canonical(img)
    gray = to_grayscale(resized)
    clahe_img = apply_clahe(gray)
    denoised = denoise(clahe_img)
    return {
        "original": img,
        "resized": resized,
        "gray": gray,
        "clahe": clahe_img,
        "denoised": denoised,
    }
