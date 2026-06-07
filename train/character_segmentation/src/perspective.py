"""
Perspective correction via dominant bright-region homography.

Approach: the plate background is the largest bright blob in a cropped plate
image. minAreaRect on that blob yields 4 corners; warpPerspective maps them
to a canonical 440x140 rectangle (full perspective correction, not just rotation).
"""

from __future__ import annotations

import cv2
import numpy as np

OUT_W = 440
OUT_H = 140
CANONICAL = (OUT_W, OUT_H)
MIN_AREA_RATIO = 0.20   # blob must cover at least 20% of the image


def _order_corners(pts: np.ndarray) -> np.ndarray:
    """Sort to: top-left, top-right, bottom-right, bottom-left."""
    pts = pts.reshape(4, 2).astype(np.float32)
    rect = np.zeros((4, 2), dtype=np.float32)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]   # TL: smallest x+y
    rect[2] = pts[np.argmax(s)]   # BR: largest x+y
    d = np.diff(pts, axis=1).ravel()
    rect[1] = pts[np.argmin(d)]   # TR: smallest y-x
    rect[3] = pts[np.argmax(d)]   # BL: largest y-x
    return rect


def _bright_blob_quad(gray: np.ndarray) -> np.ndarray | None:
    """
    Finds the 4 corners of the dominant bright region (plate background).
    Returns (4,2) float32 array or None.
    """
    h, w = gray.shape
    img_area = h * w

    # Keep bright pixels — plate background is lighter than characters
    # Use Otsu to find the threshold automatically
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # If most pixels are bright, invert (dark-on-white plate → keep white background)
    # If most pixels are dark, keep as is (white chars on dark → background is dark, skip)
    if np.mean(binary) < 127:
        # More dark than bright — plate background is light, chars detected as white
        # This means Otsu separated chars(white) from bg(dark). Invert to get bg.
        binary = cv2.bitwise_not(binary)

    # Morphological closing to fill character holes in the background blob
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=3)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < img_area * MIN_AREA_RATIO:
        return None

    rect = cv2.minAreaRect(largest)
    # Reject quads that don't look like a plate (aspect ratio 1.5–6.0).
    # When auto_correct detects vehicle bodywork instead of the plate background
    # (triggered by Otsu inversion on crops with significant dark surroundings),
    # minAreaRect returns a near-square or oddly-shaped rectangle.  Falling back
    # to a simple resize is safer than warping the wrong region.
    rw, rh = rect[1]
    ar = max(rw, rh) / max(min(rw, rh), 1.0)
    if not (1.5 <= ar <= 6.0):
        return None
    box = cv2.boxPoints(rect)   # 4 corners of the oriented bounding rectangle
    return box.astype(np.float32)


def _warp_to_canonical(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    src = _order_corners(quad)
    dst = np.array(
        [[0, 0], [OUT_W - 1, 0], [OUT_W - 1, OUT_H - 1], [0, OUT_H - 1]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(img, M, (OUT_W, OUT_H), flags=cv2.INTER_LANCZOS4)


# ── Public API ────────────────────────────────────────────────────────────────

def auto_correct(img: np.ndarray) -> tuple[np.ndarray, bool]:
    """
    Detects the dominant bright region, computes a homography to a
    440x140 rectangle, and returns the corrected image.
    Returns (corrected_img, was_corrected).
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img.copy()
    quad = _bright_blob_quad(gray)

    if quad is None:
        fallback = cv2.resize(img, CANONICAL, interpolation=cv2.INTER_LANCZOS4)
        return fallback, False

    try:
        corrected = _warp_to_canonical(img, quad)
        return corrected, True
    except cv2.error:
        fallback = cv2.resize(img, CANONICAL, interpolation=cv2.INTER_LANCZOS4)
        return fallback, False


def correct_from_quad(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Direct warpPerspective from known 4 corner points (e.g. from detector)."""
    return _warp_to_canonical(img, quad)


# Legacy aliases
def find_plate_quad(img: np.ndarray) -> np.ndarray | None:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return _bright_blob_quad(gray)


def correct_perspective(img: np.ndarray, quad: np.ndarray) -> np.ndarray:
    return correct_from_quad(img, quad)


# ── Standalone test ───────────────────────────────────────────────────────────
# cd train/character_segmentation
# python src/perspective.py <plate.jpg> [debug_dir/]

if __name__ == "__main__":
    import sys
    from pathlib import Path

    if len(sys.argv) < 2:
        print("Usage: python src/perspective.py <plate.jpg> [output_dir]")
        sys.exit(1)

    img_path = Path(sys.argv[1])
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else img_path.parent / "persp_debug"
    out_dir.mkdir(parents=True, exist_ok=True)

    src_img = cv2.imread(str(img_path))
    if src_img is None:
        print(f"Cannot read: {img_path}")
        sys.exit(1)

    gray_src = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY) if src_img.ndim == 3 else src_img
    quad = _bright_blob_quad(gray_src)
    corrected_img, was_corrected = auto_correct(src_img)

    print(f"Input  : {src_img.shape[1]}x{src_img.shape[0]}")
    print(f"Quad   : {quad.tolist() if quad is not None else 'None (no blob found)'}")
    print(f"Corrected: {was_corrected}")
    print(f"Output : {corrected_img.shape[1]}x{corrected_img.shape[0]}")

    # Debug: draw the detected quad on the original
    overlay = src_img.copy()
    if quad is not None:
        ordered = _order_corners(quad).astype(int)
        cv2.polylines(overlay, [ordered.reshape(-1, 1, 2)], True, (0, 255, 0), 2)
        for i, (px, py) in enumerate(ordered):
            cv2.circle(overlay, (px, py), 5, (0, 0, 255), -1)
            cv2.putText(overlay, ["TL","TR","BR","BL"][i], (px+4, py-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1)

    # 3-column panel: original+quad | binary blob | corrected
    _, binary_vis = cv2.threshold(gray_src, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if np.mean(binary_vis) < 127:
        binary_vis = cv2.bitwise_not(binary_vis)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    blob_vis = cv2.morphologyEx(binary_vis, cv2.MORPH_CLOSE, kernel, iterations=3)
    blob_bgr = cv2.cvtColor(blob_vis, cv2.COLOR_GRAY2BGR)

    def _fit_h(image, target_h):
        s = target_h / image.shape[0]
        return cv2.resize(image, (max(1, int(image.shape[1] * s)), target_h))

    panel = np.hstack([
        _fit_h(overlay, OUT_H),
        _fit_h(blob_bgr, OUT_H),
        _fit_h(corrected_img, OUT_H),
    ])

    out_path = out_dir / f"{img_path.stem}_persp_debug.jpg"
    cv2.imwrite(str(out_path), panel)
    print(f"\nSaved: {out_path}")
    print("Columns: [original + quad corners] [bright blob] [warpPerspective result]")
