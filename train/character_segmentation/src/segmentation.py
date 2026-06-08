"""
Main segmentation orchestrator.

Pipeline:
  1. Preprocess  (resize, grayscale, CLAHE, denoise)
  2. Perspective correction (optional, skips gracefully if no quad found)
  3. Binarize    (adaptive → Otsu fallback, morphological cleanup)
  4. Connected Components — find candidate character boxes
  5. Fix                  — merge / split to reach 7 chars
  6. Crop chars           — extract 32×32 grayscale images for the CNN
  7. Debug output (optional)

Public API:
    result = segment(img_or_path, debug_dir=None)
    result["chars"]   → list of {x, y, w, h, image}
    result["images"]  → list of 32×32 grayscale numpy arrays (CNN-ready)
    result["method"]  → which segmentation method was used
    result["confidence"] → float [0, 1]
    result["n_chars"] → number of chars found
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from . import band_cleanup, binarization, connected_components, preprocessing, validation
from .visualization import (
    draw_bboxes,
    make_mosaic,
    save_chars,
    save_debug,
)

CHAR_SIZE = 32          # output size for CNN
EXPECTED_CHARS = 7      # ABC-1234


# ── Character crop ────────────────────────────────────────────────────────────

def clean_binary_char_crop(crop: np.ndarray) -> np.ndarray:
    """
    Removes small lateral foreground fragments inside an individual character
    crop. This is intentionally per-character, not plate-wide, so strict cleanup
    does not erase weak letters from the full segmentation stage.
    """
    if crop is None or crop.size == 0:
        return crop

    binary = np.where(crop > 0, 255, 0).astype(np.uint8)
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if num_labels <= 2:
        return binary

    areas = stats[1:, cv2.CC_STAT_AREA]
    max_area = int(areas.max()) if len(areas) else 0
    if max_area <= 0:
        return binary

    h, w = binary.shape[:2]
    center_x = w / 2.0
    min_area = max(3, int(max_area * 0.10))

    candidates = []
    for label in range(1, num_labels):
        area = int(stats[label, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        cx = float(centroids[label][0])
        center_penalty = min(abs(cx - center_x) / max(center_x, 1.0), 1.0)
        score = area * (1.0 - center_penalty * 0.75)
        candidates.append((score, label, area, cx))

    if not candidates:
        return binary

    candidates.sort(reverse=True)
    best_score, best_label, _best_area, _best_cx = candidates[0]

    keep = np.zeros_like(binary)
    kept_any = False
    for score, label, area, cx in candidates:
        near_center = abs(cx - center_x) <= w * 0.25
        strong_candidate = score >= best_score * 0.55

        if label == best_label or (near_center and strong_candidate):
            keep[labels == label] = 255
            kept_any = True

    if not kept_any:
        return binary

    ys, xs = np.where(keep > 0)
    if len(xs) == 0 or len(ys) == 0:
        return binary

    margin = 1
    x1 = max(0, int(xs.min()) - margin)
    x2 = min(w, int(xs.max()) + margin + 1)
    y1 = max(0, int(ys.min()) - margin)
    y2 = min(h, int(ys.max()) + margin + 1)
    return keep[y1:y2, x1:x2]


def crop_char(
    gray: np.ndarray,
    bbox: dict,
    size: int = CHAR_SIZE,
    pad_ratio: float = 0.10,
    pad_value: int = 255,
) -> np.ndarray:
    """
    Crops a character from the grayscale image, adds padding, and resizes to
    size×size. Returns a uint8 grayscale array suitable for the CNN.
    """
    h, w = gray.shape[:2]
    x, y, bw, bh = bbox["x"], bbox["y"], bbox["w"], bbox["h"]

    # Padding
    px = max(1, int(bw * pad_ratio))
    py = max(1, int(bh * pad_ratio))
    x1 = max(0, x - px)
    y1 = max(0, y - py)
    x2 = min(w, x + bw + px)
    y2 = min(h, y + bh + py)

    crop = gray[y1:y2, x1:x2]
    if crop.size == 0:
        return np.full((size, size), 128, dtype=np.uint8)

    if pad_value == 0:
        crop = clean_binary_char_crop(crop)

    # Pad to square before resizing to avoid distortion
    ch, cw = crop.shape
    if ch != cw:
        side = max(ch, cw)
        canvas = np.full((side, side), pad_value, dtype=np.uint8)
        dy = (side - ch) // 2
        dx = (side - cw) // 2
        canvas[dy : dy + ch, dx : dx + cw] = crop
        crop = canvas

    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


# ── Core segmentation logic ───────────────────────────────────────────────────

def _pick_best_binary(binarized: dict) -> np.ndarray:
    return binarized["best"]


# ── Public API ────────────────────────────────────────────────────────────────

def segment(
    img_input: str | Path | np.ndarray,
    debug_dir: str | Path | None = None,
    expected_chars: int = EXPECTED_CHARS,
    use_perspective: bool = True,
    crop_source: str = "gray",
) -> dict:
    """
    Segments characters from a cropped plate image.

    Parameters
    ----------
    img_input      : file path or BGR numpy array
    debug_dir      : if set, saves debug images to this directory
    expected_chars : expected number of alphanumeric characters (default 7)
    use_perspective: attempt perspective correction (default True)

    Returns
    -------
    {
        "chars"     : list of {x, y, w, h, image (32×32 uint8 gray)},
        "images"    : list of 32×32 arrays (shortcut for CNN input),
        "method"    : str,
        "confidence": float,
        "n_chars"   : int,
        "note"      : str,
    }
    """

    # ── 1. Load ───────────────────────────────────────────────────────────────
    if isinstance(img_input, (str, Path)):
        raw = preprocessing.load(str(img_input))
    else:
        raw = img_input.copy()

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    stages: dict[str, np.ndarray] = {}
    # ESTA PARTE PARECE QUE NO ESTA GENERANDO BIEN EL CAMBIO DE PERSPECTIVA
    # DEBERIA PODER CONVERITR LA IMAGEN, DE ALGO QUE ESTA "VIRADO", A QUE SE VEA DE FRENTE
    if use_perspective:
        from . import perspective as _persp
        corrected, was_corrected = _persp.auto_correct(raw)
        stages["0_perspective_correction"] = corrected
    else:
        corrected, was_corrected = raw, False

    prep = preprocessing.run(corrected)
    stages["1_original"] = prep["resized"]
    stages["2_grayscale"] = prep["gray"]
    stages["3_clahe"] = prep["clahe"]
    stages["4_denoised"] = prep["denoised"]

    gray = prep["gray"]           # used for final crops
    denoised = prep["denoised"]   # used for binarization

    # ── 3. Binarize ───────────────────────────────────────────────────────────
    binarized = binarization.run(denoised)
    binary = _pick_best_binary(binarized)
    selected_binarization = binarized.get("method", "otsu")
    stages["5_binary_selected"] = binary

    # ── 3b. Isolate main character band ──────────────────────────────────────
    # Zeros out rows that belong to ECUADOR text, plate border, screws, and
    # any noise above/below the principal character row. Connected components
    # receives a clean single-band binary instead of the full noisy plate.
    binary = band_cleanup.isolate_text_band(binary)
    stages["7b_band_isolated"] = binary
    binary = band_cleanup.remove_small_blobs(binary)
    stages["7c_cleaned"] = binary

    # ── 4. Connected Components segmentation ─────────────────────────────────
    bboxes = connected_components.segment(binary)
    method_used = "connected_components"
    stages["8_cc_bboxes"] = draw_bboxes(
        prep["resized"], bboxes, color=(0, 140, 255)
    )

    # ── 5. Validate & fix ─────────────────────────────────────────────────────
    fixed, confidence, note = validation.validate_and_fix(bboxes, binary, expected_chars)

    # Clip all bbox coordinates to valid image bounds.
    # Splits and merges can produce x<0 or x+w > img_w on extreme-perspective
    # plates; crop_char handles this internally but visualization and the
    # returned bbox list should carry valid coordinates.
    img_h_px, img_w_px = gray.shape[:2]
    clipped = []
    for b in fixed:
        x = max(0, b["x"])
        y = max(0, b["y"])
        x2 = min(img_w_px, b["x"] + b["w"])
        y2 = min(img_h_px, b["y"] + b["h"])
        if x2 - x > 1 and y2 - y > 1:
            clipped.append({"x": x, "y": y, "w": x2 - x, "h": y2 - y})
    fixed = clipped if clipped else fixed

    stages["10_final_bboxes"] = draw_bboxes(
        prep["resized"], fixed, color=(0, 255, 100), thickness=2, label_idx=True
    )

    # ── 7. Crop characters ────────────────────────────────────────────────────
    # crop_source="binary" returns white characters over black background,
    # matching the CNN_MODEL_PATH project classifier training contract.
    if crop_source == "binary":
        source_img = binary
        pad_value = 0
    elif crop_source == "clahe":
        source_img = prep["clahe"]
        pad_value = 255
    else:
        source_img = prep["gray"]
        pad_value = 255
    char_imgs = [crop_char(source_img, b, size=CHAR_SIZE, pad_value=pad_value) for b in fixed]

    # Build a preview strip of all chars
    if char_imgs:
        from .visualization import chars_strip
        stages["11_chars_strip"] = chars_strip(char_imgs, size=CHAR_SIZE)

    # ── 8. Debug output ───────────────────────────────────────────────────────
    if debug_dir is not None:
        save_debug(debug_dir, stages)

    # ── 9. Build result ───────────────────────────────────────────────────────
    chars_out = []
    for b, img in zip(fixed, char_imgs):
        chars_out.append({
            "x": b["x"], "y": b["y"],
            "w": b["w"], "h": b["h"],
            "image": img,
        })

    return {
        "chars": chars_out,
        "images": char_imgs,
        "method": method_used,
        "confidence": round(confidence, 3),
        "n_chars": len(chars_out),
        "note": note,
        "perspective_corrected": was_corrected,
        "binarization_method": selected_binarization,
        "_stages": stages,    # available for custom visualizations
    }
