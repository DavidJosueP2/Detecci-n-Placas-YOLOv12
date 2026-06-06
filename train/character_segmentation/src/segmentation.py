"""
Main segmentation orchestrator.

Pipeline:
  1. Preprocess  (resize, grayscale, CLAHE, denoise)
  2. Perspective correction (optional, skips gracefully if no quad found)
  3. Binarize    (adaptive → Otsu fallback, morphological cleanup)
  4. Method A    — Vertical Projection
     └─ validate → if 5-9 chars, accept
  5. Method B    — Connected Components  (fallback)
     └─ validate → if 5-9 chars, accept
  6. Fix         — merge / split to reach 7 chars
  7. Crop chars  — extract 32×32 grayscale images for the CNN
  8. Debug output (optional)

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

from . import binarization, connected_components, preprocessing, projection, validation
from .visualization import (
    draw_bboxes,
    make_mosaic,
    projection_image,
    save_chars,
    save_debug,
)

CHAR_SIZE = 32          # output size for CNN
EXPECTED_CHARS = 7      # ABC-1234
VALID_MIN = 4           # below this → definitely failed
VALID_MAX = 10          # above this → too noisy


# ── Character crop ────────────────────────────────────────────────────────────

def crop_char(
    gray: np.ndarray,
    bbox: dict,
    size: int = CHAR_SIZE,
    pad_ratio: float = 0.10,
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

    # Pad to square before resizing to avoid distortion
    ch, cw = crop.shape
    if ch != cw:
        side = max(ch, cw)
        canvas = np.full((side, side), 255, dtype=np.uint8)
        dy = (side - ch) // 2
        dx = (side - cw) // 2
        canvas[dy : dy + ch, dx : dx + cw] = crop
        crop = canvas

    return cv2.resize(crop, (size, size), interpolation=cv2.INTER_AREA)


# ── Core segmentation logic ───────────────────────────────────────────────────

def _try_method(
    binary: np.ndarray,
    method: str,
) -> tuple[list[dict], str]:
    if method == "projection":
        return projection.segment(binary), "projection"
    return connected_components.segment(binary), "connected_components"


def _is_acceptable(bboxes: list[dict]) -> bool:
    return VALID_MIN <= len(bboxes) <= VALID_MAX


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
    stages["5_binary_adaptive"] = binarized["adaptive"]
    stages["6_binary_otsu"] = binarized["otsu"]
    stages["7_binary_best"] = binary

    # ── 3b. Isolate main character band ──────────────────────────────────────
    # Zeros out rows that belong to ECUADOR text, plate border, screws, and
    # any noise above/below the principal character row. Both projection and
    # CC receive a clean single-band binary instead of the full noisy plate.
    from . import projection as _proj_mod
    binary = _proj_mod.isolate_text_band(binary)
    stages["7b_band_isolated"] = binary
    binary = _proj_mod.remove_small_blobs(binary)
    stages["7c_cleaned"] = binary

    # ── 4. Projection segmentation ────────────────────────────────────────────
    proj_bboxes, _ = _try_method(binary, "projection")

    proj_arr = _proj_mod.smooth(_proj_mod.compute(binary))
    stages["8_projection"] = projection_image(proj_arr, height=60)
    stages["8b_projection_bboxes"] = draw_bboxes(
        prep["resized"], proj_bboxes, color=(0, 220, 0)
    )

    method_used = "projection"
    bboxes = proj_bboxes

    # ── 5. Fallback: Connected Components ─────────────────────────────────────
    if not _is_acceptable(bboxes):
        cc_bboxes, _ = _try_method(binary, "connected_components")
        stages["9_cc_bboxes"] = draw_bboxes(
            prep["resized"], cc_bboxes, color=(0, 140, 255)
        )

        # Pick the method that gives a count closer to expected
        proj_dist = abs(len(proj_bboxes) - expected_chars)
        cc_dist = abs(len(cc_bboxes) - expected_chars)

        if cc_dist < proj_dist or not _is_acceptable(bboxes):
            bboxes = cc_bboxes
            method_used = "connected_components"

    # ── 6. Validate & fix ─────────────────────────────────────────────────────
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
    # crop_source="clahe" uses CLAHE-enhanced image (more consistent with
    # what the segmentation "saw"); switch only after retraining on new crops.
    source_img = prep["clahe"] if crop_source == "clahe" else prep["gray"]
    char_imgs = [crop_char(source_img, b, size=CHAR_SIZE) for b in fixed]

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
        "_stages": stages,    # available for custom visualizations
    }
