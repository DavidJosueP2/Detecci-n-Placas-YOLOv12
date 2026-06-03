"""
Connected-components segmentation — fallback when projection fails.

Two-pass adaptive filter: calibrates thresholds from the actual components
found in the image instead of using fixed image-height ratios. This handles
perspective-warped plates where character sizes vary significantly left-to-right.
"""

import cv2
import numpy as np


def segment(binary: np.ndarray) -> list[dict]:
    """
    Returns character bounding boxes found via connected components.

    Pass 1 — very loose filters: collect everything that could be a character.
    Pass 2 — adaptive: keep only components whose height is within
              [0.25, 2.0] × median height of the top candidates.

    This tolerates 2-3× height variation across a perspective-warped plate
    without tuning any threshold manually.
    """
    h, w = binary.shape
    img_area = h * w

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    # ── Pass 1: loose collection ──────────────────────────────────────────────
    candidates = []
    for i in range(1, num_labels):
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])

        # Reject sub-pixel noise and anything larger than half the image
        if not (img_area * 0.0005 <= area <= img_area * 0.50):
            continue

        # Reject horizontal lines (plate border top/bottom: aspect >> 4)
        # and vertical thin strokes (aspect << 0.05)
        aspect = bw / max(bh, 1)
        if not (0.05 <= aspect <= 4.0):
            continue

        candidates.append({"x": x, "y": y, "w": bw, "h": bh, "_area": area})

    if not candidates:
        return []

    # ── Calibrate from the largest (most likely = main characters) ────────────
    # Sort by area; the main plate characters dominate over ECUADOR text / noise
    candidates.sort(key=lambda c: c["_area"], reverse=True)
    # Use top-15 by area to compute the reference height
    top = candidates[:15]
    med_h = float(np.median([c["h"] for c in top]))

    # ── Pass 2: adaptive height filter ───────────────────────────────────────
    # [0.25 × med_h, 2.0 × med_h] tolerates 4× size variation (e.g. T vs 7
    # on a plate with 30-40° lateral perspective tilt).
    bboxes = []
    for c in candidates:
        h_ratio = c["h"] / med_h
        aspect = c["w"] / max(c["h"], 1)

        if not (0.25 <= h_ratio <= 2.0):
            continue
        if not (0.08 <= aspect <= 3.0):
            continue

        bboxes.append({"x": c["x"], "y": c["y"], "w": c["w"], "h": c["h"]})

    return bboxes
