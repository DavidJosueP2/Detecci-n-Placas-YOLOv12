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

    # ── Calibrate from the largest component ─────────────────────────────────
    # Sort by area; the largest-area component is used as height reference.
    # Main plate chars are both taller AND wider than ECUADOR chars → they always
    # have larger area → candidates[0] is always a main char.
    #
    # WHY not median of top-15 (previous approach):
    #   If band isolation fails and ECUADOR chars are in the binary, they can
    #   occupy 8+ of the top-15 slots. Median([ECUADOR×8, main×7]) ≈ ECUADOR height.
    #   Pass 2 then filters with [0.25×ECUADOR_h, 2.0×ECUADOR_h] = [4px, 30px],
    #   which excludes main chars at 55px (above upper bound 30px) — wrong.
    #
    # Using the single largest component's height as reference:
    #   ref_h = height of the largest-area candidate (always a main char)
    #   Lower bound = 0.25 × ref_h → ECUADOR chars (18% of ref_h) are filtered
    #   Upper bound = 2.5 × ref_h → tolerates 2.5× size variation from perspective
    candidates.sort(key=lambda c: c["_area"], reverse=True)
    ref_h = float(candidates[0]["h"])

    # ── Pass 2: adaptive height filter ───────────────────────────────────────
    bboxes = []
    for c in candidates:
        h_ratio = c["h"] / ref_h
        aspect = c["w"] / max(c["h"], 1)

        if not (0.25 <= h_ratio <= 2.5):
            continue
        if not (0.08 <= aspect <= 3.0):
            continue

        bboxes.append({"x": c["x"], "y": c["y"], "w": c["w"], "h": c["h"]})

    return bboxes
