"""
Structural validation for Ecuadorian plate format (ABC-1234 → 7 chars).

Handles:
  - Removing the dash separator
  - Merging fragmented characters (too many boxes)
  - Splitting fused characters (too few boxes)
  - Confidence scoring
"""

import numpy as np

EXPECTED = 7  # ABC-1234 without the dash separator


# ── Utilities ─────────────────────────────────────────────────────────────────

def sort_by_x(bboxes: list[dict]) -> list[dict]:
    return sorted(bboxes, key=lambda b: b["x"])


def _median_h(bboxes: list[dict]) -> float:
    return float(np.median([b["h"] for b in bboxes])) if bboxes else 1.0


def _median_w(bboxes: list[dict]) -> float:
    return float(np.median([b["w"] for b in bboxes])) if bboxes else 1.0


# ── Dash removal ──────────────────────────────────────────────────────────────

def remove_dashes(bboxes: list[dict]) -> list[dict]:
    """
    The plate dash '-' has a very high width/height ratio (>> 2) and
    height much smaller than the average character height.
    Remove any such box.
    """
    if len(bboxes) <= EXPECTED:
        return bboxes  # don't remove if we're already at/below target

    med_h = _median_h(bboxes)
    result = []
    for b in bboxes:
        aspect = b["w"] / max(b["h"], 1)
        is_dash = aspect > 2.5 and b["h"] < med_h * 0.45
        if not is_dash:
            result.append(b)
    return result


# ── Merge fragmented characters ───────────────────────────────────────────────

def merge_nearby(bboxes: list[dict], gap_thresh: float | None = None) -> list[dict]:
    """
    Merges consecutive boxes whose horizontal gap is smaller than gap_thresh.
    Default gap_thresh = 0.35 × median character width.
    """
    if len(bboxes) < 2:
        return bboxes

    bboxes = sort_by_x(bboxes)
    if gap_thresh is None:
        gap_thresh = _median_w(bboxes) * 0.35

    merged = [bboxes[0].copy()]
    for b in bboxes[1:]:
        prev = merged[-1]
        prev_x2 = prev["x"] + prev["w"]
        gap = b["x"] - prev_x2
        if gap <= gap_thresh:
            # Merge: extend the previous box to include b
            new_x2 = max(prev_x2, b["x"] + b["w"])
            new_y1 = min(prev["y"], b["y"])
            new_y2 = max(prev["y"] + prev["h"], b["y"] + b["h"])
            merged[-1] = {
                "x": prev["x"],
                "y": new_y1,
                "w": new_x2 - prev["x"],
                "h": new_y2 - new_y1,
            }
        else:
            merged.append(b.copy())
    return merged


# ── Split fused characters ────────────────────────────────────────────────────

def split_wide(
    binary: np.ndarray,
    bboxes: list[dict],
    expected: int = EXPECTED,
) -> list[dict]:
    """
    If we have fewer boxes than expected, try to split the widest box(es)
    at the narrowest vertical projection point within each box.
    """
    bboxes = sort_by_x(bboxes)
    deficit = expected - len(bboxes)
    if deficit <= 0:
        return bboxes

    med_w = _median_w(bboxes)

    # Sort by width descending and split the widest ones first
    sorted_by_w = sorted(range(len(bboxes)), key=lambda i: bboxes[i]["w"], reverse=True)

    result = list(bboxes)
    splits_done = 0

    for idx in sorted_by_w:
        if splits_done >= deficit:
            break
        b = result[idx]
        # Only split if box is significantly wider than median
        if b["w"] < med_w * 1.6:
            continue

        n_splits = min(2, deficit - splits_done + 1)  # split into at most 3 pieces
        pieces = _split_box_by_projection(binary, b, n_splits)
        if len(pieces) > 1:
            result[idx : idx + 1] = pieces
            splits_done += len(pieces) - 1
            # Recompute median after split
            med_w = _median_w(result)

    return sort_by_x(result)


def _split_box_by_projection(
    binary: np.ndarray,
    bbox: dict,
    n_splits: int,
) -> list[dict]:
    """
    Splits a bbox into n_splits+1 pieces by finding n_splits deepest valleys
    in the vertical projection of the region.
    Falls back to equal-width split if no clear valleys are found.
    """
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    region = binary[y : y + h, x : x + w]
    proj = np.sum(region > 0, axis=0).astype(np.float32)

    if proj.max() == 0:
        return _equal_split(bbox, n_splits + 1)

    # Smooth projection and find the n_splits local minima
    from scipy.signal import argrelmin
    smooth = np.convolve(proj, np.ones(3) / 3, mode="same")

    # Find local minima (valleys)
    try:
        minima = argrelmin(smooth, order=3)[0]
    except Exception:
        minima = np.array([], dtype=int)

    if len(minima) == 0:
        return _equal_split(bbox, n_splits + 1)

    # Pick the n_splits deepest valleys
    valley_vals = smooth[minima]
    best_valleys = minima[np.argsort(valley_vals)[:n_splits]]
    best_valleys = np.sort(best_valleys)

    # Build split points in the original image coordinate space
    split_xs = [x + int(v) for v in best_valleys]
    return _build_pieces(binary, bbox, split_xs)


def _equal_split(bbox: dict, n: int) -> list[dict]:
    """Fallback: divide bbox into n equal-width pieces."""
    piece_w = max(1, bbox["w"] // n)
    pieces = []
    for i in range(n):
        px = bbox["x"] + i * piece_w
        pw = piece_w if i < n - 1 else bbox["x"] + bbox["w"] - px
        pieces.append({"x": px, "y": bbox["y"], "w": pw, "h": bbox["h"]})
    return pieces


def _build_pieces(
    binary: np.ndarray,
    bbox: dict,
    split_xs: list[int],
) -> list[dict]:
    """Build bboxes from split points (absolute x coordinates)."""
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    boundaries = [x] + split_xs + [x + w]
    pieces = []
    for i in range(len(boundaries) - 1):
        px1, px2 = boundaries[i], boundaries[i + 1]
        pw = px2 - px1
        if pw < 2:
            continue
        # Recompute tight vertical bounds within this strip
        strip = binary[y : y + h, px1:px2]
        rows = np.any(strip > 0, axis=1)
        if rows.any():
            py1 = y + int(np.argmax(rows))
            py2 = y + h - 1 - int(np.argmax(rows[::-1]))
            pieces.append({"x": px1, "y": py1, "w": pw, "h": py2 - py1 + 1})
        else:
            pieces.append({"x": px1, "y": y, "w": pw, "h": h})
    return pieces


# ── Geometric noise filter ────────────────────────────────────────────────────

def filter_noise_bboxes(bboxes: list[dict]) -> list[dict]:
    """
    Removes geometrically implausible bboxes before any merge is attempted.

    Targets ECUADOR residue, border fragments, and screw components that
    inflate the count beyond expected — causing merge_nearby to kick in
    with increasingly wide thresholds that then fuse valid characters.

    Two criteria (both must hold to KEEP a bbox):
      1. Height >= 30% of median height.
         Handles small residue from ECUADOR text or border slivers.
         30% is safe even under 4× perspective scaling: if the largest
         char is 80px and the smallest is 30px, median ≈ 55px → min = 16px.
         ECUADOR chars are typically < 15px tall after band isolation.

      2. Vertical center within 1.5 × median_height from the band's
         median center Y.
         Handles components that survived band isolation but are vertically
         displaced (top-border fragment, bottom noise).
    """
    if len(bboxes) <= 2:
        return bboxes

    med_h = _median_h(bboxes)
    centers_y = [b["y"] + b["h"] / 2.0 for b in bboxes]
    median_cy = float(np.median(centers_y))

    result = []
    for b in bboxes:
        if b["h"] < med_h * 0.30:
            continue
        cy = b["y"] + b["h"] / 2.0
        if abs(cy - median_cy) > med_h * 1.5:
            continue
        result.append(b)

    # Safety: never return empty — if filter removes everything, keep original
    return result if result else bboxes


# ── Main entry point ──────────────────────────────────────────────────────────

def confidence_score(n: int, expected: int = EXPECTED) -> float:
    if n == expected:
        return 1.0
    if abs(n - expected) == 1:
        return 0.85
    if abs(n - expected) == 2:
        return 0.65
    return max(0.0, 1.0 - abs(n - expected) * 0.15)


def validate_and_fix(
    bboxes: list[dict],
    binary: np.ndarray,
    expected: int = EXPECTED,
) -> tuple[list[dict], float, str]:
    """
    Returns (fixed_bboxes, confidence, note).
    """
    if not bboxes:
        return [], 0.0, "no_detections"

    bboxes = sort_by_x(bboxes)
    bboxes = remove_dashes(bboxes)
    n = len(bboxes)

    if n == expected:
        return bboxes, 1.0, "exact"

    if n > expected:
        # Step 1: remove noise geometrically BEFORE any merge.
        # If CC already found the right chars + a few noise blobs, this
        # resolves the count without touching valid characters at all.
        filtered = filter_noise_bboxes(bboxes)
        filtered = sort_by_x(filtered)
        if len(filtered) == expected:
            return filtered, 1.0, "noise_filtered"
        if expected <= len(filtered) < n:
            # Noise filter reduced count but didn't reach target — work with
            # the cleaner set for any subsequent merge
            bboxes = filtered
            n = len(bboxes)

        if n == expected:
            return bboxes, 1.0, "exact_after_filter"

        # Step 2: merge only with TIGHT thresholds (max 0.35 × median_w).
        # The old loop went up to 1.20 × median_w which is wider than the
        # inter-character gap and fused valid adjacent characters.
        for factor in (0.20, 0.35):
            gap = _median_w(bboxes) * factor
            merged = merge_nearby(bboxes, gap)
            if len(merged) == expected:
                return merged, 1.0, f"merged_{factor}"
            if len(merged) < expected:
                # Merged too far — use previous (less merged) result
                break

        # Step 3: if still over expected, keep the N boxes closest to the
        # band center (drop outliers, not adjacent characters).
        if n > expected:
            med_cy = float(np.median([b["y"] + b["h"] / 2.0 for b in bboxes]))
            bboxes_sorted_by_dist = sorted(
                bboxes, key=lambda b: abs(b["y"] + b["h"] / 2.0 - med_cy)
            )
            kept = sort_by_x(bboxes_sorted_by_dist[:expected])
            return kept, confidence_score(len(kept), expected), "trimmed_outliers"

        return bboxes, confidence_score(n, expected), "best_effort"

    # n < expected: try splitting wide boxes
    fixed = split_wide(binary, bboxes, expected)
    return fixed, confidence_score(len(fixed), expected), f"split_{len(fixed)}"
