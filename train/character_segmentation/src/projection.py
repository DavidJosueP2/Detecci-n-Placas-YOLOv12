"""

Vertical projection segmentation.

Computes a histogram of foreground pixels per column and finds character

regions between the valleys.

"""

import cv2

import numpy as np


def compute(binary: np.ndarray) -> np.ndarray:
    """Returns number of foreground pixels per column (1-D array, length = image width)."""

    return np.sum(binary > 0, axis=0).astype(np.float32)


def smooth(proj: np.ndarray, kernel: int = 3) -> np.ndarray:
    """

    Moving-average smoothing to bridge thin intra-character gaps (e.g. 'I').

    Kernel reduced from 5 to 3: a kernel of 5 spreads ±2 columns into gaps,

    which causes threshold to classify gap columns as 'in character'.

    Kernel 3 only fills single-column holes, leaving real inter-char gaps intact.

    """

    k = np.ones(kernel) / kernel

    return np.convolve(proj, k, mode="same")


def find_char_regions(
    proj: np.ndarray,
    threshold_ratio: float = 0.10,
    min_width: int = 4,
) -> list[tuple[int, int]]:
    """

    Returns (x_start, x_end) pairs for each character region.

    A column is part of a character if proj[col] >= threshold_ratio * max(proj).



    threshold_ratio raised from 0.04 to 0.10:

    After band isolation the binary is clean, so a column must contribute at

    least 10% of the tallest column's pixel count to be "active". Combined with

    the smaller smooth kernel this keeps inter-character gaps open even when

    there are 2-3 residual noise pixels per gap column.

    """

    if proj.max() == 0:
        return []

    threshold = proj.max() * threshold_ratio

    in_char = proj >= threshold

    regions = []

    start = None

    for i, active in enumerate(in_char):
        if active and start is None:
            start = i

        elif not active and start is not None:
            if i - start >= min_width:
                regions.append((start, i - 1))

            start = None

    if start is not None and len(proj) - start >= min_width:
        regions.append((start, len(proj) - 1))

    return regions


def bboxes_from_regions(
    binary: np.ndarray,
    regions: list[tuple[int, int]],
) -> list[dict]:
    """

    Converts (x_start, x_end) regions into tight bounding boxes using

    the actual foreground pixel extent per region.

    """

    h = binary.shape[0]

    bboxes = []

    for x1, x2 in regions:
        col_slice = binary[:, x1 : x2 + 1]

        rows = np.any(col_slice > 0, axis=1)

        if not rows.any():
            continue

        y1 = int(np.argmax(rows))

        y2 = int(h - 1 - np.argmax(rows[::-1]))

        bboxes.append({"x": x1, "y": y1, "w": x2 - x1 + 1, "h": y2 - y1 + 1})

    return bboxes


def segment(binary: np.ndarray) -> list[dict]:
    """Full projection pipeline: binary → list of {x, y, w, h}."""

    proj = smooth(compute(binary))

    regions = find_char_regions(proj)

    return bboxes_from_regions(binary, regions)


def remove_small_blobs(binary: np.ndarray, height_ratio: float = 0.25) -> np.ndarray:
    """
    Removes connected components whose height is below height_ratio × reference height.

    WHY height-based instead of area-based:

    Area-based calibration (previous versions) fails in two opposite directions:
      - Fixed area ratio (0.005×image): removes ECUADOR chars on standard plates ✓,
        but also removes valid chars on extreme-perspective plates where right-side
        chars can be 3× smaller in area.
      - Adaptive area (keep_ratio × median): median is contaminated when isolate_text_band
        includes ECUADOR in the band — ECUADOR chars inflate the component count, pull
        down the median, and the threshold becomes too low to filter anything.

    Height-based calibration avoids both failure modes:
      - Reference = height of the LARGEST-AREA component (always a main char, never
        ECUADOR: a main char is both taller and wider than an ECUADOR char, so it
        always dominates by area).
      - Keep: height ≥ height_ratio × ref_height.

    Typical values:
      - Main char (frontal plate): 50-80px → kept ✓
      - Main char (extreme perspective, smallest): 0.30-0.40 × tallest → kept ✓
      - ECUADOR char: 0.15-0.22 × tallest char → filtered ✓
      - Noise dot: 0.02-0.05 × tallest char → filtered ✓

    height_ratio=0.25 sits safely between ECUADOR chars (~0.18) and the smallest
    valid perspective-warped chars (~0.30), giving 0.07 margin on each side.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        binary, connectivity=8
    )
    if num_labels < 2:
        return binary

    # Sort by area descending; the largest-area component is used as height reference.
    # It is always a main plate character because main chars are both taller AND wider
    # than ECUADOR chars, so they have significantly larger area.
    areas = sorted(
        [(i, int(stats[i, cv2.CC_STAT_AREA])) for i in range(1, num_labels)],
        key=lambda x: x[1],
        reverse=True,
    )
    ref_idx = areas[0][0]
    ref_h = int(stats[ref_idx, cv2.CC_STAT_HEIGHT])
    min_h = max(4, int(ref_h * height_ratio))

    out = np.zeros_like(binary)
    for i, _ in areas:
        if int(stats[i, cv2.CC_STAT_HEIGHT]) >= min_h:
            out[labels == i] = 255
    return out


def isolate_text_band(binary: np.ndarray, pad_ratio: float = 0.08) -> np.ndarray:
    """
    Finds the main character band and returns a copy of binary with rows
    outside that band zeroed out.

    Design:
      kernel=7, threshold=0.15: same as the reference state (9e99ef9).
        - kernel=7 bridges up to 3-row gaps in row projection, which is necessary
          to avoid splitting a single character row into multiple micro-bands due
          to intra-character sparse rows (hollows inside 'A', 'R', '8', etc.).
        - threshold=0.15 is permissive enough to include borderline character rows
          without including empty inter-band gaps.

      Band selection: DENSEST band, not tallest.
        - 9e99ef9 selected the "tallest" band. This fails when the ECUADOR band
          and the main-char band merge into one tall band (kernel=7 bridges the gap).
        - Selecting by total foreground pixels (density) is more robust:
          7 main chars × ~600px² ≈ 4200 foreground pixels
          7 ECUADOR chars × ~150px² ≈ 1050 foreground pixels
          The main-char band always has ≥3× more ink, regardless of band height.
        - When both bands merge into one: the single merged band is selected anyway,
          which is the correct fallback (remove_small_blobs handles ECUADOR removal).
        - When bands are separate: the denser (main-char) band is selected correctly.
    """
    row_proj = np.sum(binary > 0, axis=1).astype(np.float32)
    row_proj = np.convolve(row_proj, np.ones(7) / 7, mode="same")

    if row_proj.max() == 0:
        return binary

    threshold = row_proj.max() * 0.15
    active = row_proj >= threshold

    bands: list[tuple[int, int]] = []
    in_band = False
    start = 0
    for i, is_active in enumerate(active):
        if is_active and not in_band:
            start = i
            in_band = True
        elif not is_active and in_band:
            bands.append((start, i - 1))
            in_band = False
    if in_band:
        bands.append((start, len(active) - 1))

    if not bands:
        return binary

    # Select the band with most foreground pixels (main chars >> ECUADOR in ink density).
    y1, y2 = max(bands, key=lambda b: int(np.sum(binary[b[0] : b[1] + 1] > 0)))

    pad = max(2, int(binary.shape[0] * pad_ratio))
    y1 = max(0, y1 - pad)
    y2 = min(binary.shape[0] - 1, y2 + pad)

    masked = binary.copy()
    masked[:y1, :] = 0
    masked[y2 + 1 :, :] = 0
    return masked
