"""
Connected-components segmentation — fallback when projection fails.
Filters components by area, aspect ratio, and height relative to the image.
"""

import cv2
import numpy as np


def segment(binary: np.ndarray) -> list[dict]:
    """
    Returns character bounding boxes found via connected components.
    Filters out noise (too small / too large) and non-character shapes
    (wrong aspect ratio or height).
    """
    h, w = binary.shape
    img_area = h * w

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    bboxes = []
    for i in range(1, num_labels):  # 0 is background
        x = int(stats[i, cv2.CC_STAT_LEFT])
        y = int(stats[i, cv2.CC_STAT_TOP])
        bw = int(stats[i, cv2.CC_STAT_WIDTH])
        bh = int(stats[i, cv2.CC_STAT_HEIGHT])
        area = int(stats[i, cv2.CC_STAT_AREA])

        # Area filter: character must be between 0.3% and 20% of total image area
        if not (img_area * 0.003 <= area <= img_area * 0.20):
            continue

        # Height filter: character should span 25%–95% of image height
        h_ratio = bh / h
        if not (0.25 <= h_ratio <= 0.95):
            continue

        # Aspect ratio filter: exclude anything extremely wide or narrow
        aspect = bw / max(bh, 1)
        if not (0.10 <= aspect <= 2.5):
            continue

        bboxes.append({"x": x, "y": y, "w": bw, "h": bh})

    return bboxes
