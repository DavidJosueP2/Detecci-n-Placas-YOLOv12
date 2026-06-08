import cv2
import numpy as np


def plate_crop_quality(crop):
    if crop is None or crop.size == 0:
        return 0.0

    height, width = crop.shape[:2]
    if height < 8 or width < 24:
        return 0.0

    scale = min(1.0, 220.0 / max(1, width))
    sample = crop
    if scale < 1.0:
        sample = cv2.resize(
            crop,
            (max(1, int(width * scale)), max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    gray = cv2.cvtColor(sample, cv2.COLOR_BGR2GRAY) if len(sample.shape) == 3 else sample
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    brightness = float(gray.mean())

    sharpness_score = min(45.0, sharpness / 4.0)
    contrast_score = min(25.0, contrast * 0.7)
    size_score = min(20.0, width / 7.0, height / 2.2)
    aspect = width / max(1.0, height)
    aspect_score = 10.0 if 1.8 <= aspect <= 6.4 else 4.0
    brightness_penalty = 0.0 if 35.0 <= brightness <= 225.0 else 8.0

    return max(
        0.0,
        sharpness_score + contrast_score + size_score + aspect_score - brightness_penalty,
    )


def plate_crop_cut_risk(crop):
    if crop is None or crop.size == 0:
        return 1.0

    height, width = crop.shape[:2]
    if height < 12 or width < 36:
        return 1.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    if width > 240:
        scale = 240.0 / width
        gray = cv2.resize(
            gray,
            (240, max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )
        height, width = gray.shape[:2]

    edge_w = max(3, int(width * 0.08))
    edge_h = max(3, int(height * 0.10))
    left = gray[:, :edge_w]
    right = gray[:, width - edge_w :]
    top = gray[:edge_h, :]
    bottom = gray[height - edge_h :, :]
    center = gray[edge_h : max(edge_h + 1, height - edge_h), edge_w : max(edge_w + 1, width - edge_w)]

    center_contrast = max(1.0, float(center.std()))
    left_pressure = min(1.0, float(left.std()) / center_contrast)
    right_pressure = min(1.0, float(right.std()) / center_contrast)
    top_pressure = min(1.0, float(top.std()) / center_contrast)
    bottom_pressure = min(1.0, float(bottom.std()) / center_contrast)

    horizontal_asymmetry = abs(left_pressure - right_pressure)
    vertical_asymmetry = abs(top_pressure - bottom_pressure) * 0.35
    one_sided_pressure = max(left_pressure, right_pressure)
    if one_sided_pressure < 0.85:
        return min(1.0, horizontal_asymmetry + vertical_asymmetry)

    return min(1.0, horizontal_asymmetry * 1.8 + vertical_asymmetry)


def plate_crop_ghost_risk(crop):
    if crop is None or crop.size == 0:
        return 0.0

    height, width = crop.shape[:2]
    if height < 16 or width < 54:
        return 0.0

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
    if width > 220:
        scale = 220.0 / width
        gray = cv2.resize(
            gray,
            (220, max(1, int(height * scale))),
            interpolation=cv2.INTER_AREA,
        )

    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    edges = np.abs(edges)
    mean = float(edges.mean())
    if mean <= 0.1:
        return 0.0

    edges = np.minimum(edges / mean, 5.0)
    best_corr = 0.0
    for shift in range(3, min(18, edges.shape[1] // 3)):
        left = edges[:, :-shift]
        right = edges[:, shift:]
        left_centered = left - left.mean()
        right_centered = right - right.mean()
        denom = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
        if denom <= 1e-6:
            continue
        corr = float((left_centered * right_centered).sum() / denom)
        best_corr = max(best_corr, corr)

    return max(0.0, min(1.0, best_corr))
