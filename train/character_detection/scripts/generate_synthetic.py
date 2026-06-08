#!/usr/bin/env python3
"""
Generates synthetic character images by augmenting real samples from the
CNN_Letter_Dataset.  Each synthetic image is a degraded copy of a randomly
selected real image, keeping the standardised license-plate font intact.

Source:  data/CNN_Letter_Dataset/{class}/*.jpg  (real images, 75×100 px)
Output:  data/CNN_Letter_Dataset/{class}/synth_{char}_{n}.png  (32×32 px)

The output lands in the same directory that train_config.yaml points to
(data/CNN_Letter_Dataset), so the training pipeline picks up the new images
without any config changes.

Usage:
    python scripts/generate_synthetic.py
    python scripts/generate_synthetic.py --n 200
    python scripts/generate_synthetic.py --source data/CNN_Letter_Dataset \\
                                         --output data/CNN_Letter_Dataset
    python scripts/generate_synthetic.py --clear   # removes existing synth_* first

    # Balance class O (only 50 real images) with geometric augmentation:
    python scripts/generate_synthetic.py --class-n O:1050 --geo-classes O --clear-classes O

Degradation knobs (edit DegradationConfig below or pass CLI flags):
    --gaussian-blur-prob    probability of Gaussian blur      (default 0.40)
    --motion-blur-prob      probability of motion blur        (default 0.30)
    --resolution-drop-prob  probability of resolution drop    (default 0.50)
    --jpeg-prob             probability of JPEG compression   (default 0.35)
    --noise-sigma-max       max std of additive Gaussian noise (default 20)

Per-class overrides:
    --class-n  CLASS:N    override target count for one class (repeatable)
    --geo-classes  A,B,C  use geometric augmentation only (no blur/JPEG) for these
    --clear-classes A,B   delete existing synth_* only for these classes
"""

import argparse
import io
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageFilter

# ── Constants ────────────────────────────────────────────────────────────────

CLASSES    = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
OUTPUT_SIZE = 32   # pixels — must match CNN input (train_config.yaml img_size)


# ── Degradation configuration ─────────────────────────────────────────────────
#
# Edit this class to tune the augmentation pipeline, or pass CLI flags.
#
# Probability fields (float 0–1): chance of applying the step per image.
# Range fields (tuple):           (min, max) for uniform sampling.

@dataclass
class DegradationConfig:
    # ── Brightness / contrast ─────────────────────────────────────────────────
    # alpha multiplies pixel values (contrast); beta offsets brightness.
    # Applied to every image — keep mild to preserve legibility.
    brightness_alpha_range: tuple = (0.70, 1.35)
    brightness_beta_range:  tuple = (-30.0, 30.0)

    # ── Gaussian noise (sensor noise / ISO grain) ─────────────────────────────
    # Applied to every image; sigma sampled uniformly from [min, max].
    noise_sigma_range: tuple = (3.0, 20.0)

    # ── Gaussian blur (defocus / lens softness) ───────────────────────────────
    gaussian_blur_prob:         float = 0.40
    gaussian_blur_radius_range: tuple = (0.3, 1.2)    # PIL GaussianBlur radius

    # ── Motion blur (vehicle movement / camera shake) ─────────────────────────
    # 1-D uniform kernel, horizontal (60 %) or vertical (40 %).
    motion_blur_prob:       float = 0.30
    motion_blur_size_range: tuple = (2, 4)             # kernel length in pixels

    # ── Resolution drop (low-res sensor → resize to 32×32) ───────────────────
    # Downscale to intermediate size then upscale back to OUTPUT_SIZE.
    # Reproduces detail loss when a plate region is enlarged to CNN input size.
    # Min raised from 14→18: at 14px confusable pairs (I/X, B/8) become
    # indistinguishable blobs that train the network to be uncertain.
    resolution_drop_prob:       float = 0.35
    resolution_drop_size_range: tuple = (18, 24)       # intermediate resolution

    # ── JPEG compression (video codec / streaming artifacts) ─────────────────
    # Min quality raised from 40→55: quality<50 introduces DCT block artifacts
    # that add spurious diagonal energy to vertical strokes (I→X confusion).
    jpeg_prob:          float = 0.35
    jpeg_quality_range: tuple = (55, 75)               # lower = more artifacts


DEGRADATION_CFG = DegradationConfig()


# ── Geometric augmentation configuration ─────────────────────────────────────
#
# Used instead of DegradationConfig for classes that need structural diversity
# rather than optical degradation (e.g. class O with only 50 real sources).
# All optical degradations (blur, JPEG, motion) are intentionally absent here
# because they are already applied at training-time by the data augmentation
# pipeline in dataset.py.

@dataclass
class GeometricAugConfig:
    # Rotation — small angles only; O is ~symmetric so ±12° covers useful range
    rotation_prob:  float = 0.70
    rotation_range: tuple = (-12.0, 12.0)     # degrees

    # Anisotropic scale — simulate different font widths / heights
    scale_prob:     float = 0.55
    scale_x_range:  tuple = (0.80, 1.18)      # horizontal stretch/squeeze
    scale_y_range:  tuple = (0.85, 1.15)      # vertical stretch/squeeze

    # Translation — small crop shifts within the 32×32 canvas
    translate_prob:    float = 0.50
    translate_px_range: tuple = (-3, 3)       # same range for x and y (pixels)

    # Morphological stroke width — erosion (thinner) or dilation (thicker)
    # Uses PIL MinFilter (thicker dark strokes) / MaxFilter (thinner dark strokes)
    morph_prob:        float = 0.45
    morph_filter_size: int   = 3              # kernel size for Min/MaxFilter

    # Mild contrast/brightness — avoid optical artifacts, just intensity shifts
    contrast_prob:        float = 0.60
    contrast_alpha_range: tuple = (0.80, 1.20)
    contrast_beta_range:  tuple = (-20.0, 20.0)


GEO_CFG = GeometricAugConfig()


# ── Source image loading ──────────────────────────────────────────────────────

def _load_source(path: Path) -> Image.Image:
    """
    Load a source image and normalise it to OUTPUT_SIZE × OUTPUT_SIZE.

    Source images are 75×100 px (portrait).  We pad to a square first so
    the character is not squashed, then resize with LANCZOS.
    """
    img = Image.open(path).convert("L")
    w, h = img.size
    if w != h:
        side = max(w, h)
        canvas = Image.new("L", (side, side), color=255)
        canvas.paste(img, ((side - w) // 2, (side - h) // 2))
        img = canvas
    return img.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.LANCZOS)


# ── Degradation primitives ────────────────────────────────────────────────────

def _motion_blur_arr(arr: np.ndarray, kernel_size: int) -> np.ndarray:
    """1-D uniform motion blur, horizontal (60 %) or vertical (40 %)."""
    k = np.ones(kernel_size, dtype=np.float32) / kernel_size
    out = np.empty(arr.shape, dtype=np.float32)
    if random.random() < 0.6:
        for i in range(arr.shape[0]):
            out[i] = np.convolve(arr[i].astype(np.float32), k, mode="same")
    else:
        for j in range(arr.shape[1]):
            out[:, j] = np.convolve(arr[:, j].astype(np.float32), k, mode="same")
    return out


def _resolution_drop(img: Image.Image, target_size: int) -> Image.Image:
    """Downsample → upsample to simulate resizing a low-res crop to 32×32."""
    small = img.resize((target_size, target_size), Image.BILINEAR)
    return small.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.BILINEAR)


def _jpeg_compress(img: Image.Image, quality: int) -> Image.Image:
    """Round-trip through JPEG to introduce DCT block artifacts."""
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("L")


# ── Augmentation pipeline ─────────────────────────────────────────────────────

def _augment(img: Image.Image, cfg: DegradationConfig) -> Image.Image:
    """
    Apply a random combination of photometric and optical degradations.

    Order mirrors the real capture pipeline:
      1. Resolution drop    — simulate low-res sensor / small crop
      2. JPEG compression   — video codec / streaming artifact
      3. Gaussian blur      — lens defocus / depth-of-field falloff
      4. Motion blur        — vehicle/camera movement during exposure
      5. Brightness/contrast jitter
      6. Gaussian noise     — sensor noise (always applied, intensity varies)
    """
    # ── 1. Resolution drop ────────────────────────────────────────────────────
    if random.random() < cfg.resolution_drop_prob:
        target = random.randint(*cfg.resolution_drop_size_range)
        img = _resolution_drop(img, target)

    arr = np.array(img, dtype=np.float32)

    # ── 2. JPEG compression ───────────────────────────────────────────────────
    if random.random() < cfg.jpeg_prob:
        quality = random.randint(*cfg.jpeg_quality_range)
        img_tmp = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
        arr = np.array(_jpeg_compress(img_tmp, quality), dtype=np.float32)

    # ── 3. Gaussian blur (defocus) ────────────────────────────────────────────
    if random.random() < cfg.gaussian_blur_prob:
        radius = random.uniform(*cfg.gaussian_blur_radius_range)
        img_tmp = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
        arr = np.array(
            img_tmp.filter(ImageFilter.GaussianBlur(radius=radius)),
            dtype=np.float32,
        )

    # ── 4. Motion blur ────────────────────────────────────────────────────────
    if random.random() < cfg.motion_blur_prob:
        size = random.randint(*cfg.motion_blur_size_range)
        arr = _motion_blur_arr(arr, size)

    # ── 5. Brightness / contrast jitter ──────────────────────────────────────
    alpha = random.uniform(*cfg.brightness_alpha_range)
    beta  = random.uniform(*cfg.brightness_beta_range)
    arr = arr * alpha + beta

    # ── 6. Gaussian noise ─────────────────────────────────────────────────────
    sigma = random.uniform(*cfg.noise_sigma_range)
    arr += np.random.randn(*arr.shape) * sigma

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# ── Geometric augmentation ────────────────────────────────────────────────────

def _augment_geometric(img: Image.Image, gcfg: GeometricAugConfig) -> Image.Image:
    """
    Apply geometric/structural transforms without optical degradations.

    Designed for classes with very few real sources (e.g. O: 50 images)
    where we need structural diversity, not more blur/noise/JPEG artifacts.
    Those are already applied at training-time by dataset.py augmentations.

    Order:
      1. Rotation          — small angles for orientation diversity
      2. Anisotropic scale — simulate different font widths / heights
      3. Translation       — random crop shift within canvas
      4. Stroke width      — MinFilter (thicker) / MaxFilter (thinner strokes)
      5. Mild contrast     — intensity shifts without optical artifacts
    """
    # ── 1. Rotation ───────────────────────────────────────────────────────────
    if random.random() < gcfg.rotation_prob:
        angle = random.uniform(*gcfg.rotation_range)
        # fillcolor=255 keeps background white (dark chars on white background)
        img = img.rotate(angle, resample=Image.BILINEAR, fillcolor=255)

    arr = np.array(img, dtype=np.float32)

    # ── 2. Anisotropic scale ──────────────────────────────────────────────────
    if random.random() < gcfg.scale_prob:
        sx = random.uniform(*gcfg.scale_x_range)
        sy = random.uniform(*gcfg.scale_y_range)
        new_w = max(16, min(int(round(OUTPUT_SIZE * sx)), OUTPUT_SIZE))
        new_h = max(16, min(int(round(OUTPUT_SIZE * sy)), OUTPUT_SIZE))
        img_tmp = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
        img_tmp = img_tmp.resize((new_w, new_h), Image.BILINEAR)
        canvas = Image.new("L", (OUTPUT_SIZE, OUTPUT_SIZE), color=255)
        canvas.paste(img_tmp, ((OUTPUT_SIZE - new_w) // 2, (OUTPUT_SIZE - new_h) // 2))
        arr = np.array(canvas, dtype=np.float32)

    # ── 3. Translation ────────────────────────────────────────────────────────
    if random.random() < gcfg.translate_prob:
        tx = random.randint(*gcfg.translate_px_range)
        ty = random.randint(*gcfg.translate_px_range)
        shifted = np.full((OUTPUT_SIZE, OUTPUT_SIZE), 255.0, dtype=np.float32)
        sx0 = max(0, -tx);  sx1 = min(OUTPUT_SIZE, OUTPUT_SIZE - tx)
        sy0 = max(0, -ty);  sy1 = min(OUTPUT_SIZE, OUTPUT_SIZE - ty)
        dx0 = max(0,  tx);  dy0 = max(0, ty)
        shifted[dy0: dy0 + (sy1 - sy0), dx0: dx0 + (sx1 - sx0)] = arr[sy0:sy1, sx0:sx1]
        arr = shifted

    # ── 4. Stroke width (morphological) ──────────────────────────────────────
    if random.random() < gcfg.morph_prob:
        img_tmp = Image.fromarray(arr.clip(0, 255).astype(np.uint8))
        if random.random() < 0.5:
            # MinFilter spreads dark pixels → thicker strokes
            img_tmp = img_tmp.filter(ImageFilter.MinFilter(gcfg.morph_filter_size))
        else:
            # MaxFilter spreads white pixels → thinner strokes
            img_tmp = img_tmp.filter(ImageFilter.MaxFilter(gcfg.morph_filter_size))
        arr = np.array(img_tmp, dtype=np.float32)

    # ── 5. Mild contrast / brightness ─────────────────────────────────────────
    if random.random() < gcfg.contrast_prob:
        alpha = random.uniform(*gcfg.contrast_alpha_range)
        beta  = random.uniform(*gcfg.contrast_beta_range)
        arr = arr * alpha + beta

    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


# ── Main generation loop ──────────────────────────────────────────────────────

def generate(
    source_root: Path,
    output_root: Path,
    n_per_class: int,
    clear: bool,
    cfg: DegradationConfig,
    class_n_overrides: dict | None = None,
    geo_classes: set | None = None,
    clear_classes: set | None = None,
    gcfg: GeometricAugConfig | None = None,
) -> None:
    print(f"Source : {source_root.resolve()}")
    print(f"Output : {output_root.resolve()}")
    print(f"Target : {n_per_class} synthetic images per class (default)\n")

    if class_n_overrides:
        print("Per-class overrides:")
        for k, v in sorted(class_n_overrides.items()):
            print(f"  {k}: {v}")
        print()

    if geo_classes:
        print(f"Geometric-only augmentation for: {sorted(geo_classes)}")
        if gcfg:
            print(f"  rotation     prob={gcfg.rotation_prob}  "
                  f"range={gcfg.rotation_range}")
            print(f"  scale        prob={gcfg.scale_prob}  "
                  f"x={gcfg.scale_x_range}  y={gcfg.scale_y_range}")
            print(f"  translate    prob={gcfg.translate_prob}  "
                  f"px={gcfg.translate_px_range}")
            print(f"  morph        prob={gcfg.morph_prob}  "
                  f"kernel={gcfg.morph_filter_size}")
            print(f"  contrast     prob={gcfg.contrast_prob}  "
                  f"alpha={gcfg.contrast_alpha_range}")
        print()

    print("Optical degradation config (non-geo classes):")
    print(f"  resolution_drop  prob={cfg.resolution_drop_prob}  "
          f"size={cfg.resolution_drop_size_range}")
    print(f"  jpeg_compress    prob={cfg.jpeg_prob}  "
          f"quality={cfg.jpeg_quality_range}")
    print(f"  gaussian_blur    prob={cfg.gaussian_blur_prob}  "
          f"radius={cfg.gaussian_blur_radius_range}")
    print(f"  motion_blur      prob={cfg.motion_blur_prob}  "
          f"kernel={cfg.motion_blur_size_range}")
    print(f"  brightness       alpha={cfg.brightness_alpha_range}  "
          f"beta={cfg.brightness_beta_range}")
    print(f"  noise_sigma      range={cfg.noise_sigma_range}")
    print()

    total_generated = 0
    _geo_classes = geo_classes or set()
    _class_n = class_n_overrides or {}
    _clear_classes = clear_classes or set()

    for char in CLASSES:
        src_dir = source_root / char
        out_dir = output_root / char
        out_dir.mkdir(parents=True, exist_ok=True)

        # Collect real source images (exclude previously generated synth files)
        sources = [
            p for p in src_dir.iterdir()
            if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
            and not p.name.startswith("synth_")
        ]
        if not sources:
            print(f"  {char}: no source images found in {src_dir} — skipping.")
            continue

        # Clear: global --clear flag, or per-class --clear-classes
        if clear or char in _clear_classes:
            for f in out_dir.glob("synth_*.png"):
                f.unlink()

        target = _class_n.get(char, n_per_class)
        existing = len(list(out_dir.glob("synth_*.png")))
        needed = max(0, target - existing)
        if needed == 0:
            print(f"  {char}: already has {existing} synth images (target {target}), skipping.")
            continue

        use_geo = char in _geo_classes
        aug_fn = (lambda img: _augment_geometric(img, gcfg or GEO_CFG)) if use_geo \
                 else (lambda img: _augment(img, cfg))
        aug_label = "geometric" if use_geo else "optical"

        count = 0
        while count < needed:
            src_path = random.choice(sources)
            try:
                img = _load_source(src_path)
                img = aug_fn(img)
                idx = existing + count
                img.save(out_dir / f"synth_{char}_{idx:05d}.png")
                count += 1
            except Exception:
                continue

        total_generated += count
        print(f"  {char} [{aug_label}]: generated {count}  "
              f"(real sources: {len(sources)}, total synth: {existing + count})")

    print(f"\nDone. Total new images: {total_generated}")
    print(f"Output root: {output_root.resolve()}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(
        description="Augment real CNN_Letter_Dataset images into synthetic training samples",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--source", default="data/CNN_Letter_Dataset",
                   help="Source dataset root (ImageFolder layout, real images only)")
    p.add_argument("--output", default="data/CNN_Letter_Dataset",
                   help="Output root — defaults to same as source so training "
                        "picks up synth images without config changes")
    p.add_argument("--n", type=int, default=200,
                   help="Default target number of synthetic images per class")
    p.add_argument("--clear", action="store_true",
                   help="Delete ALL existing synth_* files before generating")
    p.add_argument("--class-n", action="append", default=[], metavar="CLASS:N",
                   help="Per-class count override, e.g. --class-n O:1050  (repeatable)")
    p.add_argument("--geo-classes", default=None, metavar="A,B,C",
                   help="Comma-separated classes to augment with geometric transforms "
                        "only (no blur/JPEG/noise), e.g. --geo-classes O")
    p.add_argument("--clear-classes", default=None, metavar="A,B,C",
                   help="Delete synth_* only for these classes before generating, "
                        "e.g. --clear-classes O")

    deg = p.add_argument_group("optical degradation probabilities (0.0 – 1.0)")
    deg.add_argument("--gaussian-blur-prob",   type=float, default=None, metavar="P",
                     help="Probability of Gaussian blur per image")
    deg.add_argument("--motion-blur-prob",     type=float, default=None, metavar="P",
                     help="Probability of motion blur per image")
    deg.add_argument("--resolution-drop-prob", type=float, default=None, metavar="P",
                     help="Probability of resolution downscale/upscale per image")
    deg.add_argument("--jpeg-prob",            type=float, default=None, metavar="P",
                     help="Probability of JPEG compression per image")
    deg.add_argument("--noise-sigma-max",      type=float, default=None, metavar="S",
                     help="Max std of additive Gaussian noise")

    return p.parse_args()


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent.parent)

    args = _parse_args()

    cfg = DegradationConfig()
    if args.gaussian_blur_prob   is not None:
        cfg.gaussian_blur_prob   = args.gaussian_blur_prob
    if args.motion_blur_prob     is not None:
        cfg.motion_blur_prob     = args.motion_blur_prob
    if args.resolution_drop_prob is not None:
        cfg.resolution_drop_prob = args.resolution_drop_prob
    if args.jpeg_prob            is not None:
        cfg.jpeg_prob            = args.jpeg_prob
    if args.noise_sigma_max      is not None:
        cfg.noise_sigma_range    = (cfg.noise_sigma_range[0], args.noise_sigma_max)

    # Parse --class-n O:1050 into {"O": 1050}
    class_n_overrides: dict[str, int] = {}
    for token in args.class_n:
        parts = token.split(":")
        if len(parts) != 2 or not parts[1].isdigit():
            raise ValueError(f"--class-n must be CLASS:N (e.g. O:1050), got: {token!r}")
        class_n_overrides[parts[0].upper()] = int(parts[1])

    geo_classes = (
        {c.strip().upper() for c in args.geo_classes.split(",") if c.strip()}
        if args.geo_classes else None
    )
    clear_classes = (
        {c.strip().upper() for c in args.clear_classes.split(",") if c.strip()}
        if args.clear_classes else None
    )

    generate(
        Path(args.source),
        Path(args.output),
        args.n,
        args.clear,
        cfg,
        class_n_overrides=class_n_overrides or None,
        geo_classes=geo_classes,
        clear_classes=clear_classes,
        gcfg=GEO_CFG,
    )
