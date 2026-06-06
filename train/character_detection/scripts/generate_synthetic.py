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

Degradation knobs (edit DegradationConfig below or pass CLI flags):
    --gaussian-blur-prob    probability of Gaussian blur      (default 0.40)
    --motion-blur-prob      probability of motion blur        (default 0.30)
    --resolution-drop-prob  probability of resolution drop    (default 0.50)
    --jpeg-prob             probability of JPEG compression   (default 0.35)
    --noise-sigma-max       max std of additive Gaussian noise (default 20)
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
    resolution_drop_prob:       float = 0.50
    resolution_drop_size_range: tuple = (14, 24)       # intermediate resolution

    # ── JPEG compression (video codec / streaming artifacts) ─────────────────
    jpeg_prob:          float = 0.35
    jpeg_quality_range: tuple = (40, 75)               # lower = more artifacts


DEGRADATION_CFG = DegradationConfig()


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


# ── Main generation loop ──────────────────────────────────────────────────────

def generate(
    source_root: Path,
    output_root: Path,
    n_per_class: int,
    clear: bool,
    cfg: DegradationConfig,
) -> None:
    print(f"Source : {source_root.resolve()}")
    print(f"Output : {output_root.resolve()}")
    print(f"Target : {n_per_class} synthetic images per class\n")

    print("Degradation config:")
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

        if clear:
            for f in out_dir.glob("synth_*.png"):
                f.unlink()

        existing = len(list(out_dir.glob("synth_*.png")))
        needed = max(0, n_per_class - existing)
        if needed == 0:
            print(f"  {char}: already has {existing} synth images, skipping.")
            continue

        count = 0
        while count < needed:
            src_path = random.choice(sources)
            try:
                img = _load_source(src_path)
                img = _augment(img, cfg)
                idx = existing + count
                img.save(out_dir / f"synth_{char}_{idx:05d}.png")
                count += 1
            except Exception:
                continue

        total_generated += count
        print(f"  {char}: generated {count}  (real sources: {len(sources)}, "
              f"total synth in dir: {existing + count})")

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
                   help="Target number of synthetic images per class")
    p.add_argument("--clear", action="store_true",
                   help="Delete existing synth_* files before generating")

    deg = p.add_argument_group("degradation probabilities (0.0 – 1.0)")
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

    generate(Path(args.source), Path(args.output), args.n, args.clear, cfg)
