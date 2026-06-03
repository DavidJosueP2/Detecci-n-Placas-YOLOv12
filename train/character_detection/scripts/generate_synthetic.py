#!/usr/bin/env python3
"""
Generates synthetic character images that simulate the appearance of
characters on Ecuadorian license plates.

Output structure (ImageFolder format):
    data/processed/
        A/  synth_A_00000.png  synth_A_00001.png  ...
        B/  ...
        0/  ...

Usage:
    python scripts/generate_synthetic.py
    python scripts/generate_synthetic.py --n 800 --output data/processed
    python scripts/generate_synthetic.py --clear     # wipe existing synth files first
"""

import argparse
import random
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ── Constants ────────────────────────────────────────────────────────────────

CLASSES = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
CANVAS = 40          # render on a slightly larger canvas then crop/resize to 32
OUTPUT_SIZE = 32

# Font paths that produce plate-like characters (bold monospace / sans-serif).
# The generator scans system fonts and prioritises these substrings.
PREFERRED_KEYWORDS = [
    "LiberationMono-Bold",
    "LiberationSans-Bold",
    "JetBrainsMono",
    "UbuntuSansMono",
    "Ubuntu-Bold",
    "DejaVuSansMono-Bold",
    "FreeMono",
    "FreeSansBold",
    "Hack-Bold",
]

FONT_SEARCH_ROOTS = [
    Path("/usr/share/fonts"),
    Path("/usr/local/share/fonts"),
    Path.home() / ".fonts",
    Path.home() / ".local/share/fonts",
]


# ── Font discovery ────────────────────────────────────────────────────────────

def _find_fonts() -> list[Path]:
    found: list[Path] = []
    for root in FONT_SEARCH_ROOTS:
        if root.exists():
            found.extend(root.rglob("*.ttf"))
            found.extend(root.rglob("*.TTF"))
            found.extend(root.rglob("*.otf"))

    # Sort so preferred fonts come first
    def _priority(p: Path) -> int:
        name = p.stem
        for i, kw in enumerate(PREFERRED_KEYWORDS):
            if kw.lower() in name.lower():
                return i
        return len(PREFERRED_KEYWORDS)

    return sorted(found, key=_priority)


def _usable_fonts(fonts: list[Path], test_sizes: list[int]) -> list[tuple[Path, int]]:
    """Return (font_path, size) pairs that can render all 36 chars without error."""
    usable = []
    for fp in fonts:
        for size in test_sizes:
            try:
                f = ImageFont.truetype(str(fp), size)
                # Quick sanity-check: render a digit and a letter
                _render("A", f)
                _render("5", f)
                usable.append((fp, size))
            except Exception:
                continue
        if len(usable) >= 60:  # enough variety; stop scanning
            break
    return usable


# ── Rendering helpers ─────────────────────────────────────────────────────────

def _render(char: str, font: ImageFont.FreeTypeFont, invert: bool = False) -> Image.Image:
    """Render a single character centered on a CANVAS×CANVAS image."""
    bg = 255 if not invert else 0
    fg = 0 if not invert else 255

    img = Image.new("L", (CANVAS, CANVAS), color=bg)
    draw = ImageDraw.Draw(img)

    try:
        bbox = font.getbbox(char)
        w = bbox[2] - bbox[0]
        h = bbox[3] - bbox[1]
        x = (CANVAS - w) // 2 - bbox[0]
        y = (CANVAS - h) // 2 - bbox[1]
    except AttributeError:
        # Older Pillow fallback
        w, h = font.getsize(char)  # type: ignore[attr-defined]
        x = (CANVAS - w) // 2
        y = (CANVAS - h) // 2

    draw.text((x, y), char, font=font, fill=fg)
    return img.resize((OUTPUT_SIZE, OUTPUT_SIZE), Image.LANCZOS)


def _augment(img: Image.Image) -> Image.Image:
    """Apply random photometric degradations that simulate real plate capture."""
    arr = np.array(img, dtype=np.float32)

    # Brightness / contrast jitter
    alpha = random.uniform(0.75, 1.30)   # contrast
    beta = random.uniform(-25, 25)        # brightness
    arr = arr * alpha + beta
    arr = np.clip(arr, 0, 255)

    # Gaussian noise
    sigma = random.uniform(3, 18)
    arr += np.random.randn(*arr.shape) * sigma
    arr = np.clip(arr, 0, 255).astype(np.uint8)

    img = Image.fromarray(arr)

    # Occasional mild blur (simulate motion or defocus)
    if random.random() < 0.25:
        try:
            from PIL import ImageFilter
            radius = random.uniform(0.3, 1.0)
            img = img.filter(ImageFilter.GaussianBlur(radius=radius))
        except Exception:
            pass

    return img


# ── Main generation loop ──────────────────────────────────────────────────────

def generate(output_root: Path, n_per_class: int, clear: bool) -> None:
    print("Scanning system fonts …")
    all_fonts = _find_fonts()
    if not all_fonts:
        print("  WARNING: no TTF/OTF fonts found; output will use PIL default font.")

    font_sizes = [20, 22, 24, 26, 28]
    usable = _usable_fonts(all_fonts, font_sizes)

    if usable:
        print(f"  Found {len(usable)} usable (font, size) combinations.")
        # Show the first few so the user knows which fonts are driving the data
        for fp, sz in usable[:5]:
            print(f"    {fp.name}  size={sz}")
        if len(usable) > 5:
            print(f"    … and {len(usable)-5} more")
    else:
        print("  No usable fonts found; using PIL default.")

    total_generated = 0

    for char in CLASSES:
        out_dir = output_root / char
        out_dir.mkdir(parents=True, exist_ok=True)

        if clear:
            for f in out_dir.glob("synth_*.png"):
                f.unlink()

        existing = len(list(out_dir.glob("synth_*.png")))
        needed = max(0, n_per_class - existing)
        if needed == 0:
            print(f"  {char}: already has {existing} synth images, skipping.")
            continue

        count = 0
        font_pool = usable if usable else [(None, 0)]
        font_idx = 0

        while count < needed:
            fp, sz = font_pool[font_idx % len(font_pool)]
            font_idx += 1

            try:
                pil_font = (
                    ImageFont.truetype(str(fp), sz) if fp else ImageFont.load_default()
                )
            except Exception:
                continue

            for invert in (False, True):
                if count >= needed:
                    break
                try:
                    img = _render(char, pil_font, invert=invert)
                    img = _augment(img)
                    idx = existing + count
                    img.save(out_dir / f"synth_{char}_{idx:05d}.png")
                    count += 1
                except Exception:
                    continue

        total_generated += count
        print(f"  {char}: generated {count}  (total in dir: {existing + count})")

    print(f"\nDone. Total new images: {total_generated}")
    print(f"Dataset root: {output_root.resolve()}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def _parse_args():
    p = argparse.ArgumentParser(description="Generate synthetic character images")
    p.add_argument("--output", default="data/processed",
                   help="Output directory (ImageFolder format)")
    p.add_argument("--n", type=int, default=500,
                   help="Target number of synthetic images per class (default: 500)")
    p.add_argument("--clear", action="store_true",
                   help="Delete existing synth_* files before generating")
    return p.parse_args()


if __name__ == "__main__":
    # Run from the character_detection/ directory
    import os
    os.chdir(Path(__file__).parent.parent)

    args = _parse_args()
    generate(Path(args.output), args.n, args.clear)
