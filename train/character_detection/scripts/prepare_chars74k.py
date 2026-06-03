#!/usr/bin/env python3
"""
Converts the raw Chars74K dataset into the ImageFolder format expected by train.py.

Chars74K source:
    http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/
    Download EnglishImg.tgz and extract to  data/chars74k_raw/

Expected input structure:
    data/chars74k_raw/
        English/
            Img/
                Sample001/   ← digit '0'
                Sample002/   ← digit '1'
                ...
                Sample010/   ← digit '9'
                Sample011/   ← letter 'A'
                ...
                Sample036/   ← letter 'Z'
                Sample037/   ← letter 'a'  (merged → 'A')
                ...
                Sample062/   ← letter 'z'  (merged → 'Z')

Output (added to ImageFolder root, prefixed 'c74k_' to avoid overwriting synth files):
    data/processed/
        A/  c74k_img001-00001.png  …
        0/  c74k_img001-00001.png  …
        …

Usage:
    python scripts/prepare_chars74k.py
    python scripts/prepare_chars74k.py --input data/chars74k_raw --output data/processed
"""

import argparse
import shutil
from pathlib import Path

from PIL import Image


# Chars74K class index → our character label
# Sample001..010 → digits 0..9
# Sample011..036 → uppercase A..Z
# Sample037..062 → lowercase a..z → merged into uppercase A..Z
def _sample_to_char(sample_num: int) -> str | None:
    if 1 <= sample_num <= 10:
        return str(sample_num - 1)          # '0'..'9'
    elif 11 <= sample_num <= 36:
        return chr(ord("A") + sample_num - 11)  # 'A'..'Z'
    elif 37 <= sample_num <= 62:
        return chr(ord("A") + sample_num - 37)  # lowercase → uppercase
    return None


_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".ppm"}


def prepare(input_root: Path, output_root: Path) -> None:
    img_root = input_root / "English" / "Img"
    if not img_root.exists():
        raise FileNotFoundError(
            f"Chars74K not found at: {img_root}\n"
            "Download EnglishImg.tgz from "
            "http://www.ee.surrey.ac.uk/CVSSP/demos/chars74k/ "
            "and extract to data/chars74k_raw/"
        )

    total = 0
    sample_dirs = sorted(img_root.iterdir())

    for sample_dir in sample_dirs:
        if not sample_dir.is_dir() or not sample_dir.name.startswith("Sample"):
            continue

        try:
            num = int(sample_dir.name.replace("Sample", ""))
        except ValueError:
            continue

        char = _sample_to_char(num)
        if char is None:
            continue

        out_dir = output_root / char
        out_dir.mkdir(parents=True, exist_ok=True)

        for img_path in sorted(sample_dir.iterdir()):
            if img_path.suffix.lower() not in _IMG_EXTS:
                continue

            dest = out_dir / f"c74k_{img_path.stem}.png"
            if dest.exists():
                continue  # skip already prepared files

            try:
                img = Image.open(img_path).convert("L")
                img = img.resize((32, 32), Image.LANCZOS)
                img.save(dest)
                total += 1
            except Exception as e:
                print(f"  Skipping {img_path.name}: {e}")

        print(f"  Sample{num:03d} → '{char}'")

    print(f"\nDone. {total} images written to {output_root.resolve()}")


def _parse_args():
    p = argparse.ArgumentParser(description="Prepare Chars74K for training")
    p.add_argument("--input", default="data/chars74k_raw",
                   help="Path to extracted Chars74K root (contains English/Img/)")
    p.add_argument("--output", default="data/processed",
                   help="Output ImageFolder root")
    return p.parse_args()


if __name__ == "__main__":
    import os
    os.chdir(Path(__file__).parent.parent)

    args = _parse_args()
    prepare(Path(args.input), Path(args.output))
