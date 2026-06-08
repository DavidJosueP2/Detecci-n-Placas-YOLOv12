#!/usr/bin/env python3
"""
CLI for plate character segmentation.

Usage:
    python segment_plate.py --image plate.jpg
    python segment_plate.py --image plate.jpg --debug
    python segment_plate.py --image plate.jpg --debug --debug-dir debug/my_plate
    python segment_plate.py --image plate.jpg --save-chars output/chars/
    python segment_plate.py --image plate.jpg --no-perspective
"""

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Run from this script's directory so src.* imports resolve
os.chdir(Path(__file__).parent)
sys.path.insert(0, str(Path(__file__).parent))

from src.segmentation import segment
from src.visualization import save_chars


def _print_result(result: dict, image_path: str) -> None:
    print(f"\n{'─' * 50}")
    print(f"  Image      : {image_path}")
    print(f"  Method     : {result['method']}")
    print(f"  Confidence : {result['confidence']:.2f}")
    print(f"  Characters : {result['n_chars']}   (note: {result['note']})")
    print(
        f"  Perspective: {'corrected' if result['perspective_corrected'] else 'unchanged'}"
    )
    print(f"{'─' * 50}")
    for i, ch in enumerate(result["chars"]):
        print(
            f"  [{i}]  x={ch['x']:3d}  y={ch['y']:3d}  w={ch['w']:3d}  h={ch['h']:3d}"
        )
    print()


def main() -> int:
    p = argparse.ArgumentParser(
        description="Segment characters from a cropped plate image"
    )
    p.add_argument("--image", required=True, help="Path to cropped plate image")
    p.add_argument(
        "--debug", action="store_true", help="Save all intermediate debug images"
    )
    p.add_argument(
        "--debug-dir",
        default=None,
        help="Directory for debug images (default: debug/<stem>/)",
    )
    p.add_argument(
        "--save-chars", default=None, help="Directory to save individual char images"
    )
    p.add_argument(
        "--no-perspective", action="store_true", help="Skip perspective correction"
    )
    p.add_argument(
        "--expected",
        type=int,
        default=7,
        help="Expected number of characters (default: 7)",
    )
    args = p.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        print(f"Error: image not found: {img_path}")
        return 1

    debug_dir = None
    if args.debug:
        debug_dir = Path(args.debug_dir or f"debug/{img_path.stem}")

    result = segment(
        img_path,
        debug_dir=debug_dir,
        expected_chars=args.expected,
        use_perspective=not args.no_perspective,
    )

    _print_result(result, str(img_path))

    if args.save_chars and result["images"]:
        save_chars(args.save_chars, result["images"])

    if result["n_chars"] == 0:
        print("WARNING: no characters detected.")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
