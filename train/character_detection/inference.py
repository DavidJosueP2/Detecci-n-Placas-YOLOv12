#!/usr/bin/env python3
"""
Single-character inference.

Usage:
    python inference.py --image path/to/char.png
    python inference.py --image char.png --position letter
    python inference.py --image char.png --position digit
    python inference.py --image char.png --topk 5

Position constraint (based on plate structure ABC-1234):
    --position letter   → restrict output to A-Z  (indices 0-25)
    --position digit    → restrict output to 0-9  (indices 26-35)
    (omit)              → unrestricted top-1 of all 36 classes
"""

import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Run from the character_detection/ directory so src.* imports resolve
os.chdir(Path(__file__).parent)

from src.model import CLASSES, CharCNN  # noqa: E402

IMG_SIZE = 32

_PREPROCESS = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])


def load_model(checkpoint_path: str, device: torch.device) -> CharCNN:
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    cfg = ckpt.get("cfg", {})
    mcfg = cfg.get("model", {"num_classes": 36, "dropout1": 0.5, "dropout2": 0.3})

    model = CharCNN(
        num_classes=mcfg.get("num_classes", 36),
        dropout1=mcfg.get("dropout1", 0.5),
        dropout2=mcfg.get("dropout2", 0.3),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return model.to(device)


def predict(
    image: Image.Image,
    model: CharCNN,
    device: torch.device,
    position: str | None = None,
    top_k: int = 3,
) -> dict:
    tensor = _PREPROCESS(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = F.softmax(logits, dim=1)[0].cpu()

    if position == "letter":
        probs[26:] = 0.0
        probs = probs / probs.sum()
    elif position == "digit":
        probs[:26] = 0.0
        probs = probs / probs.sum()

    top_indices = probs.topk(top_k).indices.tolist()

    return {
        "prediction": CLASSES[top_indices[0]],
        "confidence": float(probs[top_indices[0]]),
        "top_k": [(CLASSES[i], float(probs[i])) for i in top_indices],
    }


def main() -> None:
    p = argparse.ArgumentParser(description="Character inference")
    p.add_argument("--image", required=True, help="Path to character image")
    p.add_argument(
        "--model", default="outputs/models/best_model.pth",
        help="Path to .pth checkpoint",
    )
    p.add_argument(
        "--position", choices=["letter", "digit"],
        default=None,
        help="Restrict prediction to letters (A-Z) or digits (0-9)",
    )
    p.add_argument("--topk", type=int, default=3)
    args = p.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        sys.exit(
            f"Model not found: {model_path}\n"
            "Run train.py first to generate outputs/models/best_model.pth"
        )

    img_path = Path(args.image)
    if not img_path.exists():
        sys.exit(f"Image not found: {img_path}")
    img = Image.open(img_path)

    tensor = _PREPROCESS(img)

    from torchvision.transforms.functional import to_pil_image

    preview = tensor.squeeze(0)
    preview = (preview * 0.5) + 0.5

    to_pil_image(preview).save("debug_input.png")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(str(model_path), device)
    image = Image.open(img_path)

    result = predict(image, model, device, position=args.position, top_k=args.topk)

    print(f"\nImage      : {img_path}")
    print(f"Prediction : {result['prediction']}  ({result['confidence']*100:.1f}%)")
    if args.position:
        print(f"Constraint : {args.position}s only")
    print(f"Top-{args.topk}:")
    for char, prob in result["top_k"]:
        bar = "█" * int(prob * 30)
        print(f"  {char}  {prob*100:5.1f}%  {bar}")


if __name__ == "__main__":
    main()
