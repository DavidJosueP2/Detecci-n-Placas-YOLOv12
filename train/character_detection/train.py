#!/usr/bin/env python3
"""
Main training entry point for the character CNN.

Usage (from train/character_detection/):
    python train.py
    python train.py --config configs/train_config.yaml
    python train.py --config configs/train_config.yaml --seed 123
"""

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Deterministic ops (slight speed cost; remove if training is too slow)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(config_path: str, seed: int) -> None:
    # Always run relative to this script's directory
    os.chdir(Path(__file__).parent)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    set_seed(seed)

    # Create output directories
    for key in ("model_dir", "logs_dir", "plots_dir"):
        Path(cfg["paths"][key]).mkdir(parents=True, exist_ok=True)

    # ── Imports after chdir so src.* resolves correctly ──────────────────────
    from src.dataset import build_dataloaders
    from src.model import CLASSES, build_model
    from src.trainer import Trainer
    from src.utils import (
        plot_confusion_matrix,
        plot_training_curves,
        print_classification_report,
    )

    # ── Dataset ───────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"Config : {config_path}")
    print(f"Device : {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"Seed   : {seed}")
    print(f"{'='*60}\n")

    train_loader, val_loader, test_loader = build_dataloaders(cfg)

    # ── Model ─────────────────────────────────────────────────────────────────
    mcfg = cfg["model"]
    model = build_model(mcfg)
    arch = mcfg.get("arch", "CharCNN")
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model  : {arch}")
    print(f"Params : {n_params:,}\n")

    # ── Training ──────────────────────────────────────────────────────────────
    trainer = Trainer(model, cfg)
    history = trainer.fit(train_loader, val_loader)

    # ── Test evaluation ───────────────────────────────────────────────────────
    print(f"\n{'='*60}  TEST SET  {'='*60}")
    test_acc, preds, labels = trainer.evaluate(test_loader, load_best=True)
    print(f"Test Accuracy: {test_acc * 100:.2f}%\n")

    print_classification_report(labels, preds, CLASSES)
    plot_training_curves(history, cfg["paths"]["plots_dir"])
    plot_confusion_matrix(labels, preds, CLASSES, cfg["paths"]["plots_dir"])

    print(f"\nBest model : {cfg['paths']['best_model']}")
    print("Plots      :", cfg["paths"]["plots_dir"])


def _parse_args():
    p = argparse.ArgumentParser(description="Train character CNN")
    p.add_argument("--config", default="configs/train_config.yaml")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    main(args.config, args.seed)
