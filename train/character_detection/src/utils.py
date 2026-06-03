from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix


def plot_confusion_matrix(
    labels: list,
    preds: list,
    class_names: List[str],
    output_dir: str,
) -> None:
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    # Row-normalize so each cell shows recall per class
    row_sums = cm.sum(axis=1, keepdims=True).clip(min=1)
    cm_norm = cm.astype(float) / row_sums

    fig, ax = plt.subplots(figsize=(14, 12))
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0,
        vmax=1,
        ax=ax,
        annot_kws={"size": 7},
    )
    ax.set_xlabel("Predicted", fontsize=12)
    ax.set_ylabel("True", fontsize=12)
    ax.set_title("Confusion Matrix — row-normalized recall", fontsize=13)
    plt.tight_layout()

    out = Path(output_dir) / "confusion_matrix.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def plot_training_curves(history: dict, output_dir: str) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

    ax_loss.plot(epochs, history["train_loss"], label="train")
    ax_loss.plot(epochs, history["val_loss"], label="val")
    ax_loss.set_title("Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.legend()
    ax_loss.grid(alpha=0.3)

    ax_acc.plot(epochs, history["train_acc"], label="train")
    ax_acc.plot(epochs, history["val_acc"], label="val")
    ax_acc.set_title("Accuracy")
    ax_acc.set_xlabel("Epoch")
    ax_acc.set_ylim(0, 1)
    ax_acc.legend()
    ax_acc.grid(alpha=0.3)

    plt.tight_layout()
    out = Path(output_dir) / "training_curves.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved: {out}")


def print_classification_report(
    labels: list,
    preds: list,
    class_names: List[str],
) -> None:
    print("\n--- Classification Report ---")
    print(
        classification_report(
            labels, preds,
            target_names=class_names,
            digits=3,
            zero_division=0,
        )
    )

    # Show the 10 worst-performing classes so debugging is fast
    cm = confusion_matrix(labels, preds, labels=list(range(len(class_names))))
    per_class = []
    for i, name in enumerate(class_names):
        s = cm[i].sum()
        per_class.append((name, cm[i, i] / s if s else 0.0, s))

    print("--- Lowest per-class accuracy (top 10 worst) ---")
    for name, acc, support in sorted(per_class, key=lambda x: x[1])[:10]:
        print(f"  {name}  acc={acc:.3f}  support={support}")
