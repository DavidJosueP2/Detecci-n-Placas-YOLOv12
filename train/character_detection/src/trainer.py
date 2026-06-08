import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm


class EarlyStopping:
    """Stops training when val_accuracy stops improving."""

    def __init__(self, patience: int = 15, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best = None
        self.counter = 0
        self.triggered = False

    def step(self, score: float) -> bool:
        if self.best is None or score > self.best + self.min_delta:
            self.best = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.triggered = True
        return self.triggered


class Trainer:
    def __init__(self, model: nn.Module, cfg: dict):
        self.model = model
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        t = cfg["training"]
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(),
            lr=t["learning_rate"],
            weight_decay=t["weight_decay"],
        )
        # Reduce LR when val accuracy plateaus for 5 epochs
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", factor=0.5, patience=5
        )

        self.best_model_path = Path(cfg["paths"]["best_model"])
        self.best_val_acc = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
    ) -> Dict[str, List[float]]:
        print(f"Device: {self.device}")
        early_stop = EarlyStopping(
            patience=self.cfg["training"]["early_stopping_patience"],
            min_delta=self.cfg["training"]["min_delta"],
        )
        epochs = self.cfg["training"]["epochs"]
        history: Dict[str, List[float]] = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
        }

        for epoch in range(1, epochs + 1):
            t0 = time.time()
            train_loss, train_acc = self._train_epoch(train_loader)
            val_loss, val_acc = self._eval_epoch(val_loader)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_acc"].append(train_acc)
            history["val_acc"].append(val_acc)

            # Save checkpoint when val accuracy improves
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self._save_checkpoint(epoch, val_acc)
                marker = " ✓"
            else:
                marker = ""

            self.scheduler.step(val_acc)

            print(
                f"[{epoch:3d}/{epochs}] "
                f"loss {train_loss:.4f} acc {train_acc:.4f} | "
                f"val_loss {val_loss:.4f} val_acc {val_acc:.4f} | "
                f"best {self.best_val_acc:.4f} | "
                f"{time.time()-t0:.1f}s{marker}"
            )

            if early_stop.step(val_acc):
                print(
                    f"Early stopping at epoch {epoch}  "
                    f"(no improvement for {early_stop.patience} epochs)"
                )
                break

        return history

    def evaluate(
        self,
        loader: DataLoader,
        load_best: bool = True,
    ) -> Tuple[float, List[int], List[int]]:
        if load_best and self.best_model_path.exists():
            ckpt = torch.load(self.best_model_path, map_location=self.device)
            self.model.load_state_dict(ckpt["model_state_dict"])
            print(f"Loaded best model (epoch {ckpt['epoch']}, val_acc {ckpt['val_acc']:.4f})")

        _, acc = self._eval_epoch(loader)
        preds, labels = self._collect_predictions(loader)
        return acc, preds, labels

    # ------------------------------------------------------------------
    # Internal loops
    # ------------------------------------------------------------------

    def _train_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss, correct, total = 0.0, 0, 0

        for imgs, targets in tqdm(loader, desc="  train", leave=False):
            imgs, targets = imgs.to(self.device), targets.to(self.device)
            self.optimizer.zero_grad()
            logits = self.model(imgs)
            loss = self.criterion(logits, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * imgs.size(0)
            correct += (logits.argmax(1) == targets).sum().item()
            total += imgs.size(0)

        return total_loss / total, correct / total

    def _eval_epoch(self, loader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss, correct, total = 0.0, 0, 0

        with torch.no_grad():
            for imgs, targets in tqdm(loader, desc="  eval ", leave=False):
                imgs, targets = imgs.to(self.device), targets.to(self.device)
                logits = self.model(imgs)
                loss = self.criterion(logits, targets)

                total_loss += loss.item() * imgs.size(0)
                correct += (logits.argmax(1) == targets).sum().item()
                total += imgs.size(0)

        return total_loss / total, correct / total

    def _collect_predictions(self, loader: DataLoader) -> Tuple[List[int], List[int]]:
        self.model.eval()
        preds, labels = [], []
        with torch.no_grad():
            for imgs, targets in loader:
                imgs = imgs.to(self.device)
                preds.extend(self.model(imgs).argmax(1).cpu().tolist())
                labels.extend(targets.tolist())
        return preds, labels

    def _save_checkpoint(self, epoch: int, val_acc: float):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "val_acc": val_acc,
                "cfg": self.cfg,
            },
            self.best_model_path,
        )
