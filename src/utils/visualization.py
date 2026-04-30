"""
src/utils/visualization.py
───────────────────────────
Plotting helpers used during and after training.
"""

import json
import os
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_training_curves(history: Dict, save_dir: str):
    """
    Plot accuracy, F1, and loss curves from a training history dict.

    history format:
        {
            "train_accuracy": [...],
            "val_accuracy":   [...],
            "train_loss":     [...],
            "val_loss":       [...],
            "train_f1":       [...],
            "val_f1":         [...],
        }
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    epochs = range(1, len(history["train_accuracy"]) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # Accuracy
    ax = axes[0]
    ax.plot(epochs, history["train_accuracy"], label="Train Accuracy")
    ax.plot(epochs, history["val_accuracy"],   label="Val Accuracy")
    ax.set_title("Accuracy")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True)

    # F1
    ax = axes[1]
    ax.plot(epochs, history["train_f1"], label="Train F1")
    ax.plot(epochs, history["val_f1"],   label="Val F1")
    ax.set_title("F1-Score (Weighted)")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("F1")
    ax.legend()
    ax.grid(True)

    # Loss
    ax = axes[2]
    ax.plot(epochs, history["train_loss"], label="Train Loss")
    ax.plot(epochs, history["val_loss"],   label="Val Loss")
    ax.set_title("Loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    fig.savefig(save_dir / "training_curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training curves → {save_dir}/training_curves.png")


def plot_confusion_matrix(
    cm: List[List[int]],
    class_names: List[str],
    save_path: str,
    title: str = "Confusion Matrix",
):
    cm = np.array(cm)
    # Normalise rows to percentages for readability
    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-8)

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        cm_norm,
        annot=cm,          # show raw counts
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved confusion matrix → {save_path}")
