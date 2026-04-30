"""
train.py
────────
Train AVT-CA on the RAVDESS dataset.

Usage:
    python train.py                      # uses defaults from config.py
    python train.py --data ./data/RAVDESS --epochs 128 --batch_size 8

Outputs (saved in CHECKPOINT_DIR):
    best_model.pt          model with highest validation accuracy
    last_model.pt          model at final epoch
    checkpoint_ep{N}.pt    periodic checkpoints every N epochs
    config_used.json       config snapshot
    history.json           per-epoch metrics
    logs/training_curves.png
    logs/confusion_matrix_best.png
"""

import argparse
import json
import logging
import os
import random
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# ── Local imports ────────────────────────────────────────────
from config import Config
from src.data.ravdess_dataset import RAVDESSDataset, EMOTION_NAMES
from src.models.avtca import AVTCA
from src.utils.metrics import RunningMetrics
from src.utils.visualization import plot_training_curves, plot_confusion_matrix

# ── Logging setup ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def save_checkpoint(model, optimizer, epoch, metrics, path):
    torch.save({
        "epoch":     epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics":   metrics,
    }, path)


def load_checkpoint(path, model, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    if optimizer and "optimizer_state_dict" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
    return ckpt.get("epoch", 0), ckpt.get("metrics", {})


# ─────────────────────────────────────────────────────────────
# One-epoch training / validation
# ─────────────────────────────────────────────────────────────

def run_epoch(model, loader, criterion, optimizer, device, train: bool):
    model.train() if train else model.eval()
    metrics = RunningMetrics()
    context = torch.enable_grad() if train else torch.no_grad()

    with context:
        for audio, video, labels in tqdm(loader, desc="train" if train else "val ", leave=False):
            audio  = audio.to(device, non_blocking=True)   # (B, n_mels, T_a)
            video  = video.to(device, non_blocking=True)   # (B, T_v, 3, H, W)
            labels = labels.to(device, non_blocking=True)  # (B,)

            logits = model(audio, video)                   # (B, num_classes)
            loss   = criterion(logits, labels)

            if train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            metrics.update(logits, labels, loss=loss.item())

    return metrics.compute(EMOTION_NAMES)


# ─────────────────────────────────────────────────────────────
# Main training function
# ─────────────────────────────────────────────────────────────

def train(cfg: Config):
    set_seed(cfg.SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log.info(f"Using device: {device}")

    # ── Directories ──────────────────────────────────────────
    ckpt_dir = Path(cfg.CHECKPOINT_DIR)
    log_dir  = Path(cfg.LOG_DIR)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── Datasets ─────────────────────────────────────────────
    log.info("Loading RAVDESS dataset …")
    common_kwargs = dict(
        root_dir        = cfg.DATA_ROOT,
        train_ratio     = cfg.TRAIN_RATIO,
        num_frames      = cfg.NUM_FRAMES,
        frame_size      = (cfg.FRAME_H, cfg.FRAME_W),
        n_mels          = cfg.N_MELS,
        sr              = cfg.SAMPLE_RATE,
        max_audio_len   = cfg.MAX_AUDIO_LEN,
        modality_filter = cfg.MODALITY_FILTER,
        seed            = cfg.SEED,
        cache_dir       = cfg.CACHE_DIR,
    )
    train_ds = RAVDESSDataset(split="train", **common_kwargs)
    val_ds   = RAVDESSDataset(split="val",   **common_kwargs)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=cfg.PIN_MEMORY,
    )
    log.info(f"Train: {len(train_ds)} samples  |  Val: {len(val_ds)} samples")

    # ── Model ────────────────────────────────────────────────
    model = AVTCA(
        num_classes            = cfg.NUM_CLASSES,
        n_mels                 = cfg.N_MELS,
        d_model                = cfg.D_MODEL,
        num_heads              = cfg.NUM_HEADS,
        num_transformer_layers = cfg.NUM_TRANSFORMER_LAYERS,
        ffn_dim                = cfg.FFN_DIM,
        dropout                = cfg.DROPOUT,
        cnn_ch                 = cfg.CNN_CH,
    ).to(device)

    n_params = model.count_parameters()
    log.info(f"Model parameters: {n_params:,}")

    # ── Optimizer & scheduler ─────────────────────────────────
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.WEIGHT_DECAY,
    )
    # Cosine annealing for stability (not in paper, but helps with LR=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.NUM_EPOCHS, eta_min=1e-5
    )
    criterion = nn.CrossEntropyLoss()

    # ── Save config snapshot ──────────────────────────────────
    config_dict = {k: str(v) for k, v in vars(cfg).items()}
    (ckpt_dir / "config_used.json").write_text(json.dumps(config_dict, indent=2))

    # ── Training loop ─────────────────────────────────────────
    history = {
        "train_accuracy": [], "val_accuracy": [],
        "train_f1":       [], "val_f1":       [],
        "train_loss":     [], "val_loss":     [],
    }
    best_val_acc  = 0.0
    patience_cnt  = 0

    for epoch in range(1, cfg.NUM_EPOCHS + 1):
        log.info(f"\n── Epoch {epoch}/{cfg.NUM_EPOCHS}  (lr={scheduler.get_last_lr()[0]:.2e})")

        train_m = run_epoch(model, train_loader, criterion, optimizer, device, train=True)
        val_m   = run_epoch(model, val_loader,   criterion, optimizer, device, train=False)
        scheduler.step()

        # ── Log ───────────────────────────────────────────────
        log.info(
            f"  Train  acc={train_m['accuracy']:.4f}  "
            f"f1={train_m['f1_weighted']:.4f}  "
            f"loss={train_m.get('avg_loss', 0):.4f}"
        )
        log.info(
            f"  Val    acc={val_m['accuracy']:.4f}  "
            f"f1={val_m['f1_weighted']:.4f}  "
            f"loss={val_m.get('avg_loss', 0):.4f}"
        )

        # ── History ───────────────────────────────────────────
        history["train_accuracy"].append(train_m["accuracy"])
        history["val_accuracy"]  .append(val_m["accuracy"])
        history["train_f1"]      .append(train_m["f1_weighted"])
        history["val_f1"]        .append(val_m["f1_weighted"])
        history["train_loss"]    .append(train_m.get("avg_loss", 0))
        history["val_loss"]      .append(val_m.get("avg_loss", 0))

        # ── Save best ─────────────────────────────────────────
        if val_m["accuracy"] > best_val_acc:
            best_val_acc = val_m["accuracy"]
            patience_cnt = 0
            save_checkpoint(model, optimizer, epoch, val_m, ckpt_dir / "best_model.pt")
            log.info(f"  ✓ New best val acc = {best_val_acc:.4f}  — saved best_model.pt")

            # Save best confusion matrix
            plot_confusion_matrix(
                val_m["confusion_matrix"],
                EMOTION_NAMES[: cfg.NUM_CLASSES],
                str(log_dir / "confusion_matrix_best.png"),
                title=f"Confusion Matrix – Epoch {epoch} (val acc {best_val_acc:.4f})",
            )
        else:
            patience_cnt += 1
            if patience_cnt >= cfg.EARLY_STOP_PATIENCE:
                log.info(f"Early stopping triggered after {epoch} epochs.")
                break

        # ── Periodic checkpoint ───────────────────────────────
        if epoch % cfg.SAVE_EVERY_N_EPOCHS == 0:
            save_checkpoint(
                model, optimizer, epoch, val_m,
                ckpt_dir / f"checkpoint_ep{epoch:04d}.pt"
            )

        # ── Flush history ─────────────────────────────────────
        (ckpt_dir / "history.json").write_text(json.dumps(history, indent=2))

    # ── Save final model ─────────────────────────────────────
    save_checkpoint(model, optimizer, epoch, val_m, ckpt_dir / "last_model.pt")
    log.info("Saved last_model.pt")

    # ── Final plots ───────────────────────────────────────────
    plot_training_curves(history, str(log_dir))
    log.info(f"\nTraining complete.  Best val accuracy: {best_val_acc:.4f}")
    log.info(f"Checkpoints → {ckpt_dir}")

    return history


# ─────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train AVT-CA on RAVDESS")
    parser.add_argument("--data",       type=str, help="Override DATA_ROOT")
    parser.add_argument("--ckpt_dir",   type=str, help="Override CHECKPOINT_DIR")
    parser.add_argument("--epochs",     type=int, help="Override NUM_EPOCHS")
    parser.add_argument("--batch_size", type=int, help="Override BATCH_SIZE")
    parser.add_argument("--lr",         type=float, help="Override LEARNING_RATE")
    parser.add_argument("--resume",     type=str, default="",
                        help="Path to checkpoint to resume from")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    cfg  = Config()

    # CLI overrides
    if args.data:       cfg.DATA_ROOT        = args.data
    if args.ckpt_dir:   cfg.CHECKPOINT_DIR   = args.ckpt_dir
    if args.epochs:     cfg.NUM_EPOCHS       = args.epochs
    if args.batch_size: cfg.BATCH_SIZE       = args.batch_size
    if args.lr:         cfg.LEARNING_RATE    = args.lr

    train(cfg)
