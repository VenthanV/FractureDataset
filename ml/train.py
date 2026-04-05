"""
train.py — Two-phase transfer learning training loop.

Phase 1: Frozen backbone, train head only  (PHASE1_EPOCHS, PHASE1_LR)
Phase 2: Unfreeze last N blocks, fine-tune (PHASE2_EPOCHS, PHASE2_LR)

Usage:
    python train.py
"""

import csv
import io
import os
import time
import torch.multiprocessing
# Python 3.13 + macOS: POSIX shared memory (libshm) crashes with num_workers > 0
# file_system strategy avoids this while still enabling parallel data loading
torch.multiprocessing.set_sharing_strategy('file_system')
from pathlib import Path

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from sklearn.metrics import roc_auc_score

from .config import (
    DEVICE, SPLITS_CSV, CHECKPOINT_DIR, LOG_DIR, RANDOM_SEED,
    PHASE1_EPOCHS, PHASE1_LR, PHASE1_WEIGHT_DECAY,
    PHASE2_EPOCHS, PHASE2_LR, PHASE2_WEIGHT_DECAY,
    UNFREEZE_LAST_N, EARLY_STOP_PATIENCE, EARLY_STOP_MIN_DELTA,
    LABEL_SMOOTHING, USE_SCHEDULER, HEAD_DROPOUT,
    P2_SCHEDULER, P2_PLATEAU_FACTOR, P2_PLATEAU_PATIENCE,
    MODEL_NAME, IMG_SIZE, GRAD_CLIP_MAX_NORM,
    CLASS_WEIGHT_FRACTURE,
)
from .dataloader import get_dataloaders
from .model import build_model, freeze_backbone, unfreeze_last_n


def _resolve(cfg: dict, key: str, default):
    """Return cfg[key] if present, else default."""
    return cfg.get(key, default)


def set_seed(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)


def run_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: str,
    phase: str,
    scaler: "torch.cuda.amp.GradScaler | None" = None,
) -> tuple[float, float, float]:
    """
    One full pass over the dataset.

    Args:
        scaler: GradScaler for mixed-precision training (CUDA only).
                Pass None on MPS/CPU — autocast is skipped automatically.

    Returns:
        (avg_loss, accuracy, auc_roc)
    """
    is_train   = phase == "train"
    use_amp    = device == "cuda"   # autocast only on CUDA; MPS fp16 is unstable
    model.train(is_train)

    total_loss  = 0.0
    correct     = 0
    total       = 0
    all_probs:  list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []

    with torch.set_grad_enabled(is_train):
        for images, labels in tqdm(loader, desc=phase, leave=False):
            images = images.to(device)
            labels = labels.to(device)

            if is_train:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(device, enabled=use_amp):
                logits = model(images)             # (B, 2)
                loss   = criterion(logits, labels)

            if is_train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    # Gradient clipping prevents exploding gradients when unfrozen
                    # layers receive backprop for the first time in Phase 2
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_MAX_NORM)
                    optimizer.step()

            # cast to float32 before softmax — safe for both fp16 (CUDA) and fp32 (MPS)
            probs  = torch.softmax(logits.float(), dim=1)[:, 1]  # P(fracture)
            preds  = logits.argmax(dim=1)

            total_loss += loss.item() * images.size(0)
            correct    += (preds == labels).sum().item()
            total      += images.size(0)
            # Accumulate tensors; convert to numpy once after the loop
            all_probs.append(probs.detach().cpu())
            all_labels.append(labels.cpu())

    avg_loss   = total_loss / total
    accuracy   = correct / total
    probs_np   = torch.cat(all_probs).numpy()
    labels_np  = torch.cat(all_labels).numpy()
    try:
        auc = roc_auc_score(labels_np, probs_np)
    except ValueError:
        auc = 0.5  # fallback if only one class present (shouldn't occur)

    return avg_loss, accuracy, auc


def _save_checkpoint(path: Path, model: nn.Module) -> None:
    """Save model weights wrapped with architecture metadata."""
    torch.save(
        {"state_dict": model.state_dict(), "model_name": MODEL_NAME, "img_size": IMG_SIZE},
        path,
    )


def _load_state_dict(path: Path) -> dict:
    """Load a checkpoint saved as either a plain state_dict or a wrapped dict."""
    ck = torch.load(path, map_location=DEVICE)
    return ck["state_dict"] if isinstance(ck, dict) and "state_dict" in ck else ck


class EarlyStopping:
    """
    Monitors val AUC and saves the best checkpoint.
    Sets should_stop=True after `patience` epochs without improvement.
    """

    def __init__(
        self,
        patience: int,
        min_delta: float,
        checkpoint_path: Path,
        verbose: bool = True,
    ):
        self.patience        = patience
        self.min_delta       = min_delta
        self.checkpoint_path = checkpoint_path
        self.verbose         = verbose
        self.best_auc        = 0.0
        self.counter         = 0
        self.should_stop     = False

    def step(self, val_auc: float, model: nn.Module) -> None:
        if val_auc > self.best_auc + self.min_delta:
            self.best_auc = val_auc
            self.counter  = 0
            _save_checkpoint(self.checkpoint_path, model)
            if self.verbose:
                print(f"  [checkpoint] val AUC improved to {val_auc:.4f} — saved.", flush=True)
        else:
            self.counter += 1
            if self.verbose:
                print(f"  [early stop] no improvement — {self.counter}/{self.patience}", flush=True)
            if self.counter >= self.patience:
                self.should_stop = True


def train_phase(
    phase_name:    str,
    model:         nn.Module,
    dataloaders:   dict,
    criterion:     nn.Module,
    lr:            float,
    weight_decay:  float,
    n_epochs:      int,
    stopper:       EarlyStopping,
    log_writer:    csv.DictWriter,
    log_file_ref:  io.IOBase,
    device:        str,
    scheduler_type: str = "cosine",
    trial=None,
    epoch_offset:  int = 0,
    verbose:       bool = True,
) -> float:
    """
    Generic training loop shared by Phase 1 and Phase 2.

    Args:
        scheduler_type: "cosine"  — CosineAnnealingLR (Phase 1 default)
                        "plateau" — ReduceLROnPlateau on val AUC (Phase 2 default)
                        "none"    — no scheduler
        trial:          Optuna trial object for pruning (None during normal training).
        epoch_offset:   Added to epoch index when reporting to Optuna, so Phase 2
                        steps continue from where Phase 1 left off.
        verbose:        Whether to print epoch summaries (set False during HPO).

    Returns:
        Best val AUC seen in this phase.
    """
    # Build optimizer from currently unfrozen params only
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    if not USE_SCHEDULER or scheduler_type == "none":
        scheduler = None
    elif scheduler_type == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer, mode="max",
            factor=P2_PLATEAU_FACTOR,
            patience=P2_PLATEAU_PATIENCE,
        )
    else:  # "cosine"
        scheduler = CosineAnnealingLR(optimizer, T_max=n_epochs)

    # Mixed-precision scaler — CUDA only (MPS fp16 triggers Metal shader spikes)
    scaler = torch.cuda.amp.GradScaler() if device == "cuda" else None

    best_phase_auc = 0.0

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        tr_loss, tr_acc, tr_auc = run_epoch(
            model, dataloaders["train"], criterion, optimizer, device, "train",
            scaler=scaler,
        )
        vl_loss, vl_acc, vl_auc = run_epoch(
            model, dataloaders["val"],   criterion, None,      device, "val",
        )

        if scheduler is not None:
            if scheduler_type == "plateau":
                scheduler.step(vl_auc)   # ReduceLROnPlateau monitors val AUC
            else:
                scheduler.step()

        best_phase_auc = max(best_phase_auc, vl_auc)

        elapsed = time.time() - t0
        if verbose:
            print(
                f"[{phase_name}] {epoch:02d}/{n_epochs}  {elapsed:.0f}s  |"
                f"  train loss={tr_loss:.4f} acc={tr_acc:.3f} auc={tr_auc:.3f}"
                f"  |  val loss={vl_loss:.4f} acc={vl_acc:.3f} auc={vl_auc:.3f}",
                flush=True,
            )

        log_writer.writerow({
            "phase":      phase_name,
            "epoch":      epoch,
            "train_loss": f"{tr_loss:.4f}",
            "train_acc":  f"{tr_acc:.4f}",
            "train_auc":  f"{tr_auc:.4f}",
            "val_loss":   f"{vl_loss:.4f}",
            "val_acc":    f"{vl_acc:.4f}",
            "val_auc":    f"{vl_auc:.4f}",
        })
        log_file_ref.flush()   # write every epoch, not just at close

        stopper.step(vl_auc, model)
        if stopper.should_stop:
            if verbose:
                print(f"  [early stop] triggered at epoch {epoch} — stopping {phase_name}.", flush=True)
            break

        # Optuna pruning hook — skipped entirely when trial is None
        if trial is not None:
            import optuna
            trial.report(vl_auc, step=epoch_offset + epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

    return best_phase_auc


def train_model(cfg_override: dict | None = None) -> float:
    """
    Run the two-phase training loop.

    Args:
        cfg_override: Optional dict of hyperparameter overrides.
                      Keys match config.py constant names (e.g. "PHASE1_LR").
                      Special keys:
                        "_trial"        — Optuna trial object (enables pruning)
                        "_trial_number" — isolates checkpoints per trial
                        "_verbose"      — set False to suppress print output
                      If None, all values come from config.py defaults.

    Returns:
        Best val AUC achieved across both phases.
    """
    cfg = cfg_override or {}

    # Resolve hyperparameters: cfg_override wins, config.py is the fallback
    p1_lr           = _resolve(cfg, "PHASE1_LR",           PHASE1_LR)
    p1_wd           = _resolve(cfg, "PHASE1_WEIGHT_DECAY",  PHASE1_WEIGHT_DECAY)
    p1_epochs       = _resolve(cfg, "PHASE1_EPOCHS",        PHASE1_EPOCHS)
    p2_lr           = _resolve(cfg, "PHASE2_LR",            PHASE2_LR)
    p2_wd           = _resolve(cfg, "PHASE2_WEIGHT_DECAY",  PHASE2_WEIGHT_DECAY)
    p2_epochs       = _resolve(cfg, "PHASE2_EPOCHS",        PHASE2_EPOCHS)
    unfreeze_n      = _resolve(cfg, "UNFREEZE_LAST_N",      UNFREEZE_LAST_N)
    label_smoothing = _resolve(cfg, "LABEL_SMOOTHING",      LABEL_SMOOTHING)
    patience        = _resolve(cfg, "EARLY_STOP_PATIENCE",  EARLY_STOP_PATIENCE)
    dropout1        = _resolve(cfg, "DROPOUT1",             HEAD_DROPOUT)
    dropout2        = _resolve(cfg, "DROPOUT2",             HEAD_DROPOUT)
    trial           = cfg.get("_trial",        None)
    trial_num       = cfg.get("_trial_number", None)
    verbose         = cfg.get("_verbose",      True)

    set_seed(RANDOM_SEED)
    if verbose:
        print(f"Device: {DEVICE}", flush=True)

    dataloaders = get_dataloaders(SPLITS_CSV)

    model = build_model(dropout1=dropout1, dropout2=dropout2)
    model = model.to(DEVICE)

    weight_fracture = _resolve(cfg, "CLASS_WEIGHT_FRACTURE", CLASS_WEIGHT_FRACTURE)
    if weight_fracture != 1.0:
        weight_tensor = torch.tensor([1.0, weight_fracture], dtype=torch.float32, device=DEVICE)
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing, weight=weight_tensor)
        if verbose:
            print(f"[loss] Asymmetric class weights: normal=1.0  fracture={weight_fracture}", flush=True)
    else:
        criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    # Checkpoint dir: isolated per Optuna trial to prevent overwrites
    ckpt_dir = CHECKPOINT_DIR / f"trial_{trial_num}" if trial_num is not None else CHECKPOINT_DIR
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    LOG_DIR.mkdir(parents=True, exist_ok=True)

    best_ckpt = ckpt_dir / "best_model.pth"
    last_ckpt = ckpt_dir / "last_model.pth"
    log_path  = LOG_DIR / "train_log.csv"
    log_fields = [
        "phase", "epoch",
        "train_loss", "train_acc", "train_auc",
        "val_loss",   "val_acc",   "val_auc",
    ]

    skip_phase1 = bool(int(os.environ.get("SKIP_PHASE1", "0")))

    # During HPO, discard CSV logs to avoid clutter; normal training writes to file
    log_open_mode = "a" if skip_phase1 else "w"
    log_file: io.IOBase = io.StringIO() if trial_num is not None else open(log_path, log_open_mode, newline="")

    with log_file:
        log_writer = csv.DictWriter(log_file, fieldnames=log_fields)
        if trial_num is None and not skip_phase1:
            log_writer.writeheader()

        # ── Phase 1: frozen backbone ──────────────────────────────────────
        if skip_phase1:
            # Load existing best checkpoint and skip straight to Phase 2
            if not best_ckpt.exists():
                raise FileNotFoundError(f"SKIP_PHASE1=1 but no checkpoint at {best_ckpt}")
            freeze_backbone(model)
            model.load_state_dict(_load_state_dict(best_ckpt))
            p1_best_auc = 0.0
            if verbose:
                print(f"SKIP_PHASE1: loaded checkpoint from {best_ckpt}", flush=True)
        else:
            if verbose:
                print("\n" + "=" * 60, flush=True)
                print("PHASE 1 — training head with frozen backbone", flush=True)
                print("=" * 60, flush=True)
            freeze_backbone(model)

            stopper_p1 = EarlyStopping(
                patience=patience,
                min_delta=EARLY_STOP_MIN_DELTA,
                checkpoint_path=best_ckpt,
                verbose=verbose,
            )
            p1_best_auc = train_phase(
                "Phase1", model, dataloaders, criterion,
                lr=p1_lr, weight_decay=p1_wd,
                n_epochs=p1_epochs,
                stopper=stopper_p1, log_writer=log_writer, log_file_ref=log_file,
                device=DEVICE, scheduler_type="cosine",
                trial=trial, epoch_offset=0, verbose=verbose,
            )

        # ── Phase 2: unfreeze last N blocks ───────────────────────────────
        if verbose:
            print("\n" + "=" * 60, flush=True)
            print(f"PHASE 2 — fine-tuning last {unfreeze_n} backbone blocks", flush=True)
            print("=" * 60, flush=True)
        unfreeze_last_n(model, unfreeze_n)

        if not skip_phase1:
            # Start Phase 2 from best Phase 1 weights, not the final epoch
            model.load_state_dict(_load_state_dict(best_ckpt))
            if verbose:
                print(f"Loaded best Phase 1 checkpoint from {best_ckpt}", flush=True)

        stopper_p2 = EarlyStopping(
            patience=patience,
            min_delta=EARLY_STOP_MIN_DELTA,
            checkpoint_path=best_ckpt,   # overwrites with best Phase 2 result
            verbose=verbose,
        )
        p2_best_auc = train_phase(
            "Phase2", model, dataloaders, criterion,
            lr=p2_lr, weight_decay=p2_wd,
            n_epochs=p2_epochs,
            stopper=stopper_p2, log_writer=log_writer, log_file_ref=log_file,
            device=DEVICE, scheduler_type=_resolve(cfg, "P2_SCHEDULER", P2_SCHEDULER),
            trial=trial, epoch_offset=p1_epochs, verbose=verbose,
        )

    _save_checkpoint(last_ckpt, model)

    # ── Kaggle: copy best checkpoint to /kaggle/working/ root ─────────────
    # Files in /kaggle/working/ appear in the notebook Output tab and can be
    # downloaded directly. Without this the .pth is buried in a subdirectory
    # and disappears when the session is restarted.
    from .config import _ON_KAGGLE, _KAGGLE_WORKING
    if _ON_KAGGLE and best_ckpt.exists():
        import shutil
        kaggle_out = _KAGGLE_WORKING / "best_model.pth"
        shutil.copy2(best_ckpt, kaggle_out)
        if verbose:
            print(f"\n[Kaggle] Checkpoint copied → {kaggle_out}", flush=True)
            print(f"         Download via the Output tab or:", flush=True)
            print(f"         from IPython.display import FileLink; FileLink('best_model.pth')", flush=True)

    if verbose:
        print(f"\nDone.  Best model → {best_ckpt}", flush=True)
        print(f"       Last model → {last_ckpt}", flush=True)
        print(f"       Training log → {log_path}", flush=True)

    return max(p1_best_auc, p2_best_auc)


def main() -> None:
    train_model()


if __name__ == "__main__":
    main()
