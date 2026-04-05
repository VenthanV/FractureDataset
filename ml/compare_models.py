"""
compare_models.py — Quick backbone comparison.

Trains ONLY the classification head (backbone frozen) for a small number of
epochs per model and reports val AUC. Use this to pick the best backbone
before committing to full Phase 1+2 training.

Two candidate lists are provided:
    LOCAL_CANDIDATES  — small/fast models for M4 MacBook
    CLOUD_CANDIDATES  — larger models for Kaggle GPU

Usage:
    # Compare local candidates (default)
    python compare_models.py

    # Compare cloud candidates
    python compare_models.py --mode cloud

    # Compare a custom list
    python compare_models.py --models efficientnet_b0 densenet121 convnext_tiny
"""

import argparse
import time
from pathlib import Path

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.optim import AdamW
from sklearn.metrics import roc_auc_score
import pandas as pd

from .config import (
    DEVICE, SPLITS_CSV, RANDOM_SEED,
    BATCH_SIZE, PHASE1_LR, PHASE1_WEIGHT_DECAY, LABEL_SMOOTHING, LOG_DIR,
)
from .dataloader import get_dataloaders
from .model import build_model, freeze_backbone

# ── Candidate model lists ──────────────────────────────────────────────────

LOCAL_CANDIDATES = [
    # name                  why try it
    "efficientnet_b0",      # our baseline — fast, 5.3M params
    "efficientnet_b2",      # +1-2% vs b0, ~9M params, still fast on M4
    "densenet121",          # CheXNet backbone — designed for chest X-rays
    "convnext_tiny",        # modern CNN, often beats EfficientNet
    "resnet50",             # classic baseline, huge medical imaging literature
    "inception_v3",         # multi-scale features, good at detail
]

CLOUD_CANDIDATES = [
    # name                     why try it
    "tf_efficientnetv2_m",     # our planned cloud model
    "tf_efficientnetv2_s",     # lighter V2 — faster iteration
    "convnext_base",           # strong modern CNN, 89M params
    "convnext_small",          # lighter ConvNeXt
    "densenet169",             # larger DenseNet variant
    "swin_small_patch4_window7_224",  # hierarchical ViT, good on medical images
    "inception_resnet_v2",     # multi-scale + residual connections
]

# ── Quick eval settings ────────────────────────────────────────────────────
QUICK_EPOCHS = 5   # number of head-only epochs per model — enough to rank them


def set_seed(seed: int) -> None:
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if DEVICE == "cuda":
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, device, phase):
    is_train = phase == "train"
    model.train(is_train)
    total_loss, correct, total = 0.0, 0, 0
    all_probs, all_labels = [], []

    with torch.set_grad_enabled(is_train):
        for images, labels in tqdm(loader, desc=phase, leave=False):
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss   = criterion(logits, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            probs  = torch.softmax(logits, dim=1)[:, 1]
            preds  = logits.argmax(dim=1)
            total_loss += loss.item() * images.size(0)
            correct    += (preds == labels).sum().item()
            total      += images.size(0)
            all_probs.extend(probs.detach().cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    avg_loss = total_loss / total
    accuracy = correct / total
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5
    return avg_loss, accuracy, auc


def evaluate_model(model_name: str, dataloaders: dict) -> dict:
    """
    Train head-only for QUICK_EPOCHS and return best val AUC.
    """
    print(f"\n{'─'*55}", flush=True)
    print(f"  Model: {model_name}", flush=True)
    print(f"{'─'*55}", flush=True)

    try:
        model = build_model(model_name=model_name, pretrained=True)
    except Exception as e:
        print(f"  SKIP — could not load model: {e}", flush=True)
        return {"model": model_name, "val_auc": None, "val_acc": None,
                "params_M": None, "time_s": None, "error": str(e)}

    model = model.to(DEVICE)
    freeze_backbone(model)

    n_params = sum(p.numel() for p in model.parameters()) / 1e6

    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=PHASE1_LR,
        weight_decay=PHASE1_WEIGHT_DECAY,
    )

    best_val_auc = 0.0
    best_val_acc = 0.0
    t_start = time.time()

    for epoch in range(1, QUICK_EPOCHS + 1):
        tr_loss, tr_acc, tr_auc = run_epoch(
            model, dataloaders["train"], criterion, optimizer, DEVICE, "train"
        )
        vl_loss, vl_acc, vl_auc = run_epoch(
            model, dataloaders["val"],   criterion, None,      DEVICE, "val"
        )
        print(
            f"  epoch {epoch}/{QUICK_EPOCHS} | "
            f"train auc={tr_auc:.3f} | val auc={vl_auc:.3f} acc={vl_acc:.3f}",
            flush=True,
        )
        if vl_auc > best_val_auc:
            best_val_auc = vl_auc
            best_val_acc = vl_acc

    elapsed = time.time() - t_start

    return {
        "model":    model_name,
        "params_M": round(n_params, 1),
        "val_auc":  round(best_val_auc, 4),
        "val_acc":  round(best_val_acc, 4),
        "time_s":   round(elapsed, 0),
        "error":    None,
    }


def print_results_table(results: list[dict]) -> None:
    valid = [r for r in results if r["val_auc"] is not None]
    valid.sort(key=lambda r: r["val_auc"], reverse=True)

    print("\n" + "=" * 65, flush=True)
    print("COMPARISON RESULTS  (head-only, {} epochs per model)".format(QUICK_EPOCHS), flush=True)
    print("=" * 65, flush=True)
    print(f"{'Rank':<5} {'Model':<40} {'Params':>7} {'Val AUC':>8} {'Val Acc':>8} {'Time':>7}", flush=True)
    print("─" * 65, flush=True)
    for rank, r in enumerate(valid, 1):
        marker = " ← WINNER" if rank == 1 else ""
        print(
            f"{rank:<5} {r['model']:<40} {r['params_M']:>6.1f}M "
            f"{r['val_auc']:>8.4f} {r['val_acc']:>8.4f} {r['time_s']:>6.0f}s"
            f"{marker}",
            flush=True,
        )

    failed = [r for r in results if r["val_auc"] is None]
    if failed:
        print("\nSkipped (load error):", flush=True)
        for r in failed:
            print(f"  {r['model']}: {r['error']}", flush=True)

    print("=" * 65, flush=True)
    if valid:
        winner = valid[0]
        print(f"\nRecommendation: use '{winner['model']}' for full training.", flush=True)
        print(f"  → Set MODEL_NAME = \"{winner['model']}\" in config.py", flush=True)
        print(f"    (or export MODEL_NAME=\"{winner['model']}\" for cloud)\n", flush=True)


def main() -> None:
    global QUICK_EPOCHS   # must be declared before any use of the name

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=["local", "cloud"], default="local",
        help="Which candidate list to use (default: local)"
    )
    parser.add_argument(
        "--models", nargs="+", default=None,
        help="Custom list of timm model names to compare"
    )
    parser.add_argument(
        "--epochs", type=int, default=QUICK_EPOCHS,
        help=f"Head-only epochs per model (default: {QUICK_EPOCHS})"
    )
    args = parser.parse_args()
    QUICK_EPOCHS = args.epochs

    if args.models:
        candidates = args.models
    elif args.mode == "cloud":
        candidates = CLOUD_CANDIDATES
    else:
        candidates = LOCAL_CANDIDATES

    set_seed(RANDOM_SEED)
    print(f"Device  : {DEVICE}", flush=True)
    print(f"Mode    : {args.mode}", flush=True)
    print(f"Epochs  : {QUICK_EPOCHS} per model", flush=True)
    print(f"Models  : {candidates}\n", flush=True)

    assert SPLITS_CSV.exists(), (
        f"splits.csv not found at {SPLITS_CSV}\n"
        "Run python dataset.py first."
    )

    dataloaders = get_dataloaders(SPLITS_CSV)
    results = []

    progress_path = LOG_DIR / "compare_progress.txt"
    progress_path.parent.mkdir(parents=True, exist_ok=True)

    for i, model_name in enumerate(candidates, 1):
        set_seed(RANDOM_SEED)  # reset seed before each model for fair comparison
        result = evaluate_model(model_name, dataloaders)
        results.append(result)

        # Write a one-line summary so `tail -f logs/compare_progress.txt` shows live status
        auc_str = f"{result['val_auc']:.4f}" if result["val_auc"] is not None else "FAILED"
        summary = (
            f"[{i}/{len(candidates)}] {model_name:<40}  val_auc={auc_str}"
            f"  time={result['time_s']}s\n"
        )
        with open(progress_path, "a") as pf:
            pf.write(summary)
        print(f"  → logged to {progress_path}", flush=True)

    print_results_table(results)

    # Save results CSV
    out_path = LOG_DIR / "model_comparison.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(out_path, index=False)
    print(f"Results saved → {out_path}", flush=True)


if __name__ == "__main__":
    main()
