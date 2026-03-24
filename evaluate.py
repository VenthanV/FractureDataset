"""
evaluate.py — Evaluate the best checkpoint on the held-out test split.

Metrics reported:
    AUC-ROC, sensitivity, specificity, accuracy, PPV (precision), NPV
    Optimal decision threshold via Youden's J statistic
    Confusion matrix and ROC curve saved to logs/

Usage:
    python evaluate.py
    python evaluate.py --save-json      # also writes logs/eval_results.json
"""

import argparse
import json
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe for MPS and servers
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

from config import DEVICE, SPLITS_CSV, CHECKPOINT_DIR, LOG_DIR
from dataloader import get_dataloaders
from model import build_model


def run_inference(model, loader, device) -> dict:
    """Run inference and collect labels, probabilities, and predictions."""
    model.eval()
    all_labels: list[int]   = []
    all_probs:  list[float] = []
    all_preds:  list[int]   = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            logits = model(images)
            probs  = torch.softmax(logits, dim=1)[:, 1]  # P(fracture)
            preds  = logits.argmax(dim=1)

            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    return {
        "labels": np.array(all_labels),
        "probs":  np.array(all_probs),
        "preds":  np.array(all_preds),
    }


def compute_metrics(results: dict) -> dict:
    """
    Compute clinical and statistical metrics.

    Medical context:
        - Sensitivity (recall for fracture class): fraction of actual fractures
          correctly detected. HIGH sensitivity = low miss rate.
          This is the PRIMARY metric for fracture screening.
        - Specificity: fraction of normal cases correctly identified.
          HIGH specificity = low false alarm rate.
        - For fracture detection, sensitivity > specificity in importance:
          missing a fracture is clinically worse than a false alarm.
        - NPV (negative predictive value): probability that a "normal" prediction
          is truly normal — critical for ruling-out use cases.
    """
    labels = results["labels"]
    probs  = results["probs"]
    preds  = results["preds"]

    auc = roc_auc_score(labels, probs)
    cm  = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy    = (tp + tn) / (tp + tn + fp + fn)
    ppv         = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # precision
    npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0

    return {
        "auc_roc":          auc,
        "sensitivity":      sensitivity,
        "specificity":      specificity,
        "accuracy":         accuracy,
        "ppv":              ppv,
        "npv":              npv,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "confusion_matrix": cm,
    }


def find_optimal_threshold(labels, probs) -> tuple[float, float, float]:
    """
    Find the probability threshold maximising Youden's J = sensitivity + specificity - 1.
    Standard method for selecting a clinical operating point from an ROC curve.

    Returns:
        (threshold, sensitivity_at_threshold, specificity_at_threshold)
    """
    fpr, tpr, thresholds = roc_curve(labels, probs)
    j        = tpr - fpr
    best_idx = int(np.argmax(j))
    return float(thresholds[best_idx]), float(tpr[best_idx]), float(1 - fpr[best_idx])


def plot_roc_curve(labels, probs, save_path) -> None:
    fpr, tpr, _ = roc_curve(labels, probs)
    auc = roc_auc_score(labels, probs)
    opt_thresh, opt_sens, opt_spec = find_optimal_threshold(labels, probs)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#4c9be8", lw=2, label=f"ROC  (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random classifier")
    ax.scatter(
        [1 - opt_spec], [opt_sens],
        color="#e05c5c", zorder=5, s=80,
        label=f"Optimal threshold  sens={opt_sens:.3f}  spec={opt_spec:.3f}",
    )
    ax.set_xlabel("1 − Specificity  (FPR)")
    ax.set_ylabel("Sensitivity  (TPR)")
    ax.set_title("ROC Curve — Test Set", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"ROC curve  → {save_path}")


def plot_confusion_matrix(cm, save_path) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)
    classes    = ["Normal", "Fracture"]
    tick_marks = range(len(classes))
    ax.set_xticks(list(tick_marks))
    ax.set_xticklabels(classes)
    ax.set_yticks(list(tick_marks))
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix — Test Set", fontsize=13)
    thresh = cm.max() / 2
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center", fontsize=14, fontweight="bold",
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Confusion matrix → {save_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate fracture detection model")
    parser.add_argument(
        "--save-json",
        action="store_true",
        help="Save all metrics to logs/eval_results.json (read by /model/stats API endpoint)",
    )
    args, _ = parser.parse_known_args()

    print(f"Device: {DEVICE}")

    best_ckpt = CHECKPOINT_DIR / "best_model.pth"
    assert best_ckpt.exists(), (
        f"Checkpoint not found at {best_ckpt}\n"
        "Run train.py first."
    )

    model = build_model(pretrained=False)
    model.load_state_dict(torch.load(best_ckpt, map_location=DEVICE))
    model = model.to(DEVICE)
    print(f"Loaded checkpoint: {best_ckpt}")

    dataloaders = get_dataloaders(SPLITS_CSV)
    results     = run_inference(model, dataloaders["test"], DEVICE)
    metrics     = compute_metrics(results)

    print("\n" + "=" * 55)
    print("TEST SET RESULTS")
    print("=" * 55)
    print(f"AUC-ROC      : {metrics['auc_roc']:.4f}")
    print(f"Sensitivity  : {metrics['sensitivity']:.4f}  ← primary metric (fracture recall)")
    print(f"Specificity  : {metrics['specificity']:.4f}")
    print(f"Accuracy     : {metrics['accuracy']:.4f}")
    print(f"PPV          : {metrics['ppv']:.4f}  (precision)")
    print(f"NPV          : {metrics['npv']:.4f}")
    print(f"\nConfusion matrix:")
    print(f"  TP={metrics['tp']}  FP={metrics['fp']}")
    print(f"  FN={metrics['fn']}  TN={metrics['tn']}")

    opt_thresh, opt_sens, opt_spec = find_optimal_threshold(
        results["labels"], results["probs"]
    )
    print(f"\nOptimal threshold (Youden's J): {opt_thresh:.3f}")
    print(f"  → sensitivity={opt_sens:.3f}  specificity={opt_spec:.3f}")

    print("\nClassification report:")
    print(classification_report(
        results["labels"], results["preds"],
        target_names=["Normal", "Fracture"],
    ))

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    plot_roc_curve(results["labels"], results["probs"], LOG_DIR / "roc_curve.png")
    plot_confusion_matrix(metrics["confusion_matrix"],  LOG_DIR / "confusion_matrix.png")

    if args.save_json:
        # Count split sizes from the dataloaders' datasets
        n_train = len(dataloaders["train"].dataset)
        n_val   = len(dataloaders["val"].dataset)
        n_test  = len(dataloaders["test"].dataset)

        eval_results = {
            "auc_roc":           round(float(metrics["auc_roc"]),      4),
            "sensitivity":       round(float(metrics["sensitivity"]),   4),
            "specificity":       round(float(metrics["specificity"]),   4),
            "accuracy":          round(float(metrics["accuracy"]),      4),
            "ppv":               round(float(metrics["ppv"]),           4),
            "npv":               round(float(metrics["npv"]),           4),
            "tp":                int(metrics["tp"]),
            "fp":                int(metrics["fp"]),
            "fn":                int(metrics["fn"]),
            "tn":                int(metrics["tn"]),
            "optimal_threshold": round(opt_thresh, 3),
            "n_train":           n_train,
            "n_val":             n_val,
            "n_test":            n_test,
            "model_name":        str(model.backbone.__class__.__name__),
            # Raw test-set scores — used by the frontend to compute metrics
            # live for any threshold without a round-trip to the server.
            "test_probs":        [round(float(p), 5) for p in results["probs"]],
            "test_labels":       [int(l) for l in results["labels"]],
        }

        json_path = LOG_DIR / "eval_results.json"
        json_path.write_text(json.dumps(eval_results, indent=2))
        print(f"\nEval results → {json_path}")


if __name__ == "__main__":
    main()
