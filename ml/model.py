"""
model.py — timm backbone with a binary classification head and freeze utilities.
"""

import timm
import torch
import torch.nn as nn

from .config import MODEL_NAME, NUM_CLASSES, HEAD_HIDDEN_DIM


def build_model(
    model_name: str = MODEL_NAME,
    pretrained: bool = True,
    dropout1: float = 0.4,
    dropout2: float = 0.4,
) -> "FractureClassifier":
    """
    Build a FractureClassifier from any timm model.

    timm.create_model(..., num_classes=0) removes the original classifier head
    and exposes raw feature embeddings. We attach our own binary head on top.

    Args:
        model_name: Any timm identifier.
                    Local:  "efficientnet_b0"
                    Cloud:  "tf_efficientnetv2_m" or "convnext_base"
        pretrained: Load ImageNet pretrained weights.
        dropout1:   Dropout rate before the first linear layer in the head.
        dropout2:   Dropout rate before the final linear layer in the head.
    """
    backbone = timm.create_model(model_name, pretrained=pretrained, num_classes=0)
    return FractureClassifier(
        backbone=backbone,
        num_features=backbone.num_features,
        dropout1=dropout1,
        dropout2=dropout2,
    )


class FractureClassifier(nn.Module):
    """
    Thin wrapper: timm backbone + custom binary classification head.

    Head design:
        Dropout(d1) → Linear(features→256) → BatchNorm1d → SiLU → Dropout(d2) → Linear(256→2)

    BatchNorm1d stabilises training when the backbone output distribution shifts
    during Phase 2 fine-tuning. SiLU (Swish) is used by EfficientNet internally
    and avoids the dying-ReLU problem; empirically outperforms GELU on small heads.

    Why CrossEntropy + 2 outputs instead of BCEWithLogitsLoss + 1 output?
        Integrates cleanly with label_smoothing and standard torch patterns.
        AUC-ROC uses softmax(logits)[:, 1] as the fracture probability.
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_features: int,
        dropout1: float = 0.4,
        dropout2: float = 0.4,
    ):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Dropout(p=dropout1),
            nn.Linear(num_features, HEAD_HIDDEN_DIM),
            nn.BatchNorm1d(HEAD_HIDDEN_DIM),
            nn.SiLU(),
            nn.Dropout(p=dropout2),
            nn.Linear(HEAD_HIDDEN_DIM, NUM_CLASSES),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.backbone(x)    # (B, num_features)
        return self.head(features)     # (B, 2)


def freeze_backbone(model: FractureClassifier) -> None:
    """
    Freeze all backbone parameters. Only the head trains (Phase 1).
    Call this before Phase 1 training begins.
    """
    for param in model.backbone.parameters():
        param.requires_grad = False
    for param in model.head.parameters():
        param.requires_grad = True
    _print_trainable(model, "Backbone frozen")


def unfreeze_last_n(model: FractureClassifier, n: int) -> None:
    """
    Unfreeze the last N block groups of the backbone for Phase 2 fine-tuning.

    For efficientnet_b0:     backbone.blocks has 9 MBConv groups  → unfreeze[-2:]
    For tf_efficientnetv2_m: backbone.blocks has 6 Fused-MBConv stages → unfreeze[-2:]

    Also always unfreezes the normalization / projection layers immediately
    before global average pooling (bn2, conv_head, bn_head, norm) — these
    are critical for domain adaptation and cheap to unfreeze.

    Args:
        model: FractureClassifier instance (backbone must already be frozen)
        n:     number of top-level block groups to unfreeze from the end
    """
    # Start fully frozen (should already be from Phase 1, but be explicit)
    for param in model.backbone.parameters():
        param.requires_grad = False

    # Locate block list — all EfficientNet variants expose .blocks
    if hasattr(model.backbone, "blocks"):
        blocks = list(model.backbone.blocks)
    else:
        blocks = list(model.backbone.children())

    for block in blocks[-n:]:
        for param in block.parameters():
            param.requires_grad = True

    # Always unfreeze the layers just before global pooling
    for attr in ("bn2", "norm", "conv_head", "bn_head", "conv_stem", "bn1"):
        layer = getattr(model.backbone, attr, None)
        if layer is not None and attr not in ("conv_stem", "bn1"):  # keep early layers frozen
            for param in layer.parameters():
                param.requires_grad = True

    for param in model.head.parameters():
        param.requires_grad = True

    _print_trainable(model, f"Last {n} blocks unfrozen")


def _print_trainable(model: FractureClassifier, label: str) -> None:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"[model] {label} — trainable: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")
