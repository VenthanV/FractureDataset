"""
predictor.py — Model singleton, inference pipeline, and Grad-CAM overlay.

Loaded once at FastAPI startup via lifespan. Subsequent requests reuse the
same model instance — no repeated disk I/O or weight initialisation.
"""

import io
import json
import sys
import base64
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# Add parent directory to sys.path so we can import from the project root
_API_DIR = Path(__file__).parent
_ROOT    = _API_DIR.parent
sys.path.insert(0, str(_ROOT))

import cv2

from ml.config import (
    CHECKPOINT_DIR, MODEL_NAME, IMG_SIZE,
    NORMALIZE_MEAN, NORMALIZE_STD, DEVICE,
    EVAL_RESIZE_RATIO, GRADCAM_IMAGE_WEIGHT,
)
from ml.model import build_model

try:
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.image import show_cam_on_image
    _GRADCAM_AVAILABLE = True
except ImportError:
    _GRADCAM_AVAILABLE = False


def _resolve_checkpoint_meta(ckpt_path: Path) -> tuple[str, int, dict]:
    """
    Load checkpoint and resolve model_name + img_size.

    Priority:
    1. Metadata embedded in checkpoint dict (new format from train.py)
    2. Sidecar JSON next to the .pth file  (best_model_config.json)
    3. config.py defaults (MODEL_NAME, IMG_SIZE)

    Returns (model_name, img_size, state_dict).
    """
    raw = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)

    if isinstance(raw, dict) and "state_dict" in raw:
        # New format: {"state_dict": ..., "model_name": ..., "img_size": ...}
        state      = raw["state_dict"]
        model_name = raw.get("model_name", MODEL_NAME)
        img_size   = int(raw.get("img_size", IMG_SIZE))
    else:
        # Legacy format: plain state_dict
        state = raw
        sidecar = ckpt_path.with_name(ckpt_path.stem + "_config.json")
        if sidecar.exists():
            meta       = json.loads(sidecar.read_text())
            model_name = meta.get("model_name", MODEL_NAME)
            img_size   = int(meta.get("img_size", IMG_SIZE))
        else:
            model_name = MODEL_NAME
            img_size   = IMG_SIZE

    return model_name, img_size, state


def _get_val_transform(img_size: int) -> transforms.Compose:
    """Same deterministic pipeline used during evaluation — no augmentation."""
    eval_size = int(img_size * EVAL_RESIZE_RATIO)
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(eval_size),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
    ])


class FracturePredictor:
    """
    Singleton wrapper around the trained FractureClassifier.

    Handles:
    - Checkpoint loading (once at startup)
    - Preprocessing (same val-transform as training)
    - Inference with configurable threshold
    - Grad-CAM++ overlay generation
    """

    def __init__(self) -> None:
        best_ckpt = CHECKPOINT_DIR / "best_model.pth"
        if not best_ckpt.exists():
            import os
            from huggingface_hub import hf_hub_download
            hf_repo = os.getenv("HF_REPO_ID", "VenthanVi/fracture-detection")
            print(f"[predictor] Checkpoint not found locally — downloading from {hf_repo}")
            CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
            hf_hub_download(
                repo_id=hf_repo,
                filename="best_model.pth",
                local_dir=str(CHECKPOINT_DIR),
            )

        model_name, img_size, state = _resolve_checkpoint_meta(best_ckpt)
        print(f"[predictor] architecture={model_name}  img_size={img_size}")

        self.model = build_model(pretrained=False, model_name=model_name)
        self.model.load_state_dict(state)
        self.model = self.model.to(DEVICE)
        self.model.eval()

        self.transform  = _get_val_transform(img_size)
        self.device     = DEVICE
        self.model_name = model_name
        self.img_size   = img_size

        # Resolve Grad-CAM target layer: last conv block of the backbone
        self._target_layer = None
        print(f"[predictor] grad-cam available: {_GRADCAM_AVAILABLE}")
        print(f"[predictor] backbone type: {type(self.model.backbone).__name__}")
        print(f"[predictor] has blocks: {hasattr(self.model.backbone, 'blocks')}")
        if _GRADCAM_AVAILABLE and hasattr(self.model.backbone, "blocks"):
            last_stage = self.model.backbone.blocks[-1]
            print(f"[predictor] last_stage type: {type(last_stage).__name__}")
            if hasattr(last_stage, "__getitem__"):
                self._target_layer = [last_stage[-1]]
            else:
                self._target_layer = [last_stage]
        print(f"[predictor] target_layer: {self._target_layer}")

        print(f"[predictor] Model loaded from {best_ckpt}  device={DEVICE}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, image_bytes: bytes, threshold: float = 0.5, n_tta: int = 1) -> dict:
        """
        Run inference on raw image bytes.

        Args:
            n_tta: Number of Test-Time Augmentation views (1 = off, 5 = recommended).
                   TTA averages probabilities over horizontally flipped and rotated
                   views. Improves stability ~1–2% AUC at ~N× inference cost.

        Returns:
            {
                "label":          "fracture" | "normal",
                "probability":    float (P(fracture)),
                "threshold_used": float,
                "gradcam_image":  "data:image/png;base64,...",
            }
        """
        pil_gray = Image.open(io.BytesIO(image_bytes)).convert("L")
        tensor   = self.transform(pil_gray).unsqueeze(0).to(self.device)  # (1, 3, H, W)

        with torch.no_grad():
            if n_tta > 1:
                from ml.dataloader import get_tta_transforms
                tta_tfs  = get_tta_transforms(img_size=self.img_size, n_views=n_tta)
                views    = torch.stack([t(pil_gray) for t in tta_tfs]).to(self.device)
                logits   = self.model(views)
                prob     = float(torch.softmax(logits, dim=1)[:, 1].mean().cpu())
            else:
                logits = self.model(tensor)
                prob   = float(torch.softmax(logits, dim=1)[0, 1].cpu())

        label       = "fracture" if prob >= threshold else "normal"
        gradcam_b64 = self._gradcam(tensor, pil_gray)  # always from standard transform

        return {
            "label":          label,
            "probability":    round(prob, 4),
            "threshold_used": threshold,
            "gradcam_image":  gradcam_b64,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _gradcam(self, tensor: torch.Tensor, pil_gray: Image.Image) -> str:
        """
        Generate a Grad-CAM++ heatmap overlay and return it as a base64 PNG.

        Accepts the already-computed input tensor and decoded PIL image to
        avoid re-decoding the original bytes.

        If pytorch_grad_cam is not installed or no target layer is available,
        returns an empty-string placeholder so the rest of the API still works.
        """
        if not _GRADCAM_AVAILABLE or self._target_layer is None:
            return ""

        try:
            return self._run_gradcam(tensor, pil_gray)
        except Exception as e:
            import traceback
            print(f"[predictor] Grad-CAM failed: {e}")
            traceback.print_exc()
            return ""

    def _run_gradcam(self, tensor: torch.Tensor, pil_gray: Image.Image) -> str:
        orig_w, orig_h = pil_gray.size  # original image dimensions

        # GradCAM++ runs at model input size (self.img_size × self.img_size)
        with GradCAMPlusPlus(
            model=self.model,
            target_layers=self._target_layer,
        ) as cam:
            targets = None  # uses highest-scoring class by default
            grayscale_cam = cam(input_tensor=tensor, targets=targets)
            grayscale_cam = grayscale_cam[0]          # (img_size, img_size)

        # Resize cam back to original image dimensions so proportions match
        cam_resized = np.array(
            Image.fromarray((grayscale_cam * 255).astype(np.uint8)).resize(
                (orig_w, orig_h), Image.BILINEAR
            ),
            dtype=np.float32,
        ) / 255.0

        # Original image as float RGB for the overlay
        rgb_img = np.array(pil_gray.convert("RGB"), dtype=np.float32) / 255.0

        # COLORMAP_INFERNO: warm red/yellow tones — perceptually correct for medical images
        overlay = show_cam_on_image(
            rgb_img, cam_resized,
            use_rgb=True,
            colormap=cv2.COLORMAP_INFERNO,
            image_weight=GRADCAM_IMAGE_WEIGHT,
        )

        # Encode as base64 PNG
        buf = io.BytesIO()
        Image.fromarray(overlay).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"


# Module-level singleton — populated in FastAPI lifespan
predictor: FracturePredictor | None = None
