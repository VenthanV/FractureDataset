"""
predictor.py — Model singleton, inference pipeline, and Grad-CAM overlay.

Loaded once at FastAPI startup via lifespan. Subsequent requests reuse the
same model instance — no repeated disk I/O or weight initialisation.
"""

import io
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

from config import (
    CHECKPOINT_DIR, MODEL_NAME, IMG_SIZE,
    NORMALIZE_MEAN, NORMALIZE_STD, DEVICE,
)
from model import build_model

try:
    from pytorch_grad_cam import GradCAMPlusPlus
    from pytorch_grad_cam.utils.image import show_cam_on_image
    _GRADCAM_AVAILABLE = True
except ImportError:
    _GRADCAM_AVAILABLE = False


def _get_val_transform() -> transforms.Compose:
    """Same deterministic pipeline used during evaluation — no augmentation."""
    eval_size = int(IMG_SIZE * 256 / 224)
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize(eval_size),
        transforms.CenterCrop(IMG_SIZE),
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

        self.model = build_model(pretrained=False)
        state = torch.load(best_ckpt, map_location=DEVICE)
        self.model.load_state_dict(state)
        self.model = self.model.to(DEVICE)
        self.model.eval()

        self.transform  = _get_val_transform()
        self.device     = DEVICE
        self.model_name = MODEL_NAME

        # Resolve Grad-CAM target layer: last conv block of the backbone
        self._target_layer = None
        if _GRADCAM_AVAILABLE and hasattr(self.model.backbone, "blocks"):
            last_stage = self.model.backbone.blocks[-1]
            # EfficientNetV2: blocks[-1] is a Sequential of sub-blocks → take last
            if hasattr(last_stage, "__getitem__"):
                self._target_layer = [last_stage[-1]]
            else:
                self._target_layer = [last_stage]

        print(f"[predictor] Model loaded from {best_ckpt}  device={DEVICE}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def predict(self, image_bytes: bytes, threshold: float = 0.5) -> dict:
        """
        Run inference on raw image bytes.

        Returns:
            {
                "label":          "fracture" | "normal",
                "probability":    float (P(fracture)),
                "threshold_used": float,
                "gradcam_image":  "data:image/png;base64,...",
            }
        """
        tensor = self._preprocess(image_bytes)              # (1, 3, H, W)

        with torch.no_grad():
            logits = self.model(tensor)
            prob   = float(torch.softmax(logits, dim=1)[0, 1].cpu())

        label        = "fracture" if prob >= threshold else "normal"
        gradcam_b64  = self._gradcam(image_bytes)

        return {
            "label":          label,
            "probability":    round(prob, 4),
            "threshold_used": threshold,
            "gradcam_image":  gradcam_b64,
        }

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _preprocess(self, image_bytes: bytes) -> torch.Tensor:
        """Decode bytes → PIL → val-transform → (1, 3, H, W) tensor on device."""
        image = Image.open(io.BytesIO(image_bytes)).convert("L")
        tensor = self.transform(image).unsqueeze(0).to(self.device)
        return tensor

    def _gradcam(self, image_bytes: bytes) -> str:
        """
        Generate a Grad-CAM++ heatmap overlay and return it as a base64 PNG.

        If pytorch_grad_cam is not installed or no target layer is available,
        returns an empty-string placeholder so the rest of the API still works.
        """
        if not _GRADCAM_AVAILABLE or self._target_layer is None:
            return ""

        try:
            return self._run_gradcam(image_bytes)
        except Exception as e:
            import traceback
            print(f"[predictor] Grad-CAM failed: {e}")
            traceback.print_exc()
            return ""

    def _run_gradcam(self, image_bytes: bytes) -> str:
        tensor = self._preprocess(image_bytes)  # (1, 3, H, W)

        # Build a numpy RGB image for the overlay (unnormalised, 0-1 range)
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        pil_img = pil_img.resize((IMG_SIZE, IMG_SIZE))
        rgb_img = np.array(pil_img, dtype=np.float32) / 255.0

        # GradCAM++ — target class 1 = fracture
        with GradCAMPlusPlus(
            model=self.model,
            target_layers=self._target_layer,
        ) as cam:
            targets = None  # uses highest-scoring class by default
            grayscale_cam = cam(input_tensor=tensor, targets=targets)
            grayscale_cam = grayscale_cam[0]          # (H, W)

        # Blend heatmap over the original image
        overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

        # Encode as base64 PNG
        buf = io.BytesIO()
        Image.fromarray(overlay).save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{b64}"


# Module-level singleton — populated in FastAPI lifespan
predictor: FracturePredictor | None = None
