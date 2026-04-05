"""
main.py — FastAPI application for the Fracture Detection Web App.

Endpoints:
    GET  /health           → liveness check
    POST /predict          → single-image inference + Grad-CAM
    POST /predict/batch    → multi-image inference
    GET  /model/stats      → metrics from logs/eval_results.json

Start:
    cd api && uvicorn main:app --reload --port 8000
"""

import io
import json
import sys
from pathlib import Path

import torch
from PIL import Image as PILImage

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Resolve project root so imports from config / model work
_API_DIR  = Path(__file__).parent
_ROOT     = _API_DIR.parent
sys.path.insert(0, str(_ROOT))
sys.path.insert(0, str(_API_DIR))

from ml.config import LOG_DIR, DEFAULT_THRESHOLD
import predictor as pred_module
from predictor import FracturePredictor
from schemas import HealthResponse, PredictionResponse


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Fracture Detection API",
    description="EfficientNetV2-M binary classifier for forearm fractures",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_predictor() -> FracturePredictor:
    if pred_module.predictor is None:
        pred_module.predictor = FracturePredictor()
    return pred_module.predictor


def _validate_threshold(threshold: float) -> None:
    if not 0.0 <= threshold <= 1.0:
        raise HTTPException(status_code=422, detail="threshold must be in [0, 1]")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    p = _get_predictor()
    return HealthResponse(status="ok", model=p.model_name)


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    threshold: float = Form(DEFAULT_THRESHOLD),
):
    _validate_threshold(threshold)
    image_bytes = await file.read()
    result      = _get_predictor().predict(image_bytes, threshold=threshold)
    return PredictionResponse(filename=file.filename or "upload", **result)


@app.post("/predict/batch", response_model=list[PredictionResponse])
async def predict_batch(
    files: list[UploadFile] = File(...),
    threshold: float = Form(DEFAULT_THRESHOLD),
):
    _validate_threshold(threshold)

    predictor = _get_predictor()
    filenames: list[str]      = []
    pil_grays: list           = []
    tensors:   list           = []

    for upload in files:
        image_bytes = await upload.read()
        pil_gray    = PILImage.open(io.BytesIO(image_bytes)).convert("L")
        tensor      = predictor.transform(pil_gray)
        filenames.append(upload.filename or "upload")
        pil_grays.append(pil_gray)
        tensors.append(tensor)

    batch = torch.stack(tensors).to(predictor.device)

    with torch.no_grad():
        logits = predictor.model(batch)
        probs  = torch.softmax(logits, dim=1)[:, 1].cpu().tolist()

    responses: list[PredictionResponse] = []
    for i, (fname, pil_gray, prob) in enumerate(zip(filenames, pil_grays, probs)):
        label       = "fracture" if prob >= threshold else "normal"
        gradcam_b64 = predictor._gradcam(batch[i : i + 1], pil_gray)
        responses.append(PredictionResponse(
            filename=fname,
            label=label,
            probability=round(float(prob), 4),
            threshold_used=threshold,
            gradcam_image=gradcam_b64,
        ))

    return responses


_stats_cache: dict | None = None


@app.get("/model/stats")
async def model_stats():
    global _stats_cache
    if _stats_cache is None:
        json_path = LOG_DIR / "eval_results.json"
        if not json_path.exists():
            raise HTTPException(
                status_code=404,
                detail="eval_results.json not found. Run: python evaluate.py --save-json",
            )
        _stats_cache = json.loads(json_path.read_text())
    return _stats_cache
