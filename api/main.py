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

import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Resolve project root so imports from config / model work
_API_DIR  = Path(__file__).parent
_ROOT     = _API_DIR.parent
sys.path.insert(0, str(_ROOT))

from config import LOG_DIR
import predictor as pred_module
from predictor import FracturePredictor
from schemas import HealthResponse, PredictionResponse


# ── Lifespan: load model once at startup ──────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    pred_module.predictor = FracturePredictor()
    yield
    pred_module.predictor = None


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Fracture Detection API",
    description="EfficientNetV2-M binary classifier for forearm fractures",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _get_predictor() -> FracturePredictor:
    if pred_module.predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return pred_module.predictor


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    p = _get_predictor()
    return HealthResponse(status="ok", model=p.model_name)


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    threshold: float = Form(0.5),
):
    if not 0.0 <= threshold <= 1.0:
        raise HTTPException(status_code=422, detail="threshold must be in [0, 1]")

    image_bytes = await file.read()
    result      = _get_predictor().predict(image_bytes, threshold=threshold)

    return PredictionResponse(
        filename=file.filename or "upload",
        **result,
    )


@app.post("/predict/batch", response_model=List[PredictionResponse])
async def predict_batch(
    files: List[UploadFile] = File(...),
    threshold: float = Form(0.5),
):
    if not 0.0 <= threshold <= 1.0:
        raise HTTPException(status_code=422, detail="threshold must be in [0, 1]")

    predictor = _get_predictor()
    responses: List[PredictionResponse] = []

    for upload in files:
        image_bytes = await upload.read()
        result      = predictor.predict(image_bytes, threshold=threshold)
        responses.append(PredictionResponse(
            filename=upload.filename or "upload",
            **result,
        ))

    return responses


@app.get("/model/stats")
async def model_stats():
    json_path = LOG_DIR / "eval_results.json"
    if not json_path.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "eval_results.json not found. "
                "Run: python evaluate.py --save-json"
            ),
        )
    return json.loads(json_path.read_text())
