"""
schemas.py — Pydantic request/response models for the Fracture Detection API.
"""

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    filename: str
    label: str            # "fracture" | "normal"
    probability: float    # P(fracture)
    threshold_used: float
    gradcam_image: str    # "data:image/png;base64,..."


class HealthResponse(BaseModel):
    status: str
    model: str
