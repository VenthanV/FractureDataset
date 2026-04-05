"""
schemas.py — Pydantic request/response models for the Fracture Detection API.
"""

from pydantic import BaseModel, field_validator


class PredictionResponse(BaseModel):
    filename: str
    label: str            # "fracture" | "normal"
    probability: float    # P(fracture)
    threshold_used: float
    gradcam_image: str    # "data:image/png;base64,..."


class HealthResponse(BaseModel):
    status: str
    model: str


class ThresholdForm(BaseModel):
    threshold: float = 0.5

    @field_validator("threshold")
    @classmethod
    def threshold_in_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("threshold must be in [0, 1]")
        return v
