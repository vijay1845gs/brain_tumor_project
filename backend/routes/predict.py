# predict.py — Clean Production API Layer

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional, Dict, Any
import logging

from config import get_settings
from services.predictor import run_prediction

router = APIRouter(prefix="/predict", tags=["Prediction"])
logger = logging.getLogger(__name__)
settings = get_settings()


# ─────────────────────────────────────────────
# RESPONSE MODEL
# ─────────────────────────────────────────────
class UncertaintyProfile(BaseModel):
    aleatoric: float
    epistemic: float
    dominant: str


class PredictionResponse(BaseModel):
    tumor_detected: bool
    decision_type: str                  # CONFIDENT | UNCERTAIN | FALLBACK | ERROR
    tumor_type: Optional[str] = None
    confidence: float
    uncertainty: float
    prediction_entropy: Optional[float] = None
    uncertainty_profile: Optional[UncertaintyProfile] = None
    reliability: str
    all_class_probs: Optional[Dict[str, float]] = None
    heatmap_image: Optional[str] = None
    heatmap_gradcam: Optional[str] = None
    heatmap_eigencam: Optional[str] = None
    risk_level: Optional[str] = None
    risk_color: Optional[str] = None
    clinical_note: Optional[str] = None
    recommendation: Optional[str] = None
    error: Optional[str] = None


# ─────────────────────────────────────────────
# ROUTE
# ─────────────────────────────────────────────
@router.post("/", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # Validate content type
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(
            status_code=400,
            detail="File must be an image (JPEG, PNG, etc.)."
        )

    # Read and validate file size
    image_bytes = await file.read()
    max_bytes = settings.MAX_FILE_SIZE_MB * 1024 * 1024

    if len(image_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE_MB} MB."
        )

    if len(image_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail="Empty file uploaded."
        )

    logger.info(f"Processing image: {file.filename}, size={len(image_bytes)} bytes")

    result = await run_prediction(image_bytes)

    # If the prediction pipeline itself failed, still return 200 with error field
    # (frontend can display the error gracefully)
    return PredictionResponse(**result)