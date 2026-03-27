# predict.py — Clean Production API Layer

from fastapi import APIRouter, UploadFile, File, HTTPException
from pydantic import BaseModel
from typing import Optional, List, Union

from services.predictor import run_prediction

router = APIRouter(prefix="/predict", tags=["Prediction"])


# ─────────────────────────────────────────────
# RESPONSE MODEL
# ─────────────────────────────────────────────
class PredictionResponse(BaseModel):
    tumor_detected: Union[bool, str]   # supports "uncertain"
    tumor_type: Optional[str]
    confidence: float
    uncertainty: float
    reliability: str
    all_class_probs: Optional[List[float]] = None


# ─────────────────────────────────────────────
# ROUTE
# ─────────────────────────────────────────────
@router.post("/", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        # Validate file
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image.")

        image_bytes = await file.read()

        result = await run_prediction(image_bytes)

        return PredictionResponse(**result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))