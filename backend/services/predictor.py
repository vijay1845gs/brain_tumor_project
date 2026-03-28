# services/predictor.py — FINAL PRODUCTION VERSION

import torch
import numpy as np
from PIL import Image
import io
import logging

from config import get_settings
from services.model_loader import (
    get_detection_model,
    get_classification_model,
    get_device
)
from services.preprocessing import preprocess_for_inference

# ─────────────────────────────────────────────
settings = get_settings()
DEVICE = get_device()
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# 🔥 LOAD MODELS (SINGLE SOURCE OF TRUTH)

detection_model = get_detection_model()
classification_model = get_classification_model()

logger.info(f"✅ Predictor initialized on {DEVICE}")


def preprocess(image: Image.Image):
    """
    Preprocess image using CLAHE + skull stripping + ImageNet normalization.
    Returns tensor on the correct device.
    """
    tensor, _ = preprocess_for_inference(image, size=224)
    return tensor.to(DEVICE)


# ─────────────────────────────────────────────
# 🔥 UNCERTAINTY (MC DROPOUT)

def mc_dropout(model, x, passes=5):
    """
    Monte Carlo Dropout inference.
    Enables dropout for stochastic passes, then restores eval mode.
    """
    probs = []
    was_training = model.training

    try:
        model.train()  # enable dropout for stochastic forward passes

        for _ in range(passes):
            try:
                with torch.no_grad():
                    if hasattr(model, "predict_proba"):
                        out = model.predict_proba(x)
                    else:
                        out = torch.sigmoid(model(x))

                probs.append(out.item())

            except Exception:
                probs.append(0.5)

        return float(np.mean(probs)), float(np.std(probs))

    finally:
        # 🔥 SAFETY: Always restore original mode
        if not was_training:
            model.eval()


# ─────────────────────────────────────────────
# 🔥 MAIN PIPELINE

async def run_prediction(image_bytes):
    try:
        # ── Load image ──
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x = preprocess(image)

        # ─────────────────────────
        # 🧠 DETECTION
        # ─────────────────────────
        prob, uncertainty = mc_dropout(detection_model, x)
        logger.info(f"Detection: prob={prob:.4f}, uncertainty={uncertainty:.4f}")

        HIGH_THR = settings.CONFIDENCE_THRESHOLD
        LOW_THR = settings.LOW_THRESHOLD

        if prob >= HIGH_THR:
            tumor_detected = True
            status = "HIGH"

        elif prob <= LOW_THR:
            tumor_detected = False
            status = "HIGH"

        else:
            tumor_detected = True
            status = "LOW"  # uncertain zone

        # 🔥 FN SAFETY BOOST: near threshold with high uncertainty → flag for review
        if prob > (HIGH_THR - 0.1) and uncertainty > 0.2:
            tumor_detected = True
            status = "LOW"
            logger.info("FN safety boost triggered")

        # ─────────────────────────
        # 🧠 CLASSIFICATION
        # ─────────────────────────
        tumor_type = None
        probs = None
        max_prob = 0.0

        if tumor_detected:
            try:
                raw_probs = classification_model.predict_proba(x).cpu().numpy()[0]
                probs = raw_probs
                classes = ["glioma", "meningioma", "pituitary"]

                max_prob = float(np.max(raw_probs))
                predicted_class = classes[int(np.argmax(raw_probs))]

                logger.info(f"Classification: max_prob={max_prob:.4f}, predicted={predicted_class}")

                # 🔥 SAFETY: If classification confidence is too low, reject to no_tumor
                # This prevents forced classification on ambiguous cases
                if max_prob < settings.CLASSIFICATION_CONF_THRESHOLD:
                    # Keep tumor_detected=True (detection was positive)
                    # But mark type as uncertain so clinician reviews
                    tumor_type = "uncertain"
                    logger.info("Classification confidence below threshold → marked uncertain")
                else:
                    tumor_type = predicted_class

            except Exception as cls_err:
                logger.warning(f"Classification failed: {cls_err}")
                tumor_type = None
                probs = None
                max_prob = 0.0

        # ─────────────────────────
        # 📊 RELIABILITY SCORE
        # ─────────────────────────
        if status == "LOW":
            reliability = "LOW"
        elif max_prob < 0.6:
            reliability = "MEDIUM"
        else:
            reliability = "HIGH"

        # ─────────────────────────
        # 📦 FINAL RESPONSE
        # ─────────────────────────
        # Build labeled class probabilities dictionary
        class_probs_dict = None
        if probs is not None:
            class_labels = ["glioma", "meningioma", "pituitary"]
            class_probs_dict = {
                label: round(float(p), 6) for label, p in zip(class_labels, probs)
            }

        # Determine decision type for frontend clarity
        if status == "LOW":
            decision_type = "UNCERTAIN"
        elif reliability == "HIGH":
            decision_type = "CONFIDENT"
        else:
            decision_type = "FALLBACK"

        return {
            "tumor_detected": bool(tumor_detected),
            "decision_type": decision_type,
            "tumor_type": tumor_type,
            "confidence": round(float(prob), 6),
            "uncertainty": round(float(uncertainty), 6),
            "reliability": reliability,
            "all_class_probs": class_probs_dict,
        }

    except Exception as e:
        logger.exception("Prediction pipeline failed")
        return {
            "tumor_detected": False,
            "decision_type": "ERROR",
            "tumor_type": None,
            "confidence": 0.0,
            "uncertainty": 1.0,
            "reliability": "LOW",
            "all_class_probs": None,
            "error": str(e),
        }