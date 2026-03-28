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
from services.gradcam import GradCAMPlusPlus, EigenCAM
from services.risk_analysis import get_risk_report

# ─────────────────────────────────────────────
settings = get_settings()
DEVICE = get_device()
logger = logging.getLogger(__name__)

logger.info(f"✅ Predictor initialized on {DEVICE}")


def preprocess(image: Image.Image):
    """
    Preprocess image using CLAHE + skull stripping + ImageNet normalization.
    Returns (tensor, original_np_array).
    """
    tensor, _ = preprocess_for_inference(image, size=224)
    return tensor.to(DEVICE), np.array(image.resize((224, 224)))


# ─────────────────────────────────────────────
# 🔥 UNCERTAINTY (MC DROPOUT)

def mc_dropout(model, x, passes=5):
    probs = []
    try:
        # only set Dropout layers to train, keep BatchNorm in eval
        model.eval()
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()

        for _ in range(passes):
            try:
                with torch.no_grad():
                    if hasattr(model, "predict_proba"):
                        out = model.predict_proba(x)
                    else:
                        out = torch.sigmoid(model(x))
                probs.append(float(out.detach().cpu().flatten()[0]))
            except Exception as e:
                logger.warning("mc_dropout pass failed: " + str(e))
                probs.append(0.5)

        return float(np.mean(probs)), float(np.std(probs))
    finally:
        model.eval()


# ─────────────────────────────────────────────
# 🔥 MAIN PIPELINE

async def run_prediction(image_bytes):
    try:
        # ── Load image ──
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x, original_np = preprocess(image)

        # ─────────────────────────
        # 🧠 DETECTION
        # ─────────────────────────
        detection_model = get_detection_model()
        classification_model = get_classification_model()

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
                raw_probs = classification_model.predict_proba(x).detach().cpu().numpy()[0]
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

        # ─────────────────────────
        # 🗺️ EXPLAINABILITY (Grad-CAM++ + EigenCAM)
        # ─────────────────────────
        heatmap_gradcam = None
        heatmap_eigencam = None

        cam_model = classification_model if tumor_detected else detection_model
        class_idx = None
        if tumor_detected and tumor_type and tumor_type not in ("uncertain", None):
            class_idx = ["glioma", "meningioma", "pituitary"].index(tumor_type)

        try:
            gradcam = GradCAMPlusPlus(cam_model)
            heatmap_gradcam = gradcam.generate(x, original_np, class_idx=class_idx)
            gradcam.remove_hooks()
        except Exception as e:
            logger.warning(f"Grad-CAM++ failed: {e}")

        try:
            eigencam = EigenCAM(cam_model)
            heatmap_eigencam = eigencam.generate(x, original_np, class_idx=class_idx)
            eigencam.remove_hooks()
        except Exception as e:
            logger.warning(f"EigenCAM failed: {e}")

        # Use Grad-CAM++ as primary heatmap, fallback to others if it failed
        heatmap_image = heatmap_gradcam or heatmap_eigencam

        # ─────────────────────────
        # 🏥 RISK ANALYSIS
        # ─────────────────────────
        risk = get_risk_report(
            tumor_type=tumor_type if tumor_detected else None,
            confidence=prob,
            uncertainty=uncertainty,
        )

        return {
            "tumor_detected": bool(tumor_detected),
            "decision_type": decision_type,
            "tumor_type": tumor_type,
            "confidence": round(float(prob), 6),
            "uncertainty": round(float(uncertainty), 6),
            "reliability": reliability,
            "all_class_probs": class_probs_dict,
            "heatmap_image": heatmap_image,
            "heatmap_gradcam": heatmap_gradcam,
            "heatmap_eigencam": heatmap_eigencam,
            "risk_level": risk.risk_level,
            "risk_color": risk.risk_color,
            "clinical_note": risk.clinical_note,
            "recommendation": risk.recommendation,
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
            "heatmap_image": None,
            "heatmap_gradcam": None,
            "heatmap_eigencam": None,
            "risk_level": "None",
            "risk_color": "slate",
            "clinical_note": "Prediction pipeline failed.",
            "recommendation": "Please retry or contact support.",
            "error": str(e),
        }