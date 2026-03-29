# services/predictor.py — FINAL PRODUCTION VERSION (CLAUDE + CALIBRATION + ENTROPY)

import torch
import numpy as np
from PIL import Image
import io
import logging
import math

from config import get_settings
from services.risk_analysis import get_risk_report
from services.model_loader import (
    get_detection_model,
    get_classification_model,
    get_device
)
from services.preprocessing import preprocess_for_inference
from services.gradcam import GradCAMPlusPlus, EigenCAM
from services.calibration import calibrate_classification_probs

# ─────────────────────────────────────────────
settings = get_settings()
DEVICE = get_device()
logger = logging.getLogger(__name__)

logger.info(f"✅ Predictor initialized on {DEVICE}")


# ─────────────────────────────────────────────
# PREPROCESS
# ─────────────────────────────────────────────
def preprocess(image: Image.Image):
    tensor, _ = preprocess_for_inference(image, size=224)
    return tensor.to(DEVICE), np.array(image.resize((224, 224)))


# ─────────────────────────────────────────────
# MC DROPOUT (DETECTION)
# ─────────────────────────────────────────────
def mc_dropout(model, x, passes=5):
    probs = []
    try:
        model.eval()
        for m in model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()

        for _ in range(passes):
            with torch.no_grad():
                if hasattr(model, "predict_proba"):
                    out = model.predict_proba(x)
                else:
                    out = torch.sigmoid(model(x))

            probs.append(float(out.detach().cpu().flatten()[0]))

        return float(np.mean(probs)), float(np.std(probs))

    finally:
        model.eval()


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
async def run_prediction(image_bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x, original_np = preprocess(image)

        detection_model = get_detection_model()
        classification_model = get_classification_model()

        # ───────── DETECTION ─────────
        prob, uncertainty = mc_dropout(detection_model, x)

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
            status = "LOW"

        # FN safety
        if prob > (HIGH_THR - 0.1) and uncertainty > 0.2:
            tumor_detected = True
            status = "LOW"

        # ───────── CLASSIFICATION (CALIBRATED) ─────────
        tumor_type = None
        probs = None
        max_prob = 0.0
        entropy = None

        if tumor_detected:
            try:
                # RAW LOGITS
                logits = classification_model(x)

                # CALIBRATION
                calibrated_probs = calibrate_classification_probs(
                    logits,
                    cus=uncertainty
                )

                probs = calibrated_probs
                classes = ["glioma", "meningioma", "pituitary"]

                max_prob = float(np.max(calibrated_probs))
                predicted_class = classes[int(np.argmax(calibrated_probs))]

                # 🔥 ENTROPY (NEW)
                entropy_raw = -np.sum(
                    calibrated_probs * np.log(calibrated_probs + 1e-8)
                )
                entropy = float(entropy_raw / math.log(len(classes)))  # normalized

                if max_prob < settings.CLASSIFICATION_CONF_THRESHOLD:
                    tumor_type = "uncertain"
                else:
                    tumor_type = predicted_class

            except Exception as cls_err:
                logger.warning(f"Classification failed: {cls_err}")

        # ───────── RELIABILITY ─────────
        if status == "LOW":
            reliability = "LOW"
        elif max_prob < 0.6:
            reliability = "MEDIUM"
        else:
            reliability = "HIGH"

        # ───────── OUTPUT ─────────
        class_probs_dict = None
        if probs is not None:
            class_labels = ["glioma", "meningioma", "pituitary"]
            class_probs_dict = {
                label: round(float(p), 6)
                for label, p in zip(class_labels, probs)
            }

        if status == "LOW":
            decision_type = "UNCERTAIN"
        elif reliability == "HIGH":
            decision_type = "CONFIDENT"
        else:
            decision_type = "FALLBACK"

        # ───────── EXPLAINABILITY ─────────
        heatmap_gradcam = None
        heatmap_eigencam = None

        cam_model = classification_model if tumor_detected else detection_model
        class_idx = None

        if tumor_detected and tumor_type not in ("uncertain", None):
            class_idx = ["glioma", "meningioma", "pituitary"].index(tumor_type)

        try:
            gradcam = GradCAMPlusPlus(cam_model)
            heatmap_gradcam = gradcam.generate(x, original_np, class_idx=class_idx)
            gradcam.remove_hooks()
        except Exception:
            pass

        try:
            eigencam = EigenCAM(cam_model)
            heatmap_eigencam = eigencam.generate(x, original_np, class_idx=class_idx)
            eigencam.remove_hooks()
        except Exception:
            pass

        heatmap_image = heatmap_gradcam or heatmap_eigencam

        # ───────── RISK ─────────
        risk = get_risk_report(
            tumor_type=tumor_type if tumor_detected else None,
            confidence=prob,
            uncertainty=uncertainty,
        )

        # 🔥 ENTROPY CALCULATION
        prediction_entropy = round(float(entropy), 4) if entropy is not None else None

        # 🔥 UNCERTAINTY PROFILE
        if prediction_entropy is not None:
            aleatoric = prediction_entropy
            epistemic = round(float(uncertainty), 4)
            dominant  = "aleatoric" if aleatoric > epistemic else "epistemic"
            uncertainty_profile = {
                "aleatoric": aleatoric,
                "epistemic": epistemic,
                "dominant":  dominant,
            }
        else:
            uncertainty_profile = None

        return {
            "tumor_detected": bool(tumor_detected),
            "decision_type": decision_type,
            "tumor_type": tumor_type,
            "confidence": round(float(prob), 6),
            "uncertainty": round(float(uncertainty), 6),
            "prediction_entropy": prediction_entropy,
            "uncertainty_profile": uncertainty_profile,

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
            "prediction_entropy": None,
            "uncertainty_profile": None,
            "reliability": "LOW",
            "all_class_probs": None,
            "error": str(e),
        }