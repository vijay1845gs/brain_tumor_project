# services/predictor.py — FINAL PRODUCTION VERSION

import torch
import numpy as np
from PIL import Image
import io

from torchvision import transforms

from config import get_settings
from services.model_loader import (
    get_detection_model,
    get_classification_model,
    get_device
)

# ─────────────────────────────────────────────
settings = get_settings()
DEVICE = get_device()

# ─────────────────────────────────────────────
# 🔥 LOAD MODELS (SINGLE SOURCE OF TRUTH)

detection_model = get_detection_model()
classification_model = get_classification_model()

# ─────────────────────────────────────────────
# 🔥 TRANSFORM

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def preprocess(image):
    return transform(image).unsqueeze(0).to(DEVICE)


# ─────────────────────────────────────────────
# 🔥 UNCERTAINTY (MC DROPOUT)

def mc_dropout(model, x, passes=5):
    probs = []

    for _ in range(passes):
        model.train()  # enable dropout

        try:
            if hasattr(model, "predict_proba"):
                out = model.predict_proba(x)
            else:
                out = torch.sigmoid(model(x))

            probs.append(out.item())

        except Exception:
            probs.append(0.5)

    return float(np.mean(probs)), float(np.std(probs))


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

        # 🔥 FN SAFETY BOOST
        if prob > (HIGH_THR - 0.1) and uncertainty > 0.2:
            tumor_detected = True

        # ─────────────────────────
        # 🧠 CLASSIFICATION
        # ─────────────────────────
        if tumor_detected:
            try:
                probs = classification_model.predict_proba(x).cpu().numpy()[0]
                classes = ["glioma", "meningioma", "pituitary"]

                max_prob = float(np.max(probs))

                if max_prob < settings.CLASSIFICATION_CONF_THRESHOLD:
                    tumor_type = "uncertain"
                else:
                    tumor_type = classes[int(np.argmax(probs))]

            except Exception:
                tumor_type = None
                probs = None
                max_prob = 0.0
        else:
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
        return {
            "tumor_detected": "uncertain" if status == "LOW" else tumor_detected,
            "tumor_type": tumor_type,
            "confidence": float(prob),
            "uncertainty": float(uncertainty),
            "reliability": reliability,
            "all_class_probs": probs.tolist() if probs is not None else None
        }

    except Exception as e:
        return {
            "tumor_detected": False,
            "tumor_type": None,
            "confidence": 0.0,
            "uncertainty": 1.0,
            "reliability": "LOW",
            "error": str(e)
        }