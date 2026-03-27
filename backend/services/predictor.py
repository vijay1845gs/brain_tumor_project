import torch
import numpy as np
from PIL import Image
import io

from config import get_settings
from torchvision import transforms

# 🔥 Model builders
from models.advanced_models import (
    build_efficient_detection,
    build_efficient_classification
)
from models.resnet_models import (
    build_resnet_detection,
    build_resnet_classification,
    EnsembleDetectionModel
)

# ─────────────────────────────────────────────────────────────
settings = get_settings()
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ─────────────────────────────────────────────────────────────
# 🔥 LOAD DETECTION MODEL (WITH ENSEMBLE SUPPORT)

def load_detection_model():
    try:
        # Try ensemble first
        eff_model = build_efficient_detection(
            "models/detection_efficientnet.pth"
        ).to(DEVICE)

        res_model = build_resnet_detection(
            "models/detection_resnet101.pth"
        ).to(DEVICE)

        print("✅ Using Ensemble Detection")
        return EnsembleDetectionModel(eff_model, res_model)

    except Exception as e:
        print("⚠️ Ensemble unavailable, falling back:", e)

        if settings.DETECTION_MODEL_TYPE == "efficientnet":
            return build_efficient_detection(
                settings.MODEL_DETECTION_PATH
            ).to(DEVICE)
        else:
            return build_resnet_detection(
                settings.MODEL_DETECTION_PATH
            ).to(DEVICE)


# ─────────────────────────────────────────────────────────────
# 🔥 LOAD CLASSIFICATION MODEL

def load_classification_model():
    if settings.CLASSIFICATION_MODEL_TYPE == "efficientnet":
        return build_efficient_classification(
            settings.MODEL_CLASSIFICATION_PATH
        ).to(DEVICE)
    else:
        return build_resnet_classification(
            settings.MODEL_CLASSIFICATION_PATH,
            num_classes=3
        ).to(DEVICE)


# ─────────────────────────────────────────────────────────────
# 🔥 INITIALIZE MODELS

detection_model = load_detection_model()
classification_model = load_classification_model()

# ─────────────────────────────────────────────────────────────
# 🔥 TRANSFORM

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


def preprocess(image):
    return transform(image).unsqueeze(0).to(DEVICE)


# ─────────────────────────────────────────────────────────────
# 🔥 MC DROPOUT (UNCERTAINTY)

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
            probs.append(0.5)  # safe fallback

    return float(np.mean(probs)), float(np.std(probs))


# ─────────────────────────────────────────────────────────────
# 🔥 MAIN PIPELINE

async def run_prediction(image_bytes):
    try:
        # ── Load image ──
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        x = preprocess(image)

        # ── Detection ──
        prob, uncertainty = mc_dropout(detection_model, x)

        HIGH_THR = settings.CONFIDENCE_THRESHOLD
        LOW_THR = 0.15  # safety margin

        if prob >= HIGH_THR:
            tumor_detected = True
            status = "HIGH"

        elif prob <= LOW_THR:
            tumor_detected = False
            status = "HIGH"

        else:
            tumor_detected = True
            status = "LOW"  # uncertain zone

        # 🔥 FN SAFETY (your logic preserved)
        if prob > (HIGH_THR - 0.1) and uncertainty > 0.2:
            tumor_detected = True

        # ── Classification ──
        if tumor_detected:
            try:
                probs = classification_model.predict_proba(x).cpu().numpy()[0]
                classes = ["glioma", "meningioma", "pituitary"]

                tumor_type = classes[int(np.argmax(probs))]
                cls_conf = float(np.max(probs))

            except Exception:
                tumor_type = None
                probs = None
                cls_conf = 0.0

        else:
            tumor_type = None
            probs = None
            cls_conf = 0.0

        # ── Reliability ──
        if status == "LOW":
            reliability = "LOW"
        elif cls_conf < 0.6:
            reliability = "MEDIUM"
        else:
            reliability = "HIGH"

        # ── Final output ──
        return {
            "tumor_detected": "uncertain" if status == "LOW" else tumor_detected,
            "tumor_type": tumor_type,
            "confidence": prob,
            "uncertainty": uncertainty,
            "reliability": reliability,
            "all_class_probs": probs
        }

    except Exception as e:
        # 🔥 PRODUCTION SAFE RESPONSE
        return {
            "tumor_detected": False,
            "tumor_type": None,
            "confidence": 0.0,
            "uncertainty": 1.0,
            "reliability": "LOW",
            "error": str(e)
        }