# services/model_loader.py — FINAL PRODUCTION VERSION

import os
import threading
import logging
import torch

from config import get_settings

from models.resnet_models import (
    build_resnet_detection,
    build_resnet_classification,
)

from models.advanced_models import (
    build_efficient_detection,
    build_efficient_classification,
    EnsembleDetectionModel,
    EnsembleClassificationModel,
)

# ─────────────────────────────────────────────
settings = get_settings()
logger = logging.getLogger(__name__)

_lock = threading.Lock()
_detection_model = None
_classification_model = None
_models_loaded = False


# ─────────────────────────────────────────────
# DEVICE CONTROL
# ─────────────────────────────────────────────
def _get_device():
    if settings.DEVICE == "cpu":
        return "cpu"
    elif settings.DEVICE == "cuda":
        return "cuda"
    else:
        return "cuda" if torch.cuda.is_available() else "cpu"


_device = _get_device()


# ─────────────────────────────────────────────
# LOAD MODELS
# ─────────────────────────────────────────────
def _load_models():
    global _detection_model, _classification_model, _models_loaded

    logger.info("🔄 Loading models...")

    eff_det_path = settings.MODEL_EFF_DET_PATH
    eff_cls_path = settings.MODEL_EFF_CLS_PATH

    res_det_path = settings.MODEL_RES_DET_PATH
    res_cls_path = settings.MODEL_RES_CLS_PATH

    # =========================
    # DETECTION
    # =========================
    try:
        if settings.ENSEMBLE_ENABLED and os.path.exists(eff_det_path) and os.path.exists(res_det_path):
            logger.info("🚀 ENSEMBLE Detection (EfficientNet + ResNet)")

            eff_model = build_efficient_detection(eff_det_path, _device)
            res_model = build_resnet_detection(res_det_path, _device)

            _detection_model = EnsembleDetectionModel(eff_model, res_model)

        elif os.path.exists(eff_det_path):
            logger.info(f"🚀 EfficientNet Detection: {eff_det_path}")
            _detection_model = build_efficient_detection(eff_det_path, _device)

        elif os.path.exists(res_det_path):
            logger.info(f"⚙️ ResNet Detection (fallback): {res_det_path}")
            _detection_model = build_resnet_detection(res_det_path, _device)

        else:
            logger.warning("⚠️ No detection weights found → using pretrained EfficientNet")
            _detection_model = build_efficient_detection(None, _device)

    except Exception as e:
        logger.exception("❌ Detection model failed")
        raise RuntimeError(f"Detection model error: {e}")

    # =========================
    # CLASSIFICATION
    # =========================
    try:
        if settings.ENSEMBLE_ENABLED and os.path.exists(eff_cls_path) and os.path.exists(res_cls_path):
            logger.info("🚀 ENSEMBLE Classification")

            eff_model = build_efficient_classification(eff_cls_path, _device)
            res_model = build_resnet_classification(res_cls_path, _device)

            _classification_model = EnsembleClassificationModel(eff_model, res_model)

        elif os.path.exists(eff_cls_path):
            logger.info(f"🚀 EfficientNet Classification: {eff_cls_path}")
            _classification_model = build_efficient_classification(eff_cls_path, _device)

        elif os.path.exists(res_cls_path):
            logger.info(f"⚙️ ResNet Classification (fallback): {res_cls_path}")
            _classification_model = build_resnet_classification(res_cls_path, _device)

        else:
            logger.warning("⚠️ No classification weights found → using pretrained EfficientNet")
            _classification_model = build_efficient_classification(None, _device)

    except Exception as e:
        logger.exception("❌ Classification model failed")
        raise RuntimeError(f"Classification model error: {e}")

    _models_loaded = True
    logger.info(f"✅ Models loaded on {_device}")


# ─────────────────────────────────────────────
# PUBLIC ACCESS
# ─────────────────────────────────────────────
def get_detection_model():
    global _detection_model

    if _detection_model is None:
        with _lock:
            if _detection_model is None:
                _load_models()

    return _detection_model


def get_classification_model():
    global _classification_model

    if _classification_model is None:
        with _lock:
            if _classification_model is None:
                _load_models()

    return _classification_model


def get_device():
    return _device


def models_loaded() -> bool:
    """Check if models have been successfully loaded."""
    return _models_loaded


# ─────────────────────────────────────────────
# RELOAD SUPPORT
# ─────────────────────────────────────────────
def reload_models():
    global _detection_model, _classification_model, _models_loaded

    with _lock:
        logger.info("🔄 Reloading models...")
        _detection_model = None
        _classification_model = None
        _models_loaded = False
        _load_models()

    logger.info("✅ Reload complete")