# config.py — FINAL PRODUCTION VERSION

from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):

    # =========================
    # 🎯 CONFIDENCE SETTINGS
    # =========================
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", 0.4))
    LOW_THRESHOLD: float = float(os.getenv("LOW_THRESHOLD", 0.15))
    CLASSIFICATION_CONF_THRESHOLD: float = float(os.getenv("CLASSIFICATION_CONF_THRESHOLD", 0.6))

    # =========================
    # 📦 MODEL PATHS (PRIMARY — EfficientNet)
    # =========================
    MODEL_EFF_DET_PATH: str = os.getenv(
        "MODEL_EFF_DET_PATH", "models/detection_efficientnet.pth"
    )

    MODEL_EFF_CLS_PATH: str = os.getenv(
        "MODEL_EFF_CLS_PATH", "models/classification_efficientnet.pth"
    )

    # =========================
    # 📦 MODEL PATHS (FALLBACK — ResNet101)
    # =========================
    MODEL_RES_DET_PATH: str = os.getenv(
        "MODEL_RES_DET_PATH", "models/detection_resnet101.pth"
    )

    MODEL_RES_CLS_PATH: str = os.getenv(
        "MODEL_RES_CLS_PATH", "models/classification_resnet101.pth"
    )

    # =========================
    # 🔥 ENSEMBLE CONTROL
    # =========================
    ENSEMBLE_ENABLED: bool = os.getenv("ENSEMBLE_ENABLED", "true").lower() == "true"

    # =========================
    # ⚙️ DEVICE CONTROL
    # =========================
    DEVICE: str = os.getenv("DEVICE", "auto")  # "auto" | "cpu" | "cuda"

    # =========================
    # 🧠 ADVANCED INFERENCE FLAGS
    # =========================
    ENABLE_UNCERTAINTY: bool = os.getenv("ENABLE_UNCERTAINTY", "true").lower() == "true"
    ENABLE_TTA: bool = os.getenv("ENABLE_TTA", "true").lower() == "true"

    # =========================
    # 📁 FILE LIMITS
    # =========================
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", 10))

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings():
    return Settings()