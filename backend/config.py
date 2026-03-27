from pydantic_settings import BaseSettings
from functools import lru_cache
import os


class Settings(BaseSettings):
    # ─────────────────────────────────────────────
    # 🎯 CONFIDENCE SETTINGS (aligned with your model)
    # ─────────────────────────────────────────────
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", 0.4))
    LOW_THRESHOLD: float = float(os.getenv("LOW_THRESHOLD", 0.15))
    CLASSIFICATION_CONF_THRESHOLD: float = float(os.getenv("CLASSIFICATION_CONF_THRESHOLD", 0.6))

    # ─────────────────────────────────────────────
    # 🤖 MODEL TYPE (NEW — important)
    # ─────────────────────────────────────────────
    DETECTION_MODEL_TYPE: str = os.getenv("DETECTION_MODEL_TYPE", "efficientnet")
    CLASSIFICATION_MODEL_TYPE: str = os.getenv("CLASSIFICATION_MODEL_TYPE", "efficientnet")

    # ─────────────────────────────────────────────
    # 📦 MODEL PATHS (must match training output)
    # ─────────────────────────────────────────────
    MODEL_DETECTION_PATH: str = os.getenv(
        "MODEL_DETECTION_PATH", "models/detection_efficientnet.pth"
    )

    MODEL_CLASSIFICATION_PATH: str = os.getenv(
        "MODEL_CLASSIFICATION_PATH", "models/classification_efficientnet.pth"
    )

    # ─────────────────────────────────────────────
    # 🔥 ENSEMBLE (optional but recommended)
    # ─────────────────────────────────────────────
    ENSEMBLE_ENABLED: bool = os.getenv("ENSEMBLE_ENABLED", "true").lower() == "true"

    ENSEMBLE_RESNET_PATH: str = os.getenv(
        "ENSEMBLE_RESNET_PATH", "models/detection_resnet101.pth"
    )


@lru_cache()
def get_settings():
    return Settings()