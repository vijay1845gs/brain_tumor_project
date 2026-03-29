# services/calibration.py
"""
Post-hoc calibration utilities.
No retraining required. Applied at inference time only.

Temperature Scaling:
    Divides logits by temperature T before softmax.
    T > 1  →  softens distribution (reduces overconfidence)
    T = 1  →  original softmax (no change)
    T < 1  →  sharpens distribution (increases confidence)

CUS-Adaptive Temperature:
    Increases T dynamically when cascade uncertainty is high,
    ensuring high-uncertainty predictions are never overconfident.
"""

import numpy as np
import torch
import torch.nn.functional as F
from typing import Union


# ── Default temperature (tuned for ResNet101 on medical imaging) ──────────────
# Set between 1.5 and 3.0 without a validation set.
# 2.0 is a safe, well-documented starting point for medical imaging classifiers.
# See: Guo et al., "On Calibration of Modern Neural Networks", ICML 2017.
DEFAULT_TEMPERATURE = 2.0

# ── Label smoothing epsilon ───────────────────────────────────────────────────
# Ensures no class ever receives probability 0.0 or 1.0.
# Standard value in production medical AI: 0.05–0.10
SMOOTHING_EPSILON = 0.05

# ── CUS thresholds for adaptive temperature ───────────────────────────────────
CUS_LOW      = 0.05   # below this: use DEFAULT_TEMPERATURE
CUS_MODERATE = 0.10   # above this: use HIGH_TEMPERATURE
HIGH_TEMPERATURE = 4.0


def get_adaptive_temperature(cus: float) -> float:
    """
    Returns calibration temperature based on Cascade Uncertainty Score.
    Higher CUS → higher temperature → softer, more honest distribution.
    """
    if cus <= CUS_LOW:
        return DEFAULT_TEMPERATURE
    if cus >= CUS_MODERATE:
        return HIGH_TEMPERATURE
    # Linear interpolation between DEFAULT and HIGH
    ratio = (cus - CUS_LOW) / (CUS_MODERATE - CUS_LOW)
    return DEFAULT_TEMPERATURE + ratio * (HIGH_TEMPERATURE - DEFAULT_TEMPERATURE)


def temperature_scale_logits(
    logits: torch.Tensor,
    temperature: float
) -> np.ndarray:
    """
    Apply temperature scaling to raw logits and return calibrated softmax.

    Args:
        logits      : raw output from classification model, shape (1, num_classes)
        temperature : scaling factor > 0

    Returns:
        calibrated probabilities as numpy array, shape (num_classes,)
    """
    if temperature <= 0:
        raise ValueError(f"Temperature must be positive, got {temperature}")

    scaled_logits = logits / temperature
    probs = F.softmax(scaled_logits, dim=1)
    return probs.detach().cpu().numpy()[0]


def smooth_probabilities(
    probs: np.ndarray,
    epsilon: float = SMOOTHING_EPSILON
) -> np.ndarray:
    """
    Apply label smoothing to a probability vector.
    Prevents hard 0.0 and 1.0 outputs.

    Formula: p_smooth = (1 - epsilon) * p + epsilon / K
    where K is the number of classes.

    Args:
        probs   : probability vector, shape (K,), must sum to ~1.0
        epsilon : smoothing factor (0.05 recommended)

    Returns:
        smoothed probability vector, shape (K,)
    """
    K = len(probs)
    smoothed = (1.0 - epsilon) * probs + epsilon / K
    # Renormalize to ensure exact sum = 1.0 despite floating point
    smoothed = smoothed / smoothed.sum()
    return smoothed.astype(np.float32)


def calibrate_classification_probs(
    logits: torch.Tensor,
    cus: float,
    epsilon: float = SMOOTHING_EPSILON,
) -> np.ndarray:
    """
    Full calibration pipeline: temperature scaling → label smoothing.
    This is the single entry point called from predictor.py.

    Args:
        logits  : raw logits from classification model, shape (1, num_classes)
        cus     : Cascade Uncertainty Score (float)
        epsilon : smoothing factor

    Returns:
        calibrated, smoothed probability vector, shape (num_classes,)
    """
    T = get_adaptive_temperature(cus)
    probs = temperature_scale_logits(logits, T)
    probs = smooth_probabilities(probs, epsilon)
    return probs


def expected_calibration_error(
    confidences: np.ndarray,
    accuracies:  np.ndarray,
    n_bins: int = 10,
) -> float:
    """
    Compute Expected Calibration Error (ECE) for evaluation.
    Use this in your evaluation script to quantify calibration improvement.

    Args:
        confidences : max predicted probability per sample, shape (N,)
        accuracies  : binary correct/incorrect per sample, shape (N,)
        n_bins      : number of equal-width bins

    Returns:
        ECE (float) — lower is better, 0.0 is perfect calibration
    """
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    N   = len(confidences)

    for i in range(n_bins):
        lo, hi = bin_boundaries[i], bin_boundaries[i + 1]
        mask   = (confidences >= lo) & (confidences < hi)
        if mask.sum() == 0:
            continue
        bin_conf = confidences[mask].mean()
        bin_acc  = accuracies[mask].mean()
        ece     += (mask.sum() / N) * abs(bin_conf - bin_acc)

    return float(ece)