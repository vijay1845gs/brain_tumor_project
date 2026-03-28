"""
services/gradcam.py  (UPGRADED v2)
────────────────────────────────────
Multi-method explainability engine:

1. Grad-CAM++   — gradient-weighted class activation maps (fast)
2. Score-CAM    — perturbation-based, gradient-free (more faithful)
3. EigenCAM     — uses PCA on feature maps (robust to gradient instability)

Produces dual-view outputs: heatmap overlay AND side-by-side comparison strip.

References:
  - Grad-CAM++: Chattopadhyay et al., WACV 2018
  - Score-CAM: Wang et al., CVPR 2020 Workshop
  - EigenCAM: Muhammad & Yeasin, 2020
"""

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import base64
from io import BytesIO
from PIL import Image
from typing import Optional


class GradCAMPlusPlus:
    """
    Grad-CAM++ with improved alpha weighting for more precise localisation.
    """

    def __init__(self, model: torch.nn.Module, target_layer: str = None):
        self.model = model
        self.model.eval()
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._hooks: list = []

        # Auto-detect correct target layer if not specified
        if target_layer is None:
            target_layer = self._detect_target_layer()

        self._register_hooks(target_layer)

    def _detect_target_layer(self) -> str:
        """Detect the correct target layer based on model architecture."""
        # EfficientNet uses backbone.features
        if hasattr(self.model, "backbone") and hasattr(self.model.backbone, "features"):
            return "backbone.features"
        # ResNet uses backbone.layer4
        if hasattr(self.model, "backbone") and hasattr(self.model.backbone, "layer4"):
            return "backbone.layer4"
        # Ensemble: try model_a first
        if hasattr(self.model, "model_a"):
            if hasattr(self.model.model_a, "backbone"):
                if hasattr(self.model.model_a.backbone, "features"):
                    return "model_a.backbone.features"
                if hasattr(self.model.model_a.backbone, "layer4"):
                    return "model_a.backbone.layer4"
        # Fallback
        return "backbone.layer4"

    def _get_layer(self, layer_name: str) -> torch.nn.Module:
        layer = self.model
        for part in layer_name.split("."):
            layer = getattr(layer, part)
        return layer

    def _register_hooks(self, target_layer: str):
        layer = self._get_layer(target_layer)

        def forward_hook(module, input, output):
            self._activations = output.detach()

        def backward_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach()

        self._hooks.append(layer.register_forward_hook(forward_hook))
        self._hooks.append(layer.register_full_backward_hook(backward_hook))

    def remove_hooks(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def _compute_cam(self, class_idx: Optional[int] = None) -> np.ndarray:
        grads = self._gradients        # (1, C, H, W)
        acts  = self._activations      # (1, C, H, W)

        grads_sq  = grads ** 2
        grads_cu  = grads ** 3
        sum_acts  = acts.sum(dim=(2, 3), keepdim=True)
        denom     = 2 * grads_sq + sum_acts * grads_cu
        denom     = torch.where(denom != 0, denom, torch.ones_like(denom))
        alpha     = grads_sq / denom
        weights   = (alpha * F.relu(grads)).sum(dim=(2, 3), keepdim=True)

        cam = (weights * acts).sum(dim=1).squeeze(0)
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()
        return cam

    def generate(
        self,
        input_tensor: torch.Tensor,
        original_image: np.ndarray,
        class_idx: Optional[int] = None,
    ) -> str:
        input_tensor = input_tensor.clone().requires_grad_(True)
        output = self.model(input_tensor)

        if output.shape[-1] == 1 or output.dim() == 1:
            score = output.squeeze()
            if score.dim() == 0:
                score = score
            else:
                score = score[0]
        else:
            if class_idx is None:
                class_idx = output.argmax(dim=1).item()
            score = output[0, class_idx]

        self.model.zero_grad()
        score.backward(retain_graph=True)

        cam = self._compute_cam(class_idx)
        return _overlay_cam(cam, original_image)


class ScoreCAM:
    """
    Score-CAM: gradient-free, perturbation-based activation maps.
    More faithful to model decisions than gradient methods.
    Uses channel activations as perturbation masks.
    """

    def __init__(self, model: torch.nn.Module, target_layer: str = None):
        self.model = model
        self.model.eval()
        self._activations: Optional[torch.Tensor] = None
        self._hook_handle = None

        if target_layer is None:
            target_layer = self._detect_target_layer()

        self._register_hook(target_layer)

    def _detect_target_layer(self) -> str:
        if hasattr(self.model, "backbone") and hasattr(self.model.backbone, "features"):
            return "backbone.features"
        if hasattr(self.model, "backbone") and hasattr(self.model.backbone, "layer4"):
            return "backbone.layer4"
        if hasattr(self.model, "model_a"):
            if hasattr(self.model.model_a, "backbone"):
                if hasattr(self.model.model_a.backbone, "features"):
                    return "model_a.backbone.features"
                if hasattr(self.model.model_a.backbone, "layer4"):
                    return "model_a.backbone.layer4"
        return "backbone.layer4"

    def _get_layer(self, name: str):
        layer = self.model
        for part in name.split("."):
            layer = getattr(layer, part)
        return layer

    def _register_hook(self, target_layer: str):
        layer = self._get_layer(target_layer)
        def hook(module, input, output):
            self._activations = output.detach()
        self._hook_handle = layer.register_forward_hook(hook)

    def remove_hooks(self):
        if self._hook_handle:
            self._hook_handle.remove()

    def generate(
        self,
        input_tensor: torch.Tensor,
        original_image: np.ndarray,
        class_idx: Optional[int] = None,
        max_channels: int = 64,  # limit channels for speed
    ) -> str:
        with torch.no_grad():
            baseline_out = self.model(input_tensor)
            if class_idx is None and baseline_out.shape[-1] > 1:
                class_idx = baseline_out.argmax(dim=1).item()

        acts = self._activations.squeeze(0)  # (C, H, W)
        C, H, W = acts.shape
        inp_size = input_tensor.shape[-2:]   # (224, 224)

        # Subsample channels for speed
        step = max(1, C // max_channels)
        channel_indices = list(range(0, C, step))[:max_channels]

        scores = []
        with torch.no_grad():
            for ci in channel_indices:
                # Upsample single channel map to input size
                mask = F.interpolate(
                    acts[ci].unsqueeze(0).unsqueeze(0),
                    size=inp_size, mode="bilinear", align_corners=False
                ).squeeze()
                # Normalise to [0,1]
                m_min, m_max = mask.min(), mask.max()
                if m_max > m_min:
                    mask = (mask - m_min) / (m_max - m_min)
                else:
                    mask = torch.zeros_like(mask)

                # Apply mask to input
                masked_input = input_tensor * mask.unsqueeze(0).unsqueeze(0)
                out = self.model(masked_input)

                if class_idx is not None and out.shape[-1] > 1:
                    score = torch.softmax(out, dim=1)[0, class_idx].item()
                else:
                    score = torch.sigmoid(out).squeeze().item()
                scores.append(score)

        scores_t = torch.tensor(scores)
        cam = torch.zeros(H, W)
        for i, ci in enumerate(channel_indices):
            cam += scores_t[i] * acts[ci].cpu()

        cam = F.relu(cam)
        cam = cam.numpy()
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return _overlay_cam(cam, original_image)


class EigenCAM:
    """
    EigenCAM: uses the first principal component of the feature map
    as the CAM. Fast, gradient-free, and works well with CNNs.
    Particularly useful when gradients are unstable.
    """

    def __init__(self, model: torch.nn.Module, target_layer: str = None):
        self.model = model
        self.model.eval()
        self._activations: Optional[torch.Tensor] = None
        self._hook_handle = None

        if target_layer is None:
            target_layer = self._detect_target_layer()

        layer = self._get_layer(target_layer)
        self._hook_handle = layer.register_forward_hook(
            lambda m, i, o: setattr(self, "_activations", o.detach())
        )

    def _detect_target_layer(self) -> str:
        if hasattr(self.model, "backbone") and hasattr(self.model.backbone, "features"):
            return "backbone.features"
        if hasattr(self.model, "backbone") and hasattr(self.model.backbone, "layer4"):
            return "backbone.layer4"
        if hasattr(self.model, "model_a"):
            if hasattr(self.model.model_a, "backbone"):
                if hasattr(self.model.model_a.backbone, "features"):
                    return "model_a.backbone.features"
                if hasattr(self.model.model_a.backbone, "layer4"):
                    return "model_a.backbone.layer4"
        return "backbone.layer4"

    def _get_layer(self, name: str):
        layer = self.model
        for part in name.split("."):
            layer = getattr(layer, part)
        return layer

    def remove_hooks(self):
        if self._hook_handle:
            self._hook_handle.remove()

    def generate(
        self,
        input_tensor: torch.Tensor,
        original_image: np.ndarray,
        class_idx: Optional[int] = None,
    ) -> str:
        with torch.no_grad():
            _ = self.model(input_tensor)

        acts = self._activations.squeeze(0)  # (C, H, W)
        acts_flat = acts.view(acts.shape[0], -1).cpu().numpy()  # (C, H*W)

        # PCA — first principal component
        acts_centered = acts_flat - acts_flat.mean(axis=1, keepdims=True)
        _, _, Vt = np.linalg.svd(acts_centered, full_matrices=False)
        first_pc = Vt[0]  # (H*W,)
        cam = first_pc.reshape(acts.shape[1], acts.shape[2])
        cam = np.maximum(cam, 0)  # relu
        cam -= cam.min()
        if cam.max() > 0:
            cam /= cam.max()

        return _overlay_cam(cam, original_image)


# ── Shared utilities ──────────────────────────────────────────────────────────

def _overlay_cam(cam: np.ndarray, original_image: np.ndarray, alpha: float = 0.45) -> str:
    """
    Resize CAM to original image size and blend with colourmap.
    Returns base64-encoded PNG string.
    """
    h, w = original_image.shape[:2]
    cam_resized = cv2.resize(cam, (w, h))

    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    if original_image.dtype != np.uint8:
        original_image = (original_image * 255).astype(np.uint8)
    if len(original_image.shape) == 2:
        original_image = np.stack([original_image] * 3, axis=2)

    overlay = cv2.addWeighted(original_image, 1 - alpha, heatmap, alpha, 0)
    pil_img = Image.fromarray(overlay)
    buf = BytesIO()
    pil_img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{b64}"


def generate_comparison_strip(
    original: np.ndarray,
    gradcam_b64: str,
    scorecam_b64: str,
) -> str:
    """
    Returns a side-by-side strip: [original | grad-cam++ | score-cam].
    Encoded as base64 PNG for frontend display.
    """
    def decode(b64: str) -> np.ndarray:
        data = base64.b64decode(b64.split(",", 1)[1])
        img = Image.open(BytesIO(data)).convert("RGB")
        return np.array(img)

    orig = cv2.resize(original, (224, 224))
    gcam = decode(gradcam_b64)
    scam = decode(scorecam_b64)

    strip = np.concatenate([orig, gcam, scam], axis=1)

    labels = ["Original", "Grad-CAM++", "Score-CAM"]
    x_positions = [112, 336, 560]
    for label, x in zip(labels, x_positions):
        cv2.putText(strip, label, (x - 40, 215), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

    pil_strip = Image.fromarray(strip)
    buf = BytesIO()
    pil_strip.save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"
