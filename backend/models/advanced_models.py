# models/advanced_models.py

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b4, EfficientNet_B4_Weights

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
TUMOR_CLASSES = ["glioma", "meningioma", "pituitary"]

# ─────────────────────────────────────────────
# SIMPLE + STRONG HEAD (NO OVERFIT)
# ─────────────────────────────────────────────
class EfficientHead(nn.Module):
    def __init__(self, in_features, num_classes, dropout=0.4):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.net(x)

# ─────────────────────────────────────────────
# DETECTION MODEL
# ─────────────────────────────────────────────
class EfficientDetectionModel(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()

        weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b4(weights=weights)

        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()

        self.backbone = backbone
        self.head = EfficientHead(in_features, 1, dropout=0.4)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

    def predict_proba(self, x):
        logits = self.forward(x)
        return torch.sigmoid(logits).squeeze(1)

# ─────────────────────────────────────────────
# CLASSIFICATION MODEL
# ─────────────────────────────────────────────
class EfficientClassificationModel(nn.Module):
    def __init__(self, pretrained=True, num_classes=3):
        super().__init__()

        weights = EfficientNet_B4_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = efficientnet_b4(weights=weights)

        in_features = backbone.classifier[1].in_features
        backbone.classifier = nn.Identity()

        self.backbone = backbone
        self.head = EfficientHead(in_features, num_classes, dropout=0.35)

    def forward(self, x):
        features = self.backbone(x)
        return self.head(features)

    def predict_proba(self, x):
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

# ─────────────────────────────────────────────
# BUILD FUNCTIONS
# ─────────────────────────────────────────────
def build_efficient_detection(weights_path=None, device="cpu"):
    model = EfficientDetectionModel(pretrained=(weights_path is None))

    if weights_path:
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model


def build_efficient_classification(weights_path=None, device="cpu"):
    model = EfficientClassificationModel(pretrained=(weights_path is None))

    if weights_path:
        state = torch.load(weights_path, map_location=device)
        model.load_state_dict(state)

    model.to(device)
    model.eval()
    return model


# =========================================================
# 🔥 ENSEMBLE MODELS (PRODUCTION CRITICAL)
# =========================================================

class EnsembleDetectionModel(nn.Module):
    def __init__(self, model_a, model_b):
        super().__init__()
        self.model_a = model_a
        self.model_b = model_b

    def predict_proba(self, x):
        p1 = self.model_a.predict_proba(x)
        p2 = self.model_b.predict_proba(x)
        return (p1 + p2) / 2.0

    def forward(self, x):
        return self.predict_proba(x)


class EnsembleClassificationModel(nn.Module):
    def __init__(self, model_a, model_b):
        super().__init__()
        self.model_a = model_a
        self.model_b = model_b

    def predict_proba(self, x):
        p1 = self.model_a.predict_proba(x)
        p2 = self.model_b.predict_proba(x)
        return (p1 + p2) / 2.0

    def forward(self, x):
        return self.predict_proba(x)