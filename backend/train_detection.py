import os
import argparse
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt

from models.advanced_models import EfficientDetectionModel
from models.resnet_models import TumorDetectionModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
BATCH_SIZE = 16

# =========================
# FOCAL LOSS (FN reduction)
# =========================
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.75):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits, targets):
        bce = nn.functional.binary_cross_entropy_with_logits(
            logits, targets.float(), reduction="none"
        )
        pt = torch.exp(-bce)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


# =========================
# MIXUP + CUTMIX
# =========================
def mixup(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    idx = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


def cutmix(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    bs, _, h, w = x.shape
    idx = torch.randperm(bs).to(x.device)

    cx, cy = np.random.randint(w), np.random.randint(h)
    cut_w = int(w * np.sqrt(1 - lam))
    cut_h = int(h * np.sqrt(1 - lam))

    x1 = np.clip(cx - cut_w // 2, 0, w)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    y2 = np.clip(cy + cut_h // 2, 0, h)

    x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
    return x, y, y[idx], lam


# =========================
# DATA
# =========================
def get_loaders(data_dir):
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(0.3, 0.3, 0.2),
        transforms.ToTensor()
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), train_tf)
    val_ds = datasets.ImageFolder(os.path.join(data_dir, "val"), val_tf)

    targets = [s[1] for s in train_ds.samples]
    class_counts = np.bincount(targets)
    weights = 1.0 / class_counts[targets]

    sampler = WeightedRandomSampler(weights, len(weights))

    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler),
        DataLoader(val_ds, batch_size=BATCH_SIZE),
    )


# =========================
# MODEL
# =========================
def get_model(name):
    if name == "efficientnet":
        return EfficientDetectionModel(pretrained=True)
    return TumorDetectionModel(pretrained=True)


# =========================
# TRAIN
# =========================
def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        if random.random() < 0.5:
            x, y_a, y_b, lam = mixup(x, y)
            outputs = model(x).squeeze(1)
            loss = lam * criterion(outputs, y_a) + (1 - lam) * criterion(outputs, y_b)
        else:
            outputs = model(x).squeeze(1)
            loss = criterion(outputs, y)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# =========================
# EVAL (FN-focused)
# =========================
def evaluate(model, loader):
    model.eval()
    y_true, y_probs = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            out = model(x).squeeze(1)
            probs = torch.sigmoid(out).cpu().numpy()

            y_probs.extend(probs)
            y_true.extend(y.numpy())

    y_probs = np.array(y_probs)
    y_true = np.array(y_true)

    thresholds = np.linspace(0.1, 0.9, 50)
    best_f1, best_thr = 0, 0.5

    for t in thresholds:
        preds = (y_probs >= t).astype(int)
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1, best_thr = f1, t

    preds = (y_probs >= best_thr).astype(int)

    return (
        precision_score(y_true, preds),
        recall_score(y_true, preds),
        best_f1,
        confusion_matrix(y_true, preds),
        best_thr
    )


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="efficientnet")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--out", default="models")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    train_loader, val_loader = get_loaders(args.data)
    model = get_model(args.model).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=3e-4,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs
    )

    criterion = FocalLoss()

    best_f1 = 0

    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, optimizer, criterion)
        prec, rec, f1, cm, thr = evaluate(model, val_loader)

        scheduler.step()

        print(f"Epoch {epoch+1} | Loss {loss:.4f} | F1 {f1:.4f} | Recall {rec:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), f"{args.out}/best_detection.pth")
            print("✅ Best model saved")

    print("Best Threshold:", thr)


if __name__ == "__main__":
    main()