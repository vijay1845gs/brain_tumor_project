# train_detection.py

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, precision_recall_curve, roc_curve
)
import matplotlib.pyplot as plt
import seaborn as sns

from models.advanced_models import EfficientDetectionModel
from models.resnet_models import TumorDetectionModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
BATCH_SIZE = 8


# =========================
# FOCAL LOSS
# =========================
class FocalLoss(nn.Module):
    def __init__(self, gamma=1.5, alpha=0.5):
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
# MODEL
# =========================
def get_model(name):
    if name == "efficientnet":
        print("🚀 Using EfficientNet")
        return EfficientDetectionModel(pretrained=True)
    else:
        print("🚀 Using ResNet101")
        return TumorDetectionModel(pretrained=True)


# =========================
# DATA
# =========================
def get_loaders(data_dir):
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
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

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    return train_loader, val_loader


# =========================
# TRAIN
# =========================
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x).squeeze(1)
        loss = criterion(out, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# =========================
# EVALUATE
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

    y_true = np.array(y_true)
    y_probs = np.array(y_probs)

    precision, recall, thresholds = precision_recall_curve(y_true, y_probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)

    best_idx = np.argmax(f1_scores)
    threshold = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    threshold = max(threshold, 0.4)

    y_pred = (y_probs >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    return acc, prec, rec, f1, cm, y_true, y_probs, y_pred, threshold


# =========================
# PLOTS
# =========================
def save_plots(y_true, y_probs, y_pred, name):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.savefig(f"{name}_cm.png")
    plt.clf()

    fpr, tpr, _ = roc_curve(y_true, y_probs)
    plt.plot(fpr, tpr)
    plt.savefig(f"{name}_roc.png")
    plt.clf()

    precision, recall, _ = precision_recall_curve(y_true, y_probs)
    plt.plot(recall, precision)
    plt.savefig(f"{name}_pr.png")
    plt.clf()


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--model", choices=["efficientnet", "resnet101"], default="efficientnet")
    parser.add_argument("--out", default="models")

    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    train_loader, val_loader = get_loaders(args.data)
    model = get_model(args.model).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = FocalLoss()

    best_f1 = 0

    for epoch in range(args.epochs):
        loss = train(model, train_loader, optimizer, criterion)
        acc, prec, rec, f1, cm, y_true, y_probs, y_pred, thr = evaluate(model, val_loader)

        print(f"Epoch {epoch+1} | Loss {loss:.4f} | F1 {f1:.4f} | Thr {thr:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), f"{args.out}/detection_{args.model}.pth")
            save_plots(y_true, y_probs, y_pred, args.model)
            print("✅ Saved best model + plots")

    print("Final Threshold:", thr)


if __name__ == "__main__":
    main()