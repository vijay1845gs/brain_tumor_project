# train_classification.py — PRODUCTION GRADE

import os
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix
)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from models.resnet_models import TumorClassificationModel

# ─────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
BATCH_SIZE = 16

# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
def get_model():
    print("🚀 Using ResNet101 for classification")
    return TumorClassificationModel(num_classes=3, pretrained=True)

# ─────────────────────────────────────────────
# DATA LOADERS
# ─────────────────────────────────────────────
def get_loaders(data_dir):
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor()
    ])

    val_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), train_tf)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"), val_tf)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=BATCH_SIZE)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    print(f"Classes: {train_ds.classes}")

    return train_loader, val_loader, train_ds.classes

# ─────────────────────────────────────────────
# TRAIN
# ─────────────────────────────────────────────
def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)

# ─────────────────────────────────────────────
# EVALUATE
# ─────────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            out = model(x)
            preds = torch.argmax(out, dim=1).cpu().numpy()

            y_pred.extend(preds)
            y_true.extend(y.numpy())

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="macro")
    rec = recall_score(y_true, y_pred, average="macro")
    f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred)

    return acc, prec, rec, f1, cm, y_true, y_pred

# ─────────────────────────────────────────────
# SAVE PLOTS
# ─────────────────────────────────────────────
def save_plots(cm, classes, out_dir):
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes,
                yticklabels=classes)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix - Classification")
    plt.savefig(f"{out_dir}/classification_cm.png")
    plt.close()

# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--out", default="models")

    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    train_loader, val_loader, classes = get_loaders(args.data)
    model = get_model().to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0
    patience = 3
    patience_counter = 0
    metrics_log = []

    for epoch in range(args.epochs):
        loss = train(model, train_loader, optimizer, criterion)
        acc, prec, rec, f1, cm, y_true, y_pred = evaluate(model, val_loader)

        print(f"\nEpoch {epoch+1}")
        print(f"Loss: {loss:.4f}")
        print(f"Acc: {acc:.4f} | Prec: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

        # ── Save metrics ──
        metrics = {
            "epoch": epoch + 1,
            "loss": loss,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        }
        metrics_log.append(metrics)

        # ── Save best model ──
        if f1 > best_f1:
            best_f1 = f1
            patience_counter = 0

            torch.save(model.state_dict(), f"{args.out}/classification_model.pth")
            save_plots(cm, classes, args.out)
            np.savetxt(f"{args.out}/classification_confusion.txt", cm, fmt="%d")

            print("✅ Saved best classification model")

        else:
            patience_counter += 1

        # ── Early stopping ──
        if patience_counter >= patience:
            print("⏹ Early stopping triggered")
            break

        # ── Save metrics.json ──
        with open(f"{args.out}/classification_metrics.json", "w") as f:
            json.dump(metrics_log, f, indent=4)

    print("\n🎯 Classification Training Complete")
    print(f"Best F1: {best_f1:.4f}")

if __name__ == "__main__":
    main()