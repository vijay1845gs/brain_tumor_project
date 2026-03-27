# train_classification.py

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from models.advanced_models import EfficientClassificationModel
from models.resnet_models import TumorClassificationModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
BATCH_SIZE = 16


def get_loaders(data_dir):
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor()
    ])

    train_ds = datasets.ImageFolder(os.path.join(data_dir, "train"), transform)
    val_ds   = datasets.ImageFolder(os.path.join(data_dir, "val"), transform)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")
    print("Classes:", train_ds.classes)

    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(val_ds, batch_size=BATCH_SIZE),
        train_ds.classes
    )


def get_model(name, num_classes):
    if name == "efficientnet":
        print("🚀 Using EfficientNet")
        return EfficientClassificationModel(pretrained=True, num_classes=num_classes).to(DEVICE)
    else:
        print("🚀 Using ResNet101")
        return TumorClassificationModel(pretrained=True, num_classes=num_classes).to(DEVICE)


def train(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0

    for x, y in loader:
        x, y = x.to(DEVICE), y.to(DEVICE)

        optimizer.zero_grad()
        out = model(x)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


def evaluate(model, loader):
    model.eval()
    y_true, y_probs = [], []

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            out = model(x)
            probs = torch.softmax(out, dim=1)

            y_probs.extend(probs.cpu().numpy())
            y_true.extend(y.numpy())

    y_probs = np.array(y_probs)
    y_pred = np.argmax(y_probs, axis=1)

    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted")
    rec  = recall_score(y_true, y_pred, average="weighted")
    f1   = f1_score(y_true, y_pred, average="weighted")
    cm   = confusion_matrix(y_true, y_pred)

    return acc, prec, rec, f1, cm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--model", choices=["efficientnet", "resnet101"], default="efficientnet")
    parser.add_argument("--out", default="models")

    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)

    train_loader, val_loader, classes = get_loaders(args.data)
    model = get_model(args.model, len(classes))

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    best_f1 = 0

    for epoch in range(args.epochs):
        loss = train(model, train_loader, optimizer, criterion)
        acc, prec, rec, f1, cm = evaluate(model, val_loader)

        print(f"Epoch {epoch+1} | Loss {loss:.4f} | Acc {acc:.4f} | F1 {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), f"{args.out}/classification_{args.model}.pth")
            print("✅ Saved best classification model")

    print("Final Confusion Matrix:\n", cm)


if __name__ == "__main__":
    main()