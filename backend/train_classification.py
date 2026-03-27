import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, accuracy_score

from models.advanced_models import EfficientClassificationModel
from models.resnet_models import TumorClassificationModel

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
BATCH_SIZE = 32


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

    return (
        DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True),
        DataLoader(val_ds, batch_size=BATCH_SIZE),
        train_ds.classes
    )


# =========================
# MODEL
# =========================
def get_model(name, num_classes):
    if name == "efficientnet":
        return EfficientClassificationModel(pretrained=True, num_classes=num_classes)
    return TumorClassificationModel(pretrained=True, num_classes=num_classes)


# =========================
# TRAIN
# =========================
def train_epoch(model, loader, optimizer, criterion):
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


# =========================
# EVAL
# =========================
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

    return accuracy_score(y_true, y_pred), f1_score(y_true, y_pred, average="weighted")


# =========================
# MAIN
# =========================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", default="efficientnet")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--out", default="models")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    train_loader, val_loader, classes = get_loaders(args.data)
    model = get_model(args.model, len(classes)).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=3e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.3)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_f1 = 0

    for epoch in range(args.epochs):
        loss = train_epoch(model, train_loader, optimizer, criterion)
        acc, f1 = evaluate(model, val_loader)

        scheduler.step()

        print(f"Epoch {epoch+1} | Loss {loss:.4f} | Acc {acc:.4f} | F1 {f1:.4f}")

        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), f"{args.out}/best_classification.pth")
            print("✅ Best classification model saved")


if __name__ == "__main__":
    main()