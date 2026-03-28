"""
Training script for the image deepfake detection model.

Usage:
    python -m training.train_image \
        --data_dir data/images \
        --epochs 30 \
        --batch_size 32 \
        --lr 1e-4 \
        --output_path models/image_model.pt
"""

import argparse
import logging
import os

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from backend.models.image_model import ImageDeepfakeDetector
from training.data_pipeline import ImageDeepfakeDataset, make_dataloaders, train_val_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, preds_all, labels_all = 0.0, [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(images)
        preds_all.extend((torch.sigmoid(logits) >= 0.5).cpu().int().tolist())
        labels_all.extend(labels.cpu().int().tolist())
    n = len(loader.dataset)
    return total_loss / n, accuracy_score(labels_all, preds_all)


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, preds_all, labels_all = 0.0, [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device).unsqueeze(1)
        logits = model(images)
        total_loss += criterion(logits, labels).item() * len(images)
        preds_all.extend((torch.sigmoid(logits) >= 0.5).cpu().int().tolist())
        labels_all.extend(labels.cpu().int().tolist())
    n = len(loader.dataset)
    flat_preds = [p[0] if isinstance(p, list) else p for p in preds_all]
    flat_labels = [l[0] if isinstance(l, list) else l for l in labels_all]
    return (
        total_loss / n,
        accuracy_score(flat_labels, flat_preds),
        precision_score(flat_labels, flat_preds, zero_division=0),
        recall_score(flat_labels, flat_preds, zero_division=0),
        f1_score(flat_labels, flat_preds, zero_division=0),
    )


def main():
    parser = argparse.ArgumentParser(description="Train image deepfake detector")
    parser.add_argument("--data_dir", default="data/images")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_path", default="models/image_model.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    dataset = ImageDeepfakeDataset(args.data_dir, split="train")
    train_set, val_set = train_val_split(dataset, val_ratio=0.2)
    train_loader, val_loader = make_dataloaders(train_set, val_set, args.batch_size)
    logger.info("Train: %d  Val: %d", len(train_set), len(val_set))

    model = ImageDeepfakeDetector(pretrained=True).to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        vl_loss, vl_acc, prec, rec, f1 = eval_epoch(model, val_loader, criterion, device)
        scheduler.step()
        logger.info(
            "Epoch %02d/%02d | tr_loss=%.4f tr_acc=%.4f | "
            "vl_loss=%.4f vl_acc=%.4f prec=%.4f rec=%.4f f1=%.4f",
            epoch, args.epochs, tr_loss, tr_acc, vl_loss, vl_acc, prec, rec, f1,
        )
        if vl_acc > best_val_acc:
            best_val_acc = vl_acc
            os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
            torch.save(model.state_dict(), args.output_path)
            logger.info("  ✓ Saved best model (val_acc=%.4f)", best_val_acc)

    logger.info("Training complete. Best val accuracy: %.4f", best_val_acc)


if __name__ == "__main__":
    main()
