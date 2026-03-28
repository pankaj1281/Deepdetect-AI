"""
Training script for the audio deepfake detection model.

Usage:
    python -m training.train_audio \
        --data_dir data/audio \
        --epochs 30 \
        --batch_size 32 \
        --lr 1e-4 \
        --output_path models/audio_model.pt
"""

import argparse
import logging
import os

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from backend.models.audio_model import AudioCNN
from training.data_pipeline import AudioDeepfakeDataset, make_dataloaders, train_val_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, preds_all, labels_all = 0.0, [], []
    for specs, labels in loader:
        specs, labels = specs.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        logits = model(specs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(specs)
        preds_all.extend((torch.sigmoid(logits) >= 0.5).cpu().int().tolist())
        labels_all.extend(labels.cpu().int().tolist())
    flat = lambda x: [v[0] if isinstance(v, list) else v for v in x]
    return total_loss / len(loader.dataset), accuracy_score(flat(labels_all), flat(preds_all))


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, preds_all, labels_all = 0.0, [], []
    for specs, labels in loader:
        specs, labels = specs.to(device), labels.to(device).unsqueeze(1)
        logits = model(specs)
        total_loss += criterion(logits, labels).item() * len(specs)
        preds_all.extend((torch.sigmoid(logits) >= 0.5).cpu().int().tolist())
        labels_all.extend(labels.cpu().int().tolist())
    flat = lambda x: [v[0] if isinstance(v, list) else v for v in x]
    fp, fl = flat(preds_all), flat(labels_all)
    n = len(loader.dataset)
    return (
        total_loss / n,
        accuracy_score(fl, fp),
        precision_score(fl, fp, zero_division=0),
        recall_score(fl, fp, zero_division=0),
        f1_score(fl, fp, zero_division=0),
    )


def main():
    parser = argparse.ArgumentParser(description="Train audio deepfake detector")
    parser.add_argument("--data_dir", default="data/audio")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_path", default="models/audio_model.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    dataset = AudioDeepfakeDataset(args.data_dir)
    train_set, val_set = train_val_split(dataset, val_ratio=0.2)
    train_loader, val_loader = make_dataloaders(train_set, val_set, args.batch_size)
    logger.info("Train: %d  Val: %d", len(train_set), len(val_set))

    model = AudioCNN().to(device)
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
