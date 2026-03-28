"""
Training script for the video deepfake detection model (CNN + LSTM).

Usage:
    python -m training.train_video \
        --data_dir data/videos \
        --epochs 20 \
        --batch_size 8 \
        --lr 1e-4 \
        --output_path models/video_model.pt
"""

import argparse
import logging
import os
import random
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from backend.models.video_model import VideoDeepfakeDetector, MAX_FRAMES, IMAGE_SIZE
from training.data_pipeline import train_val_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}
TRANSFORM = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


class VideoDeepfakeDataset(Dataset):
    def __init__(self, root: str):
        self.root = Path(root)
        self.samples = []
        for label, name in ((0, "real"), (1, "fake")):
            folder = self.root / name
            if not folder.exists():
                continue
            for p in sorted(folder.iterdir()):
                if p.suffix.lower() in VIDEO_EXTENSIONS:
                    self.samples.append((p, label))
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        frames = self._extract_frames(path)
        frame_tensor = torch.stack(frames)  # (MAX_FRAMES, C, H, W)
        return frame_tensor, torch.tensor(label, dtype=torch.float32)

    def _extract_frames(self, path: Path):
        cap = cv2.VideoCapture(str(path))
        total = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 1)
        indices = np.linspace(0, total - 1, MAX_FRAMES, dtype=int)
        frames = []
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ret, frame = cap.read()
            if ret:
                frames.append(TRANSFORM(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))))
        cap.release()
        blank = torch.zeros(3, IMAGE_SIZE, IMAGE_SIZE)
        while len(frames) < MAX_FRAMES:
            frames.append(frames[-1] if frames else blank)
        return frames[:MAX_FRAMES]


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, preds_all, labels_all = 0.0, [], []
    for frames, labels in loader:
        frames, labels = frames.to(device), labels.to(device).unsqueeze(1)
        optimizer.zero_grad()
        logits = model(frames)
        loss = criterion(logits, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * len(frames)
        preds_all.extend((torch.sigmoid(logits) >= 0.5).cpu().int().tolist())
        labels_all.extend(labels.cpu().int().tolist())
    flat = lambda x: [v[0] if isinstance(v, list) else v for v in x]
    return total_loss / len(loader.dataset), accuracy_score(flat(labels_all), flat(preds_all))


@torch.no_grad()
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, preds_all, labels_all = 0.0, [], []
    for frames, labels in loader:
        frames, labels = frames.to(device), labels.to(device).unsqueeze(1)
        logits = model(frames)
        total_loss += criterion(logits, labels).item() * len(frames)
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
    parser = argparse.ArgumentParser(description="Train video deepfake detector")
    parser.add_argument("--data_dir", default="data/videos")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output_path", default="models/video_model.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Device: %s", device)

    dataset = VideoDeepfakeDataset(args.data_dir)
    train_set, val_set = train_val_split(dataset, val_ratio=0.2)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=2)
    logger.info("Train: %d  Val: %d", len(train_set), len(val_set))

    model = VideoDeepfakeDetector(pretrained=True).to(device)
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
