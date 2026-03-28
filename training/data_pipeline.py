"""
Shared data loading pipeline for deepfake detection training.

Directory structure expected:
    data/
        images/
            real/   *.jpg *.png
            fake/   *.jpg *.png
        videos/
            real/   *.mp4
            fake/   *.mp4
        audio/
            real/   *.wav
            fake/   *.wav
"""

import os
import random
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
# Image dataset
# ──────────────────────────────────────────────────────────────────────────────

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class ImageDeepfakeDataset(Dataset):
    """Binary classification dataset for still images."""

    def __init__(self, root: str, split: str = "train", transform: Callable = None):
        self.root = Path(root)
        self.transform = transform or self._default_transform(split)
        self.samples: List[Tuple[Path, int]] = []
        for label, name in ((0, "real"), (1, "fake")):
            folder = self.root / name
            if not folder.exists():
                continue
            for p in sorted(folder.iterdir()):
                if p.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((p, label))
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.float32)

    @staticmethod
    def _default_transform(split: str):
        if split == "train":
            return transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])


# ──────────────────────────────────────────────────────────────────────────────
# Audio dataset
# ──────────────────────────────────────────────────────────────────────────────

AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}
SAMPLE_RATE = 22050
DURATION = 5
N_MELS = 128
HOP_LENGTH = 512
SPEC_SIZE = 128


class AudioDeepfakeDataset(Dataset):
    """Binary classification dataset for audio clips (spectrogram-based)."""

    def __init__(self, root: str):
        self.root = Path(root)
        self.samples: List[Tuple[Path, int]] = []
        for label, name in ((0, "real"), (1, "fake")):
            folder = self.root / name
            if not folder.exists():
                continue
            for p in sorted(folder.iterdir()):
                if p.suffix.lower() in AUDIO_EXTENSIONS:
                    self.samples.append((p, label))
        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        spec = self._load_spectrogram(path)
        tensor = torch.tensor(spec, dtype=torch.float32).unsqueeze(0)  # (1, H, W)
        return tensor, torch.tensor(label, dtype=torch.float32)

    def _load_spectrogram(self, path: Path) -> np.ndarray:
        try:
            import librosa
            y, sr = librosa.load(str(path), sr=SAMPLE_RATE, duration=DURATION, mono=True)
            mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS, hop_length=HOP_LENGTH)
            mel_db = librosa.power_to_db(mel, ref=np.max)
            mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
        except Exception:
            mel_db = np.zeros((N_MELS, 128), dtype=np.float32)
        img = Image.fromarray((mel_db * 255).astype(np.uint8)).resize((SPEC_SIZE, SPEC_SIZE))
        return np.array(img, dtype=np.float32) / 255.0


# ──────────────────────────────────────────────────────────────────────────────
# Shared utilities
# ──────────────────────────────────────────────────────────────────────────────

def train_val_split(dataset: Dataset, val_ratio: float = 0.2):
    """Split a dataset into train and validation subsets."""
    n = len(dataset)
    val_n = int(n * val_ratio)
    train_n = n - val_n
    return torch.utils.data.random_split(dataset, [train_n, val_n])


def make_dataloaders(
    train_set: Dataset,
    val_set: Dataset,
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, DataLoader]:
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    return train_loader, val_loader
