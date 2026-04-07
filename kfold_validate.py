"""
K-fold cross-validation for label quality checking.

Merges all 1599 labeled samples into one pool, runs 10 training runs
with different random 80/20 splits, and flags samples that are
consistently predicted wrong (likely mislabeled).

Usage: python kfold_validate.py
"""

import csv
import json
import math
import os
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset, WeightedRandomSampler
from PIL import Image
from torchvision import transforms
import torchvision.transforms.v2 as T

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHARSET = "abcdefghjklmnpqrstuvwxy23457"
BLANK = 0
NUM_CLASSES = len(CHARSET) + 1

IMG_W = 256
IMG_H = 48

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

char_to_idx = {c: i + 1 for i, c in enumerate(CHARSET)}
idx_to_char = {i + 1: c for i, c in enumerate(CHARSET)}

N_FOLDS = 10
PRETRAIN_EPOCHS = 30
MIXED_EPOCHS = 20
BATCH_SIZE = 32
SYNTHETIC_DIR = "dataset_synthetic/train"

# ---------------------------------------------------------------------------
# Model (same as train_phased.py)
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x):
        w = x.mean(dim=(2, 3))
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))
        return x * w.unsqueeze(2).unsqueeze(3)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.se = SEBlock(out_ch)

    def forward(self, x):
        residual = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        return F.relu(out + residual, inplace=True)


class CRNN(nn.Module):
    def __init__(self, num_classes, hidden_size=128):
        super().__init__()
        self.cnn = nn.Sequential(
            ResBlock(3, 32), nn.MaxPool2d(2, 2),
            ResBlock(32, 64), nn.MaxPool2d(2, 2), nn.Dropout2d(0.2),
            ResBlock(64, 128), nn.MaxPool2d((2, 1)), nn.Dropout2d(0.2),
            ResBlock(128, 256), nn.MaxPool2d((2, 1)), nn.Dropout2d(0.2),
            nn.AdaptiveAvgPool2d((1, None)),
        )
        self.rnn = nn.LSTM(input_size=256, hidden_size=hidden_size,
                           num_layers=1, bidirectional=True, batch_first=True)
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        conv = self.cnn(x).squeeze(2).permute(0, 2, 1)
        rnn_out, _ = self.rnn(conv)
        out = self.fc(self.dropout(rnn_out))
        return out.permute(1, 0, 2)


# ---------------------------------------------------------------------------
# CTC decode
# ---------------------------------------------------------------------------

def greedy_decode(logits):
    indices = logits.argmax(dim=2).permute(1, 0)
    results = []
    for seq in indices:
        chars = []
        prev = None
        for idx in seq.tolist():
            if idx != prev and idx != BLANK:
                chars.append(idx_to_char[idx])
            prev = idx
        results.append("".join(chars))
    return results


# ---------------------------------------------------------------------------
# Dataset — unified, indexed by sample ID
# ---------------------------------------------------------------------------

class AllSamplesDataset(Dataset):
    """All labeled samples with unique IDs for cross-referencing."""
    def __init__(self, transform=None):
        self.transform = transform
        self.samples = []  # (img_path, label, sample_id)

        # Load YOLO dataset (train + val)
        combined = Path("dataset_pseudo_v2")
        for split in ["train", "val"]:
            img_dir = combined / "images" / split
            lbl_dir = combined / "text_labels" / split
            for img_path in sorted(img_dir.glob("*.png")):
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                if lbl_path.exists():
                    label = lbl_path.read_text().strip().lower()
                    if all(c in char_to_idx for c in label):
                        sample_id = f"{split}/{img_path.name}"
                        self.samples.append((str(img_path), label, sample_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, sample_id = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        target = [char_to_idx[c] for c in label]
        return img, torch.tensor(target, dtype=torch.long), len(target), label, sample_id


class FilenameDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.samples = []
        for fname in os.listdir(img_dir):
            if not fname.lower().endswith(".png"):
                continue
            parts = fname.rsplit(".", 1)[0].split("_", 1)
            if len(parts) == 2:
                label = parts[1].lower()
                if all(c in char_to_idx for c in label):
                    self.samples.append((fname, label))
        self.samples.sort()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        fname, label = self.samples[idx]
        img = Image.open(os.path.join(self.img_dir, fname)).convert("RGB")
        if self.transform:
            img = self.transform(img)
        target = [char_to_idx[c] for c in label]
        return img, torch.tensor(target, dtype=torch.long), len(target), label, f"synth/{fname}"


def collate_fn(batch):
    images, targets, target_lengths, labels, sample_ids = zip(*batch)
    images = torch.stack(images, 0)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    targets = torch.cat(targets, 0)
    return images, targets, target_lengths, labels, sample_ids


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

train_transform = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.RandomAffine(degrees=5, translate=(0.03, 0.05), scale=(0.95, 1.05), shear=3),
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1, hue=0.05),
    transforms.ToTensor(),
    transforms.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.3),
    T.GaussianNoise(mean=0.0, sigma=0.03, clip=True),
    transforms.RandomErasing(p=0.15, scale=(0.02, 0.06)),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

val_transform = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    for images, targets, target_lengths, _, _ in loader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)
        target_lengths = target_lengths.to(DEVICE)
        logits = model(images)
        T, B, _ = logits.shape
        input_lengths = torch.full((B,), T, dtype=torch.long, device=DEVICE)
        loss = criterion(logits.log_softmax(2), targets, input_lengths, target_lengths)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()


@torch.no_grad()
def predict_all(model, loader):
    """Returns list of (sample_id, gt_label, pred_label)."""
    model.eval()
    results = []
    for images, targets, target_lengths, labels, sample_ids in loader:
        images = images.to(DEVICE)
        logits = model(images)
        preds = greedy_decode(logits)
        for sid, gt, pred in zip(sample_ids, labels, preds):
            results.append((sid, gt, pred))
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading all labeled samples...")
    all_ds = AllSamplesDataset(transform=None)  # transform applied per-split
    n = len(all_ds)
    print(f"Total: {n} samples")

    synth_ds = FilenameDataset(SYNTHETIC_DIR, transform=train_transform)
    print(f"Synthetic: {len(synth_ds)} samples")

    # Track predictions per sample across folds
    # sample_id -> list of (fold, prediction)
    predictions = defaultdict(list)

    for fold in range(N_FOLDS):
        print(f"\n{'='*60}")
        print(f"FOLD {fold+1}/{N_FOLDS}")
        print(f"{'='*60}")

        fold_start = time.time()
        # Random 80/20 split
        indices = list(range(n))
        random.seed(fold * 42 + 7)
        random.shuffle(indices)
        split = int(0.8 * n)
        train_indices = indices[:split]
        val_indices = indices[split:]

        # Create train/val datasets with appropriate transforms
        # We need to wrap with transform since base dataset has None
        train_real = TransformSubset(all_ds, train_indices, train_transform)
        val_set = TransformSubset(all_ds, val_indices, val_transform)

        # Combined train: real + synthetic with 3:1 weighting
        combined = ConcatDataset([synth_ds, train_real])
        weights = [1.0] * len(synth_ds) + [3.0 * len(synth_ds) / len(train_real)] * len(train_real)
        sampler = WeightedRandomSampler(weights, num_samples=len(combined), replacement=True)

        train_loader = DataLoader(combined, batch_size=BATCH_SIZE, sampler=sampler,
                                  num_workers=4, collate_fn=collate_fn, pin_memory=True)
        val_loader = DataLoader(val_set, batch_size=BATCH_SIZE, shuffle=False,
                                num_workers=4, collate_fn=collate_fn, pin_memory=True)

        # Also need a pretrain-only loader
        pretrain_loader = DataLoader(synth_ds, batch_size=BATCH_SIZE, shuffle=True,
                                     num_workers=4, collate_fn=collate_fn, pin_memory=True)

        # Fresh model
        model = CRNN(NUM_CLASSES, hidden_size=128).to(DEVICE)
        criterion = nn.CTCLoss(blank=BLANK, zero_infinity=True)

        # Phase 1: Pretrain
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, PRETRAIN_EPOCHS, eta_min=1e-6)
        t0 = time.time()
        for ep in range(1, PRETRAIN_EPOCHS + 1):
            train_one_epoch(model, pretrain_loader, criterion, optimizer)
            scheduler.step()
            elapsed = time.time() - t0
            eta = elapsed / ep * (PRETRAIN_EPOCHS - ep)
            print(f"    pretrain {ep:2d}/{PRETRAIN_EPOCHS}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s left)", flush=True)
        print(f"  Pretrain done in {time.time()-t0:.0f}s")

        # Phase 2: Mixed
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, MIXED_EPOCHS, eta_min=1e-6)
        t0 = time.time()
        for ep in range(1, MIXED_EPOCHS + 1):
            train_one_epoch(model, train_loader, criterion, optimizer)
            scheduler.step()
            elapsed = time.time() - t0
            eta = elapsed / ep * (MIXED_EPOCHS - ep)
            print(f"    mixed {ep:2d}/{MIXED_EPOCHS}  ({elapsed:.0f}s elapsed, ~{eta:.0f}s left)", flush=True)
        print(f"  Mixed done in {time.time()-t0:.0f}s")

        # Predict on val
        results = predict_all(model, val_loader)
        correct = sum(1 for _, gt, pred in results if gt == pred)
        total_time = time.time() - fold_start
        print(f"  Val accuracy: {correct}/{len(results)} = {100*correct/len(results):.1f}%  (fold took {total_time:.0f}s)")

        for sid, gt, pred in results:
            predictions[sid].append((fold, pred))

    # ---------------------------------------------------------------------------
    # Analysis
    # ---------------------------------------------------------------------------
    print(f"\n{'='*60}")
    print("CROSS-VALIDATION ANALYSIS")
    print(f"{'='*60}\n")

    # Build sample_id -> label mapping
    id_to_label = {s[2]: s[1] for s in all_ds.samples}

    # For each sample, count how many times it was in val and how many times wrong
    sample_stats = []
    for sid, label in id_to_label.items():
        preds = predictions.get(sid, [])
        if not preds:
            continue
        n_val = len(preds)
        n_wrong = sum(1 for _, pred in preds if pred != label)
        wrong_preds = [pred for _, pred in preds if pred != label]
        sample_stats.append({
            "sample_id": sid,
            "label": label,
            "n_val": n_val,
            "n_wrong": n_wrong,
            "error_rate": n_wrong / n_val,
            "wrong_preds": wrong_preds,
            "most_common_pred": max(set(p for _, p in preds), key=lambda x: sum(1 for _, p in preds if p == x)),
        })

    # Sort by error rate (most wrong first)
    sample_stats.sort(key=lambda s: (-s["error_rate"], -s["n_wrong"]))

    # Report
    print(f"Total samples evaluated: {len(sample_stats)}")
    always_correct = sum(1 for s in sample_stats if s["n_wrong"] == 0)
    print(f"Always correct: {always_correct}")
    sometimes_wrong = [s for s in sample_stats if s["n_wrong"] > 0]
    print(f"Wrong at least once: {len(sometimes_wrong)}")

    high_error = [s for s in sample_stats if s["error_rate"] >= 0.5]
    print(f"Wrong >=50% of the time: {len(high_error)}")
    print()

    print("LIKELY MISLABELED (wrong >=80% of the time):")
    print("-" * 90)
    print(f"{'Sample ID':40s} {'Label':12s} {'Most Pred':12s} {'Wrong':6s} {'Total':6s} {'Rate':6s}")
    print("-" * 90)
    for s in sample_stats:
        if s["error_rate"] >= 0.8:
            print(f"{s['sample_id']:40s} {s['label']:12s} {s['most_common_pred']:12s} "
                  f"{s['n_wrong']:6d} {s['n_val']:6d} {s['error_rate']:6.1%}")

    print()
    print("SUSPICIOUS (wrong 50-79% of the time):")
    print("-" * 90)
    for s in sample_stats:
        if 0.5 <= s["error_rate"] < 0.8:
            print(f"{s['sample_id']:40s} {s['label']:12s} {s['most_common_pred']:12s} "
                  f"{s['n_wrong']:6d} {s['n_val']:6d} {s['error_rate']:6.1%}")

    print()
    print("HARD BUT PROBABLY CORRECT (wrong 30-49% of the time):")
    print("-" * 90)
    for s in sample_stats:
        if 0.3 <= s["error_rate"] < 0.5:
            print(f"{s['sample_id']:40s} {s['label']:12s} {s['most_common_pred']:12s} "
                  f"{s['n_wrong']:6d} {s['n_val']:6d} {s['error_rate']:6.1%}")

    # Save full results to CSV
    csv_path = "kfold_results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["sample_id", "label", "most_common_pred",
                                                "n_wrong", "n_val", "error_rate", "wrong_preds"])
        writer.writeheader()
        for s in sample_stats:
            row = dict(s)
            row["wrong_preds"] = "|".join(row["wrong_preds"])
            writer.writerow(row)
    print(f"\nFull results saved to {csv_path}")


class TransformSubset(Dataset):
    """Subset of AllSamplesDataset with a specific transform."""
    def __init__(self, dataset, indices, transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_path, label, sample_id = self.dataset.samples[self.indices[idx]]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        target = [char_to_idx[c] for c in label]
        return img, torch.tensor(target, dtype=torch.long), len(target), label, sample_id


if __name__ == "__main__":
    main()
