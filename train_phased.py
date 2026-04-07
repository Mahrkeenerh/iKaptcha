"""
Two-phase CRNN training for Ikariam captcha recognition.

Phase 1: Pretrain on synthetic data (50k samples)
Phase 2: Mixed training — 3:1 synthetic:real weighted sampling
         with cosine warm restarts (first 75%) + SWA (last 25%)

Usage:
    python train_phased.py --phase all
    python train_phased.py --phase pretrain
    python train_phased.py --phase mixed --checkpoint best_pretrain.pth
"""

import argparse
import math
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from PIL import Image
from torchvision import transforms
import torchvision.transforms.v2 as T

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CHARSET = "abcdefghjklmnpqrstuvwxy23457"
BLANK = 0
NUM_CLASSES = len(CHARSET) + 1  # 28 chars + blank

IMG_W = 256
IMG_H = 48

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

char_to_idx = {c: i + 1 for i, c in enumerate(CHARSET)}
idx_to_char = {i + 1: c for i, c in enumerate(CHARSET)}

# Phase configs: (epochs, lr, weight_decay, batch_size, warmup_epochs)
PHASE_CONFIG = {
    "pretrain": (50, 3e-4, 1e-2, 32, 3),
    "mixed":    (40, 2e-4, 1e-2, 32, 3),
}

# Dataset paths
COMBINED_DIR = os.environ.get("DATASET_DIR", "dataset_pseudo_v2")
SYNTHETIC_TRAIN = "dataset_synthetic/train"
SYNTHETIC_VAL = "dataset_synthetic/val"
REAL_TRAIN_IMG = os.path.join(COMBINED_DIR, "images/train")
REAL_TRAIN_LBL = os.path.join(COMBINED_DIR, "text_labels/train")
REAL_VAL_IMG = os.path.join(COMBINED_DIR, "images/val")
REAL_VAL_LBL = os.path.join(COMBINED_DIR, "text_labels/val")

# SWA config
SWA_START_PCT = 0.75  # Start SWA at 75% of mixed phase
SWA_LR = 1e-4


# ---------------------------------------------------------------------------
# CTC decoding + metrics
# ---------------------------------------------------------------------------

def greedy_decode(logits):
    """Greedy CTC decode: (T, B, C) -> list of strings."""
    indices = logits.argmax(dim=2).permute(1, 0)  # (B, T)
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


def edit_distance(a, b):
    """Levenshtein distance."""
    m, n = len(a), len(b)
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            if a[i - 1] == b[j - 1]:
                dp[j] = prev
            else:
                dp[j] = 1 + min(prev, dp[j], dp[j - 1])
            prev = temp
    return dp[n]


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class TextLabelDataset(Dataset):
    """Load images with separate text label files (YOLO-converted format)."""
    def __init__(self, img_dir, lbl_dir, transform=None):
        self.img_dir = img_dir
        self.lbl_dir = lbl_dir
        self.transform = transform
        self.samples = []
        for fname in os.listdir(img_dir):
            if not fname.lower().endswith(".png"):
                continue
            name = fname.rsplit(".", 1)[0]
            lbl_path = os.path.join(lbl_dir, name + ".txt")
            if os.path.exists(lbl_path):
                with open(lbl_path) as f:
                    label = f.read().strip().lower()
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
        return img, torch.tensor(target, dtype=torch.long), len(target), label


class FilenameDataset(Dataset):
    """Load images where label is encoded in filename: {index}_{label}.png."""
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        self.samples = []
        for fname in os.listdir(img_dir):
            if not fname.lower().endswith(".png"):
                continue
            name = fname.rsplit(".", 1)[0]
            parts = name.split("_", 1)
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
        return img, torch.tensor(target, dtype=torch.long), len(target), label


def collate_fn(batch):
    images, targets, target_lengths, labels = zip(*batch)
    images = torch.stack(images, 0)
    target_lengths = torch.tensor(target_lengths, dtype=torch.long)
    targets = torch.cat(targets, 0)
    return images, targets, target_lengths, labels


# ---------------------------------------------------------------------------
# Model — CRNN with higher horizontal resolution (T=64)
# ---------------------------------------------------------------------------

class SEBlock(nn.Module):
    """Squeeze-and-Excitation: channel attention."""
    def __init__(self, channels, reduction=4):
        super().__init__()
        mid = max(channels // reduction, 8)
        self.fc1 = nn.Linear(channels, mid)
        self.fc2 = nn.Linear(mid, channels)

    def forward(self, x):
        # x: (B, C, H, W)
        w = x.mean(dim=(2, 3))          # (B, C) — squeeze
        w = F.relu(self.fc1(w), inplace=True)
        w = torch.sigmoid(self.fc2(w))   # (B, C) — excitation
        return x * w.unsqueeze(2).unsqueeze(3)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, use_se=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
        self.se = SEBlock(out_ch) if use_se else nn.Identity()

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
            # Block 1: 3x48x256 -> 32x24x128
            ResBlock(3, 32),
            nn.MaxPool2d(2, 2),

            # Block 2: 32x24x128 -> 64x12x64
            ResBlock(32, 64),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            # Block 3: 64x12x64 -> 128x6x64
            ResBlock(64, 128),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.2),

            # Block 4: 128x6x64 -> 256x3x64
            ResBlock(128, 256),
            nn.MaxPool2d((2, 1)),
            nn.Dropout2d(0.2),

            # Collapse height: 256x3x64 -> 256x1x64 -> T=64
            nn.AdaptiveAvgPool2d((1, None)),
        )
        self.rnn = nn.LSTM(
            input_size=256, hidden_size=hidden_size,
            num_layers=1, bidirectional=True, batch_first=True,
        )
        self.dropout = nn.Dropout(0.4)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        conv = self.cnn(x)
        conv = conv.squeeze(2)
        conv = conv.permute(0, 2, 1)
        rnn_out, _ = self.rnn(conv)
        out = self.dropout(rnn_out)
        out = self.fc(out)
        out = out.permute(1, 0, 2)  # (T, B, C) for CTC
        return out


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def get_train_transform():
    return transforms.Compose([
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


def get_val_transform():
    return transforms.Compose([
        transforms.Resize((IMG_H, IMG_W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


# ---------------------------------------------------------------------------
# Training / Validation
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, scheduler_per_batch=None):
    model.train()
    total_loss = 0.0
    count = 0
    for images, targets, target_lengths, _labels in loader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)
        target_lengths = target_lengths.to(DEVICE)

        logits = model(images)
        T, B, _ = logits.shape
        input_lengths = torch.full((B,), T, dtype=torch.long, device=DEVICE)
        log_probs = logits.log_softmax(2)

        loss = criterion(log_probs, targets, input_lengths, target_lengths)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        if scheduler_per_batch is not None:
            scheduler_per_batch.step()

        total_loss += loss.item() * B
        count += B

    return total_loss / count


@torch.no_grad()
def validate(model, loader, criterion):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_edit_dist = 0
    total_gt_chars = 0
    total_samples = 0

    for images, targets, target_lengths, labels in loader:
        images = images.to(DEVICE)
        targets = targets.to(DEVICE)
        target_lengths = target_lengths.to(DEVICE)

        logits = model(images)
        T, B, _ = logits.shape
        input_lengths = torch.full((B,), T, dtype=torch.long, device=DEVICE)
        log_probs = logits.log_softmax(2)

        loss = criterion(log_probs, targets, input_lengths, target_lengths)
        total_loss += loss.item() * B

        preds = greedy_decode(logits)
        for pred, gt in zip(preds, labels):
            total_samples += 1
            if pred == gt:
                total_correct += 1
            total_edit_dist += edit_distance(pred, gt)
            total_gt_chars += len(gt)

    avg_loss = total_loss / len(loader.dataset)
    seq_acc = total_correct / total_samples
    cer = total_edit_dist / max(1, total_gt_chars)
    return avg_loss, seq_acc, cer


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------

def run_pretrain(model, config, train_tf, val_tf):
    """Phase 1: pretrain on synthetic data."""
    epochs, lr, wd, batch_size, warmup = config
    train_ds = FilenameDataset(SYNTHETIC_TRAIN, transform=train_tf)
    val_ds = TextLabelDataset(REAL_VAL_IMG, REAL_VAL_LBL, transform=val_tf)
    print(f"Pretrain: {len(train_ds)} synthetic train | {len(val_ds)} real val")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=4, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, collate_fn=collate_fn, pin_memory=True)

    criterion = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # Warmup then cosine
    def lr_lambda(epoch):
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(1, epochs - warmup)
        return 0.5 * (1 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    ckpt_dir = "checkpoints_pretrain"
    os.makedirs(ckpt_dir, exist_ok=True)
    best_seq_acc = 0.0

    print(f"\n{'='*60}")
    print(f"Phase: pretrain | {epochs} epochs | LR={lr} | WD={wd} | BS={batch_size}")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, seq_acc, cer = validate(model, val_loader, criterion)
        lr_now = optimizer.param_groups[0]["lr"]
        scheduler.step()

        tag = ""
        if seq_acc > best_seq_acc:
            best_seq_acc = seq_acc
            torch.save(model.state_dict(), "best_pretrain.pth")
            tag = " *best*"

        if epoch % 10 == 0 or epoch == epochs:
            torch.save(model.state_dict(),
                       os.path.join(ckpt_dir, f"pretrain_ep{epoch:03d}.pth"))

        print(f"  [pretrain] {epoch:02d}/{epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"seq_acc={seq_acc:.1%}  cer={cer:.4f}  lr={lr_now:.1e}{tag}")

    print(f"\n  Best pretrain accuracy: {best_seq_acc:.1%}")
    return best_seq_acc


def run_mixed(model, config, train_tf, val_tf):
    """Phase 2: mixed training with 3:1 synthetic:real, OneCycleLR."""
    epochs, lr, wd, batch_size, warmup = config

    # Load datasets
    real_ds = TextLabelDataset(REAL_TRAIN_IMG, REAL_TRAIN_LBL, transform=train_tf)
    synth_ds = FilenameDataset(SYNTHETIC_TRAIN, transform=train_tf)
    val_ds = TextLabelDataset(REAL_VAL_IMG, REAL_VAL_LBL, transform=val_tf)

    # Create combined dataset with 3:1 synthetic:real weighting
    combined_ds = ConcatDataset([synth_ds, real_ds])
    weights = [1.0] * len(synth_ds) + [3.0 * len(synth_ds) / len(real_ds)] * len(real_ds)
    sampler = WeightedRandomSampler(weights, num_samples=len(combined_ds), replacement=True)

    print(f"Mixed: {len(real_ds)} real + {len(synth_ds)} synthetic | {len(val_ds)} val")
    print(f"  Weighted sampling: real 3x oversampled")

    train_loader = DataLoader(combined_ds, batch_size=batch_size, sampler=sampler,
                              num_workers=4, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                            num_workers=4, collate_fn=collate_fn, pin_memory=True)

    criterion = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # OneCycleLR: warmup then long smooth decay (no restarts, no SWA)
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.1,       # 10% warmup
        div_factor=10.0,     # initial_lr = max_lr / 10
        final_div_factor=100.0,  # min_lr = initial_lr / 100
    )

    ckpt_dir = "checkpoints_mixed"
    os.makedirs(ckpt_dir, exist_ok=True)
    best_seq_acc = 0.0

    print(f"\n{'='*60}")
    print(f"Phase: mixed | {epochs} epochs | OneCycleLR max_lr={lr} | WD={wd} | BS={batch_size}")
    print(f"  steps/epoch={steps_per_epoch}, pct_start=0.1 (warmup 10%)")
    print(f"{'='*60}\n")

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer,
                                     scheduler_per_batch=scheduler)

        val_loss, seq_acc, cer = validate(model, val_loader, criterion)
        lr_now = optimizer.param_groups[0]["lr"]

        tag = ""
        if seq_acc > best_seq_acc:
            best_seq_acc = seq_acc
            torch.save(model.state_dict(), "best_mixed.pth")
            tag = " *best*"

        if epoch % 10 == 0 or epoch == epochs:
            torch.save(model.state_dict(),
                       os.path.join(ckpt_dir, f"mixed_ep{epoch:03d}.pth"))

        print(f"  [mixed] {epoch:02d}/{epochs}  "
              f"train={train_loss:.4f}  val={val_loss:.4f}  "
              f"seq_acc={seq_acc:.1%}  cer={cer:.4f}  lr={lr_now:.1e}{tag}")

    # Save the FINAL epoch's model alongside the best (honest selection)
    torch.save(model.state_dict(), "final_mixed.pth")
    final_acc = seq_acc

    print(f"\n  Best mixed accuracy: {best_seq_acc:.1%}")
    print(f"  Final-epoch accuracy: {final_acc:.1%}  (saved as final_mixed.pth)")
    return best_seq_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Two-phase CRNN training")
    parser.add_argument("--phase", choices=["pretrain", "mixed", "all"],
                        default="all")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Checkpoint to load before starting")
    args = parser.parse_args()

    train_tf = get_train_transform()
    val_tf = get_val_transform()

    model = CRNN(NUM_CLASSES, hidden_size=128).to(DEVICE)
    if args.checkpoint:
        print(f"Loading checkpoint: {args.checkpoint}")
        model.load_state_dict(torch.load(args.checkpoint, map_location=DEVICE, weights_only=True))

    phases = ["pretrain", "mixed"] if args.phase == "all" else [args.phase]

    for phase in phases:
        config = PHASE_CONFIG[phase]

        if phase == "pretrain":
            run_pretrain(model, config, train_tf, val_tf)
            if "mixed" in phases:
                model.load_state_dict(torch.load("best_pretrain.pth",
                                      map_location=DEVICE, weights_only=True))

        elif phase == "mixed":
            run_mixed(model, config, train_tf, val_tf)

    print("\nDone! Production model saved as final_mixed.pth")


if __name__ == "__main__":
    main()
