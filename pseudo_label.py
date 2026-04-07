"""
pseudo_label.py — Run inference on all images in real_samples_unlabeled/ and real_samples/,
save predicted labels + per-sequence confidence to pseudo_labels.csv.

Confidence is defined as the mean softmax probability at each non-blank CTC timestep
along the greedily decoded path.  This gives a per-character confidence whose mean
represents how certain the model is about the full sequence.

Usage:
    python pseudo_label.py [--checkpoint best_mixed.pth] [--batch-size 64]
"""

import argparse
import csv
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# ---------------------------------------------------------------------------
# Import model components from train_phased (without executing main)
# ---------------------------------------------------------------------------
import importlib.util, sys

_spec = importlib.util.spec_from_file_location(
    "train_phased",
    os.path.join(os.path.dirname(__file__), "train_phased.py"),
)
_mod = importlib.util.module_from_spec(_spec)
# Prevent argparse from running inside train_phased on import
_orig_argv = sys.argv
sys.argv = sys.argv[:1]
_spec.loader.exec_module(_mod)
sys.argv = _orig_argv

CRNN = _mod.CRNN
CHARSET = _mod.CHARSET
BLANK = _mod.BLANK
NUM_CLASSES = _mod.NUM_CLASSES
idx_to_char = _mod.idx_to_char
IMG_H = _mod.IMG_H
IMG_W = _mod.IMG_W

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------------------------
# Transforms (val — no augmentation)
# ---------------------------------------------------------------------------

VAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


# ---------------------------------------------------------------------------
# Confidence-aware greedy decode
# ---------------------------------------------------------------------------

@torch.no_grad()
def greedy_decode_with_confidence(logits):
    """
    Greedy CTC decode that also returns per-sequence mean confidence.

    Args:
        logits: (T, B, C) raw model output (not log-softmaxed)

    Returns:
        List of (label_str, confidence_float) tuples, one per batch item.
    """
    probs = F.softmax(logits, dim=2)          # (T, B, C) — softmax over classes
    indices = logits.argmax(dim=2)             # (T, B) — greedy class per timestep
    probs_t = probs.permute(1, 0, 2)          # (B, T, C)
    indices_t = indices.permute(1, 0)         # (B, T)

    results = []
    for b in range(indices_t.shape[0]):
        seq = indices_t[b].tolist()           # length T
        prob_seq = probs_t[b]                 # (T, C)

        chars = []
        char_confidences = []
        prev = None
        for t, idx in enumerate(seq):
            if idx != prev and idx != BLANK:
                chars.append(idx_to_char[idx])
                char_confidences.append(prob_seq[t, idx].item())
            prev = idx

        label = "".join(chars)
        if char_confidences:
            confidence = sum(char_confidences) / len(char_confidences)
        else:
            confidence = 0.0

        results.append((label, confidence))

    return results


# ---------------------------------------------------------------------------
# Dataset — load raw PNGs with their file paths
# ---------------------------------------------------------------------------

def collect_png_paths(*dirs):
    """Return sorted list of absolute PNG paths from the given directories."""
    paths = []
    for d in dirs:
        d = Path(d)
        if not d.exists():
            print(f"  Warning: directory not found, skipping: {d}")
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() == ".png":
                paths.append(str(p.resolve()))
    return paths


def load_batch(paths, transform):
    """Load a list of image paths into a stacked tensor."""
    tensors = []
    valid_paths = []
    for p in paths:
        try:
            img = Image.open(p).convert("RGB")
            tensors.append(transform(img))
            valid_paths.append(p)
        except Exception as e:
            print(f"  Warning: could not load {p}: {e}")
    if not tensors:
        return None, []
    return torch.stack(tensors, 0), valid_paths


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Pseudo-label real captcha images")
    parser.add_argument("--checkpoint", default="best_mixed.pth",
                        help="Model checkpoint to load (default: best_mixed.pth)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Inference batch size (default: 64)")
    parser.add_argument("--output", default="pseudo_labels.csv",
                        help="Output CSV file (default: pseudo_labels.csv)")
    args = parser.parse_args()

    # Resolve paths relative to this script's directory
    base = Path(__file__).parent
    checkpoint_path = base / args.checkpoint
    unlabeled_dir = base / "real_samples_unlabeled"
    labeled_dir = base / "real_samples"
    output_path = base / args.output

    print(f"Device: {DEVICE}")
    print(f"Checkpoint: {checkpoint_path}")

    # Load model
    model = CRNN(NUM_CLASSES, hidden_size=128).to(DEVICE)
    state = torch.load(str(checkpoint_path), map_location=DEVICE, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    print(f"Model loaded ({sum(p.numel() for p in model.parameters()):,} params)")

    # Collect all image paths
    all_paths = collect_png_paths(unlabeled_dir, labeled_dir)
    print(f"\nImages found:")
    print(f"  real_samples_unlabeled/: {sum(1 for p in all_paths if 'real_samples_unlabeled' in p)}")
    print(f"  real_samples/:           {sum(1 for p in all_paths if '/real_samples/' in p and 'unlabeled' not in p)}")
    print(f"  Total:                   {len(all_paths)}")

    # Run inference in batches
    rows = []
    total = len(all_paths)
    bs = args.batch_size

    print(f"\nRunning inference (batch_size={bs})...")
    for start in range(0, total, bs):
        batch_paths = all_paths[start:start + bs]
        images, valid_paths = load_batch(batch_paths, VAL_TRANSFORM)
        if images is None:
            continue

        images = images.to(DEVICE)
        with torch.no_grad():
            logits = model(images)  # (T, B, C)

        decoded = greedy_decode_with_confidence(logits)
        for path, (label, confidence) in zip(valid_paths, decoded):
            rows.append((path, label, confidence))

        if (start // bs) % 10 == 0:
            pct = min(start + bs, total) / total * 100
            print(f"  {min(start + bs, total)}/{total} ({pct:.0f}%)")

    print(f"  Done — {len(rows)} images processed")

    # Write CSV
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["image_path", "predicted_label", "confidence"])
        writer.writerows(rows)
    print(f"\nSaved: {output_path}")

    # Print confidence distribution stats
    confidences = [r[2] for r in rows]
    thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 0.99]
    print("\nConfidence distribution:")
    print(f"  Total images: {len(confidences)}")
    print(f"  Mean:   {sum(confidences)/len(confidences):.4f}")
    print(f"  Median: {sorted(confidences)[len(confidences)//2]:.4f}")
    print(f"  Min:    {min(confidences):.4f}")
    print(f"  Max:    {max(confidences):.4f}")
    print()
    print("  Threshold | Pass | Pass%")
    print("  ----------+------+------")
    for t in thresholds:
        passing = sum(1 for c in confidences if c >= t)
        print(f"  >= {t:.2f}   | {passing:4d} | {passing/len(confidences)*100:.1f}%")


if __name__ == "__main__":
    main()
