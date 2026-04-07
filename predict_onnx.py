"""
Predict captcha labels using the production ONNX model.

This is the lightweight inference path — depends only on onnxruntime, Pillow,
and numpy. No PyTorch needed. Use this as the reference implementation when
porting inference to other runtimes (onnxruntime-web, etc.).

Usage:
    python predict_onnx.py samples/test1.png samples/test2.png ...
    python predict_onnx.py --model crnn.onnx samples/*.png
"""

import argparse

import numpy as np
import onnxruntime as ort
from PIL import Image

# Must match train_phased.py / export_onnx.py exactly.
CHARSET = "abcdefghjklmnpqrstuvwxy23457"
BLANK = 0
IDX_TO_CHAR = {i + 1: c for i, c in enumerate(CHARSET)}

IMG_H = 48
IMG_W = 256
MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)


def preprocess(image_path: str) -> np.ndarray:
    """Load image, resize to 48x256, normalize, return (1, 3, 48, 256) float32."""
    img = Image.open(image_path).convert("RGB").resize((IMG_W, IMG_H), Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0          # (H, W, 3)
    arr = (arr - MEAN) / STD                                 # normalize
    arr = arr.transpose(2, 0, 1)                             # (3, H, W)
    return arr[None, ...].astype(np.float32)                 # (1, 3, H, W)


def greedy_ctc_decode(logits_tbc: np.ndarray) -> list[str]:
    """CTC greedy decode on (T, B, C) logits. Collapses repeats, drops blanks."""
    indices = logits_tbc.argmax(axis=2).transpose(1, 0)      # (B, T)
    out = []
    for seq in indices:
        chars = []
        prev = None
        for idx in seq.tolist():
            if idx != prev and idx != BLANK:
                chars.append(IDX_TO_CHAR[idx])
            prev = idx
        out.append("".join(chars))
    return out


def confidence(logits_tbc: np.ndarray) -> float:
    """Mean of per-timestep max softmax — single-image only."""
    x = logits_tbc[:, 0, :]                                  # (T, C)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    p = e / e.sum(axis=1, keepdims=True)
    return float(p.max(axis=1).mean())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="crnn.onnx")
    parser.add_argument("images", nargs="+")
    args = parser.parse_args()

    session = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])

    for path in args.images:
        x = preprocess(path)
        logits = session.run(["logits"], {"input": x})[0]    # (T, 1, C)
        pred = greedy_ctc_decode(logits)[0]
        conf = confidence(logits)
        print(f"{path}: {pred}  (conf={conf:.3f})")


if __name__ == "__main__":
    main()
