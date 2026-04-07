"""
Export the production CRNN to ONNX and verify parity vs PyTorch.

Outputs:
    crnn.onnx — portable model (loadable by onnxruntime, onnxruntime-web, etc.)

Verification:
    Runs both PyTorch and ONNX Runtime predictions on the original YOLO val
    and the corrected val. Reports accuracy for each and counts any mismatches
    between the two backends.

Usage:
    python export_onnx.py [--checkpoint final_mixed.pth] [--output crnn.onnx]
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType

sys.path.insert(0, ".")
from train_phased import (
    CRNN, NUM_CLASSES, BLANK, char_to_idx, idx_to_char,
    greedy_decode, IMG_H, IMG_W,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_transform = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


def load_val(img_dir, lbl_dir):
    samples = []
    for img_path in sorted(Path(img_dir).glob("*.png")):
        lbl_path = Path(lbl_dir) / (img_path.stem + ".txt")
        if not lbl_path.exists():
            continue
        label = lbl_path.read_text().strip().lower()
        if all(c in char_to_idx for c in label):
            samples.append((str(img_path), label, img_path.stem))
    return samples


def greedy_decode_numpy(logits_tbc: np.ndarray):
    """CTC greedy decode on numpy logits with shape (T, B, C)."""
    indices = logits_tbc.argmax(axis=2).transpose(1, 0)  # (B, T)
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


def export_model(checkpoint: str, output: str) -> CRNN:
    model = CRNN(NUM_CLASSES, hidden_size=128).to(DEVICE)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE, weights_only=True))
    model.eval()

    # The last layer of model.cnn is AdaptiveAvgPool2d((1, None)), which the
    # legacy ONNX exporter rejects (output_size contains None). At this stage
    # the feature map is H=3, W=64, so a fixed AvgPool2d((3, 1)) is identical.
    last = model.cnn[-1]
    assert isinstance(last, torch.nn.AdaptiveAvgPool2d), f"unexpected last cnn layer: {last}"
    model.cnn[-1] = torch.nn.AvgPool2d(kernel_size=(3, 1)).to(DEVICE)

    # Sanity check the swap leaves outputs unchanged.
    sanity = torch.randn(2, 3, IMG_H, IMG_W, device=DEVICE)
    with torch.no_grad():
        a = model(sanity)
        model.cnn[-1] = torch.nn.AdaptiveAvgPool2d((1, None)).to(DEVICE)
        b = model(sanity)
        model.cnn[-1] = torch.nn.AvgPool2d(kernel_size=(3, 1)).to(DEVICE)
    assert torch.allclose(a, b, atol=1e-6), "AvgPool swap changed model output"

    dummy = torch.randn(1, 3, IMG_H, IMG_W, device=DEVICE)
    torch.onnx.export(
        model,
        dummy,
        output,
        input_names=["input"],
        output_names=["logits"],
        # logits shape is (T, B, C); T is fixed (64), B is dynamic.
        dynamic_axes={"input": {0: "batch"}, "logits": {1: "batch"}},
        opset_version=18,
        do_constant_folding=True,
        dynamo=False,
    )
    size_kb = Path(output).stat().st_size / 1024
    print(f"Exported {output} ({size_kb:.1f} KB)")
    return model


@torch.no_grad()
def predict_torch(model: CRNN, batch: torch.Tensor):
    logits = model(batch.to(DEVICE))  # (T, B, C)
    return logits.cpu().numpy()


def predict_onnx(session: ort.InferenceSession, batch: torch.Tensor):
    logits = session.run(["logits"], {"input": batch.numpy()})[0]
    return logits


def evaluate(samples, model, session, name: str, batch_size: int = 32):
    print(f"\n=== {name} — {len(samples)} samples ===")
    torch_correct = 0
    onnx_correct = 0
    cross_mismatch = 0
    max_logit_diff = 0.0

    for start in range(0, len(samples), batch_size):
        chunk = samples[start:start + batch_size]
        imgs = torch.stack([
            val_transform(Image.open(p).convert("RGB")) for p, _, _ in chunk
        ])
        labels = [lbl for _, lbl, _ in chunk]

        logits_t = predict_torch(model, imgs)
        logits_o = predict_onnx(session, imgs)

        diff = float(np.max(np.abs(logits_t - logits_o)))
        if diff > max_logit_diff:
            max_logit_diff = diff

        preds_t = greedy_decode_numpy(logits_t)
        preds_o = greedy_decode_numpy(logits_o)

        for pt, po, gt in zip(preds_t, preds_o, labels):
            if pt == gt:
                torch_correct += 1
            if po == gt:
                onnx_correct += 1
            if pt != po:
                cross_mismatch += 1

    n = len(samples)
    print(f"  PyTorch:  {torch_correct}/{n} = {100*torch_correct/n:.1f}%")
    print(f"  ONNX:     {onnx_correct}/{n} = {100*onnx_correct/n:.1f}%")
    print(f"  String mismatches (torch vs onnx): {cross_mismatch}")
    print(f"  Max logit |diff|: {max_logit_diff:.3e}")
    return torch_correct, onnx_correct, cross_mismatch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="final_mixed.pth")
    parser.add_argument("--output", default="crnn.onnx")
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        print(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    model = export_model(args.checkpoint, args.output)
    session = ort.InferenceSession(args.output, providers=["CPUExecutionProvider"])

    # Dynamic int8 quantization (weights only — activations stay float).
    quant_path = str(Path(args.output).with_suffix(".int8.onnx"))
    quantize_dynamic(args.output, quant_path, weight_type=QuantType.QInt8)
    fp32_kb = Path(args.output).stat().st_size / 1024
    int8_kb = Path(quant_path).stat().st_size / 1024
    print(f"Quantized {quant_path} ({int8_kb:.1f} KB, "
          f"{fp32_kb/int8_kb:.1f}x smaller than fp32)")
    session_q = ort.InferenceSession(quant_path, providers=["CPUExecutionProvider"])

    original = load_val(
        "ikariam_pirate_captcha_dataset/images/val",
        "ikariam_pirate_captcha_dataset/text_labels/val",
    )
    corrected = load_val(
        "dataset_pseudo_v2/images/val",
        "dataset_pseudo_v2/text_labels/val",
    )

    print("\n--- fp32 ONNX ---")
    r1 = evaluate(original, model, session, "Original YOLO val")
    r2 = evaluate(corrected, model, session, "Corrected val")

    print("\n--- int8 ONNX ---")
    r1q = evaluate(original, model, session_q, "Original YOLO val")
    r2q = evaluate(corrected, model, session_q, "Corrected val")

    total_mismatch = r1[2] + r2[2]
    print("\n" + "=" * 50)
    if total_mismatch == 0:
        print("fp32 PASS — ONNX matches PyTorch on every sample.")
    else:
        print(f"fp32 WARN — {total_mismatch} samples differ between backends.")
    print(f"int8 mismatches vs PyTorch: original={r1q[2]} corrected={r2q[2]}")


if __name__ == "__main__":
    main()
