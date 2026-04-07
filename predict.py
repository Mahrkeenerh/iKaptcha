"""
Predict captcha labels from image files using the trained CRNN (PyTorch).

For development and debugging. For deployment, prefer predict_onnx.py — it
has no torch dependency and matches the production ONNX file we ship.

Usage:
    python predict.py samples/test1.png samples/test2.png ...
    python predict.py --model final_mixed.pth samples/*.png
"""

import argparse

import torch
from PIL import Image
from torchvision import transforms

from train_phased import (
    CRNN, NUM_CLASSES, IMG_W, IMG_H, greedy_decode,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

val_transform = transforms.Compose([
    transforms.Resize((IMG_H, IMG_W)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])


@torch.no_grad()
def predict(model, image_path: str) -> tuple[str, float]:
    img = Image.open(image_path).convert("RGB")
    tensor = val_transform(img).unsqueeze(0).to(DEVICE)
    logits = model(tensor)  # (T, 1, C)
    pred = greedy_decode(logits)[0]
    # Confidence: mean of per-timestep max softmax along the path
    probs = logits.squeeze(1).softmax(dim=1)
    conf = probs.max(dim=1).values.mean().item()
    return pred, conf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="final_mixed.pth")
    parser.add_argument("images", nargs="+")
    args = parser.parse_args()

    model = CRNN(NUM_CLASSES, hidden_size=128).to(DEVICE)
    model.load_state_dict(torch.load(args.model, map_location=DEVICE, weights_only=True))
    model.eval()

    for path in args.images:
        label, conf = predict(model, path)
        print(f"{path}: {label}  (conf={conf:.3f})")


if __name__ == "__main__":
    main()
