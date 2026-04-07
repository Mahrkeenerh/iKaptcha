# IkaCaptcha

A CRNN that solves the Ikariam pirate fortress captcha.

| Model | Params | Original val | Corrected val |
|---|---|---|---|
| YOLOv8n (IkabotAPI baseline) | ~3 M | 78.7% | 81.2% |
| **CRNN (this repo)** | **1.66 M** | **95.0%** | **97.3%** |

Character accuracy: **99.5%**. Model file: `crnn.onnx` (6.4 MB).

## Quick start

```bash
uv venv && uv pip install -e .
python predict_onnx.py samples/test1.png samples/test2.png
```

```
samples/test1.png: b45d5eee  (conf=0.993)
samples/test2.png: aqrckd3   (conf=0.995)
```

`predict_onnx.py` depends only on `onnxruntime`, `Pillow`, and `numpy` — no PyTorch needed at inference time. Use it as the reference implementation when porting to other runtimes (browser via `onnxruntime-web`, mobile, etc).

## Character set

The captcha uses **only 28 characters**, not 36. The game server excludes visually ambiguous ones:

```
Letters (24): A B C D E F G H J K L M N P Q R S T U V W X Y
Digits  (4):  2 3 4 5 7
Excluded:     0 1 6 8 9 I O Z
```

## What's in the repo

- `crnn.onnx` — production model (ship this)
- `final_mixed.pth` — same weights, PyTorch format
- `predict_onnx.py` / `predict.py` — single-image inference (ONNX / PyTorch)
- `train_phased.py` — two-phase training script
- `export_onnx.py` — re-export PyTorch → ONNX with parity verification
- `dataset_pseudo_v2/` — production dataset (11,210 train + 298 corrected val)
- `ikariam_pirate_captcha_dataset/` — original 1,200/300 YOLO dataset (kept for baseline comparison)
- `FINDINGS.md` — full technical writeup: architecture, training, what worked / didn't, error analysis

## Credits

YOLOv8n baseline and original 1,500-sample dataset from [IkabotAPI](https://github.com/Ikabot-Collective/IkabotAPI).
