# Ikariam Captcha Solver

## Goal
Automate Ikariam pirate fortress captcha solving — either via browser extension or using ikabot.

## Key Discovery: Existing Solution
- **IkabotAPI** (https://github.com/Ikabot-Collective/IkabotAPI) — YOLOv8n object detection model, 98.9% mAP
- **ikabot** (https://github.com/Ikabot-Collective/ikabot) — Full Python bot with auto-piracy + captcha solving
- Their ONNX model is downloaded locally: `yolov8n-ikariam-pirates-mAP-0_989.onnx`
- Tested on our 9 samples: **6/9 correct** (failures are single-char confusions: i/j, x/y, 3/5)

## Character Set — ONLY 28 chars (NOT 36)
Letters (24): A B C D E F G H J K L M N P Q R S T U V W X Y
Digits (4): 2 3 4 5 7
**Missing: 0 1 6 8 9 I O Z** — excluded because visually ambiguous

## Captcha Characteristics
- Source URL: `https://s53-cz.ikariam.gameforge.com/?action=Options&function=createCaptcha&rand=<random>`
- Server-generated PNG images, ~400x76px
- 4-8 chars from the 28-char set above (case-insensitive validation)
- Characters use fancy/3D fonts, random colors, rotation, spatial warping
- Noise: colored lines, outlined ellipses overlaid on top
- Background: solid pastel/light color (varies per image)
- Fetching a new captcha invalidates the previous one for that session

## Verify Endpoint
- `POST https://s53-cz.ikariam.gameforge.com/index.php`
- Params: `action=PiracyScreen&function=capture&captchaNeeded=1&captcha=<answer>&cityId=1033&position=17&buildingLevel=1&ajax=1`
- Auth: session cookie `ikariam=<value>`

## Our Model — CRNN (production, beats YOLO baseline)

| Model | Params | Original Val | Corrected Val |
|---|---|---|---|
| YOLOv8n (theirs) | ~3M | 78.7% | 81.2% |
| **Our CRNN (final epoch)** | **1.66M** | **95.0%** | **97.3%** |

Character accuracy: **99.5%** (ours) vs 96.5% (YOLO). All our numbers are from the **final training epoch**, not cherry-picked.

### Architecture — `train_phased.py`
1.66M params CRNN:
- 4 ResBlocks with SE (Squeeze-Excitation) attention: 32→64→128→256 channels
- Last 2 blocks use MaxPool(2,1) to preserve horizontal resolution → T=64 timesteps
- AdaptiveAvgPool collapses height
- 1-layer BiLSTM (hidden=128), CTC head
- Input: 48×256 RGB, Output: T=64 × 29 classes

### Training — two phases, ~90 min total
1. **Pretrain** on 65k synthetic (50 epochs, OneCycleLR max_lr=3e-4)
2. **Mixed** real + synthetic (40 epochs, OneCycleLR max_lr=2e-4, 3× real oversample via WeightedRandomSampler)

No warm restarts, no SWA — both were counterproductive with CTC. Weight decay 1e-2, dropout 0.2 (CNN) / 0.4 (FC). Gradient clip 5.0.

### Production Dataset (`dataset_pseudo_v2/`)
- 1,200 YOLO train (original, uncorrected)
- 99 hand-reviewed real samples (from `real_samples_reviewed/`)
- 9,911 pseudo-labeled real samples at ≥0.99 confidence
- **Total: 11,210 real train + 298 corrected val**
- Plus 65k synthetic from `generate_captcha.py`

### Production Model
- **`final_mixed.pth`** — PyTorch CRNN, final epoch, 97.3% corrected val
- **`crnn.onnx`** — ONNX export of the same weights, 100% prediction parity with PyTorch on all 598 val samples. Ship this for deployment (browser extensions, other runtimes). 6.4 MB.

### Deployment
- `predict.py` — PyTorch inference (development/debug), loads `final_mixed.pth`
- `predict_onnx.py` — ONNX inference (production), loads `crnn.onnx`. Depends only on `onnxruntime` + `Pillow` + `numpy`. Use as the reference implementation when porting to other runtimes (e.g. `onnxruntime-web`).
- `export_onnx.py` — re-exports `final_mixed.pth` to `crnn.onnx` and verifies parity on both val sets.

### Key Scripts
- `train_phased.py` — training
- `generate_captcha.py`, `generate_dataset.py` — synthetic data
- `fetch_captchas.py` — raw captcha fetching (safe delays 1.0-2.5s)
- `pseudo_label.py` — confidence-scored auto-labeling
- `prepare_pseudo_train_v2.py` — builds production dataset
- `kfold_validate.py` — k-fold label quality check
- `eval_compare.py` — evaluate on both original + corrected val
- `eval_yolo.py`, `eval_yolo_corrected.py` — YOLO baseline evaluation

See `FINDINGS.md` for full details.

## Labeled Samples (in `samples/`)
| File | Label |
|------|-------|
| test1.png | b45d5eee |
| test2.png | aqrckd3 |
| test3.png | 25t352j |
| test4.png | 5jjpe4b |
| test5.png | hp5xeqf |
| test6.png | xnwcprl |
| test7.png | arh3ml |
| test8.png | dtmbpw |
| test9.png | 7dxesdjv |

## Package Management
- Use `uv` for Python, not `pip`
- Use `uv venv` for virtual environments
