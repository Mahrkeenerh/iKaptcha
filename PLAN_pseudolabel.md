# Pseudo-labeling Plan

## Goal
Use our best model (94.0% accuracy) to label ~2200 unlabeled real captchas, then retrain and evaluate on the original YOLO val split to see if more real data improves accuracy.

## Data Sources
- `real_samples_unlabeled/` — 1309 freshly fetched captchas (numbered PNGs)
- `real_samples/` — 900 previously fetched captchas (named `{label}_{index}.png` — but these labels are from an OLD model and NOT trustworthy)

## Step 1: Pseudo-label all unlabeled images
Write `pseudo_label.py` that:
1. Loads model from `best_mixed.pth` (the 94.0% model)
2. Uses the CRNN architecture from `train_phased.py` (1.66M params, hidden_size=128, NUM_CLASSES=29)
3. Runs inference on every PNG in `real_samples_unlabeled/` and `real_samples/`
4. For each image, compute per-character confidence (max softmax at each CTC timestep along the decoded path)
5. Compute sequence confidence = mean of per-char confidences
6. Save results to `pseudo_labels.csv`: `image_path, predicted_label, confidence`
7. Print stats: total labeled, confidence distribution, how many pass each threshold

## Step 2: Assemble training dataset
Write `prepare_pseudolabel_dataset.py` that:
1. Reads `pseudo_labels.csv`
2. Filters by confidence threshold (try ≥0.90)
3. Copies passing images to `dataset_pseudo/images/train/`
4. Creates text labels in `dataset_pseudo/text_labels/train/` (UPPERCASE, matching existing format)
5. Prints how many samples passed the filter

## Step 3: Train on original YOLO train + pseudo-labeled + 100 hand-labeled
Use `train_phased.py` with modifications:
- Training data: YOLO original train (1200 from `ikariam_pirate_captcha_dataset/images/train/`) + pseudo-labeled + reviewed samples (99 from `real_samples_reviewed/`, with corrections applied: `j6pj3f`→`jgpj3f`, exclude `qw7b9m`)
- Val data: Original YOLO val (300 from `ikariam_pirate_captcha_dataset/images/val/` with `text_labels/val/`) — UNCORRECTED, to compare against our previous 93.3%
- Same 2-phase training (pretrain on 65k synthetic, mixed with weighted sampling)
- The pseudo-labeled data should get **0.5x loss weight** (or just rely on the confidence filtering being strict enough)

## Step 4: Evaluate
- Eval on original YOLO val (300) — compare vs 93.3% baseline
- Eval on corrected val (299) — compare vs 94.0% baseline
- Use `eval_compare.py` for both evaluations

## Key Details
- Model architecture: `CRNN` from `train_phased.py` with `hidden_size=128`
- Val transform: Resize(48,256), ToTensor, Normalize([0.5]*3, [0.5]*3)
- CTC greedy decode with CHARSET = "abcdefghjklmnpqrstuvwxy23457"
- All labels should be lowercase internally, UPPERCASE in text label files

## Expected Outcome
Adding ~2000 confidence-filtered real samples should improve accuracy by +1-2% because:
- More diverse real-world examples reduce overfitting to the 1200 YOLO train samples
- Even with ~6% label noise, the volume of correct labels outweighs the noise
