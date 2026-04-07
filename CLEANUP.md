# Repository Cleanup Guide

Keep only what's needed for reproducibility + production inference. Everything else can go.

## KEEP — Production / Reproducibility

### Code (core)
- `train_phased.py` — training script (the one that works)
- `generate_captcha.py` — synthetic captcha generator (28-char charset, targeted confusables)
- `generate_dataset.py` — multiprocess synthetic data generation
- `prepare_pseudo_train_v2.py` — assembles the production training dataset
- `pseudo_label.py` — runs inference with confidence scoring, saves CSV
- `kfold_validate.py` — k-fold cross-validation for label quality checking
- `eval_compare.py` — evaluates a model on original + corrected val
- `eval_yolo.py` — YOLO baseline evaluation (original dataset)
- `eval_yolo_corrected.py` — YOLO baseline on corrected val
- `fetch_captchas.py` — fetches raw captchas from game server
- `predict.py` — single-image inference (needs update to match current arch)

### Models
- `final_mixed.pth` — **production CRNN** (97.3% corrected val / 95.0% original val)
- `best_pretrain.pth` — final synthetic-only pretrain checkpoint (used as init for mixed)
- `yolov8n-ikariam-pirates-mAP-0_989.onnx` — YOLO baseline for comparison

### Datasets
- `dataset_pseudo_v2/` — production training dataset (11,210 real + 298 corrected val)
- `dataset_synthetic/` — 65k synthetic (can be regenerated with `generate_dataset.py`)
- `ikariam_pirate_captcha_dataset/` — original YOLO dataset (keep for apples-to-apples comparison)
- `real_samples_reviewed/` — 100 hand-labeled real captchas
- `real_samples_unlabeled/` — raw fetched captchas (kept as raw data pool)
- `real_samples/` — older fetched captchas (kept as raw data pool)
- `samples/` — original 9 test samples from CLAUDE.md

### Docs
- `CLAUDE.md` — project overview and character set info
- `FINDINGS.md` — final results and design rationale
- `CLEANUP.md` — this file
- `ANALYSIS.md` — earlier analysis (if still relevant)

### Infra
- `.venv/` — Python environment
- `fonts/` — fonts used by `generate_captcha.py` (required for regeneration)

## DELETE — Intermediate / Failed Experiments

### Scripts (superseded)
- `train.py` — old single-phase training, superseded by `train_phased.py`
- `train_vit.py` — ViT-from-scratch experiment (failed, 71%)
- `train_parseq.py` — pretrained PARSeq experiment (failed, 59%)
- `finetune.py` — old fine-tuning script, not used
- `fetch_and_label.py` — old fetcher that auto-labeled on the fly (use `fetch_captchas.py` + `pseudo_label.py` instead)
- `label_server.py` + `label_ui.html` — old labeling UI (we used the HTML review pages instead)
- `test_checkpoints.py` — old script
- `test_ikabotapi.py` — one-off test
- `error_analysis.py` — written by an analysis agent, can be regenerated if needed
- `prepare_dataset.py` — old dataset builder, superseded by `prepare_pseudo_train_v2.py`
- `prepare_pseudo_train.py` — earlier version, superseded by v2
- `prepare_pseudolabel_dataset.py` — written by an agent, superseded

### Datasets (intermediate/superseded)
- `dataset/` — very old 50k synthetic from earlier work
- `dataset_combined/` — superseded by `dataset_pseudo_v2`
- `dataset_pseudo/` — intermediate, superseded
- `dataset_pseudo_experiment/` — intermediate, superseded
- `real_samples_deleted/` — empty directory
- `generated_samples/` — small test outputs from the generator

### Models (old / intermediate)
- `best_model.pth` — very old, old architecture
- `best_mixed.pth` — best-by-val-acc (we ship `final_mixed.pth` now for honest eval)
- `best_swa.pth` — stale from previous run, SWA was counterproductive
- `best_finetune.pth` — old
- `best_pretrain_mh.pth`, `best_mixed_mh.pth` — old experiment runs
- `best_vit_pretrain.pth`, `best_vit_mixed.pth` — failed ViT experiment
- `best_parseq_pretrain.pth`, `best_parseq_mixed.pth` — failed PARSeq experiment
- `best_retrain.pth`, `best_pseudo_v1.pth`, `best_production.pth` — old snapshots
- `checkpoints/` — epoch checkpoints from old runs
- `checkpoints_pretrain/`, `checkpoints_mixed/` — latest run's per-10-epoch checkpoints (keep if you want, they're big)
- `checkpoints_vit_pretrain/`, `checkpoints_vit_mixed/` — failed experiment
- `checkpoints_parseq_pretrain/`, `checkpoints_parseq_mixed/` — failed experiment

### Review folders (one-time use)
- `review_mislabels/` — round 1 review artifacts
- `review_pseudo/` — pseudo-label review artifacts
- `review_v2/` — round 2 val review artifacts

### CSVs / plans (one-time use)
- `kfold_results.csv` — from k-fold, if corrections are already applied
- `pseudo_labels.csv` — from pseudo-labeling, if dataset is already built
- `PLAN_pseudolabel.md` — was a working doc, no longer needed

## Suggested Cleanup Commands

```bash
# Superseded scripts
rm train.py train_vit.py train_parseq.py finetune.py fetch_and_label.py \
   label_server.py label_ui.html test_checkpoints.py test_ikabotapi.py \
   error_analysis.py prepare_dataset.py prepare_pseudo_train.py \
   prepare_pseudolabel_dataset.py

# Old models
rm best_model.pth best_mixed.pth best_swa.pth best_finetune.pth \
   best_pretrain_mh.pth best_mixed_mh.pth \
   best_vit_pretrain.pth best_vit_mixed.pth \
   best_parseq_pretrain.pth best_parseq_mixed.pth \
   best_retrain.pth best_pseudo_v1.pth best_production.pth

# Intermediate datasets
rm -rf dataset dataset_combined dataset_pseudo dataset_pseudo_experiment \
      real_samples_deleted generated_samples

# Old checkpoints
rm -rf checkpoints checkpoints_vit_pretrain checkpoints_vit_mixed \
       checkpoints_parseq_pretrain checkpoints_parseq_mixed

# Review folders (artifacts, corrections already applied)
rm -rf review_mislabels review_pseudo review_v2

# Working docs (info now in FINDINGS.md)
rm PLAN_pseudolabel.md
# Keep kfold_results.csv and pseudo_labels.csv if you want a reproducibility trail
```

## To Reproduce Production Model From Scratch

```bash
# 1. Regenerate synthetic data
python generate_dataset.py --count-train 65000 --count-val 3000

# 2. Fetch ~10k real captchas (takes ~5 hours at safe rate)
python fetch_captchas.py --count 10000 --cookie "YOUR_COOKIE"

# 3. Pseudo-label with an earlier model (or train one first on YOLO data alone)
python pseudo_label.py --checkpoint best_pretrain.pth

# 4. Build production dataset
python prepare_pseudo_train_v2.py

# 5. Train
DATASET_DIR=dataset_pseudo_v2 python train_phased.py --phase all

# 6. Evaluate
python eval_compare.py --checkpoint final_mixed.pth
```
