# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Context

See @README.md for project overview, technology stack, and usage instructions.

See @workflow_orchestration.md for collaboration workflow guidelines.

## Quick Reference

### Environment
```bash
conda activate gene_pred
```

### Data Preparation (Run Before Training)

```bash
# Step 1: Create two-stage homology-aware splits
python scripts/create_training_data.py \
  --positive-fasta data/ARG_DB.fasta \
  --negative-fasta data/Non_ARG_DB.fasta \
  --output-dir data \
  --seed 42
```

### Training

**Binary classification — Evaluation (train/val/test split):**
```bash
python models/binary/train.py --config configs/binary_config.yaml --mode single
```

**Binary classification — Production Model (train+val combined):**
```bash
python models/binary/train.py --config configs/binary_config.yaml --mode final
```

**Multi-class classification:**
```bash
python models/multi/train.py --config configs/multi_config.yaml
```

### Inference
```bash
# Binary classification (production model)
mkdir -p results
python models/binary/predict.py \
  -i input.fasta \
  -m checkpoints/binary/binary_final.pth \
  -o results/predictions.csv

# Multi-class classification (requires metadata.json in the same dir as the checkpoint)
mkdir -p results
python models/multi/predict.py \
  -i input.fasta \
  -m checkpoints/multi/multi_best.pth \
  -o results/classifications.csv
```

## Language Guidelines

- **Communication**: Both English and Chinese are acceptable for discussion
- **Code Changes**: All content added to the project (new files, edits, comments, docstrings, variable names) must be in **English**

## Key Reminders

- **Data Loading**: Always use CSV `sequence` column, not FASTA ID matching
- **Data Splitting**: Two-stage homology-aware splitting (MMseqs2 + Louvain) ensures homologous families never cross splits. Run `scripts/create_training_data.py` before training.
- **Data Splitting**: Single 80:10:10 train/val/test split at family level. Rare ARG categories (< 3 families) are extracted and split at sequence level to guarantee coverage.
- **Evaluation**: Use `--mode single` to train on train set, validate on val set, and evaluate on test set.
- **Production Model**: Use `--mode final` to train on train+val combined (with internal 90/10 split for early stopping). Saved as `checkpoints/binary/binary_final.pth`.
- **Multi-class Labels**: Label mapping saved in metadata.json for inference consistency
- **Class Balancing**: Uses pos_weight (binary) and class_weights + unified FocalLoss (`models/common/losses.py`) for multi-class
- **AMP**: Uses `torch.cuda.amp` (deprecated warnings are OK)
- **Enhanced Features**: Self-Attention, Multi-scale CNN, and AECR regularization are enabled by default in the standard configs
- **Threshold Tuning**: Use `--tune-threshold` in evaluate.py for imbalanced data scenarios. Metric-specific threshold files (e.g., `threshold_f1.json`, `threshold_f2.json`) are saved to avoid silent overwrites; predict.py prefers them and falls back to `threshold.json`.
- **Checkpoint Resume**: Checkpoints save optimizer and scheduler state. Use `trainer.load_checkpoint()` to resume training.
- **Safe Loading**: Use `safe_torch_load()` from `utils.common` instead of raw `torch.load()` for secure checkpoint loading. Compatible with both PyTorch >= 2.0 and older versions.
- **Base Model**: `BinaryARGClassifier` and `MultiClassARGClassifier` both inherit from `BaseARGClassifier` (`models/common/base_model.py`). Shared architecture includes CNN, BiLSTM, Attention, Pooling, and Classifier.
- **Architecture Inference**: `evaluate.py` and `predict.py` automatically infer `use_attention` and `use_cnn` from checkpoint state_dict when model_config is not available.

## Documentation Files

- `README.md`: Project overview and usage guide
- `workflow_orchestration.md`: Collaboration workflow guidelines
- `CLAUDE.md`: This file - Claude Code quick reference
