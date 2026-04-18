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
  --n-splits 5 \
  --seed 42
```

### Training

**Binary classification — K-Fold CV (recommended):**
```bash
# Set use_kfold: true in configs/binary_config.yaml
python models/binary/train.py --config configs/binary_config.yaml
```

**Binary classification — Single Split (backward compatible):**
```bash
# Set use_kfold: false in configs/binary_config.yaml
python models/binary/train.py --config configs/binary_config.yaml
```

**Multi-class classification:**
```bash
python models/multi/train.py --config configs/multi_config.yaml
```

### Inference
```bash
# Binary classification
mkdir -p results
python models/binary/predict.py \
  -i input.fasta \
  -m checkpoints/binary/binary_best.pth \
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
- **K-Fold CV**: Binary training supports 5-fold cross-validation via `use_kfold: true` in config. Checkpoints saved per fold in `checkpoints/binary/fold_{i}/`.
- **Novelty Test**: `novelty_test.csv` evaluates true generalization to distant ARG families (<30% identity to training pool).
- **Multi-class Labels**: Label mapping saved in metadata.json for inference consistency
- **Class Balancing**: Uses pos_weight (binary) and FocalLoss (multi-class)
- **AMP**: Uses `torch.cuda.amp` (deprecated warnings are OK)
- **Enhanced Features**: Self-Attention, Multi-scale CNN, and AECR regularization are enabled by default in the standard configs
- **Threshold Tuning**: Use `--tune-threshold` in evaluate.py for imbalanced data scenarios

## Documentation Files

- `README.md`: Project overview and usage guide
- `workflow_orchestration.md`: Collaboration workflow guidelines
- `CLAUDE.md`: This file - Claude Code quick reference
