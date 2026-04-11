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

### Training

**Standard (Baseline) Models:**
```bash
# Binary classification
python models/binary/train.py --config configs/binary_config.yaml

# Multi-class classification
python models/multi/train.py --config configs/multi_config.yaml
```

**Enhanced Models (with Self-Attention + CNN + AECR):**
```bash
# Binary classification with enhanced architecture
python models/binary/train.py --config configs/binary_config_enhanced.yaml

# Multi-class classification with enhanced architecture
python models/multi/train.py --config configs/multi_config_enhanced.yaml
```

### Inference
```bash
# Binary classification
python models/binary/predict.py \
  -i input.fasta \
  -m checkpoints/binary/binary_best.pth \
  -o results/predictions.csv

# Multi-class classification
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
- **Multi-class Labels**: Label mapping saved in metadata.json for inference consistency
- **Class Balancing**: Uses pos_weight (binary) and FocalLoss (multi-class)
- **AMP**: Uses `torch.cuda.amp` (deprecated warnings are OK)
- **Enhanced Features**: Self-Attention, Multi-scale CNN, AECR regularization available via enhanced configs
- **Threshold Tuning**: Use `--tune-threshold` in evaluate.py for imbalanced data scenarios
- **Data Preparation**: Use `scripts/quality_control.py` and `scripts/download_negative_samples.py` for data prep

## Documentation Files

- `README.md`: Project overview and usage guide
- `workflow_orchestration.md`: Collaboration workflow guidelines
- `CLAUDE.md`: This file - Claude Code quick reference
