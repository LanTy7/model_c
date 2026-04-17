# ARG Classification with Deep Learning

This repository contains deep learning models for **Antibiotic Resistance Gene (ARG) identification and classification**:
- **Binary classification**: Identify if a protein sequence is an ARG
- **Multi-class classification**: Classify ARGs into antibiotic resistance categories

Models use **BiLSTM + Self-Attention + Multi-scale CNN** architecture with modular design for maintainability and extensibility. These features are now enabled by default in all standard configs.

## Architecture Overview

### 1. Core Components (Default)
- **Self-Attention Mechanism**: Multi-head attention after BiLSTM for better focus on important sequence positions
- **Multi-scale CNN**: Parallel convolutions with kernel sizes [3, 5, 7] to capture local motifs at different scales
- **AECR Regularization**: Attention Entropy and Continuity Regularization for sharper, smoother attention patterns

### 2. Class Imbalance Handling
Real-world ARG prevalence is ~0.1-1%, but training data is 1:1 balanced. We implemented:
- **Custom pos_weight**: Train with `pos_weight > 1` to emphasize positive class recall
- **Threshold Tuning**: Use `models/binary/evaluate.py --tune-threshold` to auto-search optimal classification threshold (optimizing F1/F2) instead of fixed 0.5. Validation metrics during training still use 0.5.

### 3. Improved Data Pipeline
- **Quality Control**: Retain sequences with X/B/Z/J amino acids (model supports them)
- **Stratified Splitting**: Maintain category distribution in train/val/test splits

## Technology Stack

- **Deep Learning**: PyTorch (with AMP mixed precision training)
- **Sequence Processing**: Biopython (SeqIO), pandas
- **Metrics/Evaluation**: scikit-learn
- **Data Science**: numpy, pandas, matplotlib, seaborn
- **Configuration**: YAML config files
- **Environment**: Conda environment `gene_pred`

## Repository Structure

```
configs/                      # YAML configuration files
  binary_config.yaml         # Binary classification config (with attention + CNN + AECR)
  multi_config.yaml          # Multi-class classification config (with attention + CNN + AECR)

models/                       # Modular model implementations
  common/                    # Shared components
    bilstm.py               # BiLSTM backbone with self-attention
    attention.py            # Multi-head self-attention module
    multiscale_cnn.py       # Multi-scale CNN for local feature extraction
    aecr_loss.py            # AECR regularization loss
    trainer.py              # Unified training framework (AMP, early stopping, AECR)
  binary/                    # Binary classification
    model.py                # BinaryARGClassifier (CNN + Attention + AECR)
    train.py                # Training script (supports pos_weight)
    evaluate.py             # Evaluation script with threshold tuning
    predict.py              # Inference script
  multi/                     # Multi-class classification
    model.py                # MultiClassARGClassifier + FocalLoss (CNN + Attention + AECR)
    train.py                # Training script
    evaluate.py             # Evaluation script
    predict.py              # Inference script

data/                        # Dataset storage
  train.csv                 # Training data (sequence + labels)
  val.csv                   # Validation data
  test.csv                  # Test data
  train.fasta               # FASTA format (backup)
  dataset.py                # PyTorch Dataset classes

utils/                       # Utility functions
  sequence_utils.py         # One-hot encoding, sequence indexing

checkpoints/                 # Saved model checkpoints
  binary/binary_best.pth
  multi/multi_best.pth + metadata.json

logs/                        # Training logs
figures/                     # Training curves and visualizations
```

## Common Development Commands

### Environment Setup

```bash
# Activate conda environment
conda activate gene_pred
```

### Training

Training is done via command-line scripts (not notebooks):

**Binary Classification:**
```bash
python models/binary/train.py --config configs/binary_config.yaml
```

**Multi-class Classification:**
```bash
python models/multi/train.py --config configs/multi_config.yaml
```

Key configuration in YAML files:
- `model`: Architecture parameters (hidden_size, num_layers, dropout, attention heads, CNN kernels)
- `training`: Hyperparameters (epochs, batch_size, lr, etc.) including AECR regularization settings
- `data`: Data paths and preprocessing settings

Note: The default configs (`binary_config.yaml` and `multi_config.yaml`) already include Self-Attention, Multi-scale CNN, and AECR regularization.

### Inference

**Binary Classification:**
```bash
mkdir -p results
python models/binary/predict.py \
  -i input.fasta \
  -m checkpoints/binary/binary_best.pth \
  -o results/predictions.csv
```

**Multi-class Classification:**
> Note: multi-class inference requires `metadata.json` (saved during training) to load class names and `max_length`. By default, `predict.py` looks for it in the same directory as the checkpoint.

```bash
mkdir -p results
python models/multi/predict.py \
  -i input.fasta \
  -m checkpoints/multi/multi_best.pth \
  -o results/classifications.csv
```

## Key Architecture Decisions

### Model Configuration
- **Binary model**: Embedding (vocab_size=25) -> Multi-scale CNN -> BiLSTM + Self-Attention (hidden=128, 2 layers) -> FC
- **Multi-class model**: One-hot (21 dims) -> Multi-scale CNN -> BiLSTM + Self-Attention (hidden=256, 3 layers) -> FC with Focal Loss
- **AECR**: Attention Entropy and Continuity Regularization is applied during training by default

### Data Processing
- Amino acid vocabulary: 20 standard + X (unknown) + PAD
- Sequence encoding: One-hot for multi-class, embedding indices for binary
- **Critical**: Use CSV `sequence` column for loading (not FASTA matching due to ID conflicts)
- Multi-class uses masked pooling (PAD channel index=20) to avoid padding contamination

### Training Features
- **AMP**: Automatic Mixed Precision training for speed
- **Cosine Warmup**: Learning rate scheduling
- **Early Stopping**: Based on validation metric
- **Gradient Clipping**: Max norm = 1.0
- **Class Balancing**: pos_weight for binary, class_weights + FocalLoss for multi-class

### Label Management (Multi-class)
- Categories extracted from `arg_category` column in CSV
- Rare classes (below `min_samples` threshold) merged into "Others"
- Label mapping saved in metadata.json for inference consistency

## Configuration Guide

### Binary Config Example
```yaml
model:
  vocab_size: 25
  embedding_dim: 128
  hidden_size: 128
  num_layers: 2
  dropout: 0.4
  max_length: 1000
  num_attention_heads: 4
  attention_dropout: 0.1
  cnn_out_channels: 64
  cnn_kernel_sizes: [3, 5, 7]

training:
  epochs: 100
  batch_size: 128
  lr: 0.0005
  weight_decay: 0.02
  patience: 15
  lambda_aecr: 0.1
  aecr_sigma: 3.0
```

### Multi-class Config Example
```yaml
model:
  input_size: 21
  hidden_size: 256
  num_layers: 3
  dropout: 0.4
  num_attention_heads: 8
  attention_dropout: 0.1
  cnn_out_channels: 64
  cnn_kernel_sizes: [3, 5, 7]

focal_loss:
  gamma: 1.0
  label_smoothing: 0.1
  class_weight_method: "sqrt"
  class_weight_clip: [0.5, 3.0]

training:
  lambda_aecr: 0.1
  aecr_sigma: 3.0

data:
  min_samples: 40               # Merge rare categories
  max_length_percentile: 95     # Compute max length from training data percentile
```

## Performance Tuning Tips

1. **Batch Size**: 256 works well for both tasks; reduce if OOM
2. **Learning Rate**: 0.0005 is a good starting point; use warmup
3. **Hidden Size**: Binary 128, Multi-class 256
4. **Dropout**: 0.4 for both (higher if overfitting)
5. **Num Layers**: Binary 2, Multi-class 3

## Troubleshooting

### Low Validation Accuracy
- Check data loading uses CSV `sequence` column (not FASTA ID matching)
- Verify label distribution is reasonable
- Check for train/val data leakage

### Out of Memory
- Reduce batch_size
- Reduce hidden_size or num_layers
- Enable gradient accumulation

### Slow Training
- Increase num_workers in DataLoader
- Ensure AMP is enabled (use_amp: true)
- Check GPU utilization

## Important Notes

1. **Data Loading**: Always use `train.csv`/`val.csv` with `sequence` column, not FASTA files, to avoid ID conflicts
2. **Model Checkpointing**: Best model saved based on validation F1 for both binary and multi-class
3. **Random Seeds**: Fixed seed (42) used for reproducibility in training scripts
4. **Multi-class Labels**: Must handle "Others" category carefully; label mapping saved in metadata.json
5. **Mixed Precision**: Uses `torch.cuda.amp` (deprecated warnings are OK)
6. **Default Architecture**: Self-Attention, Multi-scale CNN, and AECR regularization are enabled by default in both binary and multi-class configs

## Documentation Files

- `CLAUDE.md`: Project guidance for Claude Code
- `workflow_orchestration.md`: Workflow guidelines for collaboration
- `README.md`: This file - project overview and usage guide
