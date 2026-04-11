# ARG Classification with Deep Learning

This repository contains deep learning models for **Antibiotic Resistance Gene (ARG) identification and classification**:
- **Binary classification**: Identify if a protein sequence is an ARG
- **Multi-class classification**: Classify ARGs into antibiotic resistance categories

Models use BiLSTM architecture with modular design for maintainability and extensibility.

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
  binary_config.yaml         # Binary classification config
  multi_config.yaml          # Multi-class classification config

models/                       # Modular model implementations
  common/                    # Shared components
    bilstm.py               # BiLSTM backbone, pooling, classifier head
    trainer.py              # Unified training framework (AMP, early stopping)
  binary/                    # Binary classification
    model.py                # BinaryARGClassifier
    train.py                # Training script
    predict.py              # Inference script
  multi/                     # Multi-class classification
    model.py                # MultiClassARGClassifier + FocalLoss
    train.py                # Training script
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
- `model`: Architecture parameters (hidden_size, num_layers, dropout)
- `training`: Hyperparameters (epochs, batch_size, lr, etc.)
- `data`: Data paths and preprocessing settings

### Inference

**Binary Classification:**
```bash
python models/binary/predict.py \
  -i input.fasta \
  -m checkpoints/binary/binary_best.pth \
  -o results/predictions.csv
```

**Multi-class Classification:**
```bash
python models/multi/predict.py \
  -i input.fasta \
  -m checkpoints/multi/multi_best.pth \
  -o results/classifications.csv
```

## Key Architecture Decisions

### Model Configuration
- **Binary model**: Embedding (vocab_size=22) -> BiLSTM (hidden=128, 2 layers) -> FC
- **Multi-class model**: One-hot (21 dims) -> BiLSTM (hidden=256, 3 layers) -> FC with Focal Loss

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
  vocab_size: 22
  embedding_dim: 128
  hidden_size: 128
  num_layers: 2
  dropout: 0.3

training:
  epochs: 100
  batch_size: 256
  lr: 0.0005
  weight_decay: 0.01
  patience: 15
```

### Multi-class Config Example
```yaml
model:
  input_size: 21
  hidden_size: 256
  num_layers: 3
  dropout: 0.3

focal_loss:
  gamma: 1.0
  label_smoothing: 0.1

data:
  min_samples: 40  # Merge rare categories
```

## Performance Tuning Tips

1. **Batch Size**: 256 works well for both tasks; reduce if OOM
2. **Learning Rate**: 0.0005 is a good starting point; use warmup
3. **Hidden Size**: Binary 128, Multi-class 256
4. **Dropout**: 0.3 for both (higher if overfitting)
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
2. **Model Checkpointing**: Best model saved based on validation metric (F1 for multi-class, loss for binary)
3. **Random Seeds**: Fixed seed (42) used for reproducibility in training scripts
4. **Multi-class Labels**: Must handle "Others" category carefully; label mapping saved in metadata.json
5. **Mixed Precision**: Uses `torch.cuda.amp` (deprecated warnings are OK)

## Documentation Files

- `CLAUDE.md`: Project guidance for Claude Code
- `workflow_orchestration.md`: Workflow guidelines for collaboration
- `README.md`: This file - project overview and usage guide
