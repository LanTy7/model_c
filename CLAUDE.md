# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository contains deep learning models for **Antibiotic Resistance Gene (ARG) identification and classification**:
- **Binary classification**: Identify if a protein sequence is an ARG
- **Multi-class classification**: Classify ARGs into antibiotic resistance categories

Models use BiLSTM architecture with one-hot encoding for amino acid sequences.

## Technology Stack

- **Deep Learning**: PyTorch
- **Sequence Processing**: Biopython (SeqIO)
- **Metrics/Evaluation**: scikit-learn
- **Data Science**: numpy, pandas, matplotlib, seaborn
- **Environment**: Conda environment `dl_env`
- **External Tools**: DIAMOND or MMseqs2 for homology-based validation

## Repository Structure

```
binary/                 # Binary classification (ARG vs non-ARG)
  model_train/train.ipynb    # Training notebook
  model_test/predict.ipynb   # Inference notebook

multi/                  # Multi-class classification (ARG categories)
  model_train/train.ipynb    # Training notebook
  model_test/classify.ipynb  # Inference notebook

data/                   # Dataset storage
  ARG_db_all_seq_uniq_representative_rename_2_repsent.fasta  # Main ARG database
  database_construct_description.md                         # Data documentation

scripts/                # Utility scripts
  eval_silver_standard_v0.py       # Real-world performance evaluation
  sample_len_matched_neg.py        # Length-matched negative sampling
  clean_faa_strip_stop.py          # Clean stop codons from .faa files
  clean_faa_from_prescreen_csv.py  # Batch clean based on prescreen results
  prescreen_faa_for_arg.py         # Prescreen .faa files for ARG content

results/                # Evaluation outputs
```

## Common Development Commands

### Environment Setup
```bash
# Activate conda environment
conda activate dl_env
```

### Training
Training is done via Jupyter notebooks (not command-line scripts):
- **Binary**: `binary/model_train/train.ipynb`
- **Multi-class**: `multi/model_train/train.ipynb`

Key configuration in each notebook:
- `MODEL_CONFIG`: Model architecture parameters
- `TRAIN_CONFIG`: Training hyperparameters
- `PATH_CONFIG`: Data and output paths

### Inference
- **Binary**: `binary/model_test/predict.ipynb` - Outputs `*_scores.csv`
- **Multi-class**: `multi/model_test/classify.ipynb` - Outputs `classification_results.csv`

### Data Processing

**Sample length-matched negatives** (for binary classification training):
```bash
python scripts/sample_len_matched_neg.py \
  --pos data/ARG_db_all_seq_uniq_representative_rename_2_repsent.fasta \
  --neg YOUR_NEG.faa \
  --out data/non_arg_lenmatch_r3_seed42.fasta \
  --ratio 3 --max-len 1000 --bin-size 50 --seed 42
```

**Clean stop codons from .faa files**:
```bash
# Single file
python scripts/clean_faa_strip_stop.py --in input.faa --out cleaned.faa

# Directory batch processing
python scripts/clean_faa_strip_stop.py --in input_dir/ --out output_dir/ --recursive
```

**Prescreen .faa files for ARG content** (random sampling for evaluation):
```bash
python scripts/prescreen_faa_for_arg.py \
  --in-dir /path/to/faa_files/ \
  --out-csv results/prescreen_top.csv \
  --arg-db data/ARG_db_all_seq_uniq_representative_rename_2_repsent.fasta \
  --n 50 --seed 42
```

### Evaluation (Silver Standard)

Real-world performance evaluation using homology-based silver standard:
```bash
python scripts/eval_silver_standard_v0.py \
  --arg-db data/ARG_db_all_seq_uniq_representative_rename_2_repsent.fasta \
  --hits diamond_hits.tsv \
  --binary-csv results/binary_scores.csv \
  --multi-csv results/classification_results.csv \
  --out results/eval_merged.csv \
  --mode orf
```

**Mode selection**:
- `--mode full`: For full-length proteins (uses `min(qcov, scov)`)
- `--mode orf`: For ORF fragments from contigs (uses `qcov` with relaxed `scov`)

## Key Architecture Decisions

### Model Configuration
- **Binary model**: Embedding (vocab_size=22) → BiLSTM (hidden=48, 1 layer) → FC
- **Multi-class model**: One-hot (21 dims) → BiLSTM (hidden=128, 2 layers) → FC with Focal Loss

### Data Processing
- Amino acid vocabulary: 20 standard + X (unknown) + PAD
- Sequence encoding: One-hot for multi-class, embedding indices for binary
- **Critical**: Multi-class uses masked pooling (PAD channel index=20) to avoid padding contamination

### Label Parsing (Multi-class)
ARG categories are extracted from FASTA headers: take the last `|` delimited field.
- Rare classes (below `min_samples` threshold) are merged into "Others"
- Typical `min_samples` values: 20, 30, 40, 50 (evaluated via ablation)

### Evaluation Protocol
The project uses a **silver standard** approach for real-world evaluation:
1. Run DIAMOND/MMseqs2 blastp against ARG database
2. Apply strict thresholds (`evalue <= 1e-10`, `pident >= 80`, coverage >= 0.8) to label positives
3. No relaxed hits = labeled non-ARG
4. Compare model predictions against these reference labels

See `EVAL_PROTOCOL.md` for complete evaluation methodology.

## Documentation Files

- `AGENTS.md`: Project navigation (in Chinese) - quick entry points and current status
- `EVAL_PROTOCOL.md`: Detailed evaluation protocol for reproducible research
- `WORKLOG.md`: Chronological experiment log with results and decisions
- `data/database_construct_description.md`: Dataset construction methodology

## Important Notes

1. **Path Resolution**: Training notebooks use `resolve_repo_data_file()` to auto-locate data files by searching upward for `data/` directory
2. **Random Seeds**: Fixed seed (42) is used for reproducibility in training
3. **Model Checkpointing**: Best model is saved based on validation macro-F1 (not accuracy)
4. **Logging**: Training uses explicit logger configuration for Jupyter compatibility
5. **File Naming**: Output figures use run-specific prefixes (`{run_tag}_training_results.*`) to prevent overwriting
