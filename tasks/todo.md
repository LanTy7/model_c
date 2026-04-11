# Project Tasks

## Current Tasks

*No active tasks. Add new tasks here.*

---

## Completed Tasks

### Phase: Architecture Enhancements (MCT-ARG Inspired) ✅ COMPLETED (2025-04)

#### 1. Self-Attention Mechanism ✅
- [x] Create `models/common/attention.py` with multi-head SelfAttention module
- [x] Integrate into BiLSTM backbone (`models/common/bilstm.py`)
- [x] Add `use_attention` and `num_attention_heads` config options
- [x] Update binary and multi-class models to support attention

#### 2. Multi-scale CNN ✅
- [x] Create `models/common/multiscale_cnn.py` with parallel convolutions
- [x] Support kernel sizes [3, 5, 7] for multi-scale feature extraction
- [x] Add `use_cnn` and CNN config options to models
- [x] Integrate CNN output with BiLSTM features

#### 3. AECR Regularization ✅
- [x] Create `models/common/aecr_loss.py` with AECRLoss class
- [x] Implement entropy regularization (sharper attention)
- [x] Implement local continuity (Gaussian kernel-based smoothness)
- [x] Integrate into Trainer with `use_aecr` flag

#### 4. Enhanced Configurations ✅
- [x] Create `configs/binary_config_enhanced.yaml`
- [x] Create `configs/multi_config_enhanced.yaml`
- [x] Document all new parameters (attention, CNN, AECR)

### Phase: Class Imbalance Handling ✅ COMPLETED (2025-04)

#### 1. pos_weight Configuration ✅
- [x] Modify `models/binary/train.py` to support custom pos_weight
- [x] Allow auto-calculation or custom value in config
- [x] Default pos_weight=10.0 for ARG detection emphasis

#### 2. Threshold Tuning ✅
- [x] Add `find_optimal_threshold()` to `utils/metrics.py`
- [x] Support F1, F2, precision, recall, Youden's J metrics
- [x] Add `--tune-threshold` and `--tune-metric` to evaluate.py
- [x] Display metrics at key thresholds (0.3, 0.4, 0.5, 0.6, 0.7)

### Phase: Data Pipeline Improvements ✅ COMPLETED (2025-04)

#### 1. Quality Control Updates ✅
- [x] Update `scripts/quality_control.py` to retain X/B/Z/J amino acids
- [x] Only filter sequences with >50% X or invalid characters
- [x] Support for ambiguous amino acid codes

#### 2. Taxonomy-based Negative Sampling ✅
- [x] Update `scripts/download_negative_samples.py` with taxoniq integration
- [x] Filter for prokaryotes only (Bacteria/Archaea)
- [x] Remove eukaryote-contaminated sequences
- [x] Comprehensive testing with UniProt/NCBI Taxonomy

#### 3. Data Folder Reorganization ✅
- [x] Rename existing `data/` to `data_previous/`
- [x] Create new `data_now/` for cleaned datasets
- [x] Update all config paths to use `data_now/`

### Phase: Documentation Update ✅ COMPLETED (2025-04)

#### 1. README.md Updates ✅
- [x] Add "Recent Improvements (2025-04)" section
- [x] Document Self-Attention, CNN, AECR features
- [x] Document class imbalance handling strategies
- [x] Update repository structure with new files
- [x] Add enhanced config examples

#### 2. CLAUDE.md Updates ✅
- [x] Add enhanced training commands
- [x] Update key reminders with new features
- [x] Document threshold tuning capability

---

## Completed Tasks

### Phase: Training Visualization and Comprehensive Metrics ✅ COMPLETED

#### 1. Create Visualization Module ✅
- [x] Create `utils/visualization.py` with plotting functions:
  - `plot_training_curves()` - Loss and metric curves
  - `plot_learning_rate_schedule()` - LR schedule visualization
  - `plot_confusion_matrix()` - Confusion matrix heatmap
  - `plot_per_class_metrics()` - Per-class precision/recall/F1 bar chart
  - `plot_roc_curves()` - ROC curves for binary/multi-class

#### 2. Create Metrics Utilities ✅
- [x] Create `utils/metrics.py` with:
  - `compute_comprehensive_metrics()` - Per-class and overall metrics
  - `generate_classification_report()` - Classification report with CSV export
  - `format_metrics_for_display()` - Human-readable metrics formatting

#### 3. Enhance Trainer ✅
- [x] Add `fig_dir` and `class_names` to `TrainConfig`
- [x] Track all metrics in history (precision, recall, f1, auc per epoch)
- [x] Add `_save_history()` method
- [x] Add `_generate_training_plots()` method
- [x] Add `evaluate()` method for comprehensive test evaluation

#### 4. Update Training Scripts ✅
- [x] Update `models/binary/train.py` to integrate visualization
- [x] Update `models/multi/train.py` to integrate visualization
- [x] Add test set evaluation after training

#### 5. Create Evaluation Scripts ✅
- [x] Create `models/binary/evaluate.py` - Standalone binary evaluation
- [x] Create `models/multi/evaluate.py` - Standalone multi-class evaluation

#### 6. Update Configurations ✅
- [x] Add `visualization` section to binary and multi-class configs
- [x] Add `evaluation` section to binary and multi-class configs

### Phase: Testing and Validation ✅ COMPLETED

#### Test Results (5 epochs, test configuration)

**Binary Classification:**
- Test Accuracy: 91.45%
- Test F1 Score: 0.9145
- Test ROC-AUC: 0.9701
- Generated plots: training_curves.png, lr_schedule.png, confusion_matrix.png, per_class_metrics.png, roc_curves.png

**Multi-class Classification:**
- Test Accuracy: 56.05%
- Test F1 Score (macro): 0.2123
- Test ROC-AUC: 0.8617
- Generated plots: training_curves.png, lr_schedule.png, confusion_matrix.png, per_class_metrics.png, roc_curves.png

**Files Generated:**
```
checkpoints/binary_test/
  - binary_best.pth
  - training_history.json
  - test_metrics.json

figures/binary_test/
  - training_curves.png
  - lr_schedule.png
  - confusion_matrix.png
  - per_class_metrics.png
  - roc_curves.png

results/binary_test/
  - test_metrics.json
  - classification_report.csv
  - figures/ (evaluation plots)
```

---

## Future Tasks (Optional)

- [ ] Model deployment scripts
- [ ] Docker containerization
- [ ] API service interface
- [ ] More data augmentation experiments
- [ ] Model ensemble

---

## Completed Tasks

### Phase: Model Debugging and Retraining ✅

#### 1. Debug Complete ✅

**1.1 Fix Validation Accuracy Issue**
- [x] Root cause identified: FASTA ID conflicts causing data loading errors
- [x] Fixed data loading - load from CSV `sequence` column directly
- [x] Binary validation accuracy: 12.4% → 91.6%
- [x] Multi-class data loading fix complete

#### 2. Full Training (Fixed and Retrained) ✅

**2.1 Binary Classification Training**
```bash
python models/binary/train.py --config configs/binary_config.yaml
```
- [x] Run full training (100 epochs)
- [x] Validation accuracy: **97.53%** (target >90%) ✅
- [x] Test accuracy: **97.43%** (target >90%) ✅
- [x] Save best model: checkpoints/binary/binary_best.pth

**Key Fixes:**
- Use F1 instead of loss as early stopping metric
- dropout 0.3 → 0.5 (enhanced regularization)
- weight_decay 0.01 → 0.02
- patience 15 → 10 (faster early stopping)

**2.2 Multi-class Classification Training**
```bash
python models/multi/train.py --config configs/multi_config.yaml
```
- [x] Run full training (150 epochs)
- [x] Validation Macro-F1: **0.9544** (target >0.6) ✅
- [x] Test Macro-F1: **0.9434** (target >0.6) ✅
- [x] Test accuracy: **97.52%** ✅
- [x] Save best model: checkpoints/multi/multi_best.pth
- [x] Save metadata: checkpoints/multi/metadata.json

**Key Fixes:**
- Disable aggressive class balancing sampling (use_balanced_sampling: false)
- Use mild class weights (sqrt method, clip [0.5, 3.0])
- dropout 0.3 → 0.5
- Use F1 as early stopping metric

#### 3. Test Set Evaluation Complete ✅

**3.1 Binary Classification Evaluation**
```bash
python models/binary/predict.py \
  -i data/test.fasta \
  -m checkpoints/binary/binary_best.pth \
  -o results/binary_test.csv
```
**Results:**
- Accuracy: 97.43%
- Precision: 98.93%
- Recall: 95.91%
- F1 Score: 0.9740

**3.2 Multi-class Classification Evaluation**
```bash
python models/multi/predict.py \
  -i data/test.fasta \
  -m checkpoints/multi/multi_best.pth \
  -o results/multi_test.csv
```
**Results:**
- Accuracy: 97.52%
- Macro-F1: 0.9434
- Weighted-F1: 0.9754

**Per-class F1 Scores:**

| Class | F1 | Class | F1 |
|-------|----|-------|----|
| beta-lactam | 0.99 | phenicol | 1.00 |
| quinolone | 1.00 | sulfonamide | 1.00 |
| phosphonic | 0.99 | multidrug | 0.98 |
| diaminopyrimidine | 0.98 | glycopeptide | 0.97 |
| peptide | 0.97 | MLS | 0.96 |
| aminoglycoside | 0.96 | tetracycline | 0.95 |
| aminocoumarin | 0.90 | other | 0.56 |

#### 4. Fix Records

**4.1 Issues Discovered**
1. **Multi-class data error**: Training set contained non-ARG sequences
2. **Wrong validation metric**: Using loss instead of F1 for early stopping
3. **Excessive class balancing**: Aggressive sampling caused bias toward minority classes
4. **Overfitting**: Large gap between validation and test performance

**4.2 Fixes Applied**

| File | Changes |
|------|---------|
| models/multi/train.py | Added is_arg=1 filtering, use F1 as metric_fn |
| models/binary/train.py | Use F1 as metric_fn |
| models/common/trainer.py | Support custom metric_fn for early stopping |
| utils/sequence_utils.py | Added BalancedClassSampler class |
| configs/multi_config.yaml | Disable balanced_sampling, dropout=0.5 |
| configs/binary_config.yaml | dropout=0.5, weight_decay=0.02 |
| models/binary/predict.py | Auto-infer model config |
| models/multi/predict.py | Auto-infer model config |

---

## Future Tasks (Optional)

- [ ] Model deployment scripts
- [ ] Docker containerization
- [ ] API service interface
- [ ] More data augmentation experiments
- [ ] Model ensemble
