# Project Lessons Learned

## Data Loading

### Lesson 1: Always Use CSV Sequence Column, Not FASTA ID Matching
**Problem**: FASTA ID conflicts caused data loading errors, resulting in 12.4% validation accuracy.
**Solution**: Load sequences directly from CSV `sequence` column instead of matching by FASTA ID.
**Files affected**: `models/multi/train.py`, `models/binary/train.py`

### Lesson 2: Filter Data Appropriately for Each Task
**Problem**: Multi-class training included non-ARG sequences, contaminating the training set.
**Solution**: Add `is_arg=1` filter for multi-class training to ensure only ARG sequences are used.
**Files affected**: `models/multi/train.py`

## Training Strategy

### Lesson 3: Use Task-Appropriate Metrics for Early Stopping
**Problem**: Using loss for early stopping does not reflect actual classification performance.
**Solution**: Use F1 score for early stopping in both binary and multi-class tasks.
**Implementation**: Pass `metric_fn` to `Trainer`, update `trainer.py` to support custom metrics.
**Files affected**: `models/common/trainer.py`, `models/*/train.py`

### Lesson 4: Balance Class Handling Carefully
**Problem**: Aggressive balanced sampling (`BalancedClassSampler`) caused model bias toward minority classes.
**Solution**: Disable balanced sampling, use mild class weights (sqrt method with clip [0.5, 3.0]) instead.
**Config**: `use_balanced_sampling: false` in `multi_config.yaml`
**Files affected**: `configs/multi_config.yaml`, `utils/sequence_utils.py`

## Regularization

### Lesson 5: Increase Dropout to Combat Overfitting
**Problem**: Validation-test performance gap indicated overfitting.
**Solution**: Increase dropout from 0.3 to 0.5 for both binary and multi-class models.
**Files affected**: `configs/binary_config.yaml`, `configs/multi_config.yaml`

### Lesson 6: Adjust Early Stopping Patience
**Problem**: Default patience (15) may be too long for some tasks.
**Solution**: Reduce patience to 10 for binary classification to enable faster early stopping.
**Files affected**: `configs/binary_config.yaml`

## Model Configuration

### Lesson 7: Make Prediction Scripts Robust
**Problem**: Prediction scripts required manual config specification.
**Solution**: Implement auto-inference of model configuration from checkpoint state_dict.
**Files affected**: `models/binary/predict.py`, `models/multi/predict.py`

## Multi-class Specific

### Lesson 8: Handle "Others" Category Carefully
**Observation**: "Others" category has lower F1 (0.56) compared to specific categories (0.90-1.00).
**Note**: This is expected as "Others" aggregates multiple rare classes. The model performs well on well-defined categories.

## Performance Achieved

| Task | Metric | Before Fix | After Fix | Target |
|------|--------|------------|-----------|--------|
| Binary | Val Acc | 12.4% | 97.53% | >90% |
| Binary | Test Acc | - | 97.43% | >90% |
| Multi | Val F1 | ~0.2 | 0.9544 | >0.6 |
| Multi | Test F1 | - | 0.9434 | >0.6 |

## Visualization and Metrics

### Lesson 9: Separate Visualization Module
**Approach**: Create a dedicated `utils/visualization.py` module rather than embedding plotting code in training scripts.
**Benefits**: Reusable across binary/multi-class, easier to maintain, consistent styling.
**Files**: `utils/visualization.py`, `utils/metrics.py`

### Lesson 10: Track All Metrics During Training
**Problem**: Original trainer only tracked loss and one validation metric.
**Solution**: Extended history dictionary to track all metrics (accuracy, precision, recall, F1, AUC) per epoch.
**Benefits**: Complete training curves for analysis, better understanding of model behavior.

### Lesson 11: Comprehensive Evaluation Function
**Pattern**: Add standalone `evaluate()` method to Trainer class that:
1. Runs inference on test set
2. Computes per-class and overall metrics
3. Generates confusion matrix and ROC curves
4. Saves results as JSON and CSV
**Benefits**: Standardized evaluation, reproducible results, rich visualizations.
