"""Comprehensive metrics computation utilities for ARG classification."""
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix,
    precision_recall_fscore_support, balanced_accuracy_score,
    matthews_corrcoef, cohen_kappa_score
)


def compute_comprehensive_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    class_names: Optional[List[str]] = None,
    average: str = 'macro'
) -> Dict[str, Any]:
    """
    Compute comprehensive evaluation metrics.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        y_prob: Predicted probabilities (optional, for AUC)
        class_names: List of class names (optional)
        average: Averaging method for metrics ('macro', 'micro', 'weighted')

    Returns:
        Dictionary containing all computed metrics
    """
    metrics = {}

    # Store predictions for later use
    metrics['y_true'] = y_true
    metrics['y_pred'] = y_pred
    if y_prob is not None:
        metrics['y_prob'] = y_prob

    # Overall metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)

    # Macro-averaged metrics
    metrics['precision'] = precision_score(y_true, y_pred, average=average, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, average=average, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, average=average, zero_division=0)

    # Matthews Correlation Coefficient and Cohen's Kappa
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)

    # ROC-AUC
    if y_prob is not None:
        try:
            if y_prob.ndim == 1 or y_prob.shape[1] == 1:  # Binary
                metrics['auc'] = roc_auc_score(y_true, y_prob if y_prob.ndim == 1 else y_prob[:, 0])
            else:  # Multi-class
                metrics['auc'] = roc_auc_score(
                    y_true, y_prob, multi_class='ovr', average=average
                )
        except ValueError:
            metrics['auc'] = 0.0

    # Per-class metrics
    per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    metrics['per_class_precision'] = per_class_precision.tolist()
    metrics['per_class_recall'] = per_class_recall.tolist()
    metrics['per_class_f1'] = per_class_f1.tolist()
    metrics['per_class_support'] = support.tolist()

    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()

    # Per-class metrics dictionary with names
    if class_names:
        metrics['per_class_metrics'] = {}
        for i, name in enumerate(class_names):
            if i < len(per_class_precision):
                metrics['per_class_metrics'][name] = {
                    'precision': float(per_class_precision[i]),
                    'recall': float(per_class_recall[i]),
                    'f1': float(per_class_f1[i]),
                    'support': int(support[i])
                }

    return metrics


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_path: Optional[str] = None
) -> str:
    """
    Generate sklearn-style classification report.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names (optional)
        output_path: Path to save CSV report (optional)

    Returns:
        Classification report as string
    """
    target_names = class_names if class_names else None

    report = classification_report(
        y_true, y_pred,
        target_names=target_names,
        digits=4,
        zero_division=0
    )

    # Save as CSV if output path provided
    if output_path:
        # Convert to DataFrame for CSV export
        per_class_precision, per_class_recall, per_class_f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )

        df_data = {
            'class': class_names if class_names else [f'class_{i}' for i in range(len(per_class_precision))],
            'precision': per_class_precision,
            'recall': per_class_recall,
            'f1': per_class_f1,
            'support': support.astype(int)
        }

        df = pd.DataFrame(df_data)

        # Add overall metrics
        overall = pd.DataFrame([{
            'class': 'overall',
            'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
            'support': len(y_true)
        }])

        df = pd.concat([df, overall], ignore_index=True)
        df.to_csv(output_path, index=False)

    return report


def compute_metrics_from_batches(
    all_preds: List[np.ndarray],
    all_targets: List[np.ndarray],
    all_probs: List[np.ndarray],
    class_names: Optional[List[str]] = None,
    average: str = 'macro'
) -> Dict[str, Any]:
    """
    Compute metrics from batched predictions (useful for evaluation).

    Args:
        all_preds: List of prediction arrays from each batch
        all_targets: List of target arrays from each batch
        all_probs: List of probability arrays from each batch
        class_names: List of class names (optional)
        average: Averaging method

    Returns:
        Dictionary containing all computed metrics
    """
    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)
    y_prob = np.concatenate(all_probs) if all_probs else None

    return compute_comprehensive_metrics(y_true, y_pred, y_prob, class_names, average)


def format_metrics_for_display(metrics: Dict[str, Any]) -> str:
    """
    Format metrics dictionary as a readable string.

    Args:
        metrics: Metrics dictionary from compute_comprehensive_metrics

    Returns:
        Formatted string
    """
    lines = [
        "=" * 60,
        "Evaluation Metrics",
        "=" * 60,
        f"Accuracy:           {metrics.get('accuracy', 0):.4f}",
        f"Balanced Accuracy:  {metrics.get('balanced_accuracy', 0):.4f}",
        f"Precision (macro):  {metrics.get('precision', 0):.4f}",
        f"Recall (macro):     {metrics.get('recall', 0):.4f}",
        f"F1 Score (macro):   {metrics.get('f1', 0):.4f}",
    ]

    if 'auc' in metrics:
        lines.append(f"ROC-AUC:            {metrics['auc']:.4f}")

    lines.extend([
        f"MCC:                {metrics.get('mcc', 0):.4f}",
        f"Cohen's Kappa:      {metrics.get('cohen_kappa', 0):.4f}",
        "=" * 60,
    ])

    # Per-class metrics
    if 'per_class_metrics' in metrics:
        lines.append("\nPer-Class Metrics:")
        lines.append("-" * 60)
        lines.append(f"{'Class':<20} {'Precision':>12} {'Recall':>12} {'F1':>12} {'Support':>10}")
        lines.append("-" * 60)

        for class_name, class_metrics in metrics['per_class_metrics'].items():
            lines.append(
                f"{class_name:<20} "
                f"{class_metrics['precision']:>12.4f} "
                f"{class_metrics['recall']:>12.4f} "
                f"{class_metrics['f1']:>12.4f} "
                f"{class_metrics['support']:>10d}"
            )

    lines.append("=" * 60)

    return "\n".join(lines)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = 'f1',
    beta: float = 1.0,
    thresholds: Optional[np.ndarray] = None
) -> tuple:
    """
    Find optimal classification threshold for binary classification.

    This function searches for the threshold that maximizes a given metric,
    useful for handling imbalanced datasets where the default 0.5 threshold
    may not be optimal.

    Args:
        y_true: Ground truth labels (binary: 0 or 1)
        y_prob: Predicted probabilities for the positive class
        metric: Metric to optimize ('f1', 'f2', 'precision', 'recall', 'youden')
        beta: Beta value for F-beta score (default 1.0 for F1, use 2.0 for F2)
        thresholds: Array of thresholds to try (default: 0.01 to 0.99 in 0.01 steps)

    Returns:
        Tuple of (best_threshold, best_score, threshold_metrics_df)
        where threshold_metrics_df is a DataFrame with metrics for all thresholds

    Example:
        >>> y_true = np.array([0, 0, 1, 1])
        >>> y_prob = np.array([0.1, 0.4, 0.6, 0.9])
        >>> best_thresh, best_score, _ = find_optimal_threshold(y_true, y_prob, metric='f1')
        >>> print(f"Best threshold: {best_thresh:.3f}, F1: {best_score:.4f}")
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)

    # Store metrics for each threshold
    results = []

    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)

        # Compute metrics
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)

        # Compute F-beta score
        if prec + rec > 0:
            f_beta = (1 + beta**2) * (prec * rec) / (beta**2 * prec + rec)
        else:
            f_beta = 0.0

        # Compute Youden's J statistic (sensitivity + specificity - 1)
        tn = np.sum((y_true == 0) & (y_pred == 0))
        fp = np.sum((y_true == 0) & (y_pred == 1))
        specificity = tn / max(tn + fp, 1e-10)
        youden = rec + specificity - 1

        results.append({
            'threshold': thresh,
            'precision': prec,
            'recall': rec,
            'f1': f_beta if beta == 1.0 else f_beta,  # F1 when beta=1
            f'f{beta}': f_beta,
            'youden': youden
        })

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Find best threshold based on metric
    if metric == 'f1':
        best_idx = df['f1'].idxmax()
    elif metric == 'f2':
        # Ensure f2 column exists
        if 'f2' not in df.columns:
            # Create f2 column from f_beta values or recalculate
            if f'f{beta}' in df.columns and beta == 2.0:
                df['f2'] = df[f'f{beta}']
            else:
                df['f2'] = df.apply(
                    lambda row: 5 * row['precision'] * row['recall'] /
                               max(4 * row['precision'] + row['recall'], 1e-10),
                    axis=1
                )
        best_idx = df['f2'].idxmax()
    elif metric == 'precision':
        best_idx = df['precision'].idxmax()
    elif metric == 'recall':
        best_idx = df['recall'].idxmax()
    elif metric == 'youden':
        best_idx = df['youden'].idxmax()
    else:
        raise ValueError(f"Unknown metric: {metric}. Choose from 'f1', 'f2', 'precision', 'recall', 'youden'")

    best_threshold = df.loc[best_idx, 'threshold']
    best_score = df.loc[best_idx, metric]

    return best_threshold, best_score, df


def compute_metrics_at_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, float]:
    """
    Compute metrics at a specific threshold.

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        threshold: Classification threshold

    Returns:
        Dictionary of metrics
    """
    y_pred = (y_prob >= threshold).astype(int)

    return {
        'threshold': threshold,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred)
    }
