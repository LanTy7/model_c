"""Visualization utilities for ARG classification models."""
import os
from typing import Dict, List, Optional, Any
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from sklearn.preprocessing import label_binarize


def set_plot_style():
    """Set consistent plot style."""
    sns.set_style("whitegrid")
    plt.rcParams['figure.figsize'] = (10, 6)
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['legend.fontsize'] = 10


def plot_training_curves(
    history: Dict[str, List[float]],
    fig_dir: str,
    save_format: str = 'png'
) -> None:
    """
    Plot training and validation loss and metric curves.

    Args:
        history: Dictionary containing training history
        fig_dir: Directory to save figures
        save_format: Image format (png, pdf, etc.)
    """
    set_plot_style()
    os.makedirs(fig_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss curves
    ax = axes[0, 0]
    ax.plot(epochs, history['train_loss'], 'b-', label='Train Loss', linewidth=2)
    ax.plot(epochs, history['val_loss'], 'r-', label='Val Loss', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Accuracy curves
    ax = axes[0, 1]
    if 'val_accuracy' in history and history['val_accuracy']:
        ax.plot(epochs, history['val_accuracy'], 'g-', label='Val Accuracy', linewidth=2)
        ax.set_ylabel('Accuracy')
        ax.set_title('Validation Accuracy')
        ax.legend()
        ax.grid(True, alpha=0.3)
    ax.set_xlabel('Epoch')

    # F1 score curves
    ax = axes[1, 0]
    if 'val_f1' in history and history['val_f1']:
        ax.plot(epochs, history['val_f1'], 'm-', label='Val F1', linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('F1 Score')
        ax.set_title('Validation F1 Score')
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Precision and Recall
    ax = axes[1, 1]
    if 'val_precision' in history and history['val_precision']:
        ax.plot(epochs, history['val_precision'], 'c-', label='Val Precision', linewidth=2)
    if 'val_recall' in history and history['val_recall']:
        ax.plot(epochs, history['val_recall'], 'y-', label='Val Recall', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Score')
    ax.set_title('Validation Precision and Recall')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(fig_dir, f'training_curves.{save_format}')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_learning_rate_schedule(
    history: Dict[str, List[float]],
    fig_dir: str,
    save_format: str = 'png'
) -> None:
    """
    Plot learning rate schedule over epochs.

    Args:
        history: Dictionary containing training history with 'lr' key
        fig_dir: Directory to save figures
        save_format: Image format
    """
    set_plot_style()
    os.makedirs(fig_dir, exist_ok=True)

    if 'lr' not in history or not history['lr']:
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    epochs = range(1, len(history['lr']) + 1)
    ax.plot(epochs, history['lr'], 'b-', linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Learning Rate')
    ax.set_title('Learning Rate Schedule')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    save_path = os.path.join(fig_dir, f'lr_schedule.{save_format}')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
    fig_dir: str,
    normalize: bool = True,
    save_format: str = 'png'
) -> None:
    """
    Plot confusion matrix.

    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        class_names: List of class names
        fig_dir: Directory to save figures
        normalize: Whether to normalize the confusion matrix
        save_format: Image format
    """
    set_plot_style()
    os.makedirs(fig_dir, exist_ok=True)

    cm = confusion_matrix(y_true, y_pred)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    else:
        fmt = 'd'

    fig, ax = plt.subplots(figsize=(max(8, len(class_names)), max(6, len(class_names) * 0.5)))

    sns.heatmap(
        cm,
        annot=True,
        fmt=fmt if not normalize else '.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'}
    )

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix' + (' (Normalized)' if normalize else ''))

    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    save_path = os.path.join(fig_dir, f'confusion_matrix.{save_format}')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_per_class_metrics(
    metrics_dict: Dict[str, Any],
    class_names: List[str],
    fig_dir: str,
    save_format: str = 'png'
) -> None:
    """
    Plot per-class precision, recall, and F1 as grouped bar chart.

    Args:
        metrics_dict: Dictionary containing per-class metrics
        class_names: List of class names
        fig_dir: Directory to save figures
        save_format: Image format
    """
    set_plot_style()
    os.makedirs(fig_dir, exist_ok=True)

    # Extract per-class metrics
    precision = metrics_dict.get('per_class_precision', [])
    recall = metrics_dict.get('per_class_recall', [])
    f1 = metrics_dict.get('per_class_f1', [])

    if not precision or not recall or not f1:
        return

    x = np.arange(len(class_names))
    width = 0.25

    fig, ax = plt.subplots(figsize=(max(12, len(class_names) * 0.8), 7))

    ax.bar(x - width, precision, width, label='Precision', alpha=0.8)
    ax.bar(x, recall, width, label='Recall', alpha=0.8)
    ax.bar(x + width, f1, width, label='F1 Score', alpha=0.8)

    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Per-Class Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    save_path = os.path.join(fig_dir, f'per_class_metrics.{save_format}')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_roc_curves(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    class_names: List[str],
    fig_dir: str,
    save_format: str = 'png'
) -> None:
    """
    Plot ROC curves for binary or multi-class classification.

    Args:
        y_true: Ground truth labels
        y_prob: Predicted probabilities
        class_names: List of class names
        fig_dir: Directory to save figures
        save_format: Image format
    """
    set_plot_style()
    os.makedirs(fig_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 8))

    if len(class_names) == 2:  # Binary
        # Handle binary case
        if y_prob.ndim == 2 and y_prob.shape[1] == 1:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
        else:
            fpr, tpr, _ = roc_curve(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    else:  # Multi-class
        # Binarize labels
        y_true_bin = label_binarize(y_true, classes=range(len(class_names)))

        # Compute ROC curve and ROC area for each class
        for i in range(len(class_names)):
            if y_true_bin[:, i].sum() > 0:  # Only if class exists in test set
                fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_prob[:, i])
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, linewidth=2, label=f'{class_names[i]} (AUC = {roc_auc:.3f})')

    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curves')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(fig_dir, f'roc_curves.{save_format}')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_training_report(
    history: Dict[str, List[float]],
    metrics: Optional[Dict[str, Any]],
    fig_dir: str,
    class_names: Optional[List[str]] = None,
    save_format: str = 'png'
) -> None:
    """
    Create comprehensive training report with all plots.

    Args:
        history: Training history dictionary
        metrics: Evaluation metrics dictionary (optional)
        fig_dir: Directory to save figures
        class_names: List of class names (optional)
        save_format: Image format
    """
    os.makedirs(fig_dir, exist_ok=True)

    # Generate all plots
    plot_training_curves(history, fig_dir, save_format)
    plot_learning_rate_schedule(history, fig_dir, save_format)

    if metrics and class_names:
        # Check if we have prediction data for confusion matrix
        if 'y_true' in metrics and 'y_pred' in metrics:
            plot_confusion_matrix(
                np.array(metrics['y_true']),
                np.array(metrics['y_pred']),
                class_names,
                fig_dir,
                normalize=True,
                save_format=save_format
            )

        plot_per_class_metrics(metrics, class_names, fig_dir, save_format)

        if 'y_prob' in metrics:
            plot_roc_curves(
                np.array(metrics['y_true']),
                np.array(metrics['y_prob']),
                class_names,
                fig_dir,
                save_format=save_format
            )


def save_metrics_json(metrics: Dict[str, Any], output_path: str) -> None:
    """
    Save metrics dictionary to JSON file.
    Convert numpy arrays to lists for JSON serialization.
    """
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj

    serializable_metrics = convert_to_serializable(metrics)

    with open(output_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=2)
