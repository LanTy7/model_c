#!/usr/bin/env python3
"""Evaluation script for binary ARG classification."""
import os
import sys
import json
import argparse
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.binary.model import BinaryARGClassifier
from models.common.trainer import TrainConfig
from data.dataset import BinarySequenceDataset
from utils.common import setup_logging, load_config
from utils.metrics import (
    compute_comprehensive_metrics, format_metrics_for_display,
    generate_classification_report, find_optimal_threshold, compute_metrics_at_threshold
)
from utils.visualization import (
    plot_confusion_matrix, plot_per_class_metrics, plot_roc_curves, save_metrics_json
)


def save_threshold_json(
    output_path: str,
    optimal_threshold: float,
    tune_metric: str,
    description: str = ""
) -> str:
    """Save optimal threshold to threshold.json file.

    Args:
        output_path: Path to save the threshold.json file
        optimal_threshold: The optimal threshold value
        tune_metric: The metric used to optimize the threshold
        description: Optional description of the threshold

    Returns:
        Path to the saved file
    """
    data = {
        "optimal_threshold": float(optimal_threshold),
        "tune_metric": tune_metric,
        "description": description or f"Threshold optimized for {tune_metric.upper()} score"
    }

    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    return output_path


def load_data(csv_path: str, logger) -> Tuple[List[str], List[int]]:
    """Load data from CSV (sequence column)."""
    df = pd.read_csv(csv_path)
    logger.info(f"Loaded {len(df)} entries from {csv_path}")

    # Use sequence column directly
    sequences = df['sequence'].str.upper().tolist()
    labels = df['is_arg'].astype(int).tolist()

    logger.info(f"Data: {sum(labels)} positive, {len(labels)-sum(labels)} negative")
    return sequences, labels


def load_model(checkpoint_path: str, config: dict, device: str) -> Tuple[BinaryARGClassifier, dict]:
    """Load model from checkpoint."""
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except (TypeError, RuntimeError):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Determine model_config: priority: checkpoint > YAML config > state dict inference
    if 'model_config' in checkpoint:
        # Training script saves model_config in checkpoint (most reliable)
        model_config = checkpoint['model_config']
    elif config and 'model' in config:
        # Fallback to YAML config
        model_config = {k: v for k, v in config['model'].items() if k != 'name'}
    else:
        # Try to infer from state dict
        state_dict = checkpoint['model_state_dict']
        embedding_weight = state_dict.get('embedding.weight', state_dict.get('backbone.embedding.weight'))
        if embedding_weight is not None:
            vocab_size, embedding_dim = embedding_weight.shape
        else:
            vocab_size, embedding_dim = 25, 128

        # Detect enhanced architecture (CNN + Attention)
        use_cnn = any('cnn.convs' in k for k in state_dict.keys())
        use_attention = any('backbone.attention' in k for k in state_dict.keys())

        # Infer hidden size and num_layers from LSTM weights
        hidden_size = 128
        num_layers = 1
        for key in state_dict.keys():
            if 'lstm.weight_ih_l0' in key and '_reverse' not in key:
                hidden_size = state_dict[key].shape[0] // 4
        num_layers = sum(1 for k in state_dict.keys()
                         if 'lstm.weight_ih_l' in k and '_reverse' not in k)

        model_config = {
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': 0.4,
            'max_length': 1000,
            'use_cnn': use_cnn,
            'use_attention': use_attention,
            'cnn_kernel_sizes': [3, 5, 7] if use_cnn else None,
            'cnn_out_channels': 64 if use_cnn else None,
            'num_attention_heads': 4 if use_attention else None,
        }

    model = BinaryARGClassifier(**model_config)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    return model, checkpoint


def evaluate_model(
    model: BinaryARGClassifier,
    test_loader: DataLoader,
    device: str,
    logger,
    threshold: float = 0.5
) -> Tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """Run evaluation on test set with specified threshold."""
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, mask, targets in test_loader:
            inputs = inputs.to(device)
            mask = mask.to(device)
            targets = targets.to(device)

            outputs = model(inputs, mask=mask)

            # Get probabilities
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs >= threshold).astype(int)  # Use specified threshold

            all_preds.extend(preds.flatten())
            all_targets.extend(targets.cpu().numpy().flatten())
            all_probs.append(probs.flatten())

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.concatenate(all_probs)

    # Compute metrics
    class_names = ['non-ARG', 'ARG']
    metrics = compute_comprehensive_metrics(y_true, y_pred, y_prob, class_names)

    return metrics, y_true, y_pred, y_prob


def main():
    parser = argparse.ArgumentParser(description="Evaluate binary ARG classifier")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/binary",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots"
    )
    parser.add_argument(
        "--tune-threshold",
        action="store_true",
        help="Tune classification threshold on validation set before evaluation"
    )
    parser.add_argument(
        "--tune-metric",
        type=str,
        default='f1',
        choices=['f1', 'f2', 'precision', 'recall', 'youden'],
        help="Metric to optimize when tuning threshold (default: f1)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold (used if --tune-threshold is not set)"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup
    logger = setup_logging(os.path.join(args.output_dir, 'logs'))
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    logger.info("=" * 60)
    logger.info("Binary ARG Classification Evaluation")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Checkpoint: {args.checkpoint}")

    # Check if test data exists
    if 'test_csv' not in config['data']:
        logger.error("No test_csv specified in config")
        return

    if not os.path.exists(config['data']['test_csv']):
        logger.error(f"Test data not found: {config['data']['test_csv']}")
        return

    # Load model
    logger.info("Loading model...")
    model, checkpoint = load_model(args.checkpoint, config, device)
    logger.info(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")

    # Load data
    logger.info("Loading test data...")
    test_seqs, test_labels = load_data(config['data']['test_csv'], logger)

    max_length = config['model']['max_length']
    test_dataset = BinarySequenceDataset(test_seqs, test_labels, max_length)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        persistent_workers=config['training']['num_workers'] > 0,
        pin_memory=True
    )

    # Threshold tuning (if requested) — tune on validation set to avoid data leakage
    eval_threshold = args.threshold
    if args.tune_threshold:
        logger.info("=" * 60)
        if 'val_csv' not in config['data'] or not os.path.exists(config['data']['val_csv']):
            logger.error("Validation data required for threshold tuning. Provide val_csv in config.")
            return

        logger.info(f"Tuning threshold on validation set (optimizing {args.tune_metric})...")
        val_seqs, val_labels = load_data(config['data']['val_csv'], logger)
        val_dataset = BinarySequenceDataset(val_seqs, val_labels, max_length)
        val_loader = DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=config['training']['num_workers'],
            persistent_workers=config['training']['num_workers'] > 0,
            pin_memory=True
        )

        # Collect all probabilities on validation set
        all_probs = []
        all_targets = []
        with torch.no_grad():
            for inputs, mask, targets in val_loader:
                inputs = inputs.to(device)
                mask = mask.to(device)
                outputs = model(inputs, mask=mask)
                probs = torch.sigmoid(outputs).cpu().numpy()
                all_probs.append(probs.flatten())
                all_targets.extend(targets.cpu().numpy().flatten())

        y_true_tune = np.array(all_targets)
        y_prob_tune = np.concatenate(all_probs)

        # Find optimal threshold
        best_thresh, best_score, thresh_df = find_optimal_threshold(
            y_true_tune, y_prob_tune, metric=args.tune_metric, beta=2.0 if args.tune_metric == 'f2' else 1.0
        )

        logger.info(f"Best threshold: {best_thresh:.3f} ({args.tune_metric} = {best_score:.4f})")

        # Show metrics at different thresholds
        logger.info("\nMetrics at key thresholds:")
        logger.info(f"{'Threshold':<12} {'Precision':<10} {'Recall':<10} {'F1':<10}")
        logger.info("-" * 42)
        for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
            if thresh in thresh_df['threshold'].values:
                row = thresh_df[thresh_df['threshold'] == thresh].iloc[0]
                logger.info(f"{thresh:<12.2f} {row['precision']:<10.4f} {row['recall']:<10.4f} {row['f1']:<10.4f}")

        eval_threshold = best_thresh
        logger.info("=" * 60)

    # Run evaluation
    logger.info(f"Running evaluation with threshold = {eval_threshold:.3f}...")
    metrics, y_true, y_pred, y_prob = evaluate_model(model, test_loader, device, logger, threshold=eval_threshold)

    # Add threshold info to metrics
    metrics['threshold'] = eval_threshold
    if args.tune_threshold:
        metrics['tuned_threshold'] = True
        metrics['tune_metric'] = args.tune_metric

    # Log results with threshold info
    logger.info(f"\nEvaluation threshold: {eval_threshold:.3f}")
    if args.tune_threshold:
        logger.info(f"Threshold tuned using {args.tune_metric} metric")
    logger.info("\n" + format_metrics_for_display(metrics))

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # Save threshold to checkpoint directory for later use by predict.py
    # Only save when threshold was actually tuned, to avoid overwriting a tuned value
    if args.tune_threshold:
        checkpoint_dir = os.path.dirname(os.path.abspath(args.checkpoint))
        threshold_path = os.path.join(checkpoint_dir, 'threshold.json')
        save_threshold_json(
            threshold_path,
            optimal_threshold=eval_threshold,
            tune_metric=args.tune_metric,
            description=f"Threshold optimized for {args.tune_metric.upper()} score on validation set"
        )
        logger.info(f"Threshold saved to: {threshold_path}")

    # Save metrics as JSON
    metrics_path = os.path.join(args.output_dir, 'test_metrics.json')
    save_metrics_json(metrics, metrics_path)
    logger.info(f"\nMetrics saved to: {metrics_path}")

    # Save classification report as CSV
    report_path = os.path.join(args.output_dir, 'classification_report.csv')
    report = generate_classification_report(y_true, y_pred, ['non-ARG', 'ARG'], report_path)
    logger.info(f"Classification report saved to: {report_path}")

    # Generate plots
    if not args.no_plots:
        fig_dir = os.path.join(args.output_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)

        logger.info("Generating plots...")
        plot_confusion_matrix(y_true, y_pred, ['non-ARG', 'ARG'], fig_dir)
        plot_per_class_metrics(metrics, ['non-ARG', 'ARG'], fig_dir)
        plot_roc_curves(y_true, y_prob, ['non-ARG', 'ARG'], fig_dir)
        logger.info(f"Plots saved to: {fig_dir}")

    logger.info("=" * 60)
    logger.info("Evaluation completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
