#!/usr/bin/env python3
"""Evaluation script for multi-class ARG classification."""
import os
import sys
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.multi.model import MultiClassARGClassifier
from models.common.trainer import TrainConfig
from data.dataset import MultiClassARGDataset
from utils.common import setup_logging, load_config
from utils.sequence_utils import get_max_length
from utils.metrics import compute_comprehensive_metrics, format_metrics_for_display, generate_classification_report
from utils.visualization import (
    plot_confusion_matrix, plot_per_class_metrics, plot_roc_curves, save_metrics_json
)


def load_model(checkpoint_path: str, metadata_path: str, yaml_config: dict, device: str) -> tuple:
    """Load model from checkpoint and metadata."""
    import json

    # Load metadata
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Get model config from YAML config
    num_classes = metadata['num_classes']

    if yaml_config and 'model' in yaml_config:
        model_config = {k: v for k, v in yaml_config['model'].items() if k != 'name'}
        model_config['num_classes'] = num_classes
    else:
        # Try to infer from state_dict
        state_dict = checkpoint['model_state_dict']

        # Infer hidden size from LSTM
        hidden_size = 256
        num_layers = 1
        for key in state_dict.keys():
            if 'lstm.weight_ih_l0' in key:
                hidden_size = state_dict[key].shape[0] // 4
            if 'lstm.weight_ih_l1' in key:
                num_layers = 2
            if 'lstm.weight_ih_l2' in key:
                num_layers = 3

        model_config = {
            'input_size': 21,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': 0.4,
            'num_classes': num_classes
        }

    model = MultiClassARGClassifier(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, checkpoint, metadata


def evaluate_model(
    model: MultiClassARGClassifier,
    test_loader: DataLoader,
    device: str,
    logger
) -> tuple:
    """Run evaluation on test set."""
    model.eval()

    all_preds = []
    all_targets = []
    all_probs = []

    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            outputs = model(inputs)

            # Get probabilities and predictions
            probs = torch.softmax(outputs, dim=1).cpu().numpy()
            preds = np.argmax(probs, axis=1)

            all_preds.extend(preds)
            all_targets.extend(targets.cpu().numpy())
            all_probs.append(probs)

    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.concatenate(all_probs, axis=0)

    return y_true, y_pred, y_prob


def main():
    parser = argparse.ArgumentParser(description="Evaluate multi-class ARG classifier")
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
        "--metadata",
        type=str,
        default=None,
        help="Path to metadata.json (default: same dir as checkpoint)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/multi",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip generating plots"
    )
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup
    logger = setup_logging(os.path.join(args.output_dir, 'logs'))
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    logger.info("=" * 60)
    logger.info("Multi-class ARG Classification Evaluation")
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

    # Determine metadata path
    if args.metadata is None:
        checkpoint_dir = os.path.dirname(args.checkpoint)
        args.metadata = os.path.join(checkpoint_dir, 'metadata.json')

    if not os.path.exists(args.metadata):
        logger.error(f"Metadata not found: {args.metadata}")
        return

    # Load model
    logger.info("Loading model...")
    model, checkpoint, metadata = load_model(args.checkpoint, args.metadata, config, device)
    class_names = metadata['class_names']
    label_to_idx = metadata['label_to_idx']
    max_length = metadata['max_length']

    logger.info(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"Number of classes: {len(class_names)}")

    # Load test data
    logger.info("Loading test data...")
    test_df = pd.read_csv(config['data']['test_csv'])

    # Filter: only use ARG sequences
    test_df = test_df[test_df['is_arg'] == 1].copy()
    logger.info(f"Filtered test to {len(test_df)} ARG sequences (is_arg=1)")

    test_seqs = test_df['sequence'].str.upper().tolist()
    test_labels_str = test_df['arg_category'].fillna('Others').tolist()

    # Map test labels
    test_labels = []
    for label in test_labels_str:
        if label in label_to_idx:
            test_labels.append(label_to_idx[label])
        else:
            test_labels.append(label_to_idx.get('Others', 0))
    test_labels = np.array(test_labels, dtype=np.int64)

    test_dataset = MultiClassARGDataset(test_seqs, test_labels, max_length)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        persistent_workers=config['training']['num_workers'] > 0,
        pin_memory=True
    )

    # Run evaluation
    logger.info("Running evaluation...")
    y_true, y_pred, y_prob = evaluate_model(model, test_loader, device, logger)

    # Compute metrics
    metrics = compute_comprehensive_metrics(y_true, y_pred, y_prob, class_names)

    # Log results
    logger.info("\n" + format_metrics_for_display(metrics))

    # Save results
    os.makedirs(args.output_dir, exist_ok=True)

    # Save metrics as JSON
    metrics_path = os.path.join(args.output_dir, 'test_metrics.json')
    save_metrics_json(metrics, metrics_path)
    logger.info(f"\nMetrics saved to: {metrics_path}")

    # Save classification report as CSV
    report_path = os.path.join(args.output_dir, 'classification_report.csv')
    report = generate_classification_report(y_true, y_pred, class_names, report_path)
    logger.info(f"Classification report saved to: {report_path}")

    # Generate plots
    if not args.no_plots:
        fig_dir = os.path.join(args.output_dir, 'figures')
        os.makedirs(fig_dir, exist_ok=True)

        logger.info("Generating plots...")
        plot_confusion_matrix(y_true, y_pred, class_names, fig_dir, normalize=True)
        plot_per_class_metrics(metrics, class_names, fig_dir)
        plot_roc_curves(y_true, y_prob, class_names, fig_dir)
        logger.info(f"Plots saved to: {fig_dir}")

    logger.info("=" * 60)
    logger.info("Evaluation completed!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
