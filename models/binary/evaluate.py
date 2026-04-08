#!/usr/bin/env python3
"""Evaluation script for binary ARG classification."""
import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Tuple, List

import yaml
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.binary.model import BinaryARGClassifier
from models.common.trainer import TrainConfig
from utils.sequence_utils import sequence_to_indices
from utils.metrics import compute_comprehensive_metrics, format_metrics_for_display, generate_classification_report
from utils.visualization import (
    plot_confusion_matrix, plot_per_class_metrics, plot_roc_curves, save_metrics_json
)


def setup_logging(log_dir: str, log_file: str = "evaluate.log"):
    """Setup logging."""
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(os.path.join(log_dir, log_file)),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_config(config_path: str) -> dict:
    """Load YAML configuration."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


class BinarySequenceDataset(Dataset):
    """Simple dataset for binary classification."""

    def __init__(self, sequences: List[str], labels: List[int], max_length: int):
        self.sequences = sequences
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        encoded = sequence_to_indices(seq, self.max_length)
        return torch.from_numpy(encoded), torch.tensor(label, dtype=torch.float32)


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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Use config from YAML file if available, otherwise infer from checkpoint
    if config and 'model' in config:
        model_config = {k: v for k, v in config['model'].items() if k != 'name'}
    else:
        # Try to infer from state dict
        state_dict = checkpoint['model_state_dict']
        embedding_weight = state_dict.get('embedding.weight', state_dict.get('backbone.embedding.weight'))
        if embedding_weight is not None:
            vocab_size, embedding_dim = embedding_weight.shape
        else:
            vocab_size, embedding_dim = 22, 128

        # Infer hidden size from LSTM weights
        hidden_size = 128
        num_layers = 1
        for key in state_dict.keys():
            if 'lstm.weight_ih_l0' in key:
                hidden_size = state_dict[key].shape[0] // 4
            if 'lstm.weight_ih_l1' in key:
                num_layers = 2

        model_config = {
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': 0.5,
            'max_length': 2048
        }

    model = BinaryARGClassifier(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, checkpoint


def evaluate_model(
    model: BinaryARGClassifier,
    test_loader: DataLoader,
    device: str,
    logger
) -> dict:
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

            # Get probabilities
            probs = torch.sigmoid(outputs).cpu().numpy()
            preds = (probs > 0.5).astype(int)

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

    # Load test data
    logger.info("Loading test data...")
    test_seqs, test_labels = load_data(config['data']['test_csv'], logger)

    max_length = config['model']['max_length']
    test_dataset = BinarySequenceDataset(test_seqs, test_labels, max_length)
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )

    # Run evaluation
    logger.info("Running evaluation...")
    metrics, y_true, y_pred, y_prob = evaluate_model(model, test_loader, device, logger)

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
