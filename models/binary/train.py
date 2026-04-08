#!/usr/bin/env python3
"""Training script for binary ARG classification."""
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
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.binary.model import BinaryARGClassifier
from models.common.trainer import Trainer, TrainConfig, get_cosine_schedule_with_warmup
from utils.sequence_utils import sequence_to_indices


def setup_logging(log_dir: str, log_file: str = "train.log"):
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


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def create_dataloaders(config: dict, logger) -> Tuple[DataLoader, DataLoader, float]:
    """Create train and validation dataloaders."""
    max_length = config['model']['max_length']

    # Load train data from CSV
    logger.info("Loading training data...")
    train_seqs, train_labels = load_data(config['data']['train_csv'], logger)

    # Load val data from CSV
    logger.info("Loading validation data...")
    val_seqs, val_labels = load_data(config['data']['val_csv'], logger)

    # Compute pos_weight
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    pos_weight = n_neg / max(n_pos, 1)

    # Create datasets
    train_dataset = BinarySequenceDataset(train_seqs, train_labels, max_length)
    val_dataset = BinarySequenceDataset(val_seqs, val_labels, max_length)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        pin_memory=True
    )

    return train_loader, val_loader, pos_weight


def main(config_path: str):
    """Main training function."""
    # Load config
    config = load_config(config_path)

    # Setup
    set_seed(config.get('seed', 42))
    logger = setup_logging(config['paths']['log_dir'])
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    logger.info("=" * 60)
    logger.info("Binary ARG Classification Training")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")
    logger.info(f"Config: {config_path}")

    # Create dataloaders
    train_loader, val_loader, pos_weight = create_dataloaders(config, logger)

    # Create model
    model_config = {k: v for k, v in config['model'].items() if k != 'name'}
    model = BinaryARGClassifier(**model_config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss function with pos_weight
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)
    logger.info(f"Using pos_weight: {pos_weight:.4f}")
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )

    # Scheduler
    total_steps = len(train_loader) * config['training']['epochs']
    warmup_steps = len(train_loader) * config['training']['warmup_epochs']
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    # Training config
    train_config = TrainConfig(
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay'],
        patience=config['training']['patience'],
        grad_clip=config['training']['grad_clip'],
        warmup_epochs=config['training']['warmup_epochs'],
        use_amp=config['training']['use_amp'],
        accumulation_steps=config['training']['accumulation_steps'],
        save_dir=config['paths']['save_dir'],
        device=device,
        num_workers=config['training']['num_workers'],
        fig_dir=config['paths'].get('fig_dir'),
        class_names=['non-ARG', 'ARG']
    )

    # Define metric function for early stopping (use validation F1)
    def val_metric_fn(targets, preds, probs):
        from sklearn.metrics import f1_score
        return f1_score(targets, preds, average='binary', zero_division=0)

    # Trainer
    trainer = Trainer(
        model=model,
        config=train_config,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        metric_fn=val_metric_fn,
        logger=logger
    )

    # Train
    history = trainer.train(train_loader, val_loader, save_name='binary_best.pth')

    logger.info("=" * 60)
    logger.info("Training completed!")
    logger.info(f"Best model saved to: {trainer.best_model_path}")
    logger.info(f"Best metric: {trainer.best_metric:.4f}")

    # Test set evaluation (if test data exists)
    if 'test_csv' in config['data'] and os.path.exists(config['data']['test_csv']):
        logger.info("Loading test data for evaluation...")
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

        test_metrics = trainer.evaluate(test_loader, class_names=['non-ARG', 'ARG'])

        # Log test metrics
        from utils.metrics import format_metrics_for_display
        logger.info("\n" + format_metrics_for_display(test_metrics))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train binary ARG classifier")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/binary_config.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()

    main(args.config)
