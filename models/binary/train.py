#!/usr/bin/env python3
"""Training script for binary ARG classification."""
import os
import sys
import argparse
from pathlib import Path
from typing import Tuple, List

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.binary.model import BinaryARGClassifier
from models.common.trainer import Trainer, TrainConfig, get_cosine_schedule_with_warmup
from data.dataset import BinarySequenceDataset
from utils.common import setup_logging, set_seed, load_config


class LabelSmoothedBCEWithLogitsLoss(nn.Module):
    """BCEWithLogitsLoss with label smoothing for binary classification.

    Label smoothing prevents overconfidence by replacing hard labels
    (0, 1) with soft targets (smoothing/2, 1 - smoothing/2).
    """

    def __init__(self, pos_weight=None, smoothing=0.0):
        super().__init__()
        self.smoothing = smoothing
        self.bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')

    def forward(self, inputs, targets):
        # targets: 0 or 1, shape (batch,) or (batch, 1)
        # smoothed: 0 -> smoothing/2, 1 -> 1 - smoothing/2
        if targets.dim() == 1 and inputs.dim() == 2:
            targets = targets.unsqueeze(1)
        targets = targets * (1 - self.smoothing) + self.smoothing * 0.5
        return self.bce(inputs, targets).mean()


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

    # Compute or get pos_weight
    n_pos = sum(train_labels)
    n_neg = len(train_labels) - n_pos
    auto_pos_weight = n_neg / max(n_pos, 1)

    # Check if custom pos_weight is specified in config
    custom_pos_weight = config['training'].get('pos_weight')
    if custom_pos_weight is not None:
        pos_weight = float(custom_pos_weight)
        logger.info(f"Using custom pos_weight: {pos_weight:.4f} "
                   f"(auto-calculated would be: {auto_pos_weight:.4f})")
    else:
        pos_weight = auto_pos_weight
        logger.info(f"Using auto-calculated pos_weight: {pos_weight:.4f}")

    # Create datasets
    train_dataset = BinarySequenceDataset(train_seqs, train_labels, max_length, training=True)
    val_dataset = BinarySequenceDataset(val_seqs, val_labels, max_length, training=False)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=True,
        num_workers=config['training']['num_workers'],
        persistent_workers=config['training']['num_workers'] > 0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        persistent_workers=config['training']['num_workers'] > 0,
        pin_memory=True
    )

    return train_loader, val_loader, pos_weight


def main(config_path: str):
    """Main training function."""
    # Load config
    config = load_config(config_path)

    # Setup
    set_seed(config.get('seed', 42))
    log_dir = config['paths']['log_dir']
    logger = setup_logging(log_dir)
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
    logger.info("Using default enhanced architecture (CNN + Attention)")

    # Loss function with pos_weight and optional label smoothing
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)
    logger.info(f"Using pos_weight: {pos_weight:.4f}")
    label_smoothing = config['training'].get('label_smoothing', 0.0)
    if label_smoothing > 0:
        logger.info(f"Using label smoothing: {label_smoothing}")
        criterion = LabelSmoothedBCEWithLogitsLoss(
            pos_weight=pos_weight_tensor,
            smoothing=label_smoothing
        )
    else:
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
    training_config = config['training']
    save_dir = config['paths']['save_dir']
    fig_dir = config['paths'].get('fig_dir')
    log_dir = config['paths']['log_dir']

    train_config = TrainConfig(
        epochs=training_config['epochs'],
        batch_size=training_config['batch_size'],
        lr=training_config['lr'],
        weight_decay=training_config['weight_decay'],
        patience=training_config['patience'],
        grad_clip=training_config['grad_clip'],
        warmup_epochs=training_config['warmup_epochs'],
        use_amp=training_config.get('use_amp', True),
        accumulation_steps=training_config.get('accumulation_steps', 1),
        save_dir=save_dir,
        device=device,
        num_workers=training_config.get('num_workers', 4),
        fig_dir=fig_dir,
        class_names=['non-ARG', 'ARG'],
        # AECR parameters (enabled by default)
        use_aecr=True,
        lambda_aecr=training_config.get('lambda_aecr', 0.1),
        aecr_sigma=training_config.get('aecr_sigma', 3.0),
        aecr_lambda_ent=training_config.get('aecr_lambda_ent', 1.0),
        aecr_lambda_loc=training_config.get('aecr_lambda_loc', 0.5)
    )

    # Define metric function for early stopping (use validation F2)
    # F2 emphasizes recall, which is critical for ARG detection (avoid missing ARGs)
    def val_metric_fn(targets, preds, probs):
        from sklearn.metrics import fbeta_score
        return fbeta_score(targets, preds, beta=2.0, average='binary', zero_division=0)

    # Trainer
    trainer = Trainer(
        model=model,
        config=train_config,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        metric_fn=val_metric_fn,
        logger=logger,
        model_config=model_config
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
            persistent_workers=config['training']['num_workers'] > 0,
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
