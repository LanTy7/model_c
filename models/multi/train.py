#!/usr/bin/env python3
"""Training script for multi-class ARG classification."""
import os
import sys
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.multi.model import MultiClassARGClassifier, FocalLoss
from models.common.trainer import Trainer, TrainConfig, get_cosine_schedule_with_warmup
from data.dataset import MultiClassARGDataset
from utils.common import setup_logging, set_seed, load_config
from utils.sequence_utils import compute_class_weights, get_max_length, BalancedClassSampler


def load_and_preprocess_data(csv_file: str, min_samples: int, logger):
    """Load CSV and preprocess labels."""
    logger.info(f"Loading data from {csv_file}")

    df = pd.read_csv(csv_file)

    # Filter: only use ARG sequences (is_arg=1) for multi-class classification
    df = df[df['is_arg'] == 1].copy()
    logger.info(f"Filtered to {len(df)} ARG sequences (is_arg=1)")

    # Use sequence column directly
    sequences = df['sequence'].str.upper().tolist()

    # Use arg_category column for labels
    labels = df['arg_category'].fillna('Others').tolist()

    logger.info(f"Loaded {len(sequences)} sequences")

    # Count labels
    label_counts = Counter(labels)
    logger.info(f"Original classes: {len(label_counts)}")

    # Merge rare classes
    rare_classes = [c for c, count in label_counts.items() if count < min_samples]
    if rare_classes:
        logger.info(f"Merging {len(rare_classes)} rare classes into 'Others'")
        labels = [l if l not in rare_classes else 'Others' for l in labels]

    # Create label mapping
    unique_labels = sorted(set(labels))
    if 'Others' in unique_labels:
        unique_labels.remove('Others')
        unique_labels.append('Others')

    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
    idx_to_label = {idx: label for label, idx in label_to_idx.items()}

    # Convert to indices
    y = np.array([label_to_idx[l] for l in labels], dtype=np.int64)

    # Get class counts
    class_counts = np.bincount(y)
    logger.info(f"Final classes: {len(unique_labels)}")
    for i, label in enumerate(unique_labels):
        logger.info(f"  {label}: {class_counts[i]}")

    return sequences, y, unique_labels, label_to_idx, class_counts


def create_dataloaders(config: dict, logger):
    """Create train and validation dataloaders."""
    min_samples = config['data']['min_samples']

    # Load train data
    train_seqs, train_labels, class_names, label_to_idx, class_counts = \
        load_and_preprocess_data(config['data']['train_csv'], min_samples, logger)

    # Load val data using same label mapping
    val_df = pd.read_csv(config['data']['val_csv'])

    # Filter: only use ARG sequences (is_arg=1) for multi-class classification
    val_df = val_df[val_df['is_arg'] == 1].copy()
    logger.info(f"Filtered val to {len(val_df)} ARG sequences (is_arg=1)")

    val_seqs = val_df['sequence'].str.upper().tolist()
    val_labels_str = val_df['arg_category'].fillna('Others').tolist()

    # Map val labels (unknown labels become 'Others')
    val_labels = []
    for label in val_labels_str:
        if label in label_to_idx:
            val_labels.append(label_to_idx[label])
        else:
            val_labels.append(label_to_idx.get('Others', 0))
    val_labels = np.array(val_labels, dtype=np.int64)

    # Compute max length
    max_length = get_max_length(train_seqs, config['data']['max_length_percentile'])
    logger.info(f"Max length (95th percentile): {max_length}")

    # Create datasets
    train_dataset = MultiClassARGDataset(train_seqs, train_labels, max_length)
    val_dataset = MultiClassARGDataset(val_seqs, val_labels, max_length)

    # Create dataloaders with balanced sampler for training
    use_balanced_sampling = config['training'].get('use_balanced_sampling', False)
    if use_balanced_sampling:
        logger.info("Using balanced class sampling for training")
        # Compute samples per class (median of class counts to avoid over-sampling too much)
        samples_per_class = int(np.median(class_counts))
        sampler = BalancedClassSampler(train_labels, samples_per_class=samples_per_class)
        train_loader = DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            sampler=sampler,
            num_workers=config['training']['num_workers'],
            persistent_workers=config['training']['num_workers'] > 0,
            pin_memory=True
        )
    else:
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

    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

    return train_loader, val_loader, class_names, label_to_idx, class_counts, max_length


def main(config_path: str):
    """Main training function."""
    global logger

    # Load config
    config = load_config(config_path)

    # Setup
    set_seed(config.get('seed', 42))
    log_dir = config['paths']['log_dir']
    logger = setup_logging(log_dir)
    device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')

    logger.info("=" * 60)
    logger.info("Multi-class ARG Classification Training")
    logger.info("=" * 60)
    logger.info(f"Device: {device}")

    # Create dataloaders
    train_loader, val_loader, class_names, label_to_idx, class_counts, max_length = \
        create_dataloaders(config, logger)

    num_classes = len(class_names)

    # Compute class weights
    class_weights = compute_class_weights(class_counts, method=config['focal_loss']['class_weight_method'])
    class_weights = torch.clamp(
        class_weights,
        config['focal_loss']['class_weight_clip'][0],
        config['focal_loss']['class_weight_clip'][1]
    ).to(device)
    logger.info(f"Class weights range: [{class_weights.min():.2f}, {class_weights.max():.2f}]")

    # Create model
    model_config = {k: v for k, v in config['model'].items() if k != 'name'}
    model_config['num_classes'] = num_classes
    model = MultiClassARGClassifier(**model_config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info("Using default enhanced architecture (CNN + Attention)")

    # Loss function
    criterion = FocalLoss(
        alpha=class_weights,
        gamma=config['focal_loss']['gamma'],
        label_smoothing=config['focal_loss']['label_smoothing']
    )

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['training']['lr'],
        weight_decay=config['training']['weight_decay']
    )

    # Scheduler
    # Compute steps based on optimizer steps (effective batches), not raw batches
    accumulation_steps = config['training'].get('accumulation_steps', 1)
    steps_per_epoch = (len(train_loader) + accumulation_steps - 1) // accumulation_steps
    total_steps = steps_per_epoch * config['training']['epochs']
    warmup_steps = steps_per_epoch * config['training']['warmup_epochs']
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
        class_names=class_names,
        # AECR parameters (enabled by default)
        use_aecr=True,
        lambda_aecr=training_config.get('lambda_aecr', 0.1),
        aecr_sigma=training_config.get('aecr_sigma', 3.0),
        aecr_lambda_ent=training_config.get('aecr_lambda_ent', 1.0),
        aecr_lambda_loc=training_config.get('aecr_lambda_loc', 0.5)
    )

    # Define metric function for early stopping (use validation F1)
    def val_metric_fn(targets, preds, probs):
        from sklearn.metrics import f1_score
        return f1_score(targets, preds, average='macro', zero_division=0)

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

    # Save metadata before training starts so it exists even if training is interrupted
    import json
    metadata = {
        'class_names': class_names,
        'label_to_idx': label_to_idx,
        'max_length': int(max_length),
        'num_classes': num_classes,
        'class_counts': class_counts.tolist(),
        'config': config
    }
    metadata_path = os.path.join(config['paths']['save_dir'], 'metadata.json')
    os.makedirs(config['paths']['save_dir'], exist_ok=True)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Initial metadata saved to: {metadata_path}")

    # Train
    history = trainer.train(train_loader, val_loader, save_name='multi_best.pth')

    logger.info("=" * 60)
    logger.info("Training completed!")
    logger.info(f"Best model saved to: {trainer.best_model_path}")
    logger.info(f"Best metric: {trainer.best_metric:.4f}")

    # Test set evaluation (if test data exists)
    if 'test_csv' in config['data'] and os.path.exists(config['data']['test_csv']):
        logger.info("Loading test data for evaluation...")

        # Load test data using same preprocessing
        test_df = pd.read_csv(config['data']['test_csv'])
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

        test_metrics = trainer.evaluate(test_loader, class_names=class_names)

        # Log test metrics
        from utils.metrics import format_metrics_for_display
        logger.info("\n" + format_metrics_for_display(test_metrics))

    # Update metadata with best metric after training completes
    metadata['best_metric'] = float(trainer.best_metric)
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Metadata updated with best_metric at: {metadata_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train multi-class ARG classifier")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/multi_config.yaml",
        help="Path to config file"
    )
    args = parser.parse_args()

    main(args.config)
