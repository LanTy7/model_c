#!/usr/bin/env python3
"""Training script for binary ARG classification.

Supports both single-split and k-fold cross-validation modes.
"""
import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from typing import Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.binary.model import BinaryARGClassifier
from models.common.trainer import Trainer, TrainConfig, get_cosine_schedule_with_warmup
from data.dataset import BinarySequenceDataset
from utils.common import setup_logging, set_seed, load_config
from utils.metrics import find_optimal_threshold


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


class BinaryFocalLoss(nn.Module):
    """Focal Loss for binary classification.

    Down-weights easy examples and focuses on hard negatives/positives.
    Better than pos_weighted BCE for extreme imbalance.

    Reference: Lin et al. "Focal Loss for Dense Object Detection" ICCV 2017
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, pos_weight=None, label_smoothing: float = 0.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        # inputs: logits, shape (batch,) or (batch, 1)
        # targets: 0 or 1, shape (batch,) or (batch, 1)
        if targets.dim() == 1 and inputs.dim() == 2:
            targets = targets.unsqueeze(1).float()
        else:
            targets = targets.float()

        # Label smoothing
        if self.label_smoothing > 0:
            targets = targets * (1 - self.label_smoothing) + self.label_smoothing * 0.5

        # BCE with logits (no reduction)
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, targets, pos_weight=self.pos_weight, reduction='none'
        )

        # p_t: probability of correct class
        probs = torch.sigmoid(inputs)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Focal weight
        focal_weight = self.alpha * (1 - p_t) ** self.gamma

        return (focal_weight * bce_loss).mean()


def load_data(csv_path: str, logger=None) -> Tuple[List[str], List[int]]:
    """Load data from CSV (sequence column)."""
    df = pd.read_csv(csv_path)
    if logger:
        logger.info(f"Loaded {len(df)} entries from {csv_path}")

    # Use sequence column directly
    sequences = df['sequence'].str.upper().tolist()
    labels = df['is_arg'].astype(int).tolist()

    if logger:
        logger.info(f"Data: {sum(labels)} positive, {len(labels)-sum(labels)} negative")
    return sequences, labels


def create_dataloaders(
    train_csv: str,
    val_csv: str,
    config: dict,
    logger
) -> Tuple[DataLoader, DataLoader, float]:
    """Create train and validation dataloaders."""
    max_length = config['model']['max_length']

    # Load train data from CSV
    logger.info("Loading training data...")
    train_seqs, train_labels = load_data(train_csv, logger)

    # Load val data from CSV
    logger.info("Loading validation data...")
    val_seqs, val_labels = load_data(val_csv, logger)

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


def create_test_loader(test_csv: str, config: dict):
    """Create test dataloader."""
    max_length = config['model']['max_length']
    test_seqs, test_labels = load_data(test_csv, None)
    test_dataset = BinarySequenceDataset(test_seqs, test_labels, max_length)
    return DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=config['training']['num_workers'],
        persistent_workers=config['training']['num_workers'] > 0,
        pin_memory=True
    )


def train_single_fold(
    config: dict,
    logger,
    device: str,
    fold_idx: int = None,
    train_csv: str = None,
    val_csv: str = None,
    test_csv: str = None,
    save_name: str = 'binary_best.pth'
) -> Tuple[Dict, str]:
    """Train a single model fold.

    Returns:
        (history, best_model_path)
    """
    fold_label = f"Fold {fold_idx}" if fold_idx is not None else "Single split"
    logger.info("\n" + "=" * 60)
    logger.info(f"Training: {fold_label}")
    logger.info("=" * 60)

    # Create dataloaders
    train_loader, val_loader, pos_weight = create_dataloaders(
        train_csv, val_csv, config, logger
    )

    # Create model
    model_config = {k: v for k, v in config['model'].items() if k != 'name'}
    model = BinaryARGClassifier(**model_config)
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss function: Focal Loss (default) with fallback to BCE
    pos_weight_tensor = torch.tensor([pos_weight], dtype=torch.float32).to(device)
    label_smoothing = config['training'].get('label_smoothing', 0.0)
    use_focal = config['training'].get('use_focal_loss', True)

    if use_focal:
        focal_alpha = config['training'].get('focal_alpha', 0.25)
        focal_gamma = config['training'].get('focal_gamma', 2.0)
        criterion = BinaryFocalLoss(
            alpha=focal_alpha,
            gamma=focal_gamma,
            pos_weight=pos_weight_tensor,
            label_smoothing=label_smoothing
        )
        logger.info(f"Using Focal Loss (alpha={focal_alpha}, gamma={focal_gamma})")
    elif label_smoothing > 0:
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

    # Per-fold save dir for k-fold
    if fold_idx is not None:
        fold_save_dir = os.path.join(save_dir, f'fold_{fold_idx}')
        fold_fig_dir = os.path.join(fig_dir, f'fold_{fold_idx}') if fig_dir else None
        fold_log_dir = os.path.join(log_dir, f'fold_{fold_idx}')
        os.makedirs(fold_save_dir, exist_ok=True)
        if fold_fig_dir:
            os.makedirs(fold_fig_dir, exist_ok=True)
        os.makedirs(fold_log_dir, exist_ok=True)
    else:
        fold_save_dir = save_dir
        fold_fig_dir = fig_dir
        fold_log_dir = log_dir

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
        save_dir=fold_save_dir,
        device=device,
        num_workers=training_config.get('num_workers', 4),
        fig_dir=fold_fig_dir,
        class_names=['non-ARG', 'ARG'],
        use_aecr=True,
        lambda_aecr=training_config.get('lambda_aecr', 0.1),
        aecr_sigma=training_config.get('aecr_sigma', 3.0),
        aecr_lambda_ent=training_config.get('aecr_lambda_ent', 1.0),
        aecr_lambda_loc=training_config.get('aecr_lambda_loc', 0.5)
    )

    # Define metric function for early stopping (use validation PR-AUC)
    def val_metric_fn(targets, preds, probs):
        from sklearn.metrics import average_precision_score
        try:
            return average_precision_score(targets, probs)
        except ValueError:
            return 0.0

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
    history = trainer.train(train_loader, val_loader, save_name=save_name)

    logger.info(f"Best model saved to: {trainer.best_model_path}")
    logger.info(f"Best validation metric: {trainer.best_metric:.4f}")

    # Threshold tuning on validation set
    logger.info("Tuning classification threshold on validation set...")
    model.eval()
    val_all_probs = []
    val_all_labels = []
    with torch.no_grad():
        for batch in val_loader:
            inputs, mask, targets = batch
            inputs = inputs.to(device)
            mask = mask.to(device)
            outputs = model(inputs, mask=mask)
            probs = torch.sigmoid(outputs).cpu().numpy().flatten()
            val_all_probs.extend(probs.tolist())
            val_all_labels.extend(targets.numpy().flatten().tolist())

    val_all_probs = np.array(val_all_probs)
    val_all_labels = np.array(val_all_labels)
    optimal_threshold, best_f2, _ = find_optimal_threshold(
        val_all_labels, val_all_probs, metric='f2', beta=2.0
    )
    threshold_data = {
        "optimal_threshold": float(optimal_threshold),
        "tune_metric": "f2",
        "description": "Threshold optimized for F2 score on validation set"
    }
    threshold_path = os.path.join(fold_save_dir, 'threshold.json')
    with open(threshold_path, 'w') as f:
        json.dump(threshold_data, f, indent=2)
    logger.info(f"Optimal threshold: {optimal_threshold:.4f} (F2: {best_f2:.4f})")
    logger.info(f"Threshold saved to: {threshold_path}")

    # Test evaluation
    test_metrics = None
    if test_csv and os.path.exists(test_csv):
        logger.info(f"Evaluating on test set: {test_csv}")
        test_loader = create_test_loader(test_csv, config)

        threshold = 0.5
        threshold_path = os.path.join(fold_save_dir, 'threshold.json')
        if os.path.exists(threshold_path):
            with open(threshold_path) as f:
                threshold_data = json.load(f)
                threshold = threshold_data.get('optimal_threshold', 0.5)
            logger.info(f"Using tuned threshold: {threshold:.4f}")

        test_metrics = trainer.evaluate(
            test_loader, class_names=['non-ARG', 'ARG'], threshold=threshold
        )
        from utils.metrics import format_metrics_for_display
        logger.info("\n" + format_metrics_for_display(test_metrics))

    return history, trainer.best_model_path, trainer.best_metric, test_metrics


def train_kfold(config: dict, logger, device: str):
    """Train k-fold cross-validation."""
    data_dir = config['data'].get('data_dir', 'data')
    n_splits = config['data'].get('n_splits', 5)

    logger.info("=" * 60)
    logger.info(f"K-Fold Cross-Validation: {n_splits} folds")
    logger.info("=" * 60)

    fold_results = []
    all_test_metrics = []

    for fold_idx in range(n_splits):
        fold_dir = os.path.join(data_dir, f'fold_{fold_idx}')
        train_csv = os.path.join(fold_dir, 'train.csv')
        val_csv = os.path.join(fold_dir, 'val.csv')
        test_csv = os.path.join(fold_dir, 'test.csv')

        if not os.path.exists(train_csv):
            logger.warning(f"Fold {fold_idx} data not found at {train_csv}, skipping")
            continue

        history, model_path, best_metric, test_metrics = train_single_fold(
            config=config,
            logger=logger,
            device=device,
            fold_idx=fold_idx,
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
            save_name='binary_best.pth'
        )

        fold_results.append({
            'fold': fold_idx,
            'best_metric': best_metric,
            'model_path': model_path,
            'test_metrics': test_metrics
        })
        if test_metrics:
            all_test_metrics.append(test_metrics)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("K-Fold Cross-Validation Summary")
    logger.info("=" * 60)

    for result in fold_results:
        bm = result['best_metric']
        bm_str = f"{bm:.4f}" if bm is not None else "N/A"
        logger.info(f"Fold {result['fold']}: best_val_metric={bm_str}")

    if all_test_metrics:
        # Average test metrics across folds
        avg_metrics = {}
        for key in all_test_metrics[0].keys():
            if isinstance(all_test_metrics[0][key], (int, float)):
                values = [m[key] for m in all_test_metrics if key in m]
                avg_metrics[key] = sum(values) / len(values) if values else 0

        logger.info("\nAverage Test Metrics Across Folds:")
        for key, val in avg_metrics.items():
            logger.info(f"  {key}: {val:.4f}")

    # Evaluate on novelty test set if available
    novelty_csv = os.path.join(data_dir, 'novelty_test.csv')
    if os.path.exists(novelty_csv):
        logger.info("\n" + "=" * 60)
        logger.info("Novelty Test Set Evaluation")
        logger.info("=" * 60)
        evaluate_novelty_test(config, logger, device, novelty_csv, fold_results)

    # Select best fold and copy to top-level checkpoint for backward-compatible inference
    valid_results = [r for r in fold_results if r['best_metric'] is not None]
    if valid_results:
        best_result = max(valid_results, key=lambda r: r['best_metric'])
        best_fold_idx = best_result['fold']
        best_model_src = best_result['model_path']

        save_dir = config['paths']['save_dir']
        os.makedirs(save_dir, exist_ok=True)

        # Copy best model
        best_model_dst = os.path.join(save_dir, 'binary_best.pth')
        if best_model_src and os.path.exists(best_model_src):
            shutil.copy2(best_model_src, best_model_dst)
            logger.info("\n" + "=" * 60)
            logger.info(f"Best model selected from Fold {best_fold_idx}")
            logger.info(f"Copied to: {best_model_dst}")

        # Copy corresponding threshold
        best_threshold_src = os.path.join(save_dir, f'fold_{best_fold_idx}', 'threshold.json')
        best_threshold_dst = os.path.join(save_dir, 'threshold.json')
        if os.path.exists(best_threshold_src):
            shutil.copy2(best_threshold_src, best_threshold_dst)
            logger.info(f"Threshold copied to: {best_threshold_dst}")
        logger.info("=" * 60)

    return fold_results


def evaluate_novelty_test(
    config: dict,
    logger,
    device: str,
    novelty_csv: str,
    fold_results: List[Dict]
):
    """Evaluate all fold models on the novelty test set."""
    from utils.metrics import compute_comprehensive_metrics, format_metrics_for_display

    logger.info(f"Novelty test set: {novelty_csv}")
    test_loader = create_test_loader(novelty_csv, config)

    novelty_metrics_list = []
    for result in fold_results:
        fold_idx = result['fold']
        model_path = result['model_path']

        if not model_path or not os.path.exists(model_path):
            logger.warning(f"Model not found for fold {fold_idx}: {model_path}")
            continue

        # Load model
        model_config = {k: v for k, v in config['model'].items() if k != 'name'}
        model = BinaryARGClassifier(**model_config)
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model = model.to(device)
        model.eval()

        # Load tuned threshold if available, otherwise use 0.5
        threshold = 0.5
        fold_save_dir = os.path.join(config['paths']['save_dir'], f'fold_{fold_idx}')
        threshold_path = os.path.join(fold_save_dir, 'threshold.json')
        if os.path.exists(threshold_path):
            with open(threshold_path) as f:
                threshold_data = json.load(f)
                threshold = threshold_data.get('optimal_threshold', 0.5)
            logger.info(f"Fold {fold_idx} using tuned threshold: {threshold:.4f}")

        # Run inference
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for batch in test_loader:
                indices, mask, labels = batch
                indices = indices.to(device)
                mask = mask.to(device)
                outputs = model(indices, mask=mask)
                probs = torch.sigmoid(outputs).cpu().numpy().flatten()
                all_probs.extend(probs.tolist())
                all_labels.extend(labels.numpy().flatten().tolist())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)
        preds = (all_probs >= threshold).astype(int)

        metrics = compute_comprehensive_metrics(all_labels, preds, all_probs)
        novelty_metrics_list.append(metrics)
        logger.info(f"\nFold {fold_idx} on novelty test:")
        logger.info(format_metrics_for_display(metrics))

    if novelty_metrics_list:
        avg_novelty = {}
        for key in novelty_metrics_list[0].keys():
            if isinstance(novelty_metrics_list[0][key], (int, float)):
                values = [m[key] for m in novelty_metrics_list if key in m]
                avg_novelty[key] = sum(values) / len(values) if values else 0

        logger.info("\n" + "=" * 60)
        logger.info("Average Novelty Test Metrics (All Folds)")
        logger.info("=" * 60)
        for key, val in avg_novelty.items():
            logger.info(f"  {key}: {val:.4f}")

        # Save summary
        summary = {
            'fold_results': [
                {
                    'fold': r['fold'],
                    'best_metric': r['best_metric'],
                    'model_path': r['model_path']
                } for r in fold_results
            ],
            'average_test_metrics': {k: round(v, 4) for k, v in avg_novelty.items()}
        }
        save_dir = config['paths']['save_dir']
        summary_path = os.path.join(save_dir, 'kfold_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f"\nK-fold summary saved to {summary_path}")


def train_final_model(config: dict, logger, device: str):
    """Train final production model on all data.

    Uses the final/ train/val split created by create_training_data.py.
    This model is trained on ~80% of all sequences (all families),
    with ~20% held out as validation for early stopping and threshold tuning.
    """
    data_dir = config['data'].get('data_dir', 'data')
    final_dir = os.path.join(data_dir, 'final')
    train_csv = os.path.join(final_dir, 'train.csv')
    val_csv = os.path.join(final_dir, 'val.csv')

    if not os.path.exists(train_csv):
        logger.error(f"Final split not found at {train_csv}. "
                     f"Run create_training_data.py first.")
        return None

    logger.info("=" * 60)
    logger.info("Final Production Model Training (All Data)")
    logger.info("=" * 60)
    logger.info(f"Training data: {train_csv}")
    logger.info(f"Validation data: {val_csv}")

    history, model_path, best_metric, _ = train_single_fold(
        config=config,
        logger=logger,
        device=device,
        fold_idx=None,
        train_csv=train_csv,
        val_csv=val_csv,
        test_csv=None,
        save_name='binary_final.pth'
    )

    logger.info("=" * 60)
    logger.info("Final production model training completed!")
    logger.info(f"Best model: {model_path}")
    logger.info(f"Best validation metric: {best_metric:.4f}")
    logger.info("=" * 60)

    return model_path


def main(config_path: str, mode: str = 'kfold'):
    """Main training function.

    Args:
        config_path: Path to config YAML file
        mode: 'kfold' for 5-fold cross-validation,
              'final' for production model on all data,
              'single' for backward-compatible single split
    """
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
    logger.info(f"Mode: {mode}")

    if mode == 'kfold':
        train_kfold(config, logger, device)
    elif mode == 'final':
        train_final_model(config, logger, device)
    else:
        # Single split mode (backward compatible)
        train_csv = config['data']['train_csv']
        val_csv = config['data']['val_csv']
        test_csv = config['data'].get('test_csv')

        history, model_path, best_metric, test_metrics = train_single_fold(
            config=config,
            logger=logger,
            device=device,
            train_csv=train_csv,
            val_csv=val_csv,
            test_csv=test_csv,
            save_name='binary_best.pth'
        )

        logger.info("=" * 60)
        logger.info("Training completed!")
        logger.info(f"Best model: {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train binary ARG classifier")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/binary_config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="kfold",
        choices=["kfold", "final", "single"],
        help="Training mode: kfold (5-fold CV, default), final (production model on all data), single (backward compatible)"
    )
    args = parser.parse_args()

    main(args.config, mode=args.mode)
