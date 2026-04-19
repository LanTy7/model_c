"""Shared training framework for ARG classification models."""
import os
import time
import logging
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)


@dataclass
class TrainConfig:
    """Training configuration."""
    epochs: int = 100
    batch_size: int = 256
    lr: float = 0.001
    weight_decay: float = 1e-4
    patience: int = 15
    grad_clip: float = 1.0
    warmup_epochs: int = 5
    use_amp: bool = True  # Automatic Mixed Precision
    accumulation_steps: int = 1  # Gradient accumulation
    save_dir: str = './checkpoints'
    device: str = 'cuda'
    num_workers: int = 4
    fig_dir: Optional[str] = None  # Directory for saving figures
    class_names: Optional[List[str]] = None  # Class names for visualization
    # AECR loss configuration (enabled by default)
    use_aecr: bool = True  # Whether to use AECR attention regularization
    lambda_aecr: float = 0.001  # Weight for AECR loss
    aecr_sigma: float = 3.0  # Gaussian kernel sigma for AECR
    aecr_lambda_ent: float = 1.0  # Entropy loss weight
    aecr_lambda_loc: float = 0.0  # Local continuity loss weight (disabled by default)


class EarlyStopping:
    """Early stopping handler."""

    def __init__(self, patience: int = 10, min_delta: float = 0.0, mode: str = 'min'):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as improvement
            mode: 'min' or 'max' for metric direction
        """
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.should_stop = False

    def __call__(self, score: float) -> bool:
        """Check if training should stop."""
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < (self.best_score - self.min_delta)
        else:
            improved = score > (self.best_score + self.min_delta)

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

        return self.should_stop


class Trainer:
    """
    Generic trainer for classification models.
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainConfig,
        criterion: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        scheduler: Optional[Any] = None,
        metric_fn: Optional[Callable] = None,
        logger: Optional[logging.Logger] = None,
        model_config: Optional[dict] = None
    ):
        """
        Args:
            model: PyTorch model
            config: Training configuration
            criterion: Loss function
            optimizer: Optimizer (default: AdamW)
            scheduler: Learning rate scheduler
            metric_fn: Function to compute validation metric
            logger: Logger instance
            model_config: Model architecture config to persist in checkpoint
        """
        self.model = model.to(config.device)
        self.config = config
        self.criterion = criterion
        self.scheduler = scheduler
        self.metric_fn = metric_fn
        self.logger = logger or logging.getLogger(__name__)
        self.model_config = model_config

        # Setup optimizer
        if optimizer is None:
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=config.lr,
                weight_decay=config.weight_decay
            )
        else:
            self.optimizer = optimizer

        # Mixed precision training
        self.scaler = GradScaler() if config.use_amp else None

        # Early stopping
        # current_metric is always "higher is better": metric_fn output or -val_loss
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            mode='max'
        )

        # Tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metric': [],
            'val_metric': [],
            'lr': [],
            'val_accuracy': [],
            'val_precision': [],
            'val_recall': [],
            'val_f1': [],
            'val_auc': []
        }
        self.best_metric = -float('inf')
        self.best_model_path = None

        os.makedirs(config.save_dir, exist_ok=True)

        # AECR loss for attention regularization
        self.aecr_criterion = None
        if config.use_aecr:
            try:
                from .aecr_loss import AECRLoss
                self.aecr_criterion = AECRLoss(
                    sigma=config.aecr_sigma,
                    lambda_ent=config.aecr_lambda_ent,
                    lambda_loc=config.aecr_lambda_loc
                )
                self.logger.info(
                    f"AECR loss enabled (lambda={config.lambda_aecr}, "
                    f"sigma={config.aecr_sigma})"
                )
            except ImportError:
                self.logger.warning("AECR loss not available. Install required.")

    def _train_epoch(self, train_loader: DataLoader) -> tuple:
        """Train for one epoch."""
        self.model.train()
        self.optimizer.zero_grad()  # Clear gradients at start
        total_loss = 0.0
        total_task_loss = 0.0
        total_aecr_loss = 0.0

        # Collect predictions and targets for training metrics
        all_preds = []
        all_targets = []
        all_probs = []

        # Check if model uses attention
        model_uses_attention = getattr(self.model, 'use_attention', False)

        num_batches = len(train_loader)
        for batch_idx, batch in enumerate(train_loader):
            # Support both 2-tuple (inputs, targets) and 3-tuple (inputs, mask, targets)
            if len(batch) == 3:
                inputs, mask, targets = batch
                mask = mask.to(self.config.device)
            else:
                inputs, targets = batch
                mask = None

            inputs = inputs.to(self.config.device)
            targets = targets.to(self.config.device)

            # Mixed precision forward pass
            if self.config.use_amp:
                with autocast():
                    # Forward pass (with attention if model supports it)
                    if model_uses_attention:
                        if mask is not None:
                            outputs, attention_weights = self.model(inputs, mask=mask, return_attention=True)
                        else:
                            outputs, attention_weights = self.model(inputs, return_attention=True)
                    else:
                        if mask is not None:
                            outputs = self.model(inputs, mask=mask)
                        else:
                            outputs = self.model(inputs)
                        attention_weights = None

                    # Determine binary vs multi-class before squeezing
                    is_binary = outputs.dim() == 2 and outputs.shape[1] == 1 and targets.dim() == 1
                    outputs_for_pred = outputs
                    if is_binary:
                        outputs = outputs.squeeze(1)

                    # Task loss
                    task_loss = self.criterion(outputs, targets)

                    # AECR loss (if enabled and attention weights available)
                    aecr_loss = 0.0
                    if self.aecr_criterion is not None and attention_weights is not None:
                        aecr_loss = self.aecr_criterion(attention_weights)
                        aecr_loss = aecr_loss * self.config.lambda_aecr

                    # Total loss
                    raw_loss = task_loss + aecr_loss
                    loss = raw_loss / self.config.accumulation_steps
            else:
                # Forward pass (with attention if model supports it)
                if model_uses_attention:
                    if mask is not None:
                        outputs, attention_weights = self.model(inputs, mask=mask, return_attention=True)
                    else:
                        outputs, attention_weights = self.model(inputs, return_attention=True)
                else:
                    if mask is not None:
                        outputs = self.model(inputs, mask=mask)
                    else:
                        outputs = self.model(inputs)
                    attention_weights = None

                # Determine binary vs multi-class before squeezing
                is_binary = outputs.dim() == 2 and outputs.shape[1] == 1 and targets.dim() == 1
                outputs_for_pred = outputs
                if is_binary:
                    outputs = outputs.squeeze(1)

                # Task loss
                task_loss = self.criterion(outputs, targets)

                # AECR loss (if enabled and attention weights available)
                aecr_loss = 0.0
                if self.aecr_criterion is not None and attention_weights is not None:
                    aecr_loss = self.aecr_criterion(attention_weights)
                    aecr_loss = aecr_loss * self.config.lambda_aecr

                # Total loss
                raw_loss = task_loss + aecr_loss
                loss = raw_loss / self.config.accumulation_steps

            # Backward pass
            if self.config.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation: step on full groups and on the last batch
            is_last_batch = (batch_idx + 1) == num_batches
            if (batch_idx + 1) % self.config.accumulation_steps == 0 or is_last_batch:
                if self.config.grad_clip > 0:
                    if self.config.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.grad_clip
                    )

                if self.config.use_amp:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad()

                if self.scheduler:
                    self.scheduler.step()

            # Collect predictions for training metrics
            with torch.no_grad():
                if is_binary:
                    probs = torch.sigmoid(outputs_for_pred).cpu().numpy().squeeze(-1)
                    preds = (probs > 0.5).astype(int)
                else:
                    probs = torch.softmax(outputs_for_pred, dim=1).cpu().numpy()
                    preds = np.argmax(probs, axis=1)

                all_preds.extend(preds.flatten())
                all_targets.extend(targets.cpu().numpy().flatten())
                all_probs.append(probs)

            total_loss += raw_loss.item()
            total_task_loss += task_loss.item()
            if isinstance(aecr_loss, torch.Tensor):
                total_aecr_loss += aecr_loss.item()

        avg_loss = total_loss / num_batches

        # Check for NaN/Inf in training loss
        if not np.isfinite(avg_loss):
            raise RuntimeError(
                f"Non-finite training loss detected: {avg_loss}. "
                f"This may indicate numerical instability from pos_weight, AMP, or AECR."
            )

        # Log loss components if AECR is used
        if self.aecr_criterion is not None:
            avg_task_loss = total_task_loss / num_batches
            avg_aecr_loss = total_aecr_loss / num_batches
            self.logger.debug(
                f"  Train Loss: {avg_loss:.4f} "
                f"(Task: {avg_task_loss:.4f}, AECR: {avg_aecr_loss:.4f})"
            )

        # Compute training metrics
        all_probs = np.concatenate(all_probs, axis=0)
        train_metrics = self._compute_metrics(all_targets, all_preds, all_probs)

        return avg_loss, train_metrics

    def _validate(self, val_loader: DataLoader) -> tuple:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []

        model_uses_attention = getattr(self.model, 'use_attention', False)

        with torch.no_grad():
            for batch in val_loader:
                # Support both 2-tuple and 3-tuple batches
                if len(batch) == 3:
                    inputs, mask, targets = batch
                    mask = mask.to(self.config.device)
                else:
                    inputs, targets = batch
                    mask = None

                inputs = inputs.to(self.config.device)
                targets = targets.to(self.config.device)

                if self.config.use_amp:
                    with autocast():
                        if model_uses_attention:
                            outputs, attention_weights = self.model(inputs, mask=mask, return_attention=True)
                        else:
                            outputs = self.model(inputs, mask=mask)
                            attention_weights = None

                        # Handle shape mismatch for binary classification
                        is_binary = outputs.dim() == 2 and outputs.shape[1] == 1 and targets.dim() == 1
                        if is_binary:
                            outputs_squeezed = outputs.squeeze(1)
                            task_loss = self.criterion(outputs_squeezed, targets)
                        else:
                            task_loss = self.criterion(outputs, targets)

                        # Add AECR loss for consistent train/val objective
                        if self.aecr_criterion is not None and attention_weights is not None:
                            aecr_loss = self.aecr_criterion(attention_weights) * self.config.lambda_aecr
                            loss = task_loss + aecr_loss
                        else:
                            loss = task_loss
                else:
                    if model_uses_attention:
                        if mask is not None:
                            outputs, attention_weights = self.model(inputs, mask=mask, return_attention=True)
                        else:
                            outputs, attention_weights = self.model(inputs, return_attention=True)
                    else:
                        if mask is not None:
                            outputs = self.model(inputs, mask=mask)
                        else:
                            outputs = self.model(inputs)
                        attention_weights = None

                    # Handle shape mismatch for binary classification
                    is_binary = outputs.dim() == 2 and outputs.shape[1] == 1 and targets.dim() == 1
                    if is_binary:
                        outputs_squeezed = outputs.squeeze(1)
                        task_loss = self.criterion(outputs_squeezed, targets)
                    else:
                        task_loss = self.criterion(outputs, targets)

                    # Add AECR loss for consistent train/val objective
                    if self.aecr_criterion is not None and attention_weights is not None:
                        aecr_loss = self.aecr_criterion(attention_weights) * self.config.lambda_aecr
                        loss = task_loss + aecr_loss
                    else:
                        loss = task_loss

                total_loss += loss.item()

                # Store predictions
                if is_binary:  # Binary classification
                    probs = torch.sigmoid(outputs).cpu().numpy().squeeze(-1)
                    preds = (probs > 0.5).astype(int)
                else:  # Multi-class
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
                    preds = np.argmax(probs, axis=1)

                all_preds.extend(preds.flatten())
                all_targets.extend(targets.cpu().numpy().flatten())
                all_probs.append(probs)

        avg_loss = total_loss / len(val_loader)
        all_probs = np.concatenate(all_probs, axis=0)

        # Compute metrics
        metrics = self._compute_metrics(all_targets, all_preds, all_probs)

        return avg_loss, metrics, np.array(all_targets), np.array(all_preds), all_probs

    def _compute_metrics(
        self,
        targets: np.ndarray,
        preds: np.ndarray,
        probs: np.ndarray
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        metrics = {
            'accuracy': accuracy_score(targets, preds),
            'precision': precision_score(targets, preds, average='macro', zero_division=0),
            'recall': recall_score(targets, preds, average='macro', zero_division=0),
            'f1': f1_score(targets, preds, average='macro', zero_division=0)
        }

        # ROC-AUC for binary or multi-class
        if probs.ndim == 1 or probs.shape[1] == 1:  # Binary
            try:
                metrics['auc'] = roc_auc_score(targets, probs)
            except ValueError:
                metrics['auc'] = 0.0
            try:
                metrics['pr_auc'] = average_precision_score(targets, probs)
            except ValueError:
                metrics['pr_auc'] = 0.0
        else:  # Multi-class
            try:
                metrics['auc'] = roc_auc_score(targets, probs, multi_class='ovr', average='macro')
            except ValueError:
                metrics['auc'] = 0.0
            metrics['pr_auc'] = 0.0

        return metrics

    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        save_name: str = 'best_model.pth'
    ) -> Dict:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            save_name: Name for saved model file

        Returns:
            Training history dictionary
        """
        self.logger.info(f"Starting training for {self.config.epochs} epochs...")
        self.logger.info(f"Device: {self.config.device}")
        self.logger.info(f"AMP enabled: {self.config.use_amp}")

        save_path = os.path.join(self.config.save_dir, save_name)

        for epoch in range(self.config.epochs):
            start_time = time.time()

            # Train
            train_loss, train_metrics = self._train_epoch(train_loader)

            # Validate
            val_loss, val_metrics, val_targets, val_preds, val_probs = self._validate(val_loader)

            # Get current metric for early stopping
            if self.metric_fn:
                current_metric = self.metric_fn(val_targets, val_preds, val_probs)
            else:
                current_metric = -val_loss  # Minimize loss

            # Track history
            self.history['train_loss'].append(train_loss)
            self.history['train_metric'].append(train_metrics.get('f1', 0))
            self.history['val_loss'].append(val_loss)
            self.history['val_metric'].append(current_metric)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])
            self.history['val_accuracy'].append(val_metrics.get('accuracy', 0))
            self.history['val_precision'].append(val_metrics.get('precision', 0))
            self.history['val_recall'].append(val_metrics.get('recall', 0))
            self.history['val_f1'].append(val_metrics.get('f1', 0))
            self.history['val_auc'].append(val_metrics.get('auc', 0))

            # Log
            epoch_time = time.time() - start_time
            self.logger.info(
                f"Epoch {epoch+1:03d}/{self.config.epochs} | "
                f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
                f"Acc: {val_metrics['accuracy']:.4f} | "
                f"F1: {val_metrics['f1']:.4f} | "
                f"LR: {self.history['lr'][-1]:.6f} | "
                f"Time: {epoch_time:.1f}s"
            )

            # Save best model
            is_best = current_metric > self.best_metric
            if is_best:
                self.best_metric = current_metric
                self.best_model_path = save_path
                self.save_checkpoint(save_path, epoch, val_metrics)
                self.logger.info(f"  -> Best model saved! (Metric: {self.best_metric:.4f})")

            # Early stopping
            if self.early_stopping(current_metric):
                self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        # Save history and generate plots if fig_dir is provided
        if self.config.fig_dir:
            self._save_history()
            self._generate_training_plots()

        return self.history

    def _save_history(self) -> None:
        """Save training history to JSON."""
        import json
        history_path = os.path.join(self.config.save_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        self.logger.info(f"Training history saved to: {history_path}")

    def _generate_training_plots(self) -> None:
        """Generate training visualization plots."""
        try:
            from utils.visualization import plot_training_curves, plot_learning_rate_schedule
            os.makedirs(self.config.fig_dir, exist_ok=True)
            plot_training_curves(self.history, self.config.fig_dir)
            plot_learning_rate_schedule(self.history, self.config.fig_dir)
            self.logger.info(f"Training plots saved to: {self.config.fig_dir}")
        except Exception as e:
            self.logger.warning(f"Failed to generate training plots: {e}")

    def evaluate(
        self,
        test_loader: DataLoader,
        class_names: Optional[List[str]] = None,
        threshold: float = 0.5
    ) -> Dict[str, Any]:
        """
        Comprehensive evaluation on test set.

        Args:
            test_loader: Test data loader
            class_names: List of class names for visualization
            threshold: Classification threshold for binary classification (default: 0.5)

        Returns:
            Dictionary containing all evaluation metrics
        """
        from utils.metrics import compute_comprehensive_metrics
        from utils.visualization import (
            create_training_report, plot_confusion_matrix,
            plot_per_class_metrics, plot_roc_curves, save_metrics_json
        )

        self.logger.info("Running test set evaluation...")
        self.model.eval()

        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for batch in test_loader:
                # Support both 2-tuple and 3-tuple batches
                if len(batch) == 3:
                    inputs, mask, targets = batch
                    mask = mask.to(self.config.device)
                else:
                    inputs, targets = batch
                    mask = None

                inputs = inputs.to(self.config.device)
                targets = targets.to(self.config.device)

                if self.config.use_amp:
                    with autocast():
                        if mask is not None:
                            outputs = self.model(inputs, mask=mask)
                        else:
                            outputs = self.model(inputs)
                else:
                    if mask is not None:
                        outputs = self.model(inputs, mask=mask)
                    else:
                        outputs = self.model(inputs)

                # Get predictions and probabilities
                is_binary = outputs.dim() == 2 and outputs.shape[1] == 1 and targets.dim() == 1

                if is_binary:
                    probs = torch.sigmoid(outputs).cpu().numpy()
                    preds = (probs > threshold).astype(int)
                else:
                    probs = torch.softmax(outputs, dim=1).cpu().numpy()
                    preds = np.argmax(probs, axis=1)

                all_preds.extend(preds.flatten())
                all_targets.extend(targets.cpu().numpy().flatten())
                all_probs.append(probs if is_binary else probs)

        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        y_prob = np.concatenate(all_probs, axis=0)

        # Compute comprehensive metrics
        class_names = class_names or self.config.class_names
        metrics = compute_comprehensive_metrics(y_true, y_pred, y_prob, class_names)

        # Generate visualizations
        if self.config.fig_dir:
            try:
                os.makedirs(self.config.fig_dir, exist_ok=True)

                if class_names:
                    plot_confusion_matrix(y_true, y_pred, class_names, self.config.fig_dir)
                    plot_per_class_metrics(metrics, class_names, self.config.fig_dir)
                    plot_roc_curves(y_true, y_prob, class_names, self.config.fig_dir)

                self.logger.info(f"Evaluation plots saved to: {self.config.fig_dir}")
            except Exception as e:
                self.logger.warning(f"Failed to generate evaluation plots: {e}")

        # Save metrics
        metrics_path = os.path.join(self.config.save_dir, 'test_metrics.json')
        save_metrics_json(metrics, metrics_path)
        self.logger.info(f"Test metrics saved to: {metrics_path}")

        return metrics

    def save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config,
            'model_config': self.model_config
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> Dict:
        """Load model checkpoint."""
        try:
            checkpoint = torch.load(path, map_location=self.config.device, weights_only=True)
        except (TypeError, RuntimeError):
            # Fallback for older PyTorch or checkpoints with custom objects
            checkpoint = torch.load(path, map_location=self.config.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        return checkpoint


def get_cosine_schedule_with_warmup(
    optimizer: torch.optim.Optimizer,
    num_warmup_steps: int,
    num_training_steps: int
):
    """
    Create a learning rate scheduler with linear warmup and cosine decay.
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(
            max(1, num_training_steps - num_warmup_steps)
        )
        return max(0.0, 0.5 * (1.0 + np.cos(np.pi * progress)))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
