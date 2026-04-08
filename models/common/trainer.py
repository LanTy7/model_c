"""Shared training framework for ARG classification models."""
import os
import time
import logging
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix
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
        logger: Optional[logging.Logger] = None
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
        """
        self.model = model.to(config.device)
        self.config = config
        self.criterion = criterion
        self.scheduler = scheduler
        self.metric_fn = metric_fn
        self.logger = logger or logging.getLogger(__name__)

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
        self.early_stopping = EarlyStopping(
            patience=config.patience,
            mode='max' if metric_fn else 'min'
        )

        # Tracking
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_metric': [],
            'val_metric': [],
            'lr': []
        }
        self.best_metric = -float('inf') if metric_fn else float('inf')
        self.best_model_path = None

        os.makedirs(config.save_dir, exist_ok=True)

    def _train_epoch(self, train_loader: DataLoader) -> tuple:
        """Train for one epoch."""
        self.model.train()
        self.optimizer.zero_grad()  # Clear gradients at start
        total_loss = 0.0
        num_batches = 0

        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs = inputs.to(self.config.device)
            targets = targets.to(self.config.device)

            # Mixed precision forward pass
            if self.config.use_amp:
                with autocast():
                    outputs = self.model(inputs)
                    # Handle shape mismatch for binary classification
                    if outputs.dim() == 2 and outputs.shape[1] == 1 and targets.dim() == 1:
                        outputs = outputs.squeeze(1)
                    loss = self.criterion(outputs, targets)
                    loss = loss / self.config.accumulation_steps
            else:
                outputs = self.model(inputs)
                # Handle shape mismatch for binary classification
                if outputs.dim() == 2 and outputs.shape[1] == 1 and targets.dim() == 1:
                    outputs = outputs.squeeze(1)
                loss = self.criterion(outputs, targets)
                loss = loss / self.config.accumulation_steps

            # Backward pass
            if self.config.use_amp:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()

            # Gradient accumulation
            if (batch_idx + 1) % self.config.accumulation_steps == 0:
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

            total_loss += loss.item() * self.config.accumulation_steps
            num_batches += 1

        avg_loss = total_loss / num_batches
        return avg_loss

    def _validate(self, val_loader: DataLoader) -> tuple:
        """Validate the model."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        all_probs = []

        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = inputs.to(self.config.device)
                targets = targets.to(self.config.device)

                if self.config.use_amp:
                    with autocast():
                        outputs = self.model(inputs)
                        # Handle shape mismatch for binary classification
                        is_binary = outputs.dim() == 2 and outputs.shape[1] == 1 and targets.dim() == 1
                        if is_binary:
                            outputs_squeezed = outputs.squeeze(1)
                            loss = self.criterion(outputs_squeezed, targets)
                        else:
                            loss = self.criterion(outputs, targets)
                else:
                    outputs = self.model(inputs)
                    # Handle shape mismatch for binary classification
                    is_binary = outputs.dim() == 2 and outputs.shape[1] == 1 and targets.dim() == 1
                    if is_binary:
                        outputs_squeezed = outputs.squeeze(1)
                        loss = self.criterion(outputs_squeezed, targets)
                    else:
                        loss = self.criterion(outputs, targets)

                total_loss += loss.item()

                # Store predictions
                if is_binary:  # Binary classification
                    probs = torch.sigmoid(outputs).cpu().numpy()
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

        return avg_loss, metrics

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
        if probs.shape[1] == 1:  # Binary
            try:
                metrics['auc'] = roc_auc_score(targets, probs)
            except ValueError:
                metrics['auc'] = 0.0
        else:  # Multi-class
            try:
                metrics['auc'] = roc_auc_score(targets, probs, multi_class='ovr', average='macro')
            except ValueError:
                metrics['auc'] = 0.0

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
            train_loss = self._train_epoch(train_loader)

            # Validate
            val_loss, val_metrics = self._validate(val_loader)

            # Get current metric for early stopping
            if self.metric_fn:
                current_metric = val_metrics.get('f1', val_metrics['accuracy'])
            else:
                current_metric = -val_loss  # Minimize loss

            # Track history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['val_metric'].append(current_metric)
            self.history['lr'].append(self.optimizer.param_groups[0]['lr'])

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
            is_best = current_metric > self.best_metric if self.metric_fn else current_metric < self.best_metric
            if is_best:
                self.best_metric = current_metric
                self.best_model_path = save_path
                self.save_checkpoint(save_path, epoch, val_metrics)
                self.logger.info(f"  -> Best model saved! (Metric: {self.best_metric:.4f})")

            # Early stopping
            if self.early_stopping(current_metric):
                self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

        return self.history

    def save_checkpoint(self, path: str, epoch: int, metrics: Dict):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str) -> Dict:
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.config.device)
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
