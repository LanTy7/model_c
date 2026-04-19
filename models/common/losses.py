"""Unified loss functions for binary and multi-class classification."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    Focal Loss for binary and multi-class classification.

    Down-weights easy examples and focuses on hard negatives/positives.
    Reference: Lin et al. "Focal Loss for Dense Object Detection" ICCV 2017

    Args:
        task_type: 'binary' or 'multiclass'
        alpha: For binary: float scalar (default 0.25). For multiclass: class weights tensor.
        gamma: Focusing parameter (higher = more focus on hard examples)
        pos_weight: For binary only: weight for positive class (passed to BCEWithLogitsLoss)
        label_smoothing: Label smoothing factor
        reduction: 'mean', 'sum', or 'none' (multiclass only; binary always uses mean)
    """

    def __init__(
        self,
        task_type: str = 'binary',
        alpha=None,
        gamma: float = 2.0,
        pos_weight=None,
        label_smoothing: float = 0.0,
        reduction: str = 'mean'
    ):
        super().__init__()
        if task_type not in ('binary', 'multiclass'):
            raise ValueError("task_type must be 'binary' or 'multiclass'")
        self.task_type = task_type
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        if self.task_type == 'binary':
            return self._forward_binary(inputs, targets)
        return self._forward_multiclass(inputs, targets)

    def _forward_binary(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: logits, shape (batch,) or (batch, 1)
        # targets: 0 or 1, shape (batch,) or (batch, 1)
        if targets.dim() == 1 and inputs.dim() == 2:
            targets = targets.unsqueeze(1).float()
        else:
            targets = targets.float()

        # Label smoothing
        smoothed_targets = targets
        if self.label_smoothing > 0:
            smoothed_targets = targets * (1 - self.label_smoothing) + self.label_smoothing * 0.5

        # BCE with logits (no reduction)
        bce_loss = F.binary_cross_entropy_with_logits(
            inputs, smoothed_targets, pos_weight=self.pos_weight, reduction='none'
        )

        # p_t: probability of correct class
        probs = torch.sigmoid(inputs)
        p_t = probs * targets + (1 - probs) * (1 - targets)

        # Focal weight: alpha_t balances positive/negative classes
        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        else:
            alpha_t = 1.0
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        return (focal_weight * bce_loss).mean()

    def _forward_multiclass(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # inputs: logits (batch, num_classes)
        # targets: class indices (batch,)
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            weight=self.alpha,
            reduction='none',
            label_smoothing=self.label_smoothing
        )

        # When label_smoothing > 0, ce_loss is no longer -log(pt), so
        # pt cannot be derived from ce_loss. Compute pt directly instead.
        if self.label_smoothing > 0:
            probs = torch.softmax(inputs, dim=1)
            pt = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        else:
            pt = torch.exp(-ce_loss)

        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
