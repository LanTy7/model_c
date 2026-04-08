"""Multi-class classification model for ARG categories."""
import torch
import torch.nn as nn
from models.common.bilstm import BiLSTMBackbone, GlobalPooling, ClassifierHead


class MultiClassARGClassifier(nn.Module):
    """
    BiLSTM-based multi-class classifier for ARG categories.
    Uses one-hot encoding for input.
    """

    def __init__(
        self,
        input_size: int = 21,  # 20 amino acids + PAD indicator
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.4,
        num_classes: int = 14
    ):
        """
        Args:
            input_size: One-hot encoding dimension (20 AA + PAD)
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            num_classes: Number of ARG categories
        """
        super().__init__()

        self.backbone = BiLSTMBackbone(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True
        )

        self.pooling = GlobalPooling(pooling_type='both')

        # Output size from backbone * 2 (max + mean pooling)
        classifier_input = self.backbone.output_size * 2

        self.classifier = ClassifierHead(
            input_size=classifier_input,
            hidden_size=hidden_size,
            num_classes=num_classes,
            dropout=dropout
        )

    def _create_mask(self, x: torch.Tensor) -> torch.Tensor:
        """
        Create mask from one-hot encoded input.
        Last channel (index 20) indicates PAD positions.
        """
        # x: (batch, seq_len, 21)
        # PAD channel is 1 for padding positions
        pad_channel = x[:, :, 20]  # (batch, seq_len)
        mask = pad_channel < 0.5  # True for valid positions
        return mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: One-hot encoded sequences (batch, seq_len, 21)

        Returns:
            Logits (batch, num_classes)
        """
        # Create mask from PAD channel
        mask = self._create_mask(x)  # (batch, seq_len)

        # BiLSTM with masking
        lstm_out = self.backbone(x, mask)  # (batch, seq_len, hidden*2)

        # Masked global pooling
        features = self.pooling(lstm_out, mask)  # (batch, hidden*4)

        # Classifier
        logits = self.classifier(features)  # (batch, num_classes)

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        logits = self.forward(x)
        return torch.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions."""
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=1)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get sequence embeddings before classification."""
        mask = self._create_mask(x)
        lstm_out = self.backbone(x, mask)
        return self.pooling(lstm_out, mask)


class FocalLoss(nn.Module):
    """
    Focal Loss for handling class imbalance.
    Reference: Lin et al., "Focal Loss for Dense Object Detection", ICCV 2017
    """

    def __init__(
        self,
        alpha: torch.Tensor = None,
        gamma: float = 2.0,
        label_smoothing: float = 0.0,
        reduction: str = 'mean'
    ):
        """
        Args:
            alpha: Class weights tensor
            gamma: Focusing parameter (higher = more focus on hard examples)
            label_smoothing: Label smoothing factor
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.reduction = reduction

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            inputs: Model logits (batch, num_classes)
            targets: Ground truth class indices (batch,)

        Returns:
            Focal loss value
        """
        ce_loss = torch.nn.functional.cross_entropy(
            inputs,
            targets,
            weight=self.alpha,
            reduction='none',
            label_smoothing=self.label_smoothing
        )

        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss
