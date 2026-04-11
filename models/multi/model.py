"""Multi-class classification model for ARG categories."""
import torch
from typing import Union, Tuple
import torch.nn as nn
from models.common.bilstm import (
    BiLSTMBackbone, BiLSTMAttentionBackbone, GlobalPooling, ClassifierHead
)


class MultiClassARGClassifier(nn.Module):
    """
    BiLSTM-based multi-class classifier for ARG categories.
    Uses one-hot encoding for input.
    Optionally supports self-attention layer between BiLSTM and pooling.
    Optionally supports multi-scale CNN before BiLSTM.
    """

    def __init__(
        self,
        input_size: int = 21,  # 20 amino acids + PAD indicator
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.4,
        num_classes: int = 14,
        use_attention: bool = False,
        num_attention_heads: int = 4,
        attention_dropout: float = 0.1,
        use_cnn: bool = False,
        cnn_out_channels: int = 64,
        cnn_kernel_sizes: list = None
    ):
        """
        Args:
            input_size: One-hot encoding dimension (20 AA + PAD)
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            num_classes: Number of ARG categories
            use_attention: Whether to add self-attention layer after BiLSTM
            num_attention_heads: Number of attention heads (if use_attention=True)
            attention_dropout: Dropout for attention weights
            use_cnn: Whether to use multi-scale CNN before BiLSTM
            cnn_out_channels: Output channels per CNN kernel size
            cnn_kernel_sizes: List of CNN kernel sizes (default: [3, 5, 7])
        """
        super().__init__()
        self.use_attention = use_attention
        self.use_cnn = use_cnn

        # Multi-scale CNN (optional)
        if use_cnn:
            from models.common.multiscale_cnn import MultiScaleCNN
            self.cnn = MultiScaleCNN(
                input_dim=input_size,
                out_channels=cnn_out_channels,
                kernel_sizes=cnn_kernel_sizes,
                dropout=dropout
            )
            lstm_input_size = self.cnn.output_dim
        else:
            self.cnn = None
            lstm_input_size = input_size

        # Choose backbone with or without attention
        if use_attention:
            self.backbone = BiLSTMAttentionBackbone(
                input_size=lstm_input_size,
                hidden_size=hidden_size,
                num_layers=num_layers,
                dropout=dropout,
                bidirectional=True,
                use_attention=True,
                num_attention_heads=num_attention_heads,
                attention_dropout=attention_dropout
            )
        else:
            self.backbone = BiLSTMBackbone(
                input_size=lstm_input_size,
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

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            x: One-hot encoded sequences (batch, seq_len, 21)
            return_attention: Whether to return attention weights (only if use_attention=True)

        Returns:
            If return_attention=False: Logits (batch, num_classes)
            If return_attention=True and use_attention=True: (logits, attention_weights)
        """
        # Create mask from PAD channel
        mask = self._create_mask(x)  # (batch, seq_len)

        # Optional multi-scale CNN preprocessing
        if self.use_cnn and self.cnn is not None:
            features = self.cnn(x, mask)  # (batch, seq_len, cnn_output_dim)
        else:
            features = x  # (batch, seq_len, input_size)

        # BiLSTM (+ Attention if enabled)
        if self.use_attention:
            lstm_out = self.backbone(features, mask, return_attention=False)
            # Apply attention separately to get weights
            if return_attention:
                _, attention_weights = self.backbone.attention(lstm_out, mask)
        else:
            lstm_out = self.backbone(features, mask)
            attention_weights = None

        # Masked global pooling
        features = self.pooling(lstm_out, mask)  # (batch, hidden*4)

        # Classifier
        logits = self.classifier(features)  # (batch, num_classes)

        if return_attention and attention_weights is not None:
            return logits, attention_weights
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        logits = self.forward(x, return_attention=False)
        if isinstance(logits, tuple):
            logits = logits[0]
        return torch.softmax(logits, dim=1)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Get class predictions."""
        probs = self.predict_proba(x)
        return torch.argmax(probs, dim=1)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get sequence embeddings before classification."""
        mask = self._create_mask(x)

        # Optional multi-scale CNN preprocessing
        if self.use_cnn and self.cnn is not None:
            features = self.cnn(x, mask)
        else:
            features = x

        if self.use_attention:
            lstm_out = self.backbone(features, mask, return_attention=False)
        else:
            lstm_out = self.backbone(features, mask)
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
