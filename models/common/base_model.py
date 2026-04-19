"""Base ARG classifier with shared architecture components."""
import torch
from typing import Union, Tuple
import torch.nn as nn

from models.common.bilstm import (
    BiLSTMAttentionBackbone, GlobalPooling, ClassifierHead
)


class BaseARGClassifier(nn.Module):
    """
    Base class for ARG classifiers (binary and multi-class).

    Contains the shared architecture:
      - Optional Multi-scale CNN
      - BiLSTMAttentionBackbone
      - GlobalPooling
      - ClassifierHead

    Subclasses must override:
      - _encode_input(self, x) -> encoded tensor
      - _create_mask(self, x) -> mask tensor (True for valid positions)

    Subclass __init__ should compute encoder_output_dim and call
    super().__init__(encoder_output_dim=..., ...).
    """

    def __init__(
        self,
        encoder_output_dim: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.4,
        num_classes: int = 1,
        use_attention: bool = True,
        num_attention_heads: int = 4,
        attention_dropout: float = 0.1,
        use_cnn: bool = True,
        cnn_out_channels: int = 64,
        cnn_kernel_sizes: list = None
    ):
        super().__init__()
        self.use_attention = use_attention
        self.use_cnn = use_cnn

        # Multi-scale CNN (optional)
        if self.use_cnn:
            from models.common.multiscale_cnn import MultiScaleCNN
            self.cnn = MultiScaleCNN(
                input_dim=encoder_output_dim,
                out_channels=cnn_out_channels,
                kernel_sizes=cnn_kernel_sizes,
                dropout=dropout
            )
            lstm_input_size = self.cnn.output_dim
        else:
            self.cnn = None
            lstm_input_size = encoder_output_dim

        # Backbone always uses attention by default
        self.backbone = BiLSTMAttentionBackbone(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=True,
            use_attention=use_attention,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout
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

    def _encode_input(self, x: torch.Tensor) -> torch.Tensor:
        """Encode raw input into feature vectors. To be overridden by subclasses."""
        raise NotImplementedError

    def _create_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Create padding mask (True for valid positions). To be overridden by subclasses."""
        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Raw input (subclass-specific)
            mask: Optional padding mask (batch, seq_len), True for valid positions.
                  If None, inferred via _create_mask.
            return_attention: Whether to return attention weights (only if use_attention=True)

        Returns:
            If return_attention=False: Logits (batch, num_classes)
            If return_attention=True and use_attention=True: (logits, attention_weights)
        """
        # Encode input
        encoded = self._encode_input(x)  # (batch, seq_len, feature_dim)

        # Create mask
        if mask is None:
            mask = self._create_mask(x)  # (batch, seq_len), True for valid positions

        # Multi-scale CNN preprocessing (optional)
        if self.use_cnn:
            features = self.cnn(encoded, mask)  # (batch, seq_len, cnn_output_dim)
        else:
            features = encoded

        # BiLSTM + Attention
        backbone_out = self.backbone(features, mask, return_attention=return_attention)
        if return_attention:
            lstm_out, attention_weights = backbone_out
        else:
            lstm_out = backbone_out
            attention_weights = None

        # Global pooling
        pooled = self.pooling(lstm_out, mask)  # (batch, hidden*4)

        # Classifier
        logits = self.classifier(pooled)  # (batch, num_classes)

        if return_attention and attention_weights is not None:
            return logits, attention_weights
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions. Subclasses may override for task-specific activation."""
        logits = self.forward(x, return_attention=False)
        if isinstance(logits, tuple):
            logits = logits[0]
        return logits

    def get_embeddings(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """Get sequence embeddings before classification."""
        encoded = self._encode_input(x)
        if mask is None:
            mask = self._create_mask(x)

        # Multi-scale CNN preprocessing (optional)
        if self.use_cnn:
            features = self.cnn(encoded, mask)
        else:
            features = encoded

        # BiLSTM + Attention
        lstm_out = self.backbone(features, mask, return_attention=False)
        return self.pooling(lstm_out, mask)
