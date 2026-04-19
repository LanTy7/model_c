"""Multi-class classification model for ARG categories."""
import torch
import torch.nn as nn

from models.common.base_model import BaseARGClassifier


class MultiClassARGClassifier(BaseARGClassifier):
    """
    BiLSTM-based multi-class classifier for ARG categories.
    Uses one-hot encoding for input.
    Optionally supports self-attention layer between BiLSTM and pooling.
    Optionally supports multi-scale CNN before BiLSTM.
    """

    def __init__(
        self,
        input_size: int = 21,  # 20 amino acids + PAD indicator
        hidden_size: int = 256,
        num_layers: int = 3,
        dropout: float = 0.4,
        num_classes: int = 14,
        use_attention: bool = True,
        num_attention_heads: int = 8,
        attention_dropout: float = 0.1,
        use_cnn: bool = True,
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
            use_attention: Whether to add self-attention layer after BiLSTM (default: True)
            num_attention_heads: Number of attention heads
            attention_dropout: Dropout for attention weights
            use_cnn: Whether to use multi-scale CNN before BiLSTM (default: True)
            cnn_out_channels: Output channels per CNN kernel size
            cnn_kernel_sizes: List of CNN kernel sizes (default: [3, 5, 7])
        """
        self.input_size = input_size

        super().__init__(
            encoder_output_dim=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=num_classes,
            use_attention=use_attention,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            use_cnn=use_cnn,
            cnn_out_channels=cnn_out_channels,
            cnn_kernel_sizes=cnn_kernel_sizes
        )

    def _encode_input(self, x: torch.Tensor) -> torch.Tensor:
        """One-hot input is already encoded."""
        return x  # (batch, seq_len, input_size)

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
