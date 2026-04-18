"""Binary classification model for ARG identification."""
import torch
from typing import Union, Tuple
import torch.nn as nn
from models.common.bilstm import (
    BiLSTMBackbone, BiLSTMAttentionBackbone, GlobalPooling, ClassifierHead
)


class BinaryARGClassifier(nn.Module):
    """
    BiLSTM-based binary classifier for ARG vs non-ARG.
    Uses embedding layer for input encoding.
    Optionally supports self-attention layer between BiLSTM and pooling.
    Optionally supports multi-scale CNN before BiLSTM.
    """

    def __init__(
        self,
        vocab_size: int = 22,
        embedding_dim: int = 128,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.4,
        max_length: int = 1000,
        use_attention: bool = True,
        num_attention_heads: int = 4,
        attention_dropout: float = 0.1,
        use_cnn: bool = True,
        cnn_out_channels: int = 64,
        cnn_kernel_sizes: list = None
    ):
        """
        Args:
            vocab_size: Vocabulary size (20 AA + X + PAD)
            embedding_dim: Embedding dimension
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            max_length: Maximum sequence length
            use_attention: Whether to add self-attention layer after BiLSTM (default: True)
            num_attention_heads: Number of attention heads
            attention_dropout: Dropout for attention weights
            use_cnn: Whether to use multi-scale CNN before BiLSTM (default: True)
            cnn_out_channels: Output channels per CNN kernel size
            cnn_kernel_sizes: List of CNN kernel sizes (default: [3, 5, 7])
        """
        super().__init__()
        self.use_attention = use_attention
        self.use_cnn = use_cnn

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0  # PAD token index
        )

        # Multi-scale CNN
        from models.common.multiscale_cnn import MultiScaleCNN
        self.cnn = MultiScaleCNN(
            input_dim=embedding_dim,
            out_channels=cnn_out_channels,
            kernel_sizes=cnn_kernel_sizes,
            dropout=dropout
        )
        lstm_input_size = self.cnn.output_dim

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
            num_classes=1,  # Binary classification
            dropout=dropout
        )

    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input indices (batch, seq_len)
            return_attention: Whether to return attention weights (only if use_attention=True)

        Returns:
            If return_attention=False: Logits (batch, 1)
            If return_attention=True and use_attention=True: (logits, attention_weights)
        """
        # Embedding
        emb = self.embedding(x)  # (batch, seq_len, embedding_dim)

        # Create mask for padding positions
        mask = (x != 0)  # (batch, seq_len), True for valid positions

        # Multi-scale CNN preprocessing
        features = self.cnn(emb, mask)  # (batch, seq_len, cnn_output_dim)

        # BiLSTM + Attention
        backbone_out = self.backbone(features, mask, return_attention=return_attention)
        if return_attention:
            lstm_out, attention_weights = backbone_out
        else:
            lstm_out = backbone_out
            attention_weights = None

        # Global pooling
        features = self.pooling(lstm_out, mask)  # (batch, hidden*4)

        # Classifier
        logits = self.classifier(features)  # (batch, 1)

        if return_attention:
            return logits, attention_weights
        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        logits = self.forward(x, return_attention=False)
        if isinstance(logits, tuple):
            logits = logits[0]
        return torch.sigmoid(logits)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get sequence embeddings before classification."""
        emb = self.embedding(x)
        mask = (x != 0)

        # Multi-scale CNN preprocessing
        features = self.cnn(emb, mask)

        # BiLSTM + Attention
        lstm_out = self.backbone(features, mask, return_attention=False)
        return self.pooling(lstm_out, mask)
