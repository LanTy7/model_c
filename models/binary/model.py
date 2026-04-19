"""Binary classification model for ARG identification."""
import torch
import torch.nn as nn

from models.common.base_model import BaseARGClassifier


class BinaryARGClassifier(BaseARGClassifier):
    """
    BiLSTM-based binary classifier for ARG vs non-ARG.
    Uses embedding layer for input encoding.
    Optionally supports self-attention layer between BiLSTM and pooling.
    Optionally supports multi-scale CNN before BiLSTM.
    """

    def __init__(
        self,
        vocab_size: int = 25,
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
            max_length: Maximum sequence length (unused by model; handled by dataset/preprocessing)
            use_attention: Whether to add self-attention layer after BiLSTM (default: True)
            num_attention_heads: Number of attention heads
            attention_dropout: Dropout for attention weights
            use_cnn: Whether to use multi-scale CNN before BiLSTM (default: True)
            cnn_out_channels: Output channels per CNN kernel size
            cnn_kernel_sizes: List of CNN kernel sizes (default: [3, 5, 7])
        """
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        super().__init__(
            encoder_output_dim=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            num_classes=1,
            use_attention=use_attention,
            num_attention_heads=num_attention_heads,
            attention_dropout=attention_dropout,
            use_cnn=use_cnn,
            cnn_out_channels=cnn_out_channels,
            cnn_kernel_sizes=cnn_kernel_sizes
        )

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0  # PAD token index
        )

    def _encode_input(self, x: torch.Tensor) -> torch.Tensor:
        """Embed input indices into dense vectors."""
        return self.embedding(x)  # (batch, seq_len, embedding_dim)

    def _create_mask(self, x: torch.Tensor) -> torch.Tensor:
        """Create mask from embedding indices."""
        return x != 0  # (batch, seq_len), True for valid positions

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        logits = self.forward(x, return_attention=False)
        if isinstance(logits, tuple):
            logits = logits[0]
        return torch.sigmoid(logits)
