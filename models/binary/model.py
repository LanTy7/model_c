"""Binary classification model for ARG identification."""
import torch
import torch.nn as nn
from models.common.bilstm import BiLSTMBackbone, GlobalPooling, ClassifierHead


class BinaryARGClassifier(nn.Module):
    """
    BiLSTM-based binary classifier for ARG vs non-ARG.
    Uses embedding layer for input encoding.
    """

    def __init__(
        self,
        vocab_size: int = 22,
        embedding_dim: int = 48,
        hidden_size: int = 48,
        num_layers: int = 1,
        dropout: float = 0.5,
        max_length: int = 1000
    ):
        """
        Args:
            vocab_size: Vocabulary size (20 AA + X + PAD)
            embedding_dim: Embedding dimension
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            max_length: Maximum sequence length
        """
        super().__init__()

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=0  # PAD token index
        )

        self.backbone = BiLSTMBackbone(
            input_size=embedding_dim,
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
            num_classes=1,  # Binary classification
            dropout=dropout
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input indices (batch, seq_len)

        Returns:
            Logits (batch, 1)
        """
        # Embedding
        emb = self.embedding(x)  # (batch, seq_len, embedding_dim)

        # BiLSTM
        lstm_out = self.backbone(emb)  # (batch, seq_len, hidden*2)

        # Global pooling
        features = self.pooling(lstm_out)  # (batch, hidden*4)

        # Classifier
        logits = self.classifier(features)  # (batch, 1)

        return logits

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Get probability predictions."""
        logits = self.forward(x)
        return torch.sigmoid(logits)

    def get_embeddings(self, x: torch.Tensor) -> torch.Tensor:
        """Get sequence embeddings before classification."""
        emb = self.embedding(x)
        lstm_out = self.backbone(emb)
        return self.pooling(lstm_out)
