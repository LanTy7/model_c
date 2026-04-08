"""Shared BiLSTM backbone and pooling layers."""
import torch
import torch.nn as nn


class BiLSTMBackbone(nn.Module):
    """
    Shared BiLSTM backbone for sequence encoding.
    Can be used with either embedding or one-hot input.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.4,
        bidirectional: bool = True
    ):
        """
        Args:
            input_size: Input feature dimension (embedding_dim or one-hot dim)
            hidden_size: LSTM hidden size
            num_layers: Number of LSTM layers
            dropout: Dropout probability
            bidirectional: Whether to use bidirectional LSTM
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )

        self.output_size = hidden_size * self.num_directions

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, input_size)
            mask: Optional mask tensor (batch, seq_len), True for valid positions

        Returns:
            LSTM output (batch, seq_len, hidden_size * num_directions)
        """
        output, _ = self.lstm(x)

        # Apply mask if provided (for variable length sequences)
        if mask is not None:
            # Expand mask for feature dimension
            mask = mask.unsqueeze(-1)  # (batch, seq_len, 1)
            output = output.masked_fill(~mask, 0.0)

        return output


class GlobalPooling(nn.Module):
    """
    Global max and average pooling with optional masking.
    """

    def __init__(self, pooling_type: str = 'both'):
        """
        Args:
            pooling_type: 'max', 'mean', or 'both'
        """
        super().__init__()
        self.pooling_type = pooling_type

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, seq_len, features)
            mask: Optional mask (batch, seq_len), True for valid positions

        Returns:
            Pooled features
        """
        if mask is None:
            # Simple pooling without mask
            if self.pooling_type == 'max':
                return torch.max(x, dim=1)[0]
            elif self.pooling_type == 'mean':
                return torch.mean(x, dim=1)
            else:  # both
                max_pool = torch.max(x, dim=1)[0]
                mean_pool = torch.mean(x, dim=1)
                return torch.cat([max_pool, mean_pool], dim=1)
        else:
            # Masked pooling
            mask = mask.unsqueeze(-1).float()  # (batch, seq_len, 1)

            if self.pooling_type == 'max':
                # Set masked positions to very negative value (FP16 compatible)
                x_masked = x.masked_fill(mask == 0, -1e4)
                return torch.max(x_masked, dim=1)[0]
            elif self.pooling_type == 'mean':
                sum_pool = (x * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp(min=1.0)
                return sum_pool / denom
            else:  # both
                # Max pooling with mask
                x_masked = x.masked_fill(mask == 0, -1e4)
                max_pool = torch.max(x_masked, dim=1)[0]

                # Mean pooling with mask
                sum_pool = (x * mask).sum(dim=1)
                denom = mask.sum(dim=1).clamp(min=1.0)
                mean_pool = sum_pool / denom

                return torch.cat([max_pool, mean_pool], dim=1)


class ClassifierHead(nn.Module):
    """
    Simple MLP classifier head.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        dropout: float = 0.4
    ):
        """
        Args:
            input_size: Input feature dimension
            hidden_size: Hidden layer dimension
            num_classes: Number of output classes
            dropout: Dropout probability
        """
        super().__init__()

        if hidden_size > 0:
            self.classifier = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size, num_classes)
            )
        else:
            # Direct classification without hidden layer
            self.classifier = nn.Linear(input_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)
