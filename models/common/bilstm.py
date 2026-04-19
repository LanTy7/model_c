"""Shared BiLSTM backbone and pooling layers."""
import torch
import torch.nn as nn
import torch.nn.functional as F


from typing import Union, Tuple
class BiLSTMAttentionBackbone(nn.Module):
    """
    BiLSTM backbone with optional self-attention layer.

    This combines BiLSTM encoding with multi-head self-attention,
    allowing the model to learn which sequence positions are most
    important for classification.

    Args:
        input_size: Input feature dimension
        hidden_size: LSTM hidden size
        num_layers: Number of LSTM layers
        dropout: Dropout probability
        bidirectional: Whether to use bidirectional LSTM
        use_attention: Whether to add self-attention layer
        num_attention_heads: Number of attention heads (if use_attention=True)
        attention_dropout: Dropout for attention weights
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.4,
        bidirectional: bool = True,
        use_attention: bool = True,
        num_attention_heads: int = 4,
        attention_dropout: float = 0.1
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.use_attention = use_attention

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )

        self.output_size = hidden_size * self.num_directions

        # Self-attention layer (enabled by default)
        self.attention = None
        if self.use_attention:
            from .attention import SelfAttention
            self.attention = SelfAttention(
                hidden_dim=self.output_size,
                num_heads=num_attention_heads,
                dropout=attention_dropout
            )

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input tensor (batch, seq_len, input_size)
            mask: Optional mask tensor (batch, seq_len), True for valid positions
            return_attention: Whether to return attention weights (only if use_attention=True)

        Returns:
            If return_attention=False: LSTM output (batch, seq_len, hidden_size * num_directions)
            If return_attention=True and use_attention=True: (output, attention_weights)
        """
        # BiLSTM encoding
        if mask is not None:
            lengths = mask.sum(dim=1).cpu()
            lengths_sorted, sort_idx = lengths.sort(descending=True)
            lengths_sorted = lengths_sorted.clamp(min=1)
            x_sorted = x[sort_idx]
            packed = nn.utils.rnn.pack_padded_sequence(
                x_sorted, lengths_sorted, batch_first=True
            )
            packed_output, _ = self.lstm(packed)
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_output, batch_first=True)
            # Unsort back to original order
            _, unsort_idx = sort_idx.sort()
            output = output[unsort_idx]
            # Pad to original max_length if needed
            if output.shape[1] < x.shape[1]:
                pad_size = x.shape[1] - output.shape[1]
                output = F.pad(output, (0, 0, 0, pad_size))
        else:
            output, _ = self.lstm(x)

        # Apply mask if provided (for variable length sequences)
        if mask is not None:
            # Expand mask for feature dimension
            mask_expanded = mask.unsqueeze(-1)  # (batch, seq_len, 1)
            output = output.masked_fill(~mask_expanded, 0.0)

        # Apply self-attention
        attention_weights = None
        if self.use_attention:
            output, attention_weights = self.attention(output, mask)
            # Re-mask padding positions after attention to prevent contamination
            if mask is not None:
                output = output.masked_fill(~mask.unsqueeze(-1), 0.0)

        if return_attention and attention_weights is not None:
            return output, attention_weights

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
                x_masked = x.masked_fill(~mask.bool(), -1e4)
                return torch.max(x_masked, dim=1)[0]
            elif self.pooling_type == 'mean':
                sum_pool = (x * mask).sum(dim=1)
                # clamp prevents division by zero; empty sequences will have mean = 0 due to sum being 0
                denom = mask.sum(dim=1).clamp(min=1.0)
                return sum_pool / denom
            else:  # both
                # Max pooling with mask
                x_masked = x.masked_fill(~mask.bool(), -1e4)
                max_pool = torch.max(x_masked, dim=1)[0]

                # Mean pooling with mask
                sum_pool = (x * mask).sum(dim=1)
                # clamp prevents division by zero; empty sequences will have mean = 0 due to sum being 0
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
