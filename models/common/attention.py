"""Self-Attention mechanism for BiLSTM outputs."""
import torch
from typing import Optional, Tuple

import torch.nn as nn
import torch.nn.functional as F


class SelfAttention(nn.Module):
    """
    Multi-head self-attention module for BiLSTM outputs.

    This module applies scaled dot-product attention over sequence positions,
    allowing the model to learn which positions are most important for classification.
    Designed to be placed between BiLSTM and pooling layers.

    Args:
        hidden_dim: Dimension of input features (from BiLSTM output)
        num_heads: Number of attention heads (must divide hidden_dim)
        dropout: Dropout probability for attention weights (default: 0.0)

    Input Shape:
        - x: (batch, seq_len, hidden_dim)
        - mask: (batch, seq_len), True for valid positions, False for padding

    Output Shape:
        - output: (batch, seq_len, hidden_dim)
        - attention_weights: (batch, num_heads, seq_len, seq_len)
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4, dropout: float = 0.0):
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** 0.5

        # Linear projections for Q, K, V
        self.q_linear = nn.Linear(hidden_dim, hidden_dim)
        self.k_linear = nn.Linear(hidden_dim, hidden_dim)
        self.v_linear = nn.Linear(hidden_dim, hidden_dim)

        # Output projection
        self.out_linear = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Apply multi-head self-attention.

        Args:
            x: Input tensor from BiLSTM (batch, seq_len, hidden_dim)
            mask: Optional padding mask (batch, seq_len), True for valid positions

        Returns:
            Tuple of:
                - attended_output: (batch, seq_len, hidden_dim)
                - attention_weights: (batch, num_heads, seq_len, seq_len) or None
        """
        batch_size, seq_len, _ = x.size()

        # Linear projections and reshape for multi-head attention
        # (batch, seq_len, hidden_dim) -> (batch, seq_len, num_heads, head_dim)
        # -> (batch, num_heads, seq_len, head_dim)
        Q = self.q_linear(x).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        K = self.k_linear(x).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        V = self.v_linear(x).view(
            batch_size, seq_len, self.num_heads, self.head_dim
        ).transpose(1, 2)

        # Scaled dot-product attention
        # (batch, num_heads, seq_len, head_dim) @ (batch, num_heads, head_dim, seq_len)
        # -> (batch, num_heads, seq_len, seq_len)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale

        # Apply mask if provided
        if mask is not None:
            # mask: (batch, seq_len) -> (batch, 1, 1, seq_len)
            # This masks out attention TO padding positions
            mask = mask.unsqueeze(1).unsqueeze(2)
            # Use -1e4 instead of -inf for FP16 compatibility
            attn_scores = attn_scores.masked_fill(~mask, -1e4)

        # Softmax over the last dimension (attention weights sum to 1)
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        # (batch, num_heads, seq_len, seq_len) @ (batch, num_heads, seq_len, head_dim)
        # -> (batch, num_heads, seq_len, head_dim)
        attn_output = torch.matmul(attn_weights, V)

        # Concatenate heads: (batch, num_heads, seq_len, head_dim)
        # -> (batch, seq_len, num_heads, head_dim) -> (batch, seq_len, hidden_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.hidden_dim
        )

        # Final linear projection
        output = self.out_linear(attn_output)

        return output, attn_weights


class AttentionPooling(nn.Module):
    """
    Attention-based pooling that uses learned attention weights
    to compute a weighted sum of sequence features.

    This can be used as an alternative to GlobalPooling when
    SelfAttention is added to the model.

    Args:
        hidden_dim: Dimension of input features
        num_heads: Number of attention heads for self-attention

    Input Shape:
        - x: (batch, seq_len, hidden_dim)
        - mask: (batch, seq_len), True for valid positions

    Output Shape:
        - pooled: (batch, hidden_dim)
    """

    def __init__(self, hidden_dim: int, num_heads: int = 4):
        super().__init__()
        self.self_attn = SelfAttention(hidden_dim, num_heads)

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Apply self-attention and return pooled representation.

        Args:
            x: Input tensor (batch, seq_len, hidden_dim)
            mask: Optional padding mask (batch, seq_len)

        Returns:
            Tuple of:
                - pooled: (batch, hidden_dim) - mean of attended features
                - attention_weights: (batch, num_heads, seq_len, seq_len)
        """
        attended, attn_weights = self.self_attn(x, mask)

        # Mean pooling over sequence length
        if mask is not None:
            mask = mask.unsqueeze(-1).float()
            pooled = (attended * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
        else:
            pooled = attended.mean(dim=1)

        return pooled, attn_weights
