"""Multi-scale CNN for local sequence feature extraction.

This module extracts local protein sequence features at multiple scales
using convolutional layers with different kernel sizes, capturing motifs
of varying lengths (e.g., 3-mers, 5-mers, 7-mers).
"""
import torch
from typing import Union, Tuple
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleCNN(nn.Module):
    """
    Multi-scale CNN for extracting local sequence features.

    Uses multiple parallel convolutional branches with different kernel sizes
to capture local patterns at different scales. This is particularly useful
    for identifying protein motifs of varying lengths.

    Args:
        input_dim: Input feature dimension (e.g., embedding dim or one-hot dim)
        out_channels: Number of output channels per kernel size (default: 64)
        kernel_sizes: List of kernel sizes for convolutions (default: [3, 5, 7])
        dropout: Dropout probability (default: 0.1)

    Input Shape:
        - x: (batch, seq_len, input_dim)

    Output Shape:
        - output: (batch, seq_len, out_channels * len(kernel_sizes))
        - Features from all scales concatenated along channel dimension

    Example:
        >>> mscnn = MultiScaleCNN(input_dim=128, out_channels=64, kernel_sizes=[3, 5, 7])
        >>> x = torch.randn(2, 100, 128)  # (batch, seq_len, features)
        >>> output = mscnn(x)  # (2, 100, 192) - 64*3 channels
    """

    def __init__(
        self,
        input_dim: int,
        out_channels: int = 64,
        kernel_sizes: list = None,
        dropout: float = 0.1
    ):
        super().__init__()

        if kernel_sizes is None:
            kernel_sizes = [3, 5, 7]

        self.input_dim = input_dim
        self.out_channels = out_channels
        self.kernel_sizes = kernel_sizes
        self.num_scales = len(kernel_sizes)

        # Create parallel convolutional branches
        self.convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for kernel_size in kernel_sizes:
            # Each branch: Conv1d -> BatchNorm -> ReLU
            conv = nn.Conv1d(
                in_channels=input_dim,
                out_channels=out_channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,  # Same padding to maintain sequence length
                bias=False
            )
            self.convs.append(conv)

            # Batch normalization per branch
            bn = nn.BatchNorm1d(out_channels)
            self.batch_norms.append(bn)

        # Dropout for regularization
        self.dropout = nn.Dropout(dropout)

        # Output dimension
        self.output_dim = out_channels * self.num_scales

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Extract multi-scale features.

        Args:
            x: Input tensor (batch, seq_len, input_dim)
            mask: Optional mask (batch, seq_len), True for valid positions

        Returns:
            Multi-scale features (batch, seq_len, out_channels * num_scales)
        """
        # x: (batch, seq_len, input_dim)
        # Need to transpose to (batch, input_dim, seq_len) for Conv1d
        x = x.transpose(1, 2)  # (batch, input_dim, seq_len)

        # Extract features at each scale
        multi_scale_features = []

        for conv, bn in zip(self.convs, self.batch_norms):
            # Conv1d: (batch, input_dim, seq_len) -> (batch, out_channels, seq_len)
            conv_out = conv(x)

            # BatchNorm: (batch, out_channels, seq_len)
            conv_out = bn(conv_out)

            # ReLU activation
            conv_out = F.relu(conv_out)

            # Transpose back: (batch, out_channels, seq_len) -> (batch, seq_len, out_channels)
            conv_out = conv_out.transpose(1, 2)

            multi_scale_features.append(conv_out)

        # Concatenate features from all scales
        # Each: (batch, seq_len, out_channels)
        # Concatenated: (batch, seq_len, out_channels * num_scales)
        output = torch.cat(multi_scale_features, dim=-1)

        # Apply dropout
        output = self.dropout(output)

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(-1)  # (batch, seq_len, 1)
            output = output.masked_fill(~mask, 0.0)

        return output
