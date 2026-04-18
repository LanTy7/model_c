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


class CNN_BiLSTM_Backbone(nn.Module):
    """
    Combined CNN + BiLSTM backbone for sequence encoding.

    First extracts multi-scale local features using CNN,
    then processes with BiLSTM for long-range dependencies.

    Args:
        input_size: Input feature dimension
        cnn_out_channels: Output channels per CNN kernel
        cnn_kernel_sizes: List of CNN kernel sizes
        hidden_size: LSTM hidden size
        num_layers: Number of LSTM layers
        dropout: Dropout probability
        bidirectional: Whether to use bidirectional LSTM
        use_attention: Whether to add self-attention after LSTM
        num_attention_heads: Number of attention heads

    Input Shape:
        - x: (batch, seq_len, input_size)
        - mask: (batch, seq_len), True for valid positions

    Output Shape:
        - output: (batch, seq_len, hidden_size * num_directions)
    """

    def __init__(
        self,
        input_size: int,
        cnn_out_channels: int = 64,
        cnn_kernel_sizes: list = None,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.4,
        bidirectional: bool = True,
        use_attention: bool = False,
        num_attention_heads: int = 4,
        attention_dropout: float = 0.1
    ):
        super().__init__()

        # Multi-scale CNN for local feature extraction
        self.cnn = MultiScaleCNN(
            input_dim=input_size,
            out_channels=cnn_out_channels,
            kernel_sizes=cnn_kernel_sizes,
            dropout=dropout
        )

        # BiLSTM for long-range dependencies
        # Input to LSTM is CNN output dimension
        lstm_input_size = self.cnn.output_dim

        self.lstm = nn.LSTM(
            input_size=lstm_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0
        )

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1
        self.output_size = hidden_size * self.num_directions
        self.use_attention = use_attention

        # Optional self-attention after LSTM
        if use_attention:
            from .attention import SelfAttention
            self.attention = SelfAttention(
                hidden_dim=self.output_size,
                num_heads=num_attention_heads,
                dropout=attention_dropout
            )
        else:
            self.attention = None

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
        return_attention: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: Input tensor (batch, seq_len, input_size)
            mask: Optional mask (batch, seq_len), True for valid positions
            return_attention: Whether to return attention weights

        Returns:
            Encoded features, optionally with attention weights
        """
        # Step 1: Multi-scale CNN feature extraction
        cnn_features = self.cnn(x, mask)  # (batch, seq_len, cnn_output_dim)

        # Step 2: BiLSTM encoding
        lstm_out, _ = self.lstm(cnn_features)  # (batch, seq_len, hidden*directions)

        # Apply mask if provided
        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)
            lstm_out = lstm_out.masked_fill(~mask_expanded, 0.0)

        # Step 3: Optional self-attention
        attention_weights = None
        if self.use_attention and self.attention is not None:
            lstm_out, attention_weights = self.attention(lstm_out, mask)

        if return_attention:
            return lstm_out, attention_weights

        return lstm_out


class ResidualBlock(nn.Module):
    """
    Residual block for CNN with skip connection.

    Helps with gradient flow in deeper CNN architectures.

    Args:
        channels: Number of channels
        kernel_size: Convolution kernel size
        dropout: Dropout probability
    """

    def __init__(self, channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()

        padding = kernel_size // 2

        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn1 = nn.BatchNorm1d(channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding=padding)
        self.bn2 = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, seq_len)

        Returns:
            Output with residual connection
        """
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # Residual connection
        out += residual
        out = F.relu(out)

        return out
