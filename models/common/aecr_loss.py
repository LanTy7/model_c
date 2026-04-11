"""AECR Loss: Attention Entropy and Continuity Regularization.

Reference: Based on MCT-ARG's dual-constraint regularization strategy.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AECRLoss(nn.Module):
    """
    Attention Entropy and Continuity Regularization Loss.

    This loss applies two constraints to attention weights:
    1. Entropy Minimization: Encourages attention weights to be more
       concentrated (sharper focus on important positions)
    2. Local Continuity Constraint: Uses Gaussian kernel to encourage
       nearby sequence positions to have similar attention weights,
       promoting spatial smoothness in attention patterns.

    Reference: MCT-ARG (Multi-Channel Transformer for ARG prediction)

    Args:
        sigma: Standard deviation for Gaussian kernel (default: 3)
        lambda_ent: Weight for entropy loss component (default: 1.0)
        lambda_loc: Weight for local continuity loss component (default: 0.5)

    Example:
        >>> criterion = AECRLoss(sigma=3, lambda_ent=1.0, lambda_loc=0.5)
        >>> attention_weights = torch.randn(2, 4, 100, 100)  # (batch, heads, seq, seq)
        >>> loss = criterion(attention_weights)
    """

    def __init__(
        self,
        sigma: float = 3.0,
        lambda_ent: float = 1.0,
        lambda_loc: float = 0.5
    ):
        super().__init__()
        self.sigma = sigma
        self.lambda_ent = lambda_ent
        self.lambda_loc = lambda_loc

        # Cache for Gaussian kernels (keyed by sequence length)
        self._gaussian_kernels = {}

    def _create_gaussian_kernel(
        self,
        seq_len: int,
        device: torch.device
    ) -> torch.Tensor:
        """
        Create a Gaussian kernel for local continuity constraint.

        The kernel encourages attention weights to be similar for nearby
        sequence positions, promoting spatial smoothness.

        Args:
            seq_len: Sequence length
            device: Device to create tensor on

        Returns:
            Normalized Gaussian kernel of shape (seq_len, seq_len)
        """
        if seq_len in self._gaussian_kernels:
            # Move cached kernel to correct device if needed
            kernel = self._gaussian_kernels[seq_len]
            if kernel.device != device:
                kernel = kernel.to(device)
                self._gaussian_kernels[seq_len] = kernel
            return kernel

        # Create coordinate grids
        i = torch.arange(seq_len, device=device, dtype=torch.float32)
        j = torch.arange(seq_len, device=device, dtype=torch.float32)
        ii, jj = torch.meshgrid(i, j, indexing='ij')

        # Gaussian kernel: exp(-(i-j)^2 / (2*sigma^2))
        diff = ii - jj
        kernel = torch.exp(-(diff ** 2) / (2 * self.sigma ** 2))

        # Normalize to sum to 1
        kernel = kernel / kernel.sum()

        # Cache for future use
        self._gaussian_kernels[seq_len] = kernel

        return kernel

    def forward(self, attention_weights: torch.Tensor) -> torch.Tensor:
        """
        Compute AECR loss from attention weights.

        Args:
            attention_weights: Attention weight tensor of shape
                (batch, num_heads, seq_len, seq_len). Should be
                post-softmax attention probabilities (sum to 1).

        Returns:
            Total AECR loss (scalar)
        """
        batch_size, num_heads, seq_len, _ = attention_weights.shape

        # Add small epsilon for numerical stability in log
        p = attention_weights + 1e-10

        # === Component 1: Entropy Minimization ===
        # We want attention weights to be concentrated (low entropy)
        # Loss = -sum(p * log(p))
        entropy_per_position = -p * torch.log(p)
        entropy_loss = entropy_per_position.mean()

        # === Component 2: Local Continuity Constraint ===
        # We want attention to be smooth (nearby positions similar)
        # Use Gaussian kernel to measure similarity to ideal smooth distribution

        # Get or create Gaussian kernel for this sequence length
        gaussian_kernel = self._create_gaussian_kernel(seq_len, attention_weights.device)

        # Compute similarity between attention weights and Gaussian kernel
        # Using einsum: for each batch and head, compute dot product
        # attention_weights: (batch, heads, seq_len, seq_len)
        # gaussian_kernel: (seq_len, seq_len)
        # Result: (batch, heads)
        local_similarity = torch.einsum(
            'bhij,ij->bh',
            attention_weights,
            gaussian_kernel
        )

        # Loss is 1 - similarity (we want to maximize similarity)
        local_loss = 1.0 - local_similarity.mean()

        # === Combine losses ===
        total_loss = (
            self.lambda_ent * entropy_loss +
            self.lambda_loc * local_loss
        )

        # Store individual components for logging (optional)
        self._last_entropy_loss = entropy_loss.item()
        self._last_local_loss = local_loss.item()

        return total_loss

    def get_last_component_losses(self) -> dict:
        """
        Get the individual loss components from the last forward pass.

        Returns:
            Dictionary with 'entropy' and 'local' loss values
        """
        return {
            'entropy': getattr(self, '_last_entropy_loss', 0.0),
            'local': getattr(self, '_last_local_loss', 0.0)
        }


class CombinedLoss(nn.Module):
    """
    Combined task loss + AECR regularization loss.

    This wrapper combines a task-specific loss (e.g., CrossEntropyLoss
    for classification) with the AECR attention regularization loss.

    Args:
        task_criterion: Primary task loss function (e.g., CrossEntropyLoss)
        aecr_criterion: AECRLoss instance for attention regularization
        lambda_aecr: Weight for AECR loss relative to task loss

    Example:
        >>> task_loss = nn.CrossEntropyLoss()
        >>> aecr_loss = AECRLoss()
        >>> combined = CombinedLoss(task_loss, aecr_loss, lambda_aecr=0.1)
        >>>
        >>> logits = model(inputs)  # Model returns logits and attention weights
        >>> attention = model.get_attention_weights()
        >>> loss = combined(logits, labels, attention)
    """

    def __init__(
        self,
        task_criterion: nn.Module,
        aecr_criterion: 'AECRLoss',
        lambda_aecr: float = 0.1
    ):
        super().__init__()
        self.task_criterion = task_criterion
        self.aecr_criterion = aecr_criterion
        self.lambda_aecr = lambda_aecr

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        attention_weights: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute combined loss.

        Args:
            predictions: Model predictions (e.g., logits)
            targets: Ground truth targets
            attention_weights: Optional attention weights for AECR loss.
                If None, only task loss is computed.

        Returns:
            Total combined loss
        """
        # Task loss
        task_loss = self.task_criterion(predictions, targets)

        # AECR loss (if attention weights provided)
        if attention_weights is not None and self.lambda_aecr > 0:
            aecr_loss = self.aecr_criterion(attention_weights)
            total_loss = task_loss + self.lambda_aecr * aecr_loss
        else:
            total_loss = task_loss

        return total_loss
