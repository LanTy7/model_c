"""Sequence encoding and preprocessing utilities."""
import numpy as np
import torch
from typing import List

# Amino acid vocabulary
AMINO_ACIDS = 'ACDEFGHIKLMNPQRSTVWY'
AA_DICT = {aa: i for i, aa in enumerate(AMINO_ACIDS)}
AA_DICT.update({
    'B': [AA_DICT['D'], AA_DICT['N']],  # Asp or Asn
    'Z': [AA_DICT['E'], AA_DICT['Q']],  # Glu or Gln
    'J': [AA_DICT['I'], AA_DICT['L']],  # Ile or Leu
    'X': 'ANY',
    'PAD': 20
})

# For embedding-based encoding (binary model)
# PAD=0, 20 AAs=1-20, X=21, B=22, Z=23, J=24  => vocab_size=25
AA_DICT_EMBED = {aa: i + 1 for i, aa in enumerate(AMINO_ACIDS)}
AA_DICT_EMBED.update({'X': 21, 'B': 22, 'Z': 23, 'J': 24, 'PAD': 0})


def _center_crop(sequence: str, max_length: int) -> str:
    """Center-crop sequence to max_length, dropping excess from both ends."""
    if len(sequence) <= max_length:
        return sequence
    start = (len(sequence) - max_length) // 2
    return sequence[start:start + max_length]


def one_hot_encode(sequence: str, max_length: int) -> np.ndarray:
    """
    Convert amino acid sequence to one-hot encoding.

    Args:
        sequence: Amino acid sequence
        max_length: Maximum sequence length (truncate/pad to this)

    Returns:
        One-hot encoded array of shape (max_length, 21)
        Last dimension: 20 amino acids + PAD channel
    """
    if len(sequence) == 0:
        raise ValueError("Cannot encode an empty sequence.")

    sequence = _center_crop(sequence, max_length)

    encoding = np.zeros((max_length, 21), dtype=np.float32)

    for i in range(len(sequence)):
        aa = sequence[i].upper()
        if aa in AA_DICT:
            idx = AA_DICT[aa]
            if isinstance(idx, list):  # Ambiguous amino acids (B, Z, J)
                for j in idx:
                    encoding[i, j] = 0.5
            elif idx == 'ANY':  # Unknown amino acid (X)
                encoding[i, :20] = 0.05
            else:
                encoding[i, idx] = 1.0
        else:
            encoding[i, :20] = 0.05  # Unknown character

    # Mark padding positions (PAD channel = 1)
    if len(sequence) < max_length:
        encoding[len(sequence):, 20] = 1.0

    return encoding


def sequence_to_indices(sequence: str, max_length: int) -> np.ndarray:
    """
    Convert amino acid sequence to embedding indices.

    Args:
        sequence: Amino acid sequence
        max_length: Maximum sequence length

    Returns:
        Array of indices of shape (max_length,)
    """
    if len(sequence) == 0:
        raise ValueError("Cannot encode an empty sequence.")

    sequence = _center_crop(sequence, max_length)

    indices = np.zeros(max_length, dtype=np.int64)
    seq_len = min(len(sequence), max_length)
    for i in range(seq_len):
        indices[i] = AA_DICT_EMBED.get(sequence[i].upper(), 21)
    return indices


def compute_class_weights(class_counts: np.ndarray, method: str = 'sqrt') -> torch.Tensor:
    """
    Compute class weights for imbalanced classification.

    Args:
        class_counts: Array of sample counts per class
        method: 'sqrt' for square root weighting, 'inverse' for inverse frequency

    Returns:
        Tensor of class weights
    """
    if method == 'sqrt':
        weights = np.sqrt(np.median(class_counts) / (class_counts + 1))
    elif method == 'inverse':
        weights = 1.0 / (class_counts + 1e-6)
        weights = weights / weights.sum() * len(weights)
    else:
        weights = np.ones(len(class_counts))

    return torch.tensor(weights, dtype=torch.float32)


def get_max_length(sequences: List[str], percentile: float = 95) -> int:
    """
    Compute sequence length at given percentile.

    Args:
        sequences: List of sequences
        percentile: Percentile to use (default 95)

    Returns:
        Length at percentile
    """
    lengths = [len(s) for s in sequences]
    return int(np.percentile(lengths, percentile))


class BalancedClassSampler(torch.utils.data.Sampler):
    """
    Sampler that samples uniformly across classes to handle class imbalance.
    Each batch will have approximately equal representation from each class.
    """
    def __init__(self, labels: np.ndarray, samples_per_class: int = None):
        """
        Args:
            labels: Array of class labels
            samples_per_class: Number of samples to draw per class per epoch.
                             If None, uses max class count.
        """
        self.labels = labels
        self.num_classes = len(np.unique(labels))

        # Group indices by class
        self.class_indices = [np.where(labels == i)[0] for i in range(self.num_classes)]
        self.class_counts = [len(idx) for idx in self.class_indices]

        # Determine samples per class
        if samples_per_class is None:
            self.samples_per_class = max(self.class_counts)
        else:
            self.samples_per_class = samples_per_class

        self.total_samples = self.samples_per_class * self.num_classes

    def __iter__(self):
        # For each class, sample with replacement if needed
        indices = []
        for class_idx in self.class_indices:
            if len(class_idx) >= self.samples_per_class:
                # Sample without replacement
                sampled = np.random.choice(class_idx, self.samples_per_class, replace=False)
            else:
                # Sample with replacement to reach target
                sampled = np.random.choice(class_idx, self.samples_per_class, replace=True)
            indices.extend(sampled)

        # Shuffle the combined indices
        np.random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return self.total_samples
