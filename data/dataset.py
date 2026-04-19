from typing import List

import torch
import numpy as np
from torch.utils.data import Dataset

from utils.sequence_utils import one_hot_encode, sequence_to_indices


class BinarySequenceDataset(Dataset):
    """Simple dataset for binary classification using embedding indices."""

    def __init__(self, sequences: List[str], labels: List[int], max_length: int, training: bool = False):
        self.sequences = sequences
        self.labels = labels
        self.max_length = max_length
        self.training = training

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        # sequence_to_indices handles center crop internally
        encoded = sequence_to_indices(seq, self.max_length)
        # Create mask: True for valid positions, False for padding
        # sequence_to_indices pads with 0 (PAD index) if seq < max_length
        effective_len = min(len(seq), self.max_length)
        mask = torch.zeros(self.max_length, dtype=torch.bool)
        mask[:effective_len] = True
        return torch.from_numpy(encoded), mask, torch.tensor(label, dtype=torch.float32)


class MultiClassARGDataset(Dataset):
    """Dataset for multi-class ARG classification using one-hot encoding."""

    def __init__(self, sequences, labels, max_length):
        self.sequences = sequences
        self.labels = labels
        self.max_length = max_length

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        label = self.labels[idx]
        encoded = one_hot_encode(seq, self.max_length)
        return torch.from_numpy(encoded), torch.tensor(label, dtype=torch.long)
