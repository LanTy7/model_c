import torch
import numpy as np
from torch.utils.data import Dataset

from utils.sequence_utils import one_hot_encode


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
