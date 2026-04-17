#!/usr/bin/env python3
"""Inference script for binary ARG classification."""
import os
import sys
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from Bio import SeqIO

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.binary.model import BinaryARGClassifier
from utils.sequence_utils import sequence_to_indices


def load_threshold_from_file(checkpoint_dir: str) -> tuple:
    """Load threshold from threshold.json if it exists.

    Args:
        checkpoint_dir: Directory containing the checkpoint

    Returns:
        Tuple of (threshold_value, source_description) or (None, None) if not found
    """
    threshold_path = os.path.join(checkpoint_dir, 'threshold.json')

    if not os.path.exists(threshold_path):
        return None, None

    try:
        with open(threshold_path, 'r') as f:
            data = json.load(f)

        threshold = data.get('optimal_threshold')
        tune_metric = data.get('tune_metric', 'unknown')
        description = data.get('description', '')

        if threshold is not None:
            source = f"threshold.json (metric: {tune_metric})"
            return float(threshold), source

        return None, None
    except (json.JSONDecodeError, IOError) as e:
        print(f"Warning: Failed to load threshold.json: {e}")
        return None, None


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Get model config from checkpoint
    if 'model_config' in checkpoint:
        # Training script saves model_config
        model_config = checkpoint['model_config']
    else:
        # Infer from state_dict shape
        state_dict = checkpoint['model_state_dict']
        # embedding.weight shape: (vocab_size, embedding_dim)
        vocab_size, embedding_dim = state_dict['embedding.weight'].shape

        # Check if enhanced architecture (CNN + Attention)
        use_cnn = any('cnn.convs' in k for k in state_dict.keys())
        use_attention = any('backbone.attention' in k for k in state_dict.keys())

        if use_cnn:
            # Enhanced model with CNN backbone
            # backbone.bilstm.backbone.lstm... or backbone.backbone.lstm...
            lstm_key = [k for k in state_dict.keys() if 'lstm.weight_ih_l0' in k and '_reverse' not in k][0]
            hidden_size = state_dict[lstm_key].shape[0] // 4
            num_layers = sum(1 for k in state_dict.keys() if 'lstm.weight_ih_l' in k and '_reverse' not in k)
        else:
            # Standard model
            hidden_size = state_dict['backbone.lstm.weight_ih_l0'].shape[0] // 4
            num_layers = sum(1 for k in state_dict.keys() if 'weight_ih_l' in k and '_reverse' not in k)

        model_config = {
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': 0.4,
            'max_length': 1000,
            'use_cnn': use_cnn,
            'use_attention': use_attention,
            'cnn_kernel_sizes': [3, 5, 7] if use_cnn else None,
            'cnn_out_channels': 64 if use_cnn else None,
            'num_attention_heads': 4 if use_attention else None,
        }

    model = BinaryARGClassifier(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, model_config


def predict_sequences(model, sequences, max_length, device, batch_size=256):
    """Predict on a list of sequences."""
    results = []

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i+batch_size]

        # Encode sequences
        encoded = np.array([
            sequence_to_indices(seq, max_length)
            for seq in batch_seqs
        ])

        inputs = torch.from_numpy(encoded).to(device)

        with torch.no_grad():
            probs = model.predict_proba(inputs).cpu().numpy().flatten()

        results.extend(probs)

    return np.array(results)


def main():
    parser = argparse.ArgumentParser(description='Predict ARG probability for sequences')
    parser.add_argument('--input', '-i', required=True, help='Input FASTA file')
    parser.add_argument('--model', '-m', required=True, help='Model checkpoint path')
    parser.add_argument('--output', '-o', required=True, help='Output CSV path')
    parser.add_argument('--device', '-d', default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--batch-size', '-b', type=int, default=256, help='Batch size')
    parser.add_argument('--threshold', '-t', type=float, default=None,
                        help='Classification threshold (overrides threshold.json if provided)')

    args = parser.parse_args()

    # Determine threshold to use
    # Priority: 1) User-provided --threshold, 2) threshold.json, 3) default 0.5
    default_threshold = 0.5

    if args.threshold is not None:
        # User provided explicit threshold
        threshold = args.threshold
        threshold_source = "command-line argument"
    else:
        # Try to load from threshold.json
        checkpoint_dir = os.path.dirname(os.path.abspath(args.model))
        loaded_threshold, json_source = load_threshold_from_file(checkpoint_dir)

        if loaded_threshold is not None:
            threshold = loaded_threshold
            threshold_source = json_source
        else:
            threshold = default_threshold
            threshold_source = f"default value ({default_threshold})"

    # Load model
    print(f"Loading model from {args.model}...")
    model, config = load_model(args.model, args.device)
    max_length = config.get('max_length', 1000)

    # Log threshold being used
    print(f"Using classification threshold: {threshold:.4f} (from {threshold_source})")

    # Load sequences
    print(f"Loading sequences from {args.input}...")
    sequences = []
    seq_ids = []
    for record in SeqIO.parse(args.input, "fasta"):
        sequences.append(str(record.seq).upper())
        seq_ids.append(record.id)

    print(f"Predicting on {len(sequences)} sequences...")
    probs = predict_sequences(model, sequences, max_length, args.device, args.batch_size)
    preds = (probs > threshold).astype(int)

    # Save results
    results_df = pd.DataFrame({
        'id': seq_ids,
        'prediction': preds,
        'probability': probs,
        'label': ['ARG' if p == 1 else 'non-ARG' for p in preds]
    })

    results_df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")
    print(f"Predicted ARGs: {sum(preds)} ({sum(preds)/len(preds)*100:.2f}%)")


if __name__ == "__main__":
    main()
