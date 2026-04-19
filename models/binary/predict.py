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
from utils.common import safe_torch_load
from utils.sequence_utils import sequence_to_indices


def load_threshold_from_file(checkpoint_dir: str) -> tuple:
    """Load threshold from threshold JSON if it exists.

    Prefers metric-specific files (e.g., threshold_f1.json) over generic threshold.json.

    Args:
        checkpoint_dir: Directory containing the checkpoint

    Returns:
        Tuple of (threshold_value, source_description) or (None, None) if not found
    """
    # Prefer metric-specific threshold files to avoid ambiguity
    metric_specific = []
    generic_path = os.path.join(checkpoint_dir, 'threshold.json')
    if os.path.isdir(checkpoint_dir):
        for fname in os.listdir(checkpoint_dir):
            if fname.startswith('threshold_') and fname.endswith('.json'):
                metric_specific.append(os.path.join(checkpoint_dir, fname))

    # Try metric-specific files first, then generic fallback
    candidates = metric_specific + ([generic_path] if os.path.exists(generic_path) else [])

    for threshold_path in candidates:
        try:
            with open(threshold_path, 'r') as f:
                data = json.load(f)

            threshold = data.get('optimal_threshold')
            tune_metric = data.get('tune_metric', 'unknown')

            if threshold is not None:
                source = f"{os.path.basename(threshold_path)} (metric: {tune_metric})"
                return float(threshold), source
        except (json.JSONDecodeError, IOError) as e:
            print(f"Warning: Failed to load {os.path.basename(threshold_path)}: {e}")
            continue

    return None, None


def load_model(checkpoint_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """Load trained model from checkpoint."""
    checkpoint = safe_torch_load(checkpoint_path, map_location=device)

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

        # Infer hidden size and num_layers from LSTM weights safely
        hidden_size = 128
        num_layers = 1
        for key in state_dict.keys():
            if 'lstm.weight_ih_l0' in key and '_reverse' not in key:
                hidden_size = state_dict[key].shape[0] // 4
                break
        num_layers = sum(1 for k in state_dict.keys()
                         if 'lstm.weight_ih_l' in k and '_reverse' not in k)

        model_config = {
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': 0.4,
            'max_length': 700,
            'use_cnn': use_cnn,
            'use_attention': use_attention,
            'cnn_kernel_sizes': [3, 5, 7] if use_cnn else None,
            'cnn_out_channels': 64 if use_cnn else None,
            'num_attention_heads': 4 if use_attention else None,
        }

    model = BinaryARGClassifier(**model_config)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
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


def predict_sequences_ensemble(models, sequences, max_length, device, batch_size=256):
    """Predict on a list of sequences using an ensemble of models.

    Returns average probabilities across all models.
    NOTE: Currently uses simple arithmetic mean. For better performance,
    consider weighting by validation metrics if available.
    """
    all_fold_probs = []
    for model in models:
        probs = predict_sequences(model, sequences, max_length, device, batch_size)
        all_fold_probs.append(probs)
    if not all_fold_probs:
        raise ValueError("No models provided for ensemble prediction")
    return np.mean(all_fold_probs, axis=0)


def main():
    parser = argparse.ArgumentParser(description='Predict ARG probability for sequences')
    parser.add_argument('--input', '-i', required=True, help='Input FASTA file')
    parser.add_argument('--model', '-m', required=True, help='Model checkpoint path (or directory for ensemble)')
    parser.add_argument('--models', nargs='+', help='Multiple model checkpoint paths for ensemble inference')
    parser.add_argument('--output', '-o', required=True, help='Output CSV path')
    parser.add_argument('--device', '-d', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device (cuda/cpu)')
    parser.add_argument('--batch-size', '-b', type=int, default=256, help='Batch size')
    parser.add_argument('--threshold', '-t', type=float, default=None,
                        help='Classification threshold (overrides threshold.json if provided)')

    args = parser.parse_args()

    # Determine model paths
    model_paths = []
    if args.models:
        # Explicit list of model paths for ensemble
        model_paths = args.models
    else:
        # Single model path
        model_paths = [args.model]

    ensemble_mode = len(model_paths) > 1

    # Determine threshold to use
    # Priority: 1) User-provided --threshold, 2) threshold.json, 3) default 0.5
    default_threshold = 0.5

    if args.threshold is not None:
        # User provided explicit threshold
        threshold = args.threshold
        threshold_source = "command-line argument"
    else:
        # Try to load from threshold.json (use first model's directory)
        checkpoint_dir = os.path.dirname(os.path.abspath(model_paths[0]))
        loaded_threshold, json_source = load_threshold_from_file(checkpoint_dir)

        if loaded_threshold is not None:
            threshold = loaded_threshold
            threshold_source = json_source
        else:
            threshold = default_threshold
            threshold_source = f"default value ({default_threshold})"

    # Load model(s)
    configs = []
    models = []
    for mp in model_paths:
        print(f"Loading model from {mp}...")
        model, config = load_model(mp, args.device)
        models.append(model)
        configs.append(config)

    # Use max_length from first model's config
    max_length = configs[0].get('max_length', 1000)

    # Log threshold being used
    print(f"Using classification threshold: {threshold:.4f} (from {threshold_source})")
    if ensemble_mode:
        print(f"Ensemble mode: {len(models)} models")

    # Load sequences
    print(f"Loading sequences from {args.input}...")
    sequences = []
    seq_ids = []
    skipped_empty = 0
    for record in SeqIO.parse(args.input, "fasta"):
        seq = str(record.seq).upper()
        if len(seq) == 0:
            skipped_empty += 1
            continue
        sequences.append(seq)
        seq_ids.append(record.id)

    if skipped_empty > 0:
        print(f"Warning: Skipped {skipped_empty} empty sequence(s)")

    print(f"Predicting on {len(sequences)} sequences...")
    if ensemble_mode:
        probs = predict_sequences_ensemble(models, sequences, max_length, args.device, args.batch_size)
    else:
        probs = predict_sequences(models[0], sequences, max_length, args.device, args.batch_size)
    preds = (probs >= threshold).astype(int)

    # Save results
    results_df = pd.DataFrame({
        'id': seq_ids,
        'prediction': preds,
        'probability': probs,
        'label': ['ARG' if p == 1 else 'non-ARG' for p in preds]
    })

    results_df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")
    if len(preds) > 0:
        print(f"Predicted ARGs: {sum(preds)} ({sum(preds)/len(preds)*100:.2f}%)")
    else:
        print("Warning: No sequences were predicted (empty input?)")


if __name__ == "__main__":
    main()
