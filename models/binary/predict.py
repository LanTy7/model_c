#!/usr/bin/env python3
"""Inference script for binary ARG classification."""
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from Bio import SeqIO

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.binary.model import BinaryARGClassifier
from utils.sequence_utils import sequence_to_indices


def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get model config from checkpoint
    if 'model_config' in checkpoint:
        # Training script saves model_config
        model_config = checkpoint['model_config']
    else:
        # Infer from state_dict shape
        state_dict = checkpoint['model_state_dict']
        # embedding.weight shape: (vocab_size, embedding_dim)
        vocab_size, embedding_dim = state_dict['embedding.weight'].shape
        # backbone.lstm.weight_ih_l0 shape: (hidden*4, embedding_dim)
        hidden_size = state_dict['backbone.lstm.weight_ih_l0'].shape[0] // 4
        # Count number of layers by checking for weight_ih_l{layer}
        num_layers = sum(1 for k in state_dict.keys() if 'weight_ih_l' in k and '_reverse' not in k)

        model_config = {
            'vocab_size': vocab_size,
            'embedding_dim': embedding_dim,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': 0.3,
            'max_length': 1000
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
    parser.add_argument('--threshold', '-t', type=float, default=0.5, help='Classification threshold')

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model}...")
    model, config = load_model(args.model, args.device)
    max_length = config.get('max_length', 1000)

    # Load sequences
    print(f"Loading sequences from {args.input}...")
    sequences = []
    seq_ids = []
    for record in SeqIO.parse(args.input, "fasta"):
        sequences.append(str(record.seq).upper())
        seq_ids.append(record.id)

    print(f"Predicting on {len(sequences)} sequences...")
    probs = predict_sequences(model, sequences, max_length, args.device, args.batch_size)
    preds = (probs > args.threshold).astype(int)

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
