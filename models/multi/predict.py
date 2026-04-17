#!/usr/bin/env python3
"""Inference script for multi-class ARG classification."""
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from Bio import SeqIO

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.multi.model import MultiClassARGClassifier
from utils.sequence_utils import one_hot_encode


def load_model(checkpoint_path: str, metadata_path: str = None, device: str = 'cuda'):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)

    # Load metadata if available
    if metadata_path and os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        class_names = metadata['class_names']
        max_length = metadata['max_length']
    else:
        # Try to get from checkpoint
        if 'class_names' in checkpoint:
            class_names = checkpoint['class_names']
            max_length = checkpoint['max_length']
        else:
            raise ValueError("Metadata not found. Please provide metadata.json path.")

    # Get model config
    if 'model_config' in checkpoint:
        model_config = checkpoint['model_config']
    else:
        # Infer from state_dict shape
        state_dict = checkpoint['model_state_dict']
        # classifier.classifier.0.weight shape: (hidden, lstm_output)
        hidden_size = state_dict['classifier.classifier.0.weight'].shape[0]
        # backbone.lstm.weight_ih_l0 shape: (hidden*4, input_size)
        input_size = state_dict['backbone.lstm.weight_ih_l0'].shape[1]
        # Count number of layers
        num_layers = sum(1 for k in state_dict.keys() if 'weight_ih_l' in k and '_reverse' not in k)

        model_config = {
            'input_size': input_size,
            'hidden_size': hidden_size,
            'num_layers': num_layers,
            'dropout': 0.4,
            'num_classes': len(class_names)
        }

    model = MultiClassARGClassifier(**model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    return model, class_names, max_length


def predict_sequences(model, sequences, max_length, device, batch_size=256):
    """Predict on a list of sequences."""
    all_probs = []
    all_preds = []

    for i in range(0, len(sequences), batch_size):
        batch_seqs = sequences[i:i+batch_size]

        # Encode sequences
        encoded = np.array([
            one_hot_encode(seq, max_length)
            for seq in batch_seqs
        ])

        inputs = torch.from_numpy(encoded).to(device)

        with torch.no_grad():
            probs = model.predict_proba(inputs).cpu().numpy()
            preds = np.argmax(probs, axis=1)

        all_probs.append(probs)
        all_preds.extend(preds)

    return np.array(all_preds), np.vstack(all_probs)


def main():
    parser = argparse.ArgumentParser(description='Predict ARG category for sequences')
    parser.add_argument('--input', '-i', required=True, help='Input FASTA file')
    parser.add_argument('--model', '-m', required=True, help='Model checkpoint path')
    parser.add_argument('--metadata', '-md', help='Metadata JSON path (optional)')
    parser.add_argument('--output', '-o', required=True, help='Output CSV path')
    parser.add_argument('--device', '-d', default='cuda', help='Device (cuda/cpu)')
    parser.add_argument('--batch-size', '-b', type=int, default=256, help='Batch size')

    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model}...")
    if args.metadata is None:
        args.metadata = os.path.join(os.path.dirname(args.model), 'metadata.json')
    model, class_names, max_length = load_model(args.model, args.metadata, args.device)

    # Load sequences
    print(f"Loading sequences from {args.input}...")
    sequences = []
    seq_ids = []
    for record in SeqIO.parse(args.input, "fasta"):
        sequences.append(str(record.seq).upper())
        seq_ids.append(record.id)

    print(f"Predicting on {len(sequences)} sequences...")
    preds, probs = predict_sequences(model, sequences, max_length, args.device, args.batch_size)

    # Save results
    results_df = pd.DataFrame({
        'id': seq_ids,
        'predicted_class': [class_names[p] for p in preds],
        'confidence': [probs[i, p] for i, p in enumerate(preds)]
    })

    # Add probability columns for each class
    for i, class_name in enumerate(class_names):
        results_df[f'prob_{class_name}'] = probs[:, i]

    results_df.to_csv(args.output, index=False)
    print(f"Results saved to {args.output}")

    # Print distribution
    print("\nPredicted class distribution:")
    counts = np.bincount(preds, minlength=len(class_names))
    for i, class_name in enumerate(class_names):
        count = counts[i]
        print(f"  {class_name}: {count} ({count/len(preds)*100:.2f}%)")


if __name__ == "__main__":
    main()
