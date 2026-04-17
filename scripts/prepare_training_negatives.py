"""
Prepare negative samples for training by sampling from refined negatives.
Creates negative_samples_for_training.fasta/csv with specified count.
"""

import os
import random
import logging
import json
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_fasta(filepath: str) -> List[Dict]:
    """Parse FASTA file"""
    sequences = []
    current_header = None
    current_seq = []

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_header and current_seq:
                    seq_id = current_header.split()[0]
                    sequences.append({
                        'id': seq_id,
                        'header': current_header,
                        'sequence': ''.join(current_seq)
                    })
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line)

        if current_header and current_seq:
            seq_id = current_header.split()[0]
            sequences.append({
                'id': seq_id,
                'header': current_header,
                'sequence': ''.join(current_seq)
            })

    return sequences


def save_fasta(sequences: List[Dict], filepath: str):
    """Save sequences to FASTA"""
    with open(filepath, 'w') as f:
        for seq in sequences:
            f.write(f">{seq['header']}\n")
            for i in range(0, len(seq['sequence']), 60):
                f.write(seq['sequence'][i:i+60] + '\n')


def save_csv(sequences: List[Dict], filepath: str):
    """Save to CSV format"""
    import csv

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sequence', 'category', 'is_arg', 'length', 'header'])

        for seq in sequences:
            writer.writerow([
                seq['sequence'],
                'non_arg',
                0,
                len(seq['sequence']),
                seq['id']
            ])


def main(
    refined_negative_fasta: str = './data/negative_samples_refined.fasta',
    output_dir: str = './data',
    target_count: int = 17338,
    seed: int = 42
):
    """
    Sample negative sequences for training from refined negatives.

    Args:
        refined_negative_fasta: Path to refined negative samples
        output_dir: Output directory
        target_count: Number of negatives to sample for training
        seed: Random seed
    """
    random.seed(seed)

    logger.info("=" * 60)
    logger.info("Preparing Negative Samples for Training")
    logger.info("=" * 60)

    # Load all refined negatives
    logger.info(f"\nLoading refined negatives from {refined_negative_fasta}")
    all_negatives = parse_fasta(refined_negative_fasta)
    logger.info(f"Total refined negatives: {len(all_negatives)}")

    # Check if we have enough
    if len(all_negatives) < target_count:
        logger.warning(f"Only {len(all_negatives)} negatives available, requested {target_count}")
        selected = all_negatives
    else:
        # Randomly sample
        selected = random.sample(all_negatives, target_count)
        logger.info(f"Randomly sampled {target_count} negatives for training")

    random.shuffle(selected)

    # Add metadata
    for seq in selected:
        seq['length'] = len(seq['sequence'])
        seq['is_arg'] = 0
        seq['category'] = 'non_arg'

    # Save files
    fasta_path = os.path.join(output_dir, 'negative_samples_for_training.fasta')
    csv_path = os.path.join(output_dir, 'negative_samples_for_training.csv')

    save_fasta(selected, fasta_path)
    save_csv(selected, csv_path)

    logger.info(f"\nSaved training negatives:")
    logger.info(f"  FASTA: {fasta_path}")
    logger.info(f"  CSV: {csv_path}")

    # Statistics
    lengths = [s['length'] for s in selected]
    stats = {
        'source': refined_negative_fasta,
        'target_count': target_count,
        'final_count': len(selected),
        'length_stats': {
            'mean': sum(lengths) / len(lengths),
            'min': min(lengths),
            'max': max(lengths)
        }
    }

    stats_path = os.path.join(output_dir, 'negative_samples_for_training_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"\nStatistics:")
    logger.info(f"  Mean length: {stats['length_stats']['mean']:.1f}")
    logger.info(f"  Min length: {stats['length_stats']['min']}")
    logger.info(f"  Max length: {stats['length_stats']['max']}")
    logger.info(f"\nSaved statistics to {stats_path}")

    logger.info("\n" + "=" * 60)
    logger.info("Negative samples for training prepared!")
    logger.info("=" * 60)

    return selected


if __name__ == "__main__":
    main(
        refined_negative_fasta='./data_now/negative_samples_refined.fasta',
        output_dir='./data_now',
        target_count=17338,  # Match positive count for 1:1 ratio
        seed=42
    )
