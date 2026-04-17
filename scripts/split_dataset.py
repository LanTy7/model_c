"""
Dataset Split Script
- Read qc_retained.fasta
- Stratified split into train/val/test (80:10:10)
- Save as CSV format
"""

import os
import re
import json
import logging
import random
from typing import List, Dict, Tuple
from collections import defaultdict, Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_fasta_with_category(filepath: str) -> List[Dict]:
    """Parse FASTA file, extract sequence, category, and metadata"""
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
                    seq_data = parse_header_and_sequence(current_header, ''.join(current_seq))
                    sequences.append(seq_data)
                current_header = line[1:]  # Remove '>'
                current_seq = []
            else:
                current_seq.append(line)

        # Don't forget the last sequence
        if current_header and current_seq:
            seq_data = parse_header_and_sequence(current_header, ''.join(current_seq))
            sequences.append(seq_data)

    return sequences


def parse_header_and_sequence(header: str, sequence: str) -> Dict:
    """Parse header to extract metadata"""
    # Extract category from |category=xxx| pattern
    category_match = re.search(r'\|category=([^|]+)', header)
    category = category_match.group(1) if category_match else 'unknown'

    # Extract length from |length=xxx| pattern
    length_match = re.search(r'\|length=(\d+)', header)
    length = int(length_match.group(1)) if length_match else len(sequence)

    # Original header (before our added metadata)
    original_header = header.split('|category=')[0].strip()

    return {
        'header': header,
        'original_header': original_header,
        'sequence': sequence,
        'category': category,
        'length': length
    }


def stratified_split(sequences: List[Dict], train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    """
    Perform stratified split by category.

    Args:
        sequences: List of sequence dictionaries
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed for reproducibility

    Returns:
        train_seqs, val_seqs, test_seqs
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    random.seed(seed)

    # Group by category
    category_groups = defaultdict(list)
    for seq in sequences:
        category_groups[seq['category']].append(seq)

    train_seqs, val_seqs, test_seqs = [], [], []

    for category, group in category_groups.items():
        n = len(group)
        if n < 3:
            logger.warning(f"Category '{category}' has only {n} sequences, putting all in train")
            train_seqs.extend(group)
            continue

        # Shuffle within category
        shuffled = group.copy()
        random.shuffle(shuffled)

        # Calculate split indices ensuring every split gets at least 1 when n >= 3
        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))
        n_test = max(1, n - n_train - n_val)

        # Redistribute from train if we exceeded total
        while n_train + n_val + n_test > n:
            if n_train > n_val and n_train > n_test and n_train > 1:
                n_train -= 1
            elif n_val > n_test and n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
            else:
                break

        # Split
        train_seqs.extend(shuffled[:n_train])
        val_seqs.extend(shuffled[n_train:n_train + n_val])
        test_seqs.extend(shuffled[n_train + n_val:n_train + n_val + n_test])

    # Final shuffle to mix categories
    random.shuffle(train_seqs)
    random.shuffle(val_seqs)
    random.shuffle(test_seqs)

    return train_seqs, val_seqs, test_seqs


def save_to_csv(sequences: List[Dict], filepath: str):
    """Save sequences to CSV format"""
    import csv

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sequence', 'category', 'length', 'original_header'])

        for seq in sequences:
            writer.writerow([
                seq['sequence'],
                seq['category'],
                seq['length'],
                seq['original_header']
            ])


def save_to_fasta(sequences: List[Dict], filepath: str):
    """Save sequences to FASTA format with category in header"""
    with open(filepath, 'w') as f:
        for i, seq in enumerate(sequences):
            # Create a clean header with index, category and original info
            header = f">seq_{i}|category={seq['category']}|length={seq['length']}|source={seq['original_header']}"
            f.write(header + '\n')

            # Write sequence in 60-character lines
            sequence = seq['sequence']
            for j in range(0, len(sequence), 60):
                f.write(sequence[j:j+60] + '\n')


def split_dataset(
    input_fasta: str = './data/ARG_DB.fasta',
    output_dir: str = './data',
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
):
    """
    Split dataset into train/val/test with stratified sampling.

    Args:
        input_fasta: Path to qc_retained.fasta
        output_dir: Output directory
        train_ratio: Training set ratio (default: 0.8)
        val_ratio: Validation set ratio (default: 0.1)
        test_ratio: Test set ratio (default: 0.1)
        seed: Random seed for reproducibility (default: 42)
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Loading sequences from {input_fasta}")
    sequences = parse_fasta_with_category(input_fasta)
    logger.info(f"Loaded {len(sequences)} sequences")

    # Log original category distribution
    category_counter = Counter([s['category'] for s in sequences])
    logger.info(f"Found {len(category_counter)} categories")
    for cat, count in category_counter.most_common():
        logger.info(f"  {cat}: {count}")

    # Perform stratified split
    logger.info(f"\nPerforming stratified split ({train_ratio:.0%}:{val_ratio:.0%}:{test_ratio:.0%}) with seed={seed}")
    train_seqs, val_seqs, test_seqs = stratified_split(
        sequences, train_ratio, val_ratio, test_ratio, seed
    )

    logger.info(f"\nSplit results:")
    logger.info(f"  Train: {len(train_seqs)} ({len(train_seqs)/len(sequences)*100:.1f}%)")
    logger.info(f"  Val:   {len(val_seqs)} ({len(val_seqs)/len(sequences)*100:.1f}%)")
    logger.info(f"  Test:  {len(test_seqs)} ({len(test_seqs)/len(sequences)*100:.1f}%)")

    # Log category distribution per split
    def log_split_distribution(name, seqs):
        counter = Counter([s['category'] for s in seqs])
        logger.info(f"\n{name} set category distribution:")
        for cat, count in counter.most_common():
            original_count = category_counter[cat]
            pct = count / original_count * 100
            logger.info(f"  {cat}: {count}/{original_count} ({pct:.1f}%)")

    log_split_distribution("Train", train_seqs)
    log_split_distribution("Val", val_seqs)
    log_split_distribution("Test", test_seqs)

    # Save to CSV
    train_csv = os.path.join(output_dir, 'train.csv')
    val_csv = os.path.join(output_dir, 'val.csv')
    test_csv = os.path.join(output_dir, 'test.csv')

    save_to_csv(train_seqs, train_csv)
    save_to_csv(val_seqs, val_csv)
    save_to_csv(test_seqs, test_csv)

    # Save to FASTA
    train_fasta = os.path.join(output_dir, 'train.fasta')
    val_fasta = os.path.join(output_dir, 'val.fasta')
    test_fasta = os.path.join(output_dir, 'test.fasta')

    save_to_fasta(train_seqs, train_fasta)
    save_to_fasta(val_seqs, val_fasta)
    save_to_fasta(test_seqs, test_fasta)

    logger.info(f"\nSaved splits to:")
    logger.info(f"  CSV:  {train_csv}, {val_csv}, {test_csv}")
    logger.info(f"  FASTA: {train_fasta}, {val_fasta}, {test_fasta}")

    # Save split statistics
    stats = {
        'total_sequences': len(sequences),
        'num_categories': len(category_counter),
        'split_ratios': {'train': train_ratio, 'val': val_ratio, 'test': test_ratio},
        'random_seed': seed,
        'splits': {
            'train': {'count': len(train_seqs), 'percentage': len(train_seqs)/len(sequences)*100},
            'val': {'count': len(val_seqs), 'percentage': len(val_seqs)/len(sequences)*100},
            'test': {'count': len(test_seqs), 'percentage': len(test_seqs)/len(sequences)*100}
        },
        'category_distribution': {
            'total': dict(category_counter.most_common()),
            'train': dict(Counter([s['category'] for s in train_seqs]).most_common()),
            'val': dict(Counter([s['category'] for s in val_seqs]).most_common()),
            'test': dict(Counter([s['category'] for s in test_seqs]).most_common())
        }
    }

    stats_path = os.path.join(output_dir, 'split_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"\nSaved split statistics to {stats_path}")
    logger.info("Dataset split completed!")

    return train_seqs, val_seqs, test_seqs


if __name__ == "__main__":
    split_dataset(
        input_fasta='./data/ARG_DB.fasta',
        output_dir='./data',
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42
    )
