"""
Quality Control Script
- Filter sequences <50 or >5000 amino acids
- Remove non-standard amino acids
- Save filtered-out and retained sequences as separate FASTA files
- Preserve original category annotations from source databases
"""

import os
import re
import logging
from typing import List, Dict, Tuple
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Standard amino acids + supported non-standard amino acids (X, B, Z, J)
# X: unknown amino acid (model handles with uniform distribution)
# B: Asp or Asn, Z: Glu or Gln, J: Ile or Leu (model handles with 0.5 weighting)
VALID_AA = set('ACDEFGHIKLMNPQRSTVWYXBZJ')

# Amino acids that are completely invalid (not supported by model)
INVALID_AA = set('0123456789-.* #\t\n\r')


def clean_sequence(sequence: str) -> str:
    """Clean sequence: remove invalid characters, keep valid amino acids (including X, B, Z, J), convert to uppercase"""
    sequence = sequence.upper().strip()
    cleaned = ''.join([aa for aa in sequence if aa in VALID_AA])
    return cleaned


def has_invalid_characters(sequence: str) -> bool:
    """Check if sequence contains characters that are not valid amino acids"""
    return any(aa not in VALID_AA for aa in sequence.upper())


def count_x_ratio(sequence: str) -> float:
    """
    Calculate the ratio of unknown amino acid 'X' in the sequence.
    Returns a value between 0.0 and 1.0.
    """
    seq_upper = sequence.upper()
    if len(seq_upper) == 0:
        return 0.0
    x_count = seq_upper.count('X')
    return x_count / len(seq_upper)


def extract_category_from_header(header: str) -> str:
    """
    Extract category from FASTA header.
    Assumes category is the last field after the last '|' character.
    """
    # Split by '|' and get the last part
    parts = header.split('|')
    if len(parts) > 1:
        # Last part may contain additional info after space, take first word
        last_part = parts[-1].strip()
        # Extract just the category name (before any space or special chars)
        category = last_part.split()[0].split(';')[0].split('#')[0].strip()
        return category if category else 'unknown'
    return 'unknown'


def parse_fasta(filepath: str) -> List[Tuple[str, str]]:
    """Parse FASTA file, return list of (header, sequence) tuples"""
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
                    sequences.append((current_header, ''.join(current_seq)))
                current_header = line[1:]  # Remove '>'
                current_seq = []
            else:
                current_seq.append(line)

        # Don't forget the last sequence
        if current_header and current_seq:
            sequences.append((current_header, ''.join(current_seq)))

    return sequences


def quality_control(
    input_fasta: str,
    output_dir: str = './data',
    min_length: int = 50,
    max_length: int = 5000,
    min_category_size: int = 50,
    max_x_ratio: float = 0.5
):
    """
    Perform quality control on protein sequences.

    Args:
        input_fasta: Path to input FASTA file
        output_dir: Output directory
        min_length: Minimum sequence length (default: 50)
        max_length: Maximum sequence length (default: 5000)
        min_category_size: Minimum sequences per category (default: 50)
                          Categories with fewer sequences are merged to "other"
        max_x_ratio: Maximum allowed ratio of unknown amino acid 'X' (default: 0.5)
                     Sequences with X ratio above this threshold are filtered out
    """
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"Loading sequences from {input_fasta}")
    raw_sequences = parse_fasta(input_fasta)
    logger.info(f"Loaded {len(raw_sequences)} raw sequences")

    # Quality control filtering
    retained = []
    filtered_out = []

    length_stats = {'too_short': 0, 'too_long': 0, 'too_many_x': 0, 'invalid_chars': 0, 'valid': 0}
    category_counter = Counter()

    for header, seq in raw_sequences:
        original_seq = seq
        seq_len = len(original_seq)

        # Check for invalid characters (not supported by model)
        if has_invalid_characters(original_seq):
            length_stats['invalid_chars'] += 1
            cleaned_seq = clean_sequence(original_seq)
            filtered_out.append({
                'header': header,
                'sequence': cleaned_seq,
                'reason': 'invalid_chars',
                'original_length': seq_len,
                'cleaned_length': len(cleaned_seq)
            })
            continue

        # Check for excessive unknown amino acids (X)
        x_ratio = count_x_ratio(original_seq)
        if x_ratio > max_x_ratio:
            length_stats['too_many_x'] += 1
            filtered_out.append({
                'header': header,
                'sequence': original_seq,
                'reason': 'too_many_x',
                'length': seq_len,
                'x_ratio': x_ratio
            })
            continue

        # Length filtering
        if seq_len < min_length:
            length_stats['too_short'] += 1
            filtered_out.append({
                'header': header,
                'sequence': original_seq,
                'reason': 'too_short',
                'length': seq_len
            })
            continue

        if seq_len > max_length:
            length_stats['too_long'] += 1
            filtered_out.append({
                'header': header,
                'sequence': original_seq,
                'reason': 'too_long',
                'length': seq_len
            })
            continue

        # Extract original category from header
        category = extract_category_from_header(header)

        # Keep sequence (with original sequence, not cleaned, since it's already valid)
        retained.append({
            'header': header,
            'sequence': original_seq,
            'category': category,
            'length': seq_len
        })

        category_counter[category] += 1
        length_stats['valid'] += 1

    # Apply category size threshold: merge small categories to "other"
    logger.info(f"\nApplying category size threshold (min={min_category_size})...")

    # Determine which categories to keep vs merge
    keep_categories = set()
    merge_categories = set()
    for cat, count in category_counter.items():
        if count >= min_category_size:
            keep_categories.add(cat)
        else:
            merge_categories.add(cat)

    logger.info(f"  Categories to keep (≥{min_category_size}): {len(keep_categories)}")
    logger.info(f"  Categories to merge to 'other' (<{min_category_size}): {len(merge_categories)}")
    if merge_categories:
        merge_list = sorted(merge_categories)
        logger.info(f"    Merged categories: {', '.join(merge_list)}")

    # Update retained sequences with merged categories
    final_category_counter = Counter()
    for item in retained:
        if item['category'] in merge_categories:
            item['category'] = 'other'
            item['original_category'] = item.get('original_category', item['category'])
        final_category_counter[item['category']] += 1

    # Replace counter for downstream use
    category_counter = final_category_counter

    # Log statistics
    logger.info("=" * 60)
    logger.info("Quality Control Statistics")
    logger.info("=" * 60)
    logger.info(f"Total input sequences: {len(raw_sequences)}")
    logger.info(f"  Too short (<{min_length} aa): {length_stats['too_short']}")
    logger.info(f"  Too long (>{max_length} aa): {length_stats['too_long']}")
    logger.info(f"  Too many X (>{max_x_ratio*100:.0f}%): {length_stats['too_many_x']}")
    logger.info(f"  Invalid characters: {length_stats['invalid_chars']}")
    logger.info(f"  Valid sequences retained: {length_stats['valid']}")
    logger.info("=" * 60)

    # Log category distribution after merging
    logger.info(f"\nFinal category distribution ({len(category_counter)} unique categories):")
    for cat, count in category_counter.most_common():
        if cat == 'other':
            logger.info(f"  {cat}: {count} (merged from {len(merge_categories)} small categories)")
        else:
            logger.info(f"  {cat}: {count}")

    # Save filtered-out sequences
    filtered_out_path = os.path.join(output_dir, 'qc_filtered_out.fasta')
    with open(filtered_out_path, 'w') as f:
        for item in filtered_out:
            f.write(f">{item['header']}|filtered_reason={item['reason']}")
            if 'length' in item:
                f.write(f"|length={item['length']}")
            if 'x_ratio' in item:
                f.write(f"|x_ratio={item['x_ratio']:.2f}")
            if 'original_length' in item:
                f.write(f"|original_length={item['original_length']}|cleaned_length={item['cleaned_length']}")
            f.write("\n")

            seq = item['sequence']
            for i in range(0, len(seq), 60):
                f.write(seq[i:i+60] + '\n')

    logger.info(f"\nSaved {len(filtered_out)} filtered-out sequences to {filtered_out_path}")

    # Save retained sequences (preserve original header + add metadata)
    retained_path = os.path.join(output_dir, 'qc_retained.fasta')
    with open(retained_path, 'w') as f:
        for item in retained:
            # Write header with category annotation (use merged category)
            f.write(f">{item['header']}|category={item['category']}|length={item['length']}\n")

            seq = item['sequence']
            for i in range(0, len(seq), 60):
                f.write(seq[i:i+60] + '\n')

    logger.info(f"Saved {len(retained)} retained sequences to {retained_path}")

    # Save summary statistics
    stats = {
        'total_input': len(raw_sequences),
        'filtered': {
            'too_short': length_stats['too_short'],
            'too_long': length_stats['too_long'],
            'too_many_x': length_stats['too_many_x'],
            'invalid_chars': length_stats['invalid_chars'],
            'total_filtered': len(filtered_out)
        },
        'retained': {
            'count': len(retained),
            'num_categories': len(category_counter),
            'categories': dict(category_counter.most_common()),
            'category_threshold': min_category_size,
            'max_x_ratio_threshold': max_x_ratio,
            'original_categories': len(keep_categories) + len(merge_categories),
            'merged_categories': sorted(list(merge_categories)) if merge_categories else [],
            'kept_categories': sorted(list(keep_categories))
        }
    }

    stats_path = os.path.join(output_dir, 'qc_stats.json')
    import json
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"Saved QC statistics to {stats_path}")
    logger.info("\nQuality Control completed!")

    return retained, filtered_out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Quality Control for ARG sequences')
    parser.add_argument('--input_fasta', type=str, default='./data/ARG_db_all_seq_uniq_representative_rename_2_repsent.fasta',
                        help='Path to input FASTA file')
    parser.add_argument('--output_dir', type=str, default='./data',
                        help='Output directory')
    parser.add_argument('--min_length', type=int, default=50,
                        help='Minimum sequence length')
    parser.add_argument('--max_length', type=int, default=5000,
                        help='Maximum sequence length')
    parser.add_argument('--min_category_size', type=int, default=50,
                        help='Minimum sequences per category')
    parser.add_argument('--max_x_ratio', type=float, default=0.5,
                        help='Maximum allowed ratio of unknown amino acid X')

    args = parser.parse_args()

    quality_control(
        input_fasta=args.input_fasta,
        output_dir=args.output_dir,
        min_length=args.min_length,
        max_length=args.max_length,
        min_category_size=args.min_category_size,
        max_x_ratio=args.max_x_ratio
    )
