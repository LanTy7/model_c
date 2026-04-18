"""
Prepare negative samples for training with strict quality filtering.

Improvements over original:
1. Header-based exclusion of borderline functional categories
   (efflux pumps, multidrug transporters, stress response, etc.)
2. Optional MMseqs2 filtering against ARG databases (CARD, Resfams, etc.)
3. Sequence length filtering (remove very short sequences)
"""

import argparse
import os
import random
import logging
import json
import re
import subprocess
from typing import List, Dict, Set

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Keywords in FASTA headers that indicate borderline functional categories.
# These are NOT ARGs but are functionally related and may confuse the model.
EXCLUDED_KEYWORDS = [
    # Efflux pumps and multidrug transporters
    'efflux', 'multidrug transporter', 'multidrug resistance protein',
    'mdt', 'acr', 'nor', 'mex', 'cmr', 'smr',
    # Drug/metabolite transporters
    'drug transporter', 'metabolite transporter',
    # Stress response (often co-occurs with ARGs)
    'stress response', 'stress protein', 'heat shock',
    # Biofilm formation (often associated with resistance)
    'biofilm', 'adhesion',
    # Toxin-antitoxin systems
    'toxin', 'antitoxin', 'addiction module',
    # Mobile genetic elements
    'transposase', 'integrase', 'recombinase',
    # Virulence factors
    'virulence', 'pathogenicity',
]


def parse_fasta(filepath: str) -> List[Dict]:
    """Parse FASTA file."""
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


def filter_by_header_keywords(sequences: List[Dict], excluded_keywords: List[str] = None) -> Tuple[List[Dict], List[Dict]]:
    """
    Exclude sequences whose headers contain excluded keywords.

    Returns:
        (kept_sequences, excluded_sequences)
    """
    if excluded_keywords is None:
        excluded_keywords = EXCLUDED_KEYWORDS

    kept = []
    excluded = []
    pattern = re.compile(r'\b(' + '|'.join(re.escape(kw) for kw in excluded_keywords) + r')\b', re.IGNORECASE)

    for seq in sequences:
        if pattern.search(seq['header']):
            excluded.append(seq)
        else:
            kept.append(seq)

    return kept, excluded


def filter_by_length(sequences: List[Dict], min_length: int = 30, max_length: int = 5000) -> List[Dict]:
    """Remove sequences that are too short or too long."""
    kept = []
    for seq in sequences:
        length = len(seq['sequence'])
        if min_length <= length <= max_length:
            kept.append(seq)
    return kept


def filter_by_mmseqs2(
    sequences: List[Dict],
    arg_db_fasta: str,
    output_dir: str,
    min_seq_id: float = 0.3,
    coverage: float = 0.5,
    threads: int = 8
) -> Tuple[List[Dict], List[Dict]]:
    """
    Remove sequences that have homology to known ARGs using MMseqs2.

    Returns:
        (kept_sequences, excluded_sequences)
    """
    if not os.path.exists(arg_db_fasta):
        logger.warning(f"ARG database not found: {arg_db_fasta}. Skipping MMseqs2 filtering.")
        return sequences, []

    logger.info(f"[MMseqs2 Filter] Searching against {arg_db_fasta}")
    os.makedirs(output_dir, exist_ok=True)

    # Write candidate sequences to temp FASTA
    candidates_fasta = os.path.join(output_dir, 'candidates_for_filtering.fasta')
    with open(candidates_fasta, 'w') as f:
        for seq in sequences:
            f.write(f">{seq['id']}\n{seq['sequence']}\n")

    # Run easy-search
    result_m8 = os.path.join(output_dir, 'mmseqs_filter_results.m8')
    tmp_dir = os.path.join(output_dir, 'tmp_filter')
    os.makedirs(tmp_dir, exist_ok=True)

    cmd = [
        'mmseqs', 'easy-search', candidates_fasta, arg_db_fasta, result_m8, tmp_dir,
        '--min-seq-id', str(min_seq_id),
        '-c', str(coverage),
        '--cov-mode', '0',
        '-s', '6.0',
        '--threads', str(threads),
        '-v', '1'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"MMseqs2 filtering failed: {result.stderr}")
        logger.warning("Keeping all sequences (filtering skipped)")
        return sequences, []

    # Parse hits
    hit_ids = set()
    with open(result_m8, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) >= 2:
                hit_ids.add(parts[0])

    kept = [s for s in sequences if s['id'] not in hit_ids]
    excluded = [s for s in sequences if s['id'] in hit_ids]

    logger.info(f"[MMseqs2 Filter] Kept {len(kept)} / {len(sequences)} sequences ({len(excluded)} excluded)")

    # Cleanup
    for fpath in [candidates_fasta, result_m8]:
        if os.path.exists(fpath):
            os.remove(fpath)
    if os.path.exists(tmp_dir):
        import shutil
        shutil.rmtree(tmp_dir)

    return kept, excluded


def save_fasta(sequences: List[Dict], filepath: str):
    """Save sequences to FASTA."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    with open(filepath, 'w') as f:
        for seq in sequences:
            f.write(f">{seq['header']}\n")
            for i in range(0, len(seq['sequence']), 60):
                f.write(seq['sequence'][i:i+60] + '\n')


def save_csv(sequences: List[Dict], filepath: str):
    """Save to CSV format."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
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
    min_length: int = 30,
    max_length: int = 5000,
    exclude_keywords: bool = True,
    excluded_keywords: List[str] = None,
    mmseqs2_filter: bool = False,
    arg_db_fasta: str = None,
    mmseqs2_min_seq_id: float = 0.3,
    seed: int = 42
):
    """
    Prepare negative samples for training with strict quality filtering.

    Args:
        refined_negative_fasta: Path to refined negative samples
        output_dir: Output directory
        target_count: Number of negatives to sample for training
        min_length: Minimum sequence length
        max_length: Maximum sequence length
        exclude_keywords: Whether to exclude sequences by header keywords
        excluded_keywords: Custom list of keywords to exclude
        mmseqs2_filter: Whether to run MMseqs2 against ARG database
        arg_db_fasta: Path to ARG database FASTA for MMseqs2 filtering
        mmseqs2_min_seq_id: Identity threshold for MMseqs2 filtering
        seed: Random seed
    """
    random.seed(seed)

    logger.info("=" * 60)
    logger.info("Preparing Negative Samples for Training (with filtering)")
    logger.info("=" * 60)

    # ------------------------------------------------------------------
    # Step 1: Load all refined negatives
    # ------------------------------------------------------------------
    logger.info(f"\n[Step 1] Loading refined negatives from {refined_negative_fasta}")
    all_negatives = parse_fasta(refined_negative_fasta)
    logger.info(f"  Total refined negatives: {len(all_negatives)}")

    # ------------------------------------------------------------------
    # Step 2: Length filtering
    # ------------------------------------------------------------------
    logger.info(f"\n[Step 2] Length filtering (min={min_length}, max={max_length})")
    all_negatives = filter_by_length(all_negatives, min_length, max_length)
    logger.info(f"  After length filter: {len(all_negatives)}")

    # ------------------------------------------------------------------
    # Step 3: Header keyword exclusion
    # ------------------------------------------------------------------
    if exclude_keywords:
        logger.info(f"\n[Step 3] Excluding borderline categories by header keywords")
        keywords = excluded_keywords if excluded_keywords else EXCLUDED_KEYWORDS
        all_negatives, excluded_by_keyword = filter_by_header_keywords(all_negatives, keywords)
        logger.info(f"  Excluded by keywords: {len(excluded_by_keyword)}")
        logger.info(f"  Remaining: {len(all_negatives)}")

    # ------------------------------------------------------------------
    # Step 4: MMseqs2 filtering against ARG database (optional)
    # ------------------------------------------------------------------
    if mmseqs2_filter and arg_db_fasta:
        logger.info(f"\n[Step 4] MMseqs2 filtering against {arg_db_fasta}")
        all_negatives, excluded_by_mmseqs2 = filter_by_mmseqs2(
            all_negatives, arg_db_fasta, output_dir,
            min_seq_id=mmseqs2_min_seq_id
        )
        logger.info(f"  Remaining after MMseqs2: {len(all_negatives)}")

    # ------------------------------------------------------------------
    # Step 5: Random sampling
    # ------------------------------------------------------------------
    logger.info(f"\n[Step 5] Random sampling target={target_count}")
    if len(all_negatives) < target_count:
        logger.warning(f"Only {len(all_negatives)} negatives available, requested {target_count}")
        selected = all_negatives
    else:
        selected = random.sample(all_negatives, target_count)
        logger.info(f"  Randomly sampled {target_count} negatives for training")

    random.shuffle(selected)

    # Add metadata
    for seq in selected:
        seq['length'] = len(seq['sequence'])
        seq['is_arg'] = 0
        seq['category'] = 'non_arg'

    # ------------------------------------------------------------------
    # Step 6: Save files
    # ------------------------------------------------------------------
    fasta_path = os.path.join(output_dir, 'negative_samples_for_training.fasta')
    csv_path = os.path.join(output_dir, 'negative_samples_for_training.csv')

    save_fasta(selected, fasta_path)
    save_csv(selected, csv_path)

    logger.info(f"\n[Step 6] Saved training negatives:")
    logger.info(f"  FASTA: {fasta_path}")
    logger.info(f"  CSV: {csv_path}")

    # Statistics
    lengths = [s['length'] for s in selected]
    stats = {
        'source': refined_negative_fasta,
        'target_count': target_count,
        'final_count': len(selected),
        'length_stats': {
            'mean': sum(lengths) / len(lengths) if lengths else 0,
            'min': min(lengths) if lengths else 0,
            'max': max(lengths) if lengths else 0
        },
        'filtering': {
            'min_length': min_length,
            'max_length': max_length,
            'exclude_keywords': exclude_keywords,
            'excluded_keywords': keywords if exclude_keywords else [],
            'mmseqs2_filter': mmseqs2_filter,
            'arg_db_fasta': arg_db_fasta,
            'mmseqs2_min_seq_id': mmseqs2_min_seq_id if mmseqs2_filter else None
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
    parser = argparse.ArgumentParser(description="Prepare negative samples for training with filtering")
    parser.add_argument("--refined-negative-fasta", type=str, default="./data/negative_samples_refined.fasta")
    parser.add_argument("--output-dir", type=str, default="./data")
    parser.add_argument("--target-count", type=int, default=17338, help="Target number of negative samples")
    parser.add_argument("--min-length", type=int, default=30, help="Minimum sequence length")
    parser.add_argument("--max-length", type=int, default=5000, help="Maximum sequence length")
    parser.add_argument("--exclude-keywords", action=argparse.BooleanOptionalAction, default=True,
                        help="Exclude sequences by header keywords (default: True)")
    parser.add_argument("--mmseqs2-filter", action=argparse.BooleanOptionalAction, default=False,
                        help="Enable MMseqs2 filtering against ARG database")
    parser.add_argument("--arg-db-fasta", type=str, default=None,
                        help="Path to ARG database FASTA for MMseqs2 filtering")
    parser.add_argument("--mmseqs2-min-seq-id", type=float, default=0.3,
                        help="Identity threshold for MMseqs2 filtering")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(
        refined_negative_fasta=args.refined_negative_fasta,
        output_dir=args.output_dir,
        target_count=args.target_count,
        min_length=args.min_length,
        max_length=args.max_length,
        exclude_keywords=args.exclude_keywords,
        mmseqs2_filter=args.mmseqs2_filter,
        arg_db_fasta=args.arg_db_fasta,
        mmseqs2_min_seq_id=args.mmseqs2_min_seq_id,
        seed=args.seed,
    )
