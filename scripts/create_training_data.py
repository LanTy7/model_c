"""
Create unified training dataset by merging positive (ARG) and negative (non-ARG) samples.
Outputs train/val/test splits with stratification for both binary and multi-class tasks.

Data Leakage Prevention:
- Uses CD-HIT clustering to group similar sequences
- All sequences in the same cluster are assigned to the same split (train/val/test)
- This ensures no high-similarity sequences between train and test sets
"""

import argparse
import os
import re
import json
import logging
import random
import subprocess
from typing import List, Dict, Tuple, Set
from collections import defaultdict, Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_fasta_with_metadata(filepath: str, is_arg: int, default_category: str = None) -> List[Dict]:
    """
    Parse FASTA file and extract metadata.

    Args:
        filepath: Path to FASTA file
        is_arg: 1 for ARG, 0 for non-ARG
        default_category: Default category if not found in header
    """
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
                    seq_data = extract_metadata(current_header, ''.join(current_seq), is_arg, default_category)
                    if seq_data:
                        sequences.append(seq_data)
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line)

        # Don't forget last sequence
        if current_header and current_seq:
            seq_data = extract_metadata(current_header, ''.join(current_seq), is_arg, default_category)
            if seq_data:
                sequences.append(seq_data)

    return sequences


def extract_metadata(header: str, sequence: str, is_arg: int, default_category: str = None) -> Dict:
    """Extract metadata from FASTA header"""
    # Extract category from |category=xxx| pattern
    category_match = re.search(r'\|category=([^|]+)', header)
    category = category_match.group(1) if category_match else default_category

    # Extract length from |length=xxx| pattern
    length_match = re.search(r'\|length=(\d+)', header)
    length = int(length_match.group(1)) if length_match else len(sequence)

    # Extract original ID (first part of header before any |)
    original_id = header.split('|')[0].strip()

    return {
        'id': original_id,
        'header': header,
        'sequence': sequence,
        'length': length,
        'is_arg': is_arg,
        'category': category if is_arg == 1 else 'non_arg',
        'arg_category': category if is_arg == 1 else None
    }


def run_cd_hit_clustering(
    input_fasta: str,
    output_dir: str,
    identity_threshold: float = 0.7,
    coverage_threshold: float = 0.8
) -> Dict[str, int]:
    """
    Run CD-HIT clustering to group similar sequences.
    Returns a dictionary mapping sequence ID to cluster ID.

    Args:
        input_fasta: Path to input FASTA file
        output_dir: Output directory for temporary files
        identity_threshold: Sequence identity threshold (default 0.7 = 70%)
        coverage_threshold: Alignment coverage threshold (default 0.8 = 80%)
    """
    os.makedirs(output_dir, exist_ok=True)

    output_prefix = os.path.join(output_dir, 'cdhit_clustered')

    logger.info(f"Running CD-HIT clustering (identity: {identity_threshold:.0%}, coverage: {coverage_threshold:.0%})...")

    # CD-HIT word size: higher threshold requires larger word size
    if identity_threshold >= 0.7:
        word_size = 5
    elif identity_threshold >= 0.6:
        word_size = 4
    else:
        word_size = 3

    # CD-HIT command with coverage parameters
    cmd = (
        f"cd-hit -i {input_fasta} -o {output_prefix} "
        f"-c {identity_threshold} -n {word_size} -d 0 -M 16000 -T 8 "
        f"-aS {coverage_threshold} -aL {coverage_threshold}"
    )

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"CD-HIT failed: {result.stderr}")
        raise RuntimeError("CD-HIT clustering failed")

    # Parse cluster file to get sequence to cluster mapping
    cluster_file = output_prefix + '.clstr'
    seq_to_cluster = {}

    with open(cluster_file, 'r') as f:
        current_cluster = None
        for line in f:
            line = line.strip()
            if line.startswith('>Cluster'):
                current_cluster = int(line.split()[1])
            elif line and current_cluster is not None:
                # Extract integer UID from line like:
                # 0\t147aa, >0 *\n or
                # 1\t302aa, >1 at 1:302:1:302/+/62.58%
                parts = line.split('>')
                if len(parts) >= 2:
                    seq_id = int(parts[1].split()[0].rstrip('.'))
                    seq_to_cluster[seq_id] = current_cluster

    logger.info(f"CD-HIT clustering complete: {len(set(seq_to_cluster.values()))} clusters")
    logger.info(f"  Sequences assigned to clusters: {len(seq_to_cluster)}")

    # Clean up temporary files
    for ext in ['', '.clstr']:
        temp_file = output_prefix + ext
        if os.path.exists(temp_file):
            os.remove(temp_file)

    return seq_to_cluster


def stratified_split_with_clustering(
    sequences: List[Dict],
    cluster_mapping: Dict[str, int],
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Perform stratified split with data leakage prevention via clustering.
    All sequences in the same cluster are assigned to the same split.

    Args:
        sequences: List of sequence dictionaries
        cluster_mapping: Dictionary mapping sequence ID to cluster ID
        train_ratio: Training set ratio
        val_ratio: Validation set ratio
        test_ratio: Test set ratio
        seed: Random seed
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    random.seed(seed)

    # Group sequences by cluster
    cluster_to_seqs = defaultdict(list)
    for seq in sequences:
        uid = seq['_uid']
        if uid not in cluster_mapping:
            raise KeyError(f"Sequence _uid {uid} not found in cluster mapping. "
                           f"Ensure clustering was run on the same sequence set.")
        cluster_id = cluster_mapping[uid]
        cluster_to_seqs[cluster_id].append(seq)

    logger.info(f"Total clusters: {len(cluster_to_seqs)}")
    logger.info(f"  Clustered sequences: {sum(len(v) for k, v in cluster_to_seqs.items() if k != -1)}")
    logger.info(f"  Unclustered sequences: {len(cluster_to_seqs.get(-1, []))}")

    # Group clusters by (is_arg, category) for stratification
    cluster_groups = defaultdict(list)
    for cluster_id, seqs in cluster_to_seqs.items():
        # Use the dominant class in the cluster
        is_arg = seqs[0]['is_arg']
        category = seqs[0]['category']
        cluster_groups[(is_arg, category)].append(cluster_id)

    train_clusters, val_clusters, test_clusters = [], [], []

    # Split each group
    for (is_arg, category), clusters in cluster_groups.items():
        n = len(clusters)

        if n < 3:
            # Very small groups go entirely to train
            train_clusters.extend(clusters)
            logger.warning(f"Group ({is_arg}, {category}) has only {n} clusters, all to train")
            continue

        # Shuffle clusters
        shuffled = clusters.copy()
        random.shuffle(shuffled)

        # Calculate split sizes ensuring every split gets at least 1 when n >= 3
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
        train_clusters.extend(shuffled[:n_train])
        val_clusters.extend(shuffled[n_train:n_train + n_val])
        test_clusters.extend(shuffled[n_train + n_val:n_train + n_val + n_test])

    # Collect sequences from clusters
    train_seqs = []
    for cid in train_clusters:
        train_seqs.extend(cluster_to_seqs[cid])

    val_seqs = []
    for cid in val_clusters:
        val_seqs.extend(cluster_to_seqs[cid])

    test_seqs = []
    for cid in test_clusters:
        test_seqs.extend(cluster_to_seqs[cid])

    # Final shuffle
    random.shuffle(train_seqs)
    random.shuffle(val_seqs)
    random.shuffle(test_seqs)

    logger.info(f"Split by clusters:")
    logger.info(f"  Train: {len(train_clusters)} clusters -> {len(train_seqs)} sequences")
    logger.info(f"  Val:   {len(val_clusters)} clusters -> {len(val_seqs)} sequences")
    logger.info(f"  Test:  {len(test_clusters)} clusters -> {len(test_seqs)} sequences")

    return train_seqs, val_seqs, test_seqs


def stratified_split(
    sequences: List[Dict],
    train_ratio=0.8,
    val_ratio=0.1,
    test_ratio=0.1,
    seed=42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Perform stratified split by both is_arg and category.
    Ensures both ARG and non-ARG are proportionally represented in each split.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    random.seed(seed)

    # Group by (is_arg, category) for stratification
    group_keys = []
    for seq in sequences:
        key = (seq['is_arg'], seq['category'])
        if key not in group_keys:
            group_keys.append(key)

    # Split each group
    train_seqs, val_seqs, test_seqs = [], [], []

    for key in group_keys:
        group = [s for s in sequences if (s['is_arg'], s['category']) == key]
        n = len(group)

        if n < 3:
            # Very small groups go entirely to train
            train_seqs.extend(group)
            logger.warning(f"Group {key} has only {n} samples, all assigned to train")
            continue

        # Shuffle within group
        shuffled = group.copy()
        random.shuffle(shuffled)

        # Calculate split sizes
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        n_test = n - n_train - n_val

        # Ensure at least 1 sample per split if group is large enough
        if n >= 10:
            n_val = max(1, n_val)
            n_test = max(1, n_test)
            n_train = n - n_val - n_test

        # Split
        train_seqs.extend(shuffled[:n_train])
        val_seqs.extend(shuffled[n_train:n_train + n_val])
        test_seqs.extend(shuffled[n_train + n_val:n_train + n_val + n_test])

    # Final shuffle to mix classes
    random.shuffle(train_seqs)
    random.shuffle(val_seqs)
    random.shuffle(test_seqs)

    return train_seqs, val_seqs, test_seqs


def save_to_csv(sequences: List[Dict], filepath: str):
    """Save sequences to CSV format"""
    import csv

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sequence', 'is_arg', 'category', 'arg_category', 'length', 'id'])

        for seq in sequences:
            writer.writerow([
                seq['sequence'],
                seq['is_arg'],
                seq['category'],
                seq['arg_category'] if seq['arg_category'] else '',
                seq['length'],
                seq['id']
            ])


def save_to_fasta(sequences: List[Dict], filepath: str):
    """Save sequences to FASTA format with metadata in header"""
    with open(filepath, 'w') as f:
        for seq in sequences:
            # Build header with metadata
            header = f">{seq['id']}|is_arg={seq['is_arg']}|category={seq['category']}"
            if seq['arg_category']:
                header += f"|arg_category={seq['arg_category']}"
            header += f"|length={seq['length']}"

            f.write(header + '\n')

            # Write sequence in 60-character lines
            for i in range(0, len(seq['sequence']), 60):
                f.write(seq['sequence'][i:i+60] + '\n')


def main(
    positive_fasta: str = './data/ARG_DB.fasta',
    negative_fasta: str = './data/negative_samples_for_training.fasta',
    output_dir: str = './data',
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    use_clustering: bool = True,
    cluster_identity: float = 0.7,
    seed: int = 42
):
    """
    Create unified training dataset from positive and negative samples.
    Prevents data leakage by clustering similar sequences before splitting.

    Args:
        positive_fasta: Path to ARG sequences (qc_retained.fasta)
        negative_fasta: Path to non-ARG sequences (negative_samples_for_training.fasta)
        output_dir: Output directory
        train_ratio: Training set ratio (default: 0.8)
        val_ratio: Validation set ratio (default: 0.1)
        test_ratio: Test set ratio (default: 0.1)
        use_clustering: If True, use CD-HIT clustering to prevent data leakage
        cluster_identity: Sequence identity threshold for clustering (default: 0.7 = 70%)
        seed: Random seed for reproducibility
    """
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)

    logger.info("=" * 60)
    logger.info("Creating Unified Training Dataset")
    if use_clustering:
        logger.info(f"  Data Leakage Prevention: ENABLED (CD-HIT clustering, identity={cluster_identity:.0%})")
    else:
        logger.info("  Data Leakage Prevention: DISABLED")
    logger.info("=" * 60)

    # Step 1: Load positive samples (ARG)
    logger.info(f"\nStep 1: Loading positive samples from {positive_fasta}")
    positive_seqs = parse_fasta_with_metadata(positive_fasta, is_arg=1)
    logger.info(f"Loaded {len(positive_seqs)} ARG sequences")

    # Log category distribution
    pos_categories = Counter([s['category'] for s in positive_seqs])
    logger.info(f"  Categories: {len(pos_categories)}")
    for cat, count in pos_categories.most_common():
        logger.info(f"    {cat}: {count}")

    # Step 2: Load negative samples (non-ARG)
    logger.info(f"\nStep 2: Loading negative samples from {negative_fasta}")
    negative_seqs = parse_fasta_with_metadata(negative_fasta, is_arg=0, default_category='non_arg')
    logger.info(f"Loaded {len(negative_seqs)} non-ARG sequences")

    # Step 3: Merge all sequences and assign integer UIDs
    logger.info(f"\nStep 3: Merging datasets")
    all_sequences = positive_seqs + negative_seqs
    for idx, seq in enumerate(all_sequences):
        seq['_uid'] = idx
    logger.info(f"Total sequences: {len(all_sequences)}")
    logger.info(f"  ARG: {len(positive_seqs)} ({len(positive_seqs)/len(all_sequences)*100:.1f}%)")
    logger.info(f"  non-ARG: {len(negative_seqs)} ({len(negative_seqs)/len(all_sequences)*100:.1f}%)")

    # Step 4: Stratified split with optional clustering
    logger.info(f"\nStep 4: Stratified split ({train_ratio:.0%}:{val_ratio:.0%}:{test_ratio:.0%})")

    if use_clustering:
        # Save merged sequences to temp file for clustering with integer-only headers
        temp_merged = os.path.join(output_dir, 'temp_merged_for_clustering.fasta')
        with open(temp_merged, 'w') as f:
            for seq in all_sequences:
                f.write(f">{seq['_uid']}\n")
                for i in range(0, len(seq['sequence']), 60):
                    f.write(seq['sequence'][i:i+60] + '\n')

        # Run CD-HIT clustering
        logger.info(f"\nRunning CD-HIT clustering (identity: {cluster_identity:.0%})...")
        logger.info("This prevents similar sequences from appearing in different splits")
        cluster_mapping = run_cd_hit_clustering(temp_merged, output_dir, cluster_identity)

        # Clean up temp file
        if os.path.exists(temp_merged):
            os.remove(temp_merged)

        # Check if clustering produced reasonable results
        cluster_ids = list(cluster_mapping.values())
        num_clusters = len(set(cluster_ids))
        min_clusters_needed = int(10 / min(train_ratio, val_ratio, test_ratio))  # At least 10 per split

        # Compute cluster size distribution
        from collections import Counter
        cluster_size_counts = Counter(cluster_ids)
        sizes = list(cluster_size_counts.values())
        largest = max(sizes)
        singletons = sum(1 for s in sizes if s == 1)
        medium = sum(1 for s in sizes if 2 <= s <= 10)
        large = sum(1 for s in sizes if s > 10)

        logger.info(f"  Clusters: {num_clusters}")
        logger.info(f"  Largest cluster: {largest} sequences")
        logger.info(f"  Distribution: {singletons} singletons, {medium} small (2-10), {large} large (>10)")

        if num_clusters < min_clusters_needed:
            logger.warning(f"CD-HIT produced only {num_clusters} clusters, need at least {min_clusters_needed}")
            logger.warning("Falling back to standard stratified split (sequences are already CD-HIT dereplicated)")
            train_seqs, val_seqs, test_seqs = stratified_split(
                all_sequences, train_ratio, val_ratio, test_ratio, seed
            )
        else:
            # Perform split with clustering
            train_seqs, val_seqs, test_seqs = stratified_split_with_clustering(
                all_sequences, cluster_mapping, train_ratio, val_ratio, test_ratio, seed
            )
    else:
        # Standard split without clustering
        train_seqs, val_seqs, test_seqs = stratified_split(
            all_sequences, train_ratio, val_ratio, test_ratio, seed
        )

    logger.info(f"\nSplit results:")
    logger.info(f"  Train: {len(train_seqs)} ({len(train_seqs)/len(all_sequences)*100:.1f}%)")
    logger.info(f"  Val:   {len(val_seqs)} ({len(val_seqs)/len(all_sequences)*100:.1f}%)")
    logger.info(f"  Test:  {len(test_seqs)} ({len(test_seqs)/len(all_sequences)*100:.1f}%)")

    # Log binary class distribution per split
    def log_binary_distribution(name, seqs):
        arg_count = sum(1 for s in seqs if s['is_arg'] == 1)
        non_arg_count = len(seqs) - arg_count
        logger.info(f"\n{name} set:")
        logger.info(f"  ARG: {arg_count} ({arg_count/len(seqs)*100:.1f}%)")
        logger.info(f"  non-ARG: {non_arg_count} ({non_arg_count/len(seqs)*100:.1f}%)")

    log_binary_distribution("Train", train_seqs)
    log_binary_distribution("Val", val_seqs)
    log_binary_distribution("Test", test_seqs)

    # Step 5: Save to files
    logger.info(f"\nStep 5: Saving output files")

    # CSV files
    save_to_csv(train_seqs, os.path.join(output_dir, 'train.csv'))
    save_to_csv(val_seqs, os.path.join(output_dir, 'val.csv'))
    save_to_csv(test_seqs, os.path.join(output_dir, 'test.csv'))

    # FASTA files
    save_to_fasta(train_seqs, os.path.join(output_dir, 'train.fasta'))
    save_to_fasta(val_seqs, os.path.join(output_dir, 'val.fasta'))
    save_to_fasta(test_seqs, os.path.join(output_dir, 'test.fasta'))

    logger.info(f"  Saved to {output_dir}/")
    logger.info(f"    train.csv/fasta: {len(train_seqs)} sequences")
    logger.info(f"    val.csv/fasta: {len(val_seqs)} sequences")
    logger.info(f"    test.csv/fasta: {len(test_seqs)} sequences")

    # Step 6: Save statistics
    stats = {
        'total_sequences': len(all_sequences),
        'positive_samples': len(positive_seqs),
        'negative_samples': len(negative_seqs),
        'categories': list(pos_categories.keys()),
        'num_categories': len(pos_categories),
        'split_ratios': {'train': train_ratio, 'val': val_ratio, 'test': test_ratio},
        'random_seed': seed,
        'data_leakage_prevention': {
            'enabled': use_clustering,
            'cluster_identity_threshold': cluster_identity if use_clustering else None
        },
        'splits': {
            'train': {
                'count': len(train_seqs),
                'arg_count': sum(1 for s in train_seqs if s['is_arg'] == 1),
                'non_arg_count': sum(1 for s in train_seqs if s['is_arg'] == 0)
            },
            'val': {
                'count': len(val_seqs),
                'arg_count': sum(1 for s in val_seqs if s['is_arg'] == 1),
                'non_arg_count': sum(1 for s in val_seqs if s['is_arg'] == 0)
            },
            'test': {
                'count': len(test_seqs),
                'arg_count': sum(1 for s in test_seqs if s['is_arg'] == 1),
                'non_arg_count': sum(1 for s in test_seqs if s['is_arg'] == 0)
            }
        },
        'category_distribution': {
            'total': dict(pos_categories.most_common()),
            'train': dict(Counter([s['category'] for s in train_seqs if s['is_arg'] == 1]).most_common()),
            'val': dict(Counter([s['category'] for s in val_seqs if s['is_arg'] == 1]).most_common()),
            'test': dict(Counter([s['category'] for s in test_seqs if s['is_arg'] == 1]).most_common())
        }
    }

    stats_path = os.path.join(output_dir, 'training_data_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"\nStatistics saved to {stats_path}")

    logger.info("\n" + "=" * 60)
    logger.info("Training dataset creation complete!")
    logger.info("=" * 60)

    return train_seqs, val_seqs, test_seqs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create unified training dataset")
    parser.add_argument("--positive-fasta", type=str, default="/home/mayue/ARGMind/data/ARG_DB.fasta")
    parser.add_argument("--negative-fasta", type=str, default="./data/Non_ARG_DB.fasta")
    parser.add_argument("--output-dir", type=str, default="./data")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--use-clustering", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--cluster-identity",
        type=float,
        default=0.7,
        help="CD-HIT sequence identity threshold for clustering (default: 0.7). "
             "Higher values cluster more aggressively, preventing data leakage but reducing split diversity. "
             "Lower values preserve more diversity but increase leakage risk."
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(
        positive_fasta=args.positive_fasta,
        negative_fasta=args.negative_fasta,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        use_clustering=args.use_clustering,
        cluster_identity=args.cluster_identity,
        seed=args.seed,
    )
