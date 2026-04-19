"""
Create unified training dataset with two-stage homology-aware splitting.
Based on DefensePredictor methodology (Science paper).

Key improvements over original CD-HIT approach:
1. MMseqs2 at 30% identity for redundancy reduction (Stage 1)
2. Sensitive all-vs-all profile search + Louvain network clustering (Stage 2)
3. Single 80:10:10 train/val/test split at family level with rare category handling
   - Homologous families never cross splits
   - Rare ARG categories (< 3 families) are extracted and split at sequence level
4. Stratified balancing by (is_arg, category) to maintain class distribution

Pipeline:
  Stage 1: MMseqs2 cluster at 30% identity / 80% coverage (redundancy removal)
  Stage 2: MMseqs2 all-vs-all search on cluster reps (--num-iterations 3, -s 7.5)
  Stage 3: Build homology network + Louvain clustering -> "families"
  Stage 4: 80:10:10 train/val/test split with family-level stratification
           Common categories: split at family level (homology constraint)
           Rare categories: extracted and split at sequence level (coverage guarantee)
  Output:  train.csv/val.csv/test.csv + training_data_stats.json
"""

import argparse
import csv
import os
import re
import json
import logging
import random
import subprocess
from typing import List, Dict, Tuple, Set
from collections import defaultdict, Counter

import networkx as nx
import community as community_louvain

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FASTA / metadata helpers
# ---------------------------------------------------------------------------

def parse_fasta_with_metadata(filepath: str, is_arg: int, default_category: str = None, seq_counter_start: int = 0) -> Tuple[List[Dict], int]:
    """Parse FASTA file and extract metadata."""
    sequences = []
    current_header = None
    current_seq = []
    seq_counter = seq_counter_start

    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith('>'):
                if current_header and current_seq:
                    seq_data = extract_metadata(
                        current_header, ''.join(current_seq),
                        is_arg, default_category, seq_id=f"seq_{seq_counter:08d}"
                    )
                    if seq_data:
                        sequences.append(seq_data)
                        seq_counter += 1
                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line)

        if current_header and current_seq:
            seq_data = extract_metadata(
                current_header, ''.join(current_seq),
                is_arg, default_category, seq_id=f"seq_{seq_counter:08d}"
            )
            if seq_data:
                sequences.append(seq_data)
                seq_counter += 1

    return sequences, seq_counter


def extract_metadata(header: str, sequence: str, is_arg: int, default_category: str = None, seq_id: str = None) -> Dict:
    """Extract metadata from FASTA header."""
    category_match = re.search(r'\|category=([^|]+)', header)
    category = category_match.group(1) if category_match else default_category

    length_match = re.search(r'\|length=(\d+)', header)
    length = int(length_match.group(1)) if length_match else len(sequence)

    if seq_id is not None:
        original_id = seq_id
    else:
        parts = header.split('|')
        if len(parts) >= 2 and parts[0].strip().lower() in ('sp', 'tr', 'db'):
            original_id = parts[1].strip() if len(parts) > 1 else parts[0].strip()
        else:
            original_id = parts[0].strip()

    return {
        'id': original_id,
        'header': header,
        'sequence': sequence,
        'length': length,
        'is_arg': is_arg,
        'category': category if is_arg == 1 else 'non_arg',
        'arg_category': category if is_arg == 1 else None
    }


def save_to_csv(sequences: List[Dict], filepath: str):
    """Save sequences to CSV format."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
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
    """Save sequences to FASTA format with metadata in header."""
    os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
    with open(filepath, 'w') as f:
        for seq in sequences:
            header = f">{seq['id']}|is_arg={seq['is_arg']}|category={seq['category']}"
            if seq['arg_category']:
                header += f"|arg_category={seq['arg_category']}"
            header += f"|length={seq['length']}"
            f.write(header + '\n')
            for i in range(0, len(seq['sequence']), 60):
                f.write(seq['sequence'][i:i+60] + '\n')


# ---------------------------------------------------------------------------
# Stage 1: MMseqs2 clustering (redundancy reduction)
# ---------------------------------------------------------------------------

def run_mmseqs_stage1(
    input_fasta: str,
    output_dir: str,
    min_seq_id: float = 0.3,
    coverage: float = 0.8,
    sensitivity: float = 6.0
) -> Tuple[Dict[int, int], Dict[int, int], str]:
    """
    Run MMseqs2 easy-cluster for redundancy reduction.

    Returns:
        seq_to_cluster:  {uid: cluster_id}
        cluster_to_rep:  {cluster_id: rep_uid}
        reps_fasta:      path to representative sequences FASTA
    """
    os.makedirs(output_dir, exist_ok=True)
    prefix = os.path.join(output_dir, 'stage1_cluster')
    tmp_dir = os.path.join(output_dir, 'tmp_stage1')
    os.makedirs(tmp_dir, exist_ok=True)

    logger.info(f"[Stage 1] MMseqs2 clustering: min-seq-id={min_seq_id}, coverage={coverage}, sensitivity={sensitivity}")

    cmd = [
        'mmseqs', 'easy-cluster', input_fasta, prefix, tmp_dir,
        '--min-seq-id', str(min_seq_id),
        '-c', str(coverage),
        '--cov-mode', '0',
        '--cluster-mode', '0',
        '-s', str(sensitivity),
        '--threads', '8',
        '-v', '1'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"MMseqs2 easy-cluster failed:\n{result.stderr}")
        raise RuntimeError("Stage 1 clustering failed")

    # Parse cluster TSV: each line is "representative\tmember"
    cluster_tsv = prefix + '_cluster.tsv'
    seq_to_cluster = {}
    cluster_to_rep = {}

    with open(cluster_tsv, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            rep_uid = int(parts[0])
            member_uid = int(parts[1])

            cluster_id = rep_uid
            if cluster_id not in cluster_to_rep:
                cluster_to_rep[cluster_id] = rep_uid

            seq_to_cluster[member_uid] = cluster_id

    reps_fasta = prefix + '_rep_seq.fasta'
    num_clusters = len(cluster_to_rep)
    logger.info(f"[Stage 1] Complete: {num_clusters} clusters, {len(seq_to_cluster)} sequences assigned")

    return seq_to_cluster, cluster_to_rep, reps_fasta


# ---------------------------------------------------------------------------
# Stage 2: Sensitive all-vs-all search on representatives
# ---------------------------------------------------------------------------

def run_mmseqs_stage2(
    reps_fasta: str,
    output_dir: str,
    num_iterations: int = 3,
    sensitivity: float = 7.5,
    coverage: float = 0.8
) -> str:
    """
    Run all-vs-all MMseqs2 iterative profile search on cluster representatives.
    This performs iterative sequence-profile searches that can detect remote
    homologs more sensitively than single-pass clustering.

    Returns:
        m8_file: path to tabular search results (query, target, pident)
    """
    tmp_dir = os.path.join(output_dir, 'tmp_stage2')
    os.makedirs(tmp_dir, exist_ok=True)
    db_dir = os.path.join(output_dir, 'stage2_db')
    os.makedirs(db_dir, exist_ok=True)

    reps_db = os.path.join(db_dir, 'repsDB')
    result_db = os.path.join(db_dir, 'resultDB')
    m8_file = os.path.join(output_dir, 'stage2_search_results.m8')

    logger.info(f"[Stage 2] MMseqs2 search: iterations={num_iterations}, sensitivity={sensitivity}, coverage={coverage}")

    # createdb
    r1 = subprocess.run(
        ['mmseqs', 'createdb', reps_fasta, reps_db, '-v', '1'],
        capture_output=True, text=True
    )
    if r1.returncode != 0:
        raise RuntimeError(f"mmseqs createdb failed: {r1.stderr}")

    # search (all-vs-all, profile search with iterations)
    r2 = subprocess.run(
        [
            'mmseqs', 'search', reps_db, reps_db, result_db, tmp_dir,
            '--num-iterations', str(num_iterations),
            '-s', str(sensitivity),
            '-c', str(coverage),
            '--cov-mode', '0',
            '--threads', '8',
            '-v', '1'
        ],
        capture_output=True, text=True
    )
    if r2.returncode != 0:
        raise RuntimeError(f"mmseqs search failed: {r2.stderr}")

    # convertalis to m8 format (query, target, pident only)
    r3 = subprocess.run(
        [
            'mmseqs', 'convertalis', reps_db, reps_db, result_db, m8_file,
            '--format-output', 'query,target,pident',
            '-v', '1'
        ],
        capture_output=True, text=True
    )
    if r3.returncode != 0:
        raise RuntimeError(f"mmseqs convertalis failed: {r3.stderr}")

    logger.info(f"[Stage 2] Complete: search results saved to {m8_file}")
    return m8_file


# ---------------------------------------------------------------------------
# Stage 3: Build homology network + Louvain clustering
# ---------------------------------------------------------------------------

def build_homology_network(
    m8_file: str,
    cluster_to_rep: Dict[int, int],
    min_identity: float = 0.30
) -> nx.Graph:
    """
    Build homology network from MMseqs2 search results.
    Nodes = cluster IDs (same as rep UIDs), edges = homology links.

    Edge condition: pident >= 30% (matching paper's coverage/identity threshold)
    """
    G = nx.Graph()
    for cluster_id in cluster_to_rep:
        G.add_node(cluster_id)

    edge_count = 0
    with open(m8_file, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split('\t')
            if len(parts) < 3:
                continue
            try:
                query_uid = int(parts[0])
                target_uid = int(parts[1])
                pident = float(parts[2])
            except (ValueError, IndexError):
                continue

            # Paper: "connected two proteins if at least one had greater than
            # 30% identity and 80% coverage"
            if pident < min_identity * 100:  # m8 pident is 0-100
                continue

            if query_uid == target_uid:
                continue

            if query_uid in cluster_to_rep and target_uid in cluster_to_rep:
                if not G.has_edge(query_uid, target_uid):
                    G.add_edge(query_uid, target_uid, weight=pident)
                    edge_count += 1

    logger.info(f"[Network] Built graph: {G.number_of_nodes()} nodes, {edge_count} edges (min_identity={min_identity})")
    return G


def run_louvain_clustering(
    network: nx.Graph,
    seed: int = 42
) -> Dict[int, int]:
    """
    Run Louvain community detection on homology network.

    Returns:
        cluster_to_family: {cluster_id: family_id}
    """
    random.seed(seed)
    partition = community_louvain.best_partition(network, weight='weight', random_state=seed)
    num_families = len(set(partition.values()))
    logger.info(f"[Louvain] Clustered into {num_families} families")
    return partition


def build_seq_to_family(
    seq_to_cluster: Dict[int, int],
    cluster_to_family: Dict[int, int]
) -> Dict[int, int]:
    """Map each sequence UID to its family ID."""
    seq_to_family = {}
    for uid, cluster_id in seq_to_cluster.items():
        family_id = cluster_to_family.get(cluster_id, cluster_id)
        seq_to_family[uid] = family_id
    return seq_to_family


# ---------------------------------------------------------------------------
# Stage 4: Train/Val/Test split with rare category handling
# ---------------------------------------------------------------------------

def identify_rare_categories(
    family_to_sequences: Dict[int, List[Dict]],
    min_families: int = 3
) -> Set[str]:
    """Identify ARG categories with fewer than min_families families."""
    category_family_counts = Counter()
    for fam_id, seqs in family_to_sequences.items():
        arg_cats = set(s['category'] for s in seqs if s['is_arg'] == 1)
        for cat in arg_cats:
            category_family_counts[cat] += 1
    rare_categories = {cat for cat, count in category_family_counts.items() if count < min_families}
    return rare_categories


def extract_rare_sequences(
    family_to_sequences: Dict[int, List[Dict]],
    rare_categories: Set[str]
) -> Tuple[List[Dict], Dict[int, List[Dict]]]:
    """
    Extract rare-category sequences from families.
    Returns (rare_pool, common_families) where common_families has rare seqs removed.
    If a family has only rare-category sequences, it's fully consumed into rare_pool.
    """
    rare_pool = []
    common_families = {}

    for fam_id, seqs in family_to_sequences.items():
        rare_seqs = [s for s in seqs if s['is_arg'] == 1 and s['category'] in rare_categories]
        common_seqs = [s for s in seqs if s not in rare_seqs]

        if rare_seqs:
            rare_pool.extend(rare_seqs)
        if common_seqs:
            common_families[fam_id] = common_seqs

    return rare_pool, common_families


def sequence_level_stratified_split(
    sequences: List[Dict],
    ratios: List[float] = None,
    seed: int = 42
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split sequences at sequence level with stratification by (is_arg, category).
    Ensures each category group has at least 1 sample in each split if group size >= 3.
    """
    if ratios is None:
        ratios = [0.8, 0.1, 0.1]
    assert abs(sum(ratios) - 1.0) < 1e-6
    random.seed(seed)

    # Group by (is_arg, category)
    group_to_seqs = defaultdict(list)
    for seq in sequences:
        group_to_seqs[(seq['is_arg'], seq['category'])].append(seq)

    splits = [[], [], []]

    for group_key, group_seqs in group_to_seqs.items():
        random.shuffle(group_seqs)
        n = len(group_seqs)

        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        n_test = n - n_train - n_val

        # Ensure at least 1 in each split if n >= 3
        if n >= 3:
            n_train = max(1, n_train)
            n_val = max(1, n_val)
            n_test = max(1, n_test)
            # Rebalance if overshoot
            while n_train + n_val + n_test > n:
                if n_train > 1:
                    n_train -= 1
                elif n_val > 1:
                    n_val -= 1
                elif n_test > 1:
                    n_test -= 1
        else:
            # For tiny groups, all go to train
            n_train = n
            n_val = 0
            n_test = 0

        splits[0].extend(group_seqs[:n_train])
        splits[1].extend(group_seqs[n_train:n_train + n_val])
        splits[2].extend(group_seqs[n_train + n_val:])

    return splits[0], splits[1], splits[2]


def stratified_family_split(
    families: List[int],
    family_to_sequences: Dict[int, List[Dict]],
    ratios: List[float] = None,
    seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split families into train/val/test with stratification.
    Stratification key = (is_arg, category) based on majority label in family.
    Uses greedy balancing with global targets to respect ratios.
    """
    if ratios is None:
        ratios = [0.8, 0.1, 0.1]
    assert abs(sum(ratios) - 1.0) < 1e-6
    random.seed(seed)

    # Determine majority label for each family
    family_labels = {}
    for fam in families:
        seqs = family_to_sequences[fam]
        labels = [(s['is_arg'], s['category']) for s in seqs]
        majority = Counter(labels).most_common(1)[0][0]
        family_labels[fam] = majority

    # Group families by stratification key
    group_to_fams = defaultdict(list)
    for fam in families:
        group_to_fams[family_labels[fam]].append(fam)

    # Compute global targets
    total_seqs = sum(len(family_to_sequences[f]) for f in families)
    total_args = sum(sum(1 for s in family_to_sequences[f] if s['is_arg'] == 1) for f in families)
    target_seqs = [int(total_seqs * r) for r in ratios]
    target_args = [int(total_args * r) for r in ratios]
    target_seqs[0] = total_seqs - sum(target_seqs[1:])
    target_args[0] = total_args - sum(target_args[1:])

    splits = [[], [], []]  # train, val, test
    split_seq_counts = [0, 0, 0]
    split_arg_counts = [0, 0, 0]

    # First pass: greedy assignment per group (shuffled order)
    group_keys = list(group_to_fams.keys())
    random.shuffle(group_keys)

    for group_key in group_keys:
        group_fams = group_to_fams[group_key]
        # Sort families by size descending for greedy assignment
        fams_sorted = sorted(group_fams, key=lambda f: len(family_to_sequences[f]), reverse=True)

        for fam in fams_sorted:
            fam_seqs = family_to_sequences[fam]
            fam_size = len(fam_seqs)
            fam_arg = sum(1 for s in fam_seqs if s['is_arg'] == 1)

            best_idx = 0
            best_score = -float('inf')
            for i in range(3):
                seq_deficit = target_seqs[i] - split_seq_counts[i]
                arg_deficit = target_args[i] - split_arg_counts[i]
                score = seq_deficit + 10.0 * arg_deficit
                if score > best_score:
                    best_score = score
                    best_idx = i

            splits[best_idx].append(fam)
            split_seq_counts[best_idx] += fam_size
            split_arg_counts[best_idx] += fam_arg

    # Second pass: iterative repair coverage - ensure each group appears in all 3 splits
    for _ in range(10):  # Max 10 iterations to converge
        all_covered = True
        for group_key in group_to_fams.keys():
            # Identify which splits have this group
            split_has_group = []
            for i in range(3):
                has_it = any(family_labels.get(f) == group_key for f in splits[i])
                split_has_group.append(has_it)

            missing_splits = [i for i, has in enumerate(split_has_group) if not has]
            if not missing_splits:
                continue

            all_covered = False
            for missing_idx in missing_splits:
                # Find the smallest family from splits that DO have this group
                candidates = []
                for i in range(3):
                    if split_has_group[i]:
                        for fam in splits[i]:
                            if family_labels.get(fam) == group_key:
                                candidates.append((fam, i, len(family_to_sequences[fam])))

                if not candidates:
                    continue  # Group has fewer families than splits - cannot cover all

                # Move the smallest candidate
                candidates.sort(key=lambda x: x[2])
                move_fam, from_idx, _ = candidates[0]

                # Remove from source
                splits[from_idx].remove(move_fam)
                split_seq_counts[from_idx] -= len(family_to_sequences[move_fam])
                split_arg_counts[from_idx] -= sum(1 for s in family_to_sequences[move_fam] if s['is_arg'] == 1)

                # Add to destination
                splits[missing_idx].append(move_fam)
                split_seq_counts[missing_idx] += len(family_to_sequences[move_fam])
                split_arg_counts[missing_idx] += sum(1 for s in family_to_sequences[move_fam] if s['is_arg'] == 1)

                # Update tracking for next missing split in same group
                split_has_group[from_idx] = any(family_labels.get(f) == group_key for f in splits[from_idx])
                split_has_group[missing_idx] = True

        if all_covered:
            break

    return splits[0], splits[1], splits[2]


def create_train_val_test_split(
    sequences: List[Dict],
    seq_to_family: Dict[int, int],
    output_dir: str,
    ratios: List[float] = None,
    rare_threshold: int = 3,
    seed: int = 42
) -> Dict:
    """
    Create a single train/val/test split with family-level homology constraint
    and rare category handling.
    """
    if ratios is None:
        ratios = [0.8, 0.1, 0.1]
    random.seed(seed)

    # Build family -> sequences mapping
    family_to_sequences = defaultdict(list)
    for seq in sequences:
        fam = seq_to_family.get(seq['_uid'])
        if fam is not None:
            family_to_sequences[fam].append(seq)

    # Identify rare categories
    rare_categories = identify_rare_categories(family_to_sequences, rare_threshold)
    if rare_categories:
        logger.info(f"Rare categories (\u003c{rare_threshold} families): {sorted(rare_categories)}")

    # Extract rare sequences
    rare_pool, common_families = extract_rare_sequences(family_to_sequences, rare_categories)
    logger.info(f"Rare pool: {len(rare_pool)} sequences extracted")
    logger.info(f"Common families: {len(common_families)} families remaining")

    # Split common families at family level
    common_family_ids = list(common_families.keys())
    train_fams, val_fams, test_fams = stratified_family_split(
        common_family_ids, common_families, ratios=ratios, seed=seed
    )

    # Split rare pool at sequence level
    rare_train, rare_val, rare_test = sequence_level_stratified_split(
        rare_pool, ratios=ratios, seed=seed + 1
    )

    # Collect sequences
    train_seqs = [s for fam in train_fams for s in common_families[fam]] + rare_train
    val_seqs = [s for fam in val_fams for s in common_families[fam]] + rare_val
    test_seqs = [s for fam in test_fams for s in common_families[fam]] + rare_test

    # Shuffle
    random.shuffle(train_seqs)
    random.shuffle(val_seqs)
    random.shuffle(test_seqs)

    # Save
    save_to_csv(train_seqs, os.path.join(output_dir, 'train.csv'))
    save_to_csv(val_seqs, os.path.join(output_dir, 'val.csv'))
    save_to_csv(test_seqs, os.path.join(output_dir, 'test.csv'))
    save_to_fasta(train_seqs, os.path.join(output_dir, 'train.fasta'))
    save_to_fasta(val_seqs, os.path.join(output_dir, 'val.fasta'))
    save_to_fasta(test_seqs, os.path.join(output_dir, 'test.fasta'))

    logger.info(f"Train: {len(train_seqs)} sequences ({sum(1 for s in train_seqs if s['is_arg']==1)} ARG)")
    logger.info(f"Val:   {len(val_seqs)} sequences ({sum(1 for s in val_seqs if s['is_arg']==1)} ARG)")
    logger.info(f"Test:  {len(test_seqs)} sequences ({sum(1 for s in test_seqs if s['is_arg']==1)} ARG)")

    # Build stats
    def _split_stats(seqs):
        return {
            'count': len(seqs),
            'arg_count': sum(1 for s in seqs if s['is_arg'] == 1),
            'non_arg_count': sum(1 for s in seqs if s['is_arg'] == 0),
            'num_families': len(set(seq_to_family.get(s['_uid']) for s in seqs if seq_to_family.get(s['_uid']) is not None))
        }

    stats = {
        'train': _split_stats(train_seqs),
        'val': _split_stats(val_seqs),
        'test': _split_stats(test_seqs),
    }

    return stats


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    positive_fasta: str = '/home/mayue/ARGMind/data/ARG_DB.fasta',
    negative_fasta: str = '/home/mayue/ARGMind/data/Non_ARG_DB.fasta',
    output_dir: str = '/home/mayue/ARGMind/data',
    n_splits: int = 5,
    stage1_min_seq_id: float = 0.30,
    stage1_coverage: float = 0.80,
    stage1_sensitivity: float = 6.0,
    stage2_num_iterations: int = 3,
    stage2_sensitivity: float = 7.5,
    stage2_coverage: float = 0.8,
    network_min_identity: float = 0.30,
    seed: int = 42
):
    """
    Create unified training dataset with two-stage homology-aware splitting.
    """
    if n_splits != 5:
        logger.warning(f"--n-splits={n_splits} is deprecated and ignored. Using single 80:10:10 split.")
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)

    logger.info("=" * 70)
    logger.info("Creating Unified Training Dataset (Two-Stage Homology-Aware)")
    logger.info("=" * 70)
    logger.info(f"  Stage 1: MMseqs2 cluster at {stage1_min_seq_id:.0%} identity / {stage1_coverage:.0%} coverage")
    logger.info(f"  Stage 2: MMseqs2 search ({stage2_num_iterations} iterations, sensitivity={stage2_sensitivity})")
    logger.info(f"  Network: Louvain clustering (min_edge_identity={network_min_identity:.0%})")
    logger.info(f"  Split: 80:10:10 train/val/test with rare category handling")
    logger.info(f"  Seed: {seed}")

    # ------------------------------------------------------------------
    # Step 1: Load sequences
    # ------------------------------------------------------------------
    logger.info(f"\n[Step 1] Loading positive samples from {positive_fasta}")
    positive_seqs, next_counter = parse_fasta_with_metadata(positive_fasta, is_arg=1)
    logger.info(f"  Loaded {len(positive_seqs)} ARG sequences")

    negative_seqs, _ = parse_fasta_with_metadata(
        negative_fasta, is_arg=0, default_category='non_arg', seq_counter_start=next_counter
    )
    logger.info(f"  Loaded {len(negative_seqs)} non-ARG sequences")
    pos_categories = Counter([s['category'] for s in positive_seqs])
    logger.info(f"  Categories: {len(pos_categories)}")
    for cat, count in pos_categories.most_common():
        logger.info(f"    {cat}: {count}")

    # Merge and assign UIDs
    all_sequences = positive_seqs + negative_seqs
    for idx, seq in enumerate(all_sequences):
        seq['_uid'] = idx
    logger.info(f"\n[Step 1] Total sequences: {len(all_sequences)} (ARG: {len(positive_seqs)}, non-ARG: {len(negative_seqs)})")

    # Write merged FASTA with integer UIDs for Stage 1
    merged_fasta = os.path.join(output_dir, 'merged_with_uids.fasta')
    with open(merged_fasta, 'w') as f:
        for seq in all_sequences:
            f.write(f">{seq['_uid']}\n")
            for i in range(0, len(seq['sequence']), 60):
                f.write(seq['sequence'][i:i+60] + '\n')

    # ------------------------------------------------------------------
    # Stage 1: MMseqs2 clustering
    # ------------------------------------------------------------------
    seq_to_cluster, cluster_to_rep, reps_fasta = run_mmseqs_stage1(
        merged_fasta, output_dir,
        min_seq_id=stage1_min_seq_id,
        coverage=stage1_coverage,
        sensitivity=stage1_sensitivity
    )

    # ------------------------------------------------------------------
    # Stage 2: Sensitive search + Network + Louvain
    # ------------------------------------------------------------------
    m8_file = run_mmseqs_stage2(
        reps_fasta, output_dir,
        num_iterations=stage2_num_iterations,
        sensitivity=stage2_sensitivity,
        coverage=stage2_coverage
    )

    network = build_homology_network(m8_file, cluster_to_rep, min_identity=network_min_identity)
    cluster_to_family = run_louvain_clustering(network, seed=seed)
    seq_to_family = build_seq_to_family(seq_to_cluster, cluster_to_family)

    # ------------------------------------------------------------------
    # Stage 3: Create train/val/test split with rare category handling
    # ------------------------------------------------------------------
    logger.info(f"\n[Stage 3] Creating train/val/test split (80:10:10) with rare category handling")
    split_stats = create_train_val_test_split(
        all_sequences, seq_to_family, output_dir,
        ratios=[0.8, 0.1, 0.1],
        rare_threshold=3,
        seed=seed
    )

    # ------------------------------------------------------------------
    # Compute and save statistics
    # ------------------------------------------------------------------
    stats = {
        'total_sequences': len(all_sequences),
        'positive_samples': len(positive_seqs),
        'negative_samples': len(negative_seqs),
        'categories': list(pos_categories.keys()),
        'num_categories': len(pos_categories),
        'stage1': {
            'tool': 'mmseqs2',
            'min_seq_id': stage1_min_seq_id,
            'coverage': stage1_coverage,
            'num_clusters': len(cluster_to_rep)
        },
        'stage2': {
            'tool': 'mmseqs2',
            'num_iterations': stage2_num_iterations,
            'sensitivity': stage2_sensitivity,
            'network_nodes': network.number_of_nodes(),
            'network_edges': network.number_of_edges(),
            'num_families': len(set(cluster_to_family.values()))
        },
        'split': split_stats
    }

    stats_path = os.path.join(output_dir, 'training_data_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"\nStatistics saved to {stats_path}")
    logger.info("\n" + "=" * 70)
    logger.info("Training dataset creation complete!")
    logger.info("=" * 70)

    return stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create unified training dataset with two-stage homology-aware splitting"
    )
    parser.add_argument("--positive-fasta", type=str, default="/home/mayue/ARGMind/data/ARG_DB.fasta")
    parser.add_argument("--negative-fasta", type=str, default="/home/mayue/ARGMind/data/Non_ARG_DB.fasta")
    parser.add_argument("--output-dir", type=str, default="/home/mayue/ARGMind/data")
    parser.add_argument("--n-splits", type=int, default=5, help="Deprecated: no longer used (kept for backward compatibility)")
    parser.add_argument("--stage1-min-seq-id", type=float, default=0.30,
                        help="Stage 1 MMseqs2 clustering identity threshold (default: 0.30)")
    parser.add_argument("--stage1-coverage", type=float, default=0.80)
    parser.add_argument("--stage1-sensitivity", type=float, default=6.0)
    parser.add_argument("--stage2-num-iterations", type=int, default=3,
                        help="Stage 2 MMseqs2 search iterations (default: 3)")
    parser.add_argument("--stage2-sensitivity", type=float, default=7.5,
                        help="Stage 2 MMseqs2 search sensitivity (default: 7.5)")
    parser.add_argument("--stage2-coverage", type=float, default=0.80)
    parser.add_argument("--network-min-identity", type=float, default=0.30,
                        help="Minimum edge identity for homology network (default: 0.30)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(
        positive_fasta=args.positive_fasta,
        negative_fasta=args.negative_fasta,
        output_dir=args.output_dir,
        n_splits=args.n_splits,
        stage1_min_seq_id=args.stage1_min_seq_id,
        stage1_coverage=args.stage1_coverage,
        stage1_sensitivity=args.stage1_sensitivity,
        stage2_num_iterations=args.stage2_num_iterations,
        stage2_sensitivity=args.stage2_sensitivity,
        stage2_coverage=args.stage2_coverage,
        network_min_identity=args.network_min_identity,
        seed=args.seed,
    )
