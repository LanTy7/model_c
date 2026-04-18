"""
Create unified training dataset with two-stage homology-aware splitting.
Based on DefensePredictor methodology (Science paper).

Key improvements over original CD-HIT approach:
1. MMseqs2 at 30% identity for redundancy reduction (Stage 1)
2. Sensitive all-vs-all profile search + Louvain network clustering (Stage 2)
3. 5-fold GroupKFold with family-level stratification
4. Dedicated novelty test set for evaluating novel ARG detection

Pipeline:
  Stage 1: MMseqs2 cluster at 30% identity / 80% coverage (redundancy removal)
  Stage 2: MMseqs2 all-vs-all search on cluster reps (--num-iterations 3, -s 7.5)
  Stage 3: Build homology network + Louvain clustering -> "families"
  Stage 4: Determine novelty test set (families with max identity < 30% to training pool)
  Stage 5: 5-fold GroupKFold on remaining families with stratified balancing
  Output:  fold_0/..fold_4/ (train/val/test) + novelty_test.csv + stats.json
"""

import argparse
import csv
import os
import re
import json
import logging
import random
import shutil
import subprocess
import tempfile
from typing import List, Dict, Tuple, Set
from collections import defaultdict, Counter

import networkx as nx
import community as community_louvain

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# FASTA / metadata helpers (unchanged from original)
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

    # Use provided seq_id (guaranteed unique) or fall back to header parsing
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

            # Use rep_uid directly as cluster_id
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
    Run all-vs-all MMseqs2 search on cluster representatives.

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
    min_identity: float = 0.2
) -> nx.Graph:
    """
    Build homology network from MMseqs2 search results.
    Nodes = cluster IDs (same as rep UIDs), edges = homology links.
    """
    G = nx.Graph()
    # Add all clusters as nodes
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

            if pident < min_identity * 100:  # m8 pident is 0-100
                continue

            # Skip self-hits
            if query_uid == target_uid:
                continue

            # Find cluster IDs for these UIDs
            # cluster_to_rep is {cluster_id: rep_uid}, so we need reverse mapping
            # Actually query_uid and target_uid are rep UIDs = cluster IDs
            # (since search was run on reps, and reps have UIDs as headers)
            if query_uid in cluster_to_rep and target_uid in cluster_to_rep:
                q_cluster = query_uid
                t_cluster = target_uid
                # Add edge with weight = pident
                if not G.has_edge(q_cluster, t_cluster):
                    G.add_edge(q_cluster, t_cluster, weight=pident)
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
    random_state = random.Random(seed)
    # Louvain doesn't accept a seed directly, but we can set Python random seed
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
# Stage 4: Determine novelty test set
# ---------------------------------------------------------------------------

def determine_novelty_test_set(
    cluster_to_family: Dict[int, int],
    cluster_to_rep: Dict[int, int],
    network: nx.Graph,
    sequences: List[Dict],
    seq_to_family: Dict[int, int],
    novelty_threshold: float = 0.30,
    novelty_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[Set[int], Set[int], Dict[int, float]]:
    """
    Determine which families go to novelty test set.

    Strategy:
    1. Sort families by size (descending)
    2. Reserve largest families as "training pool" (to ensure training diversity)
    3. For remaining families, compute max edge weight to training pool
    4. Families with max identity < novelty_threshold are "novelty candidates"
    5. Randomly select novelty_ratio of candidates as final novelty test set

    Returns:
        novelty_families: set of family IDs
        cv_families: set of family IDs for 5-fold CV
        family_max_identity: {family_id: max_identity_to_training_pool}
    """
    random.seed(seed)

    # Count family sizes
    family_sizes = Counter(seq_to_family.values())
    all_families = set(family_sizes.keys())

    # Sort families by size descending
    sorted_families = sorted(all_families, key=lambda f: family_sizes[f], reverse=True)

    # Reserve largest families as training pool (~70% of sequences)
    total_seqs = sum(family_sizes.values())
    training_pool_size = int(total_seqs * 0.70)
    training_pool = set()
    pool_seq_count = 0
    for fam in sorted_families:
        training_pool.add(fam)
        pool_seq_count += family_sizes[fam]
        if pool_seq_count >= training_pool_size:
            break

    remaining_families = all_families - training_pool
    logger.info(f"[Novelty] Training pool: {len(training_pool)} families ({pool_seq_count} seqs)")
    logger.info(f"[Novelty] Remaining families: {len(remaining_families)} families ({total_seqs - pool_seq_count} seqs)")

    # Compute max identity from each remaining family to training pool
    family_max_identity = {}
    for fam in remaining_families:
        # Get all clusters in this family
        family_clusters = [cid for cid, f in cluster_to_family.items() if f == fam]
        max_id = 0.0
        for cid in family_clusters:
            for neighbor in network.neighbors(cid):
                neighbor_family = cluster_to_family.get(neighbor, neighbor)
                if neighbor_family in training_pool:
                    weight = network[cid][neighbor].get('weight', 0)
                    max_id = max(max_id, weight / 100.0)  # convert from 0-100 to 0-1
        family_max_identity[fam] = max_id

    # Novelty candidates: families with max identity < threshold
    novelty_candidates = [f for f in remaining_families if family_max_identity[f] < novelty_threshold]
    logger.info(f"[Novelty] Candidates (max_identity < {novelty_threshold:.0%}): {len(novelty_candidates)} families")

    # Select novelty test set
    if len(novelty_candidates) == 0:
        logger.warning("No novelty candidates found! Using families with lowest max identity.")
        sorted_by_identity = sorted(remaining_families, key=lambda f: family_max_identity[f])
        n_novelty = max(1, int(len(remaining_families) * novelty_ratio))
        novelty_families = set(sorted_by_identity[:n_novelty])
    else:
        n_novelty = max(1, int(len(novelty_candidates) * novelty_ratio))
        novelty_families = set(random.sample(novelty_candidates, min(n_novelty, len(novelty_candidates))))

    cv_families = remaining_families - novelty_families

    novelty_seq_count = sum(family_sizes[f] for f in novelty_families)
    cv_seq_count = sum(family_sizes[f] for f in cv_families)
    logger.info(f"[Novelty] Final novelty test set: {len(novelty_families)} families ({novelty_seq_count} seqs)")
    logger.info(f"[Novelty] CV families: {len(cv_families)} families ({cv_seq_count} seqs)")

    return novelty_families, cv_families, family_max_identity


# ---------------------------------------------------------------------------
# Stage 5: 5-fold GroupKFold with stratified balancing
# ---------------------------------------------------------------------------

def stratified_family_split(
    families: List[int],
    family_to_sequences: Dict[int, List[Dict]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> Tuple[List[int], List[int], List[int]]:
    """
    Split a list of families into train/val/test with stratification.
    Stratification key = (is_arg, category) based on majority label in family.
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6 or test_ratio == 0.0
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

    train_fams, val_fams, test_fams = [], [], []

    for group_key, group_fams in group_to_fams.items():
        n = len(group_fams)
        if n < 3:
            train_fams.extend(group_fams)
            logger.warning(f"Group {group_key} has only {n} families, all to train")
            continue

        shuffled = group_fams.copy()
        random.shuffle(shuffled)

        n_train = max(1, int(n * train_ratio))
        n_val = max(1, int(n * val_ratio))

        if test_ratio > 0:
            n_test = max(1, n - n_train - n_val)
            while n_train + n_val + n_test > n:
                if n_train > n_val and n_train > n_test and n_train > 1:
                    n_train -= 1
                elif n_val > n_test and n_val > 1:
                    n_val -= 1
                elif n_test > 1:
                    n_test -= 1
                else:
                    break
            test_fams.extend(shuffled[n_train + n_val:n_train + n_val + n_test])
        else:
            # No test set: redistribute remaining to train/val
            while n_train + n_val > n:
                if n_train > n_val and n_train > 1:
                    n_train -= 1
                elif n_val > 1:
                    n_val -= 1
                else:
                    break
            # Any remainder goes to train
            n_train = n - n_val

        train_fams.extend(shuffled[:n_train])
        val_fams.extend(shuffled[n_train:n_train + n_val])

    return train_fams, val_fams, test_fams


def create_groupkfold_splits(
    sequences: List[Dict],
    seq_to_family: Dict[int, int],
    cv_families: Set[int],
    n_splits: int = 5,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42
) -> List[Dict[str, List[Dict]]]:
    """
    Create 5-fold splits using GroupKFold-like approach.

    Each fold contains train/val/test, where test comes from one fold-group
    and train/val come from the remaining groups.
    """
    random.seed(seed)

    # Build family -> sequences mapping for CV families only
    family_to_sequences = defaultdict(list)
    for seq in sequences:
        fam = seq_to_family.get(seq['_uid'])
        if fam in cv_families:
            family_to_sequences[fam].append(seq)

    cv_fam_list = sorted(list(cv_families))

    # Determine stratification key for each family
    family_labels = {}
    for fam in cv_fam_list:
        seqs = family_to_sequences[fam]
        labels = [(s['is_arg'], s['category']) for s in seqs]
        majority = Counter(labels).most_common(1)[0][0]
        family_labels[fam] = majority

    # Group families by stratification key, then create n_splits groups
    group_to_fams = defaultdict(list)
    for fam in cv_fam_list:
        group_to_fams[family_labels[fam]].append(fam)

    # For each stratification group, split into n_splits chunks
    fold_chunks = [[] for _ in range(n_splits)]
    for group_key, fams in group_to_fams.items():
        random.shuffle(fams)
        for i, fam in enumerate(fams):
            fold_chunks[i % n_splits].append(fam)

    # Build folds
    folds = []
    for i in range(n_splits):
        test_fams = set(fold_chunks[i])
        train_val_fams = []
        for j in range(n_splits):
            if j != i:
                train_val_fams.extend(fold_chunks[j])

        # Split train_val into train and val
        train_fams, val_fams, _ = stratified_family_split(
            train_val_fams, family_to_sequences,
            train_ratio=train_ratio / (train_ratio + val_ratio),
            val_ratio=val_ratio / (train_ratio + val_ratio),
            test_ratio=0.0,
            seed=seed + i
        )

        # Collect sequences
        train_seqs = []
        for fam in train_fams:
            train_seqs.extend(family_to_sequences[fam])

        val_seqs = []
        for fam in val_fams:
            val_seqs.extend(family_to_sequences[fam])

        test_seqs = []
        for fam in test_fams:
            test_seqs.extend(family_to_sequences[fam])

        random.shuffle(train_seqs)
        random.shuffle(val_seqs)
        random.shuffle(test_seqs)

        folds.append({
            'train': train_seqs,
            'val': val_seqs,
            'test': test_seqs,
            'train_families': set(train_fams),
            'val_families': set(val_fams),
            'test_families': test_fams
        })

    return folds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    positive_fasta: str = '/home/mayue/ARGMind/data/ARG_DB.fasta',
    negative_fasta: str = '/home/mayue/ARGMind/data/Non_ARG_DB.fasta',
    output_dir: str = '/home/mayue/ARGMind/data',
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    n_splits: int = 5,
    stage1_min_seq_id: float = 0.30,
    stage1_coverage: float = 0.80,
    stage1_sensitivity: float = 6.0,
    stage2_num_iterations: int = 3,
    stage2_sensitivity: float = 7.5,
    stage2_coverage: float = 0.8,
    network_min_identity: float = 0.20,
    novelty_threshold: float = 0.30,
    novelty_ratio: float = 0.15,
    seed: int = 42
):
    """
    Create unified training dataset with two-stage homology-aware splitting.
    """
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)

    logger.info("=" * 70)
    logger.info("Creating Unified Training Dataset (Two-Stage Homology-Aware)")
    logger.info("=" * 70)
    logger.info(f"  Stage 1: MMseqs2 cluster at {stage1_min_seq_id:.0%} identity / {stage1_coverage:.0%} coverage")
    logger.info(f"  Stage 2: MMseqs2 search ({stage2_num_iterations} iterations, sensitivity={stage2_sensitivity})")
    logger.info(f"  Network: Louvain clustering (min_edge_identity={network_min_identity:.0%})")
    logger.info(f"  Split: {n_splits}-fold GroupKFold + novelty test set (threshold={novelty_threshold:.0%})")
    logger.info(f"  Seed: {seed}")

    # ------------------------------------------------------------------
    # Step 1: Load sequences
    # ------------------------------------------------------------------
    logger.info(f"\n[Step 1] Loading positive samples from {positive_fasta}")
    positive_seqs, next_counter = parse_fasta_with_metadata(positive_fasta, is_arg=1)
    logger.info(f"  Loaded {len(positive_seqs)} ARG sequences")

    # Load negative samples with continuing counter to ensure global uniqueness
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
    logger.info(f"\n[Step 3] Total sequences: {len(all_sequences)} (ARG: {len(positive_seqs)}, non-ARG: {len(negative_seqs)})")

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
    # Stage 4: Determine novelty test set
    # ------------------------------------------------------------------
    novelty_families, cv_families, family_max_identity = determine_novelty_test_set(
        cluster_to_family, cluster_to_rep, network,
        all_sequences, seq_to_family,
        novelty_threshold=novelty_threshold,
        novelty_ratio=novelty_ratio,
        seed=seed
    )

    # ------------------------------------------------------------------
    # Stage 5: 5-fold GroupKFold
    # ------------------------------------------------------------------
    folds = create_groupkfold_splits(
        all_sequences, seq_to_family, cv_families,
        n_splits=n_splits,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed
    )

    # ------------------------------------------------------------------
    # Collect novelty test sequences
    # ------------------------------------------------------------------
    novelty_seqs = [seq for seq in all_sequences if seq_to_family.get(seq['_uid']) in novelty_families]
    random.shuffle(novelty_seqs)

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    logger.info(f"\n[Step 6] Saving outputs to {output_dir}/")

    # Save each fold
    for i, fold in enumerate(folds):
        fold_dir = os.path.join(output_dir, f'fold_{i}')
        os.makedirs(fold_dir, exist_ok=True)
        save_to_csv(fold['train'], os.path.join(fold_dir, 'train.csv'))
        save_to_csv(fold['val'], os.path.join(fold_dir, 'val.csv'))
        save_to_csv(fold['test'], os.path.join(fold_dir, 'test.csv'))
        save_to_fasta(fold['train'], os.path.join(fold_dir, 'train.fasta'))
        save_to_fasta(fold['val'], os.path.join(fold_dir, 'val.fasta'))
        save_to_fasta(fold['test'], os.path.join(fold_dir, 'test.fasta'))
        logger.info(f"  Fold {i}: train={len(fold['train'])}, val={len(fold['val'])}, test={len(fold['test'])}")

    # Save novelty test set
    if novelty_seqs:
        save_to_csv(novelty_seqs, os.path.join(output_dir, 'novelty_test.csv'))
        save_to_fasta(novelty_seqs, os.path.join(output_dir, 'novelty_test.fasta'))
        logger.info(f"  Novelty test: {len(novelty_seqs)} sequences")

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
        'novelty_test': {
            'threshold': novelty_threshold,
            'num_families': len(novelty_families),
            'num_sequences': len(novelty_seqs),
            'family_max_identity': {str(k): round(v, 4) for k, v in family_max_identity.items() if k in novelty_families}
        },
        'folds': []
    }

    for i, fold in enumerate(folds):
        fold_stats = {
            'fold': i,
            'train': {
                'count': len(fold['train']),
                'arg_count': sum(1 for s in fold['train'] if s['is_arg'] == 1),
                'non_arg_count': sum(1 for s in fold['train'] if s['is_arg'] == 0),
                'num_families': len(fold['train_families'])
            },
            'val': {
                'count': len(fold['val']),
                'arg_count': sum(1 for s in fold['val'] if s['is_arg'] == 1),
                'non_arg_count': sum(1 for s in fold['val'] if s['is_arg'] == 0),
                'num_families': len(fold['val_families'])
            },
            'test': {
                'count': len(fold['test']),
                'arg_count': sum(1 for s in fold['test'] if s['is_arg'] == 1),
                'non_arg_count': sum(1 for s in fold['test'] if s['is_arg'] == 0),
                'num_families': len(fold['test_families'])
            }
        }
        stats['folds'].append(fold_stats)

    stats_path = os.path.join(output_dir, 'training_data_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"\nStatistics saved to {stats_path}")
    logger.info("\n" + "=" * 70)
    logger.info("Training dataset creation complete!")
    logger.info("=" * 70)

    return folds, novelty_seqs, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create unified training dataset with two-stage homology-aware splitting"
    )
    parser.add_argument("--positive-fasta", type=str, default="/home/mayue/ARGMind/data/ARG_DB.fasta")
    parser.add_argument("--negative-fasta", type=str, default="/home/mayue/ARGMind/data/Non_ARG_DB.fasta")
    parser.add_argument("--output-dir", type=str, default="/home/mayue/ARGMind/data")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--n-splits", type=int, default=5, help="Number of folds for cross-validation")
    parser.add_argument("--stage1-min-seq-id", type=float, default=0.30,
                        help="Stage 1 MMseqs2 clustering identity threshold (default: 0.30)")
    parser.add_argument("--stage1-coverage", type=float, default=0.80)
    parser.add_argument("--stage1-sensitivity", type=float, default=6.0)
    parser.add_argument("--stage2-num-iterations", type=int, default=3,
                        help="Stage 2 MMseqs2 search iterations (default: 3)")
    parser.add_argument("--stage2-sensitivity", type=float, default=7.5,
                        help="Stage 2 MMseqs2 search sensitivity (default: 7.5)")
    parser.add_argument("--stage2-coverage", type=float, default=0.80)
    parser.add_argument("--network-min-identity", type=float, default=0.20,
                        help="Minimum edge identity for homology network (default: 0.20)")
    parser.add_argument("--novelty-threshold", type=float, default=0.30,
                        help="Max identity threshold for novelty test set families (default: 0.30)")
    parser.add_argument("--novelty-ratio", type=float, default=0.15,
                        help="Ratio of novelty candidates to select for novelty test set")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    main(
        positive_fasta=args.positive_fasta,
        negative_fasta=args.negative_fasta,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        n_splits=args.n_splits,
        stage1_min_seq_id=args.stage1_min_seq_id,
        stage1_coverage=args.stage1_coverage,
        stage1_sensitivity=args.stage1_sensitivity,
        stage2_num_iterations=args.stage2_num_iterations,
        stage2_sensitivity=args.stage2_sensitivity,
        stage2_coverage=args.stage2_coverage,
        network_min_identity=args.network_min_identity,
        novelty_threshold=args.novelty_threshold,
        novelty_ratio=args.novelty_ratio,
        seed=args.seed,
    )
