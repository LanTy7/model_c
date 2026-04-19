"""
Create unified training dataset with two-stage homology-aware splitting.
Based on DefensePredictor methodology (Science paper).

Key improvements over original CD-HIT approach:
1. MMseqs2 at 30% identity for redundancy reduction (Stage 1)
2. Sensitive all-vs-all profile search + Louvain network clustering (Stage 2)
3. 5-fold GroupKFold with family-level stratification
   - Each model trains on 4 folds (split into train/val), tests on 1 held-out fold
   - Homologous families never cross splits
4. Stratified balancing by (is_arg, category) to maintain class distribution

Pipeline:
  Stage 1: MMseqs2 cluster at 30% identity / 80% coverage (redundancy removal)
  Stage 2: MMseqs2 all-vs-all search on cluster reps (--num-iterations 3, -s 7.5)
  Stage 3: Build homology network + Louvain clustering -> "families"
  Stage 4: 5-fold GroupKFold on all families with stratified balancing
           Each fold: test = 1 fold-group, train+val = remaining 4 fold-groups
  Output:  fold_0/..fold_4/ (train/val/test) + stats.json
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
# Stage 4: 5-fold GroupKFold with stratified balancing
# ---------------------------------------------------------------------------

def stratified_family_split(
    families: List[int],
    family_to_sequences: Dict[int, List[Dict]],
    train_ratio: float = 0.8,
    val_ratio: float = 0.2,
    seed: int = 42
) -> Tuple[List[int], List[int]]:
    """
    Split a list of families into train/val with stratification.
    Stratification key = (is_arg, category) based on majority label in family.
    Uses greedy balancing to respect target ratio while keeping stratification.
    """
    assert abs(train_ratio + val_ratio - 1.0) < 1e-6
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

    train_fams, val_fams = [], []

    for group_key, group_fams in group_to_fams.items():
        n = len(group_fams)
        if n < 2:
            train_fams.extend(group_fams)
            continue

        # Sort families by size descending for greedy assignment
        fams_sorted = sorted(group_fams, key=lambda f: len(family_to_sequences[f]), reverse=True)

        # Greedy split into train and val respecting target ratio
        target_train_ratio = train_ratio / (train_ratio + val_ratio)
        train_size, val_size = 0, 0
        for fam in fams_sorted:
            fam_seqs = family_to_sequences[fam]
            fam_size = len(fam_seqs)
            if train_size == 0:
                train_fams.append(fam)
                train_size += fam_size
            else:
                current_train_ratio = train_size / (train_size + val_size)
                if current_train_ratio < target_train_ratio:
                    train_fams.append(fam)
                    train_size += fam_size
                else:
                    val_fams.append(fam)
                    val_size += fam_size

    return train_fams, val_fams


def create_groupkfold_splits(
    sequences: List[Dict],
    seq_to_family: Dict[int, int],
    n_splits: int = 5,
    train_val_seed: int = 42,
    seed: int = 42
) -> List[Dict[str, List[Dict]]]:
    """
    Create 5-fold splits using GroupKFold approach.

    Following DefensePredictor methodology:
    - All families are divided into 5 fold-groups using greedy balancing
    - Each fold i uses:
        * test = all sequences from fold-group i (1 fold = ~20% of data)
        * train + val = all sequences from the remaining 4 fold-groups (~80%)
        * Within train+val, split 80/20 into train and val using family-level stratification
    - Homologous families never cross splits

    Returns:
        List of 5 dicts, each with keys: 'train', 'val', 'test', 'train_families', 'val_families', 'test_families'
    """
    random.seed(seed)

    # Build family -> sequences mapping for all families
    family_to_sequences = defaultdict(list)
    for seq in sequences:
        fam = seq_to_family.get(seq['_uid'])
        if fam is not None:
            family_to_sequences[fam].append(seq)

    all_families = sorted(list(family_to_sequences.keys()))

    # Determine stratification key for each family
    family_labels = {}
    for fam in all_families:
        seqs = family_to_sequences[fam]
        labels = [(s['is_arg'], s['category']) for s in seqs]
        majority = Counter(labels).most_common(1)[0][0]
        family_labels[fam] = majority

    # Group families by stratification key, then create n_splits groups
    # Use greedy balancing to keep each chunk roughly equal in size and pos count
    group_to_fams = defaultdict(list)
    for fam in all_families:
        group_to_fams[family_labels[fam]].append(fam)

    fold_chunks = [[] for _ in range(n_splits)]
    fold_chunk_pos_counts = [0 for _ in range(n_splits)]
    fold_chunk_total_counts = [0 for _ in range(n_splits)]

    for group_key, fams in group_to_fams.items():
        # Sort families by total sequence count descending (largest first)
        fams_sorted = sorted(fams, key=lambda f: len(family_to_sequences[f]), reverse=True)
        for fam in fams_sorted:
            fam_seqs = family_to_sequences[fam]
            fam_pos = sum(1 for s in fam_seqs if s['is_arg'] == 1)
            # Greedy: assign to chunk with smallest (pos_count + total_count) sum
            scores = [fold_chunk_pos_counts[i] + fold_chunk_total_counts[i] * 0.01 for i in range(n_splits)]
            best_idx = scores.index(min(scores))
            fold_chunks[best_idx].append(fam)
            fold_chunk_pos_counts[best_idx] += fam_pos
            fold_chunk_total_counts[best_idx] += len(fam_seqs)

    # Build folds
    folds = []
    for i in range(n_splits):
        test_fams = set(fold_chunks[i])
        train_val_fams = []
        for j in range(n_splits):
            if j != i:
                train_val_fams.extend(fold_chunks[j])

        # Split train_val into train and val (80/20 of the train_val portion)
        train_fams, val_fams = stratified_family_split(
            train_val_fams, family_to_sequences,
            train_ratio=0.8,
            val_ratio=0.2,
            seed=train_val_seed + i
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
    os.makedirs(output_dir, exist_ok=True)
    random.seed(seed)

    logger.info("=" * 70)
    logger.info("Creating Unified Training Dataset (Two-Stage Homology-Aware)")
    logger.info("=" * 70)
    logger.info(f"  Stage 1: MMseqs2 cluster at {stage1_min_seq_id:.0%} identity / {stage1_coverage:.0%} coverage")
    logger.info(f"  Stage 2: MMseqs2 search ({stage2_num_iterations} iterations, sensitivity={stage2_sensitivity})")
    logger.info(f"  Network: Louvain clustering (min_edge_identity={network_min_identity:.0%})")
    logger.info(f"  Split: {n_splits}-fold GroupKFold (test=1 fold, train+val=4 folds)")
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
    # Stage 3: 5-fold GroupKFold on ALL families
    # ------------------------------------------------------------------
    folds = create_groupkfold_splits(
        all_sequences, seq_to_family,
        n_splits=n_splits,
        seed=seed
    )

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    logger.info(f"\n[Step 4] Saving outputs to {output_dir}/")

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

    return folds, stats


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create unified training dataset with two-stage homology-aware splitting"
    )
    parser.add_argument("--positive-fasta", type=str, default="/home/mayue/ARGMind/data/ARG_DB.fasta")
    parser.add_argument("--negative-fasta", type=str, default="/home/mayue/ARGMind/data/Non_ARG_DB.fasta")
    parser.add_argument("--output-dir", type=str, default="/home/mayue/ARGMind/data")
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
