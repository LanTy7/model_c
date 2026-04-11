"""
Refine negative samples using MCT-ARG methodology:
1. DIAMOND screening against ARG database (≥30% id, ≥80% cov, E≤1e-10)
2. CD-HIT dereplication (90% identity)
3. Final balanced negative dataset
"""

import os
import subprocess
import logging
import json
import random
from typing import List, Dict

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def run_diamond_blastx(
    query_fasta: str,
    db_fasta: str,
    output_dir: str = './data',
    identity_threshold: float = 30.0,
    coverage_threshold: float = 80.0,
    evalue_threshold: float = 1e-10
) -> set:
    """
    Run DIAMOND blastx to identify ARG-like sequences.
    Returns set of sequence IDs to exclude.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Build DIAMOND database from ARG sequences
    db_path = os.path.join(output_dir, 'arg_db.dmnd')

    logger.info("Building DIAMOND database from ARG sequences...")
    cmd = f"diamond makedb --in {db_fasta} -d {db_path} --threads 8"
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"DIAMOND makedb failed: {result.stderr}")
        raise RuntimeError("DIAMOND database creation failed")

    # Run DIAMOND blastp (protein vs protein)
    blast_out = os.path.join(output_dir, 'negative_vs_arg.diamond.out')

    logger.info("Running DIAMOND blastp (negative samples vs ARG database)...")
    cmd = (
        f"diamond blastp -d {db_path} -q {query_fasta} "
        f"-o {blast_out} --threads 8 "
        f"--id {identity_threshold} --query-cover {coverage_threshold} "
        f"--evalue {evalue_threshold} "
        f"--outfmt 6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore qlen slen"
    )

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"DIAMOND blastx failed: {result.stderr}")
        raise RuntimeError("DIAMOND blastx failed")

    # Parse results to get sequences to exclude
    excluded_ids = set()
    if os.path.exists(blast_out) and os.path.getsize(blast_out) > 0:
        with open(blast_out, 'r') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 13:
                    query_id = parts[0]
                    excluded_ids.add(query_id)

    logger.info(f"DIAMOND screening: {len(excluded_ids)} sequences flagged as ARG-like")

    # Clean up
    if os.path.exists(blast_out):
        os.remove(blast_out)
    if os.path.exists(db_path):
        os.remove(db_path)

    return excluded_ids


def run_cd_hit(
    input_fasta: str,
    output_fasta: str,
    identity_threshold: float = 0.90
):
    """
    Run CD-HIT to remove redundant sequences.
    """
    logger.info(f"Running CD-HIT dereplication (identity threshold: {identity_threshold})...")

    # CD-HIT adds .clstr suffix automatically
    cmd = (
        f"cd-hit -i {input_fasta} -o {output_fasta} "
        f"-c {identity_threshold} -n 5 -d 0 -M 16000 -T 8"
    )

    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error(f"CD-HIT failed: {result.stderr}")
        raise RuntimeError("CD-HIT dereplication failed")

    # Count clusters
    cluster_file = output_fasta + '.clstr'
    cluster_count = 0
    if os.path.exists(cluster_file):
        with open(cluster_file, 'r') as f:
            for line in f:
                if line.startswith('>Cluster'):
                    cluster_count += 1

    logger.info(f"CD-HIT complete: {cluster_count} clusters retained")

    # Clean up cluster file
    if os.path.exists(cluster_file):
        os.remove(cluster_file)


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
    negative_fasta: str = './data/negative_samples.fasta',
    arg_fasta: str = './data/qc_retained.fasta',
    uniprot_gz: str = './data/uniprot_sprot.fasta.gz',
    output_dir: str = './data',
    target_count: int = 17242,
    seed: int = 42
):
    """
    Refine negative samples using MCT-ARG methodology.
    """
    random.seed(seed)

    logger.info("=" * 60)
    logger.info("Refining Negative Samples (MCT-ARG Methodology)")
    logger.info("=" * 60)

    # Step 1: Load all negative samples
    logger.info(f"\nStep 1: Loading negative samples from {negative_fasta}")
    all_negatives = parse_fasta(negative_fasta)
    logger.info(f"Loaded {len(all_negatives)} candidate negative samples")

    # Step 2: DIAMOND screening against ARG database
    logger.info(f"\nStep 2: DIAMOND screening against ARG database ({arg_fasta})")
    logger.info("Parameters: ≥30% identity, ≥80% coverage, E-value ≤ 1e-10")

    excluded_ids = run_diamond_blastx(negative_fasta, arg_fasta, output_dir)

    # Filter out ARG-like sequences
    filtered_negatives = [s for s in all_negatives if s['id'] not in excluded_ids]
    logger.info(f"After DIAMOND filtering: {len(filtered_negatives)} sequences retained")
    logger.info(f"  Excluded: {len(all_negatives) - len(filtered_negatives)} ARG-like sequences")

    # Step 3: Save intermediate FASTA for CD-HIT
    temp_fasta = os.path.join(output_dir, 'temp_negatives_filtered.fasta')
    save_fasta(filtered_negatives, temp_fasta)

    # Step 4: CD-HIT dereplication (90% identity)
    logger.info(f"\nStep 3: CD-HIT dereplication (90% identity threshold)")
    cdhit_output = os.path.join(output_dir, 'temp_negatives_cdhit.fasta')
    run_cd_hit(temp_fasta, cdhit_output, identity_threshold=0.90)

    # Step 5: Load CD-HIT output
    dedup_negatives = parse_fasta(cdhit_output)
    logger.info(f"After CD-HIT: {len(dedup_negatives)} non-redundant sequences")

    # Clean up temp files
    if os.path.exists(temp_fasta):
        os.remove(temp_fasta)
    if os.path.exists(cdhit_output):
        os.remove(cdhit_output)

    # Step 6: Random sampling to target count
    if len(dedup_negatives) < target_count:
        logger.warning(f"Only {len(dedup_negatives)} negative samples available (target: {target_count})")
        final_negatives = dedup_negatives
    else:
        final_negatives = random.sample(dedup_negatives, target_count)
        logger.info(f"\nStep 4: Randomly selected {target_count} sequences")

    random.shuffle(final_negatives)

    # Step 7: Save final output
    final_fasta = os.path.join(output_dir, 'negative_samples_refined.fasta')
    final_csv = os.path.join(output_dir, 'negative_samples_refined.csv')

    save_fasta(final_negatives, final_fasta)
    save_csv(final_negatives, final_csv)

    logger.info(f"\nFinal refined negative samples saved:")
    logger.info(f"  FASTA: {final_fasta}")
    logger.info(f"  CSV: {final_csv}")

    # Statistics
    lengths = [len(s['sequence']) for s in final_negatives]
    stats = {
        'original_candidates': len(all_negatives),
        'diamond_excluded': len(excluded_ids),
        'cdhit_clusters': len(dedup_negatives),
        'final_count': len(final_negatives),
        'length_stats': {
            'mean': sum(lengths) / len(lengths),
            'min': min(lengths),
            'max': max(lengths)
        },
        'parameters': {
            'diamond_identity': 30,
            'diamond_coverage': 80,
            'diamond_evalue': '1e-10',
            'cdhit_identity': 0.90
        }
    }

    stats_path = os.path.join(output_dir, 'negative_samples_refined_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"\nStatistics saved to {stats_path}")
    logger.info(f"\nLength distribution:")
    logger.info(f"  Mean: {stats['length_stats']['mean']:.1f}")
    logger.info(f"  Min: {stats['length_stats']['min']}")
    logger.info(f"  Max: {stats['length_stats']['max']}")

    logger.info("\n" + "=" * 60)
    logger.info("Negative sample refinement complete!")
    logger.info("=" * 60)

    return final_negatives


if __name__ == "__main__":
    main(
        negative_fasta='./data/negative_samples.fasta',
        arg_fasta='./data/qc_retained.fasta',
        output_dir='./data',
        target_count=17242,
        seed=42
    )
