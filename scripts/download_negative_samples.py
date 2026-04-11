"""
Download and process UniProt Swiss-Prot as negative samples.
Excludes proteins related to antibiotic resistance, stress response, or efflux transport.
"""

import os
import gzip
import logging
import urllib.request
from typing import List, Dict, Set
from collections import Counter

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Keywords to exclude (antibiotic resistance related)
EXCLUDE_KEYWORDS = {
    # Antibiotic resistance
    'resistance', 'antibiotic', 'antimicrobial', 'resistant',
    'beta-lactamase', 'bet lactamase', 'carbapenemase',
    'aminoglycoside', 'tetracycline', 'macrolide', 'fluoroquinolone',
    'chloramphenicol', 'rifampicin', 'vancomycin', 'methicillin',
    'penicillin', 'cephalosporin', 'carbapenem', 'monobactam',
    'gentamicin', 'streptomycin', 'kanamycin', 'neomycin',
    'erythromycin', 'clarithromycin', 'azithromycin',
    'ciprofloxacin', 'levofloxacin', 'norfloxacin',
    'sulfonamide', 'trimethoprim', 'polymyxin', 'colistin',
    # Efflux pumps (often related to multidrug resistance)
    'efflux', 'multidrug', 'mdr', 'pump',
    # Stress response
    'stress', 'toxin', 'virulence',
    # Transposons and mobile elements (often carry ARGs)
    'transposon', 'transposase', 'integron', 'integrase',
}

# Valid amino acids including non-standard ones supported by model
VALID_AA = set('ACDEFGHIKLMNPQRSTVWYXBZJ')

# Eukaryotic keywords to exclude (for prokaryote filtering)
EUKARYOTE_KEYWORDS = {
    'human', 'mouse', 'arabidopsis', 'rat', 'bovine', 'yeast',
    'c. elegans', 's. pombe', 'dictyostelium', 'xenopus',
    'drosophila', 'zebrafish', 'chicken', 'pig', 'dog', 'cat',
    'rabbit', 'hamster', 'gorilla', 'chimpanzee', 'orangutan',
    'macaque', 'nematode', 'fission yeast', 'budding yeast',
    'slime mold', 'mold', 'fungi', 'fungal', 'plant', 'plants',
    'eukaryota', 'eukaryote', 'vertebrate', 'mammal', 'insect'
}


def download_uniprot_sprot(output_dir: str = './data') -> str:
    """Download UniProt Swiss-Prot FASTA file"""
    os.makedirs(output_dir, exist_ok=True)

    url = "https://ftp.uniprot.org/pub/databases/uniprot/current_release/knowledgebase/complete/uniprot_sprot.fasta.gz"
    output_path = os.path.join(output_dir, 'uniprot_sprot.fasta.gz')

    if os.path.exists(output_path):
        logger.info(f"File already exists: {output_path}")
        return output_path

    logger.info(f"Downloading UniProt Swiss-Prot from {url}")
    logger.info("This may take a few minutes...")

    # Download with progress
    def report_hook(count, block_size, total_size):
        percent = min(int(count * block_size * 100 / total_size), 100)
        if count % 100 == 0:  # Report every 100 blocks
            logger.info(f"Downloaded: {percent}%")

    try:
        urllib.request.urlretrieve(url, output_path, reporthook=report_hook)
        logger.info(f"Download complete: {output_path}")
        return output_path
    except Exception as e:
        logger.error(f"Download failed: {e}")
        raise


def is_resistance_related(header: str) -> bool:
    """Check if protein header contains resistance-related keywords"""
    header_lower = header.lower()
    return any(keyword in header_lower for keyword in EXCLUDE_KEYWORDS)


def extract_organism(header: str) -> str:
    """
    Extract organism name from UniProt FASTA header.
    UniProt format: >sp|accession|protein_name OS=Organism_name OX=...
    """
    # Look for OS= field (Organism Scientific name)
    if 'OS=' in header:
        # Extract text between OS= and the next OX= or GN= or PE= or SV=
        start = header.find('OS=') + 3
        end = len(header)
        for marker in [' OX=', ' GN=', ' PE=', ' SV=']:
            pos = header.find(marker, start)
            if pos != -1 and pos < end:
                end = pos
        return header[start:end].strip()
    return 'unknown'


def is_prokaryote(header: str, use_taxoniq: bool = True) -> bool:
    """
    Check if organism is a prokaryote (Bacteria or Archaea).

    Priority:
    1. Use taxoniq with NCBI Taxonomy (most accurate, requires taxoniq package)
    2. Fallback to keyword-based filtering if taxoniq fails

    When in doubt, EXCLUDE (return False) - safer to lose some sequences
    than include eukaryotes.
    """
    header_lower = header.lower()

    # Strategy 1: Use taxoniq if available (most accurate)
    if use_taxoniq:
        try:
            import taxoniq
            taxid = extract_taxid(header)
            if taxid:
                t = taxoniq.Taxon(taxid)
                for parent in t.ranked_lineage:
                    if parent.scientific_name == "Bacteria" or parent.scientific_name == "Archaea":
                        return True
                # Found in taxonomy but not Bacteria/Archaea
                return False
        except ImportError:
            # taxoniq not available, fall back to keywords
            pass
        except Exception:
            # taxoniq failed for this taxid, fall back to keywords
            pass

    # Strategy 2: Keyword-based fallback (less accurate but doesn't require taxoniq)
    organism = extract_organism(header).lower()

    # Check for eukaryotic keywords - exclude if found
    eukaryote_indicators = {
        'homo sapiens', 'human', 'mus musculus', 'mouse', 'rattus norvegicus', 'rat',
        'bos taurus', 'bovine', 'cattle', 'sus scrofa', 'pig', 'canis lupus', 'dog',
        'felis catus', 'cat', 'oryctolagus cuniculus', 'rabbit',
        'pan troglodytes', 'chimpanzee', 'gorilla gorilla', 'gorilla', 'pongo abelii', 'orangutan',
        'macaca mulatta', 'macaque', 'chlorocebus sabaeus', 'vervet',
        'gallus gallus', 'chicken', 'meleagris gallopavo', 'turkey',
        'danio rerio', 'zebrafish', 'xenopus laevis', 'xenopus tropicalis', 'xenopus',
        'drosophila melanogaster', 'drosophila', 'anopheles gambiae', 'mosquito',
        'caenorhabditis elegans', 'c. elegans',
        'arabidopsis thaliana', 'arabidopsis', 'oryza sativa', 'rice',
        'zea mays', 'maize', 'glycine max', 'soybean', 'solanum lycopersicum', 'tomato',
        'saccharomyces cerevisiae', 's. cerevisiae', 'brewer yeast', 'baker yeast',
        'schizosaccharomyces pombe', 's. pombe', 'fission yeast',
        'candida albicans', 'candida', 'aspergillus nidulans', 'aspergillus',
        'neurospora crassa', 'neurospora', 'dictyostelium discoideum', 'dictyostelium',
    }

    for indicator in eukaryote_indicators:
        if indicator in organism or indicator in header_lower:
            return False

    # Check for known prokaryotic genus names
    prokaryotic_genera = {
        'escherichia', 'salmonella', 'shigella', 'yersinia', 'klebsiella',
        'pseudomonas', 'acinetobacter', 'burkholderia',
        'bacillus', 'staphylococcus', 'streptococcus', 'enterococcus',
        'lactobacillus', 'corynebacterium', 'mycobacterium', 'streptomyces',
        'clostridium', 'bacteroides', 'helicobacter', 'campylobacter',
        'vibrio', 'neisseria', 'haemophilus', 'borrelia', 'treponema',
        'chlamydia', 'mycoplasma', 'thermus', 'rhizobium', 'agrobacterium',
        'azotobacter', 'cyanobacterium', 'synechocystis', 'nostoc',
        'thermococcus', 'methanococcus', 'methanobacterium',
        'sulfolobus', 'archaeoglobus', 'halobacterium', 'haloferax',
    }

    for genus in prokaryotic_genera:
        if genus in organism:
            return True

    # Default: EXCLUDE if we can't confirm it's a prokaryote
    return False


def extract_taxid(header: str) -> int:
    """Extract NCBI Taxonomy ID from OX= field in UniProt header"""
    if 'OX=' in header:
        try:
            return int(header.split('OX=')[1].split()[0])
        except (ValueError, IndexError):
            pass
    return None


def parse_fasta_gz(filepath: str, max_sequences: int = None) -> List[Dict]:
    """Parse gzipped FASTA file and extract organism information"""
    sequences = []

    with gzip.open(filepath, 'rt', encoding='utf-8', errors='ignore') as f:
        current_header = None
        current_seq = []

        for line in f:
            line = line.strip()
            if not line:
                continue

            if line.startswith('>'):
                # Save previous sequence
                if current_header and current_seq:
                    organism = extract_organism(current_header)
                    sequences.append({
                        'header': current_header,
                        'sequence': ''.join(current_seq),
                        'organism': organism
                    })

                    if max_sequences and len(sequences) >= max_sequences:
                        break

                current_header = line[1:]
                current_seq = []
            else:
                current_seq.append(line)

        # Don't forget the last sequence
        if current_header and current_seq:
            organism = extract_organism(current_header)
            sequences.append({
                'header': current_header,
                'sequence': ''.join(current_seq),
                'organism': organism
            })

    return sequences


def filter_negative_samples(
    sequences: List[Dict],
    min_length: int = 50,
    max_length: int = 5000,
    filter_prokaryote: bool = True,
    use_taxoniq: bool = True
) -> List[Dict]:
    """
    Filter sequences to exclude resistance-related proteins,
    non-prokaryotic organisms, and apply length filtering.

    Args:
        sequences: List of sequence dictionaries with 'header' and 'sequence'
        min_length: Minimum sequence length
        max_length: Maximum sequence length
        filter_prokaryote: If True, only keep prokaryotic sequences
        use_taxoniq: If True, use taxoniq with NCBI Taxonomy (more accurate)
    """
    filtered = []
    stats = {
        'total': len(sequences),
        'resistance_related': 0,
        'not_prokaryote': 0,
        'too_short': 0,
        'too_long': 0,
        'retained': 0
    }

    for seq_data in sequences:
        header = seq_data['header']
        seq = seq_data['sequence']
        seq_len = len(seq)
        organism = seq_data.get('organism', extract_organism(header))

        # Check if resistance related
        if is_resistance_related(header):
            stats['resistance_related'] += 1
            continue

        # Check if prokaryote
        if filter_prokaryote and not is_prokaryote(header, use_taxoniq=use_taxoniq):
            stats['not_prokaryote'] += 1
            continue

        # Length filtering
        if seq_len < min_length:
            stats['too_short'] += 1
            continue

        if seq_len > max_length:
            stats['too_long'] += 1
            continue

        # Check for invalid amino acids (not supported by model)
        # VALID_AA includes: 20 standard + X (unknown) + B, Z, J (ambiguous)
        if any(aa not in VALID_AA for aa in seq.upper()):
            continue

        filtered.append({
            'header': header,
            'sequence': seq,
            'length': seq_len,
            'organism': organism,
            'is_arg': 0  # Negative sample
        })
        stats['retained'] += 1

    logger.info("Filtering statistics:")
    logger.info(f"  Total input: {stats['total']}")
    logger.info(f"  Resistance related (excluded): {stats['resistance_related']}")
    logger.info(f"  Not prokaryote (excluded): {stats['not_prokaryote']}")
    logger.info(f"  Too short (<{min_length}): {stats['too_short']}")
    logger.info(f"  Too long (>{max_length}): {stats['too_long']}")
    logger.info(f"  Retained: {stats['retained']}")

    return filtered, stats


def save_to_fasta(sequences: List[Dict], filepath: str):
    """Save sequences to FASTA file"""
    with open(filepath, 'w') as f:
        for i, seq_data in enumerate(sequences):
            # Clean header and add metadata
            header = seq_data['header'].split()[0]  # Take first part of header
            f.write(f">{header}|is_arg=0|length={seq_data['length']}|negative_sample_{i}\n")

            seq = seq_data['sequence']
            for j in range(0, len(seq), 60):
                f.write(seq[j:j+60] + '\n')


def save_to_csv(sequences: List[Dict], filepath: str):
    """Save sequences to CSV file"""
    import csv

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['sequence', 'category', 'is_arg', 'length', 'header'])

        for seq_data in sequences:
            writer.writerow([
                seq_data['sequence'],
                'non_arg',  # Category for negative samples
                0,  # is_arg = False
                seq_data['length'],
                seq_data['header'].split()[0]
            ])


def main(
    output_dir: str = './data_now',
    target_count: int = None,  # Number of negative samples to select, None = keep all filtered
    seed: int = 42,
    filter_prokaryote: bool = True,
    use_taxoniq: bool = True
):
    """
    Download and prepare negative samples.

    Args:
        output_dir: Output directory
        target_count: Number of negative samples to select, None = keep all filtered sequences
        seed: Random seed for sampling
        filter_prokaryote: If True, only keep prokaryotic sequences (Bacteria/Archaea)
        use_taxoniq: If True, use taxoniq with NCBI Taxonomy for accurate classification
    """
    import random
    import json

    random.seed(seed)

    # Download UniProt Swiss-Prot
    gz_path = download_uniprot_sprot(output_dir)

    # Parse sequences
    logger.info("Parsing Swiss-Prot sequences...")
    all_sequences = parse_fasta_gz(gz_path)
    logger.info(f"Total sequences in Swiss-Prot: {len(all_sequences)}")

    # Filter negative samples
    logger.info("\nFiltering negative samples...")
    logger.info(f"  - Excluding resistance-related proteins")
    logger.info(f"  - {'Only keeping prokaryotic sequences (Bacteria/Archaea)' if filter_prokaryote else 'Keeping all organisms'}")
    if use_taxoniq:
        try:
            import taxoniq
            logger.info("  - Using taxoniq with NCBI Taxonomy for accurate classification")
        except ImportError:
            logger.warning("  - taxoniq not available, falling back to keyword-based filtering")
            use_taxoniq = False
    filtered, filter_stats = filter_negative_samples(all_sequences, filter_prokaryote=filter_prokaryote, use_taxoniq=use_taxoniq)

    # Sample or keep all filtered sequences
    if target_count is None:
        selected = filtered
        logger.info(f"\nKept all {len(selected)} filtered negative samples (target_count=None)")
    elif len(filtered) < target_count:
        logger.warning(f"Only {len(filtered)} negative samples available, requested {target_count}")
        selected = filtered
    else:
        # Randomly sample to match target count
        selected = random.sample(filtered, target_count)
        logger.info(f"\nRandomly selected {target_count} negative samples")

    # Shuffle
    random.shuffle(selected)

    # Save to files
    fasta_path = os.path.join(output_dir, 'negative_samples.fasta')
    csv_path = os.path.join(output_dir, 'negative_samples.csv')

    save_to_fasta(selected, fasta_path)
    save_to_csv(selected, csv_path)

    logger.info(f"\nSaved negative samples:")
    logger.info(f"  FASTA: {fasta_path}")
    logger.info(f"  CSV: {csv_path}")

    # Length distribution
    lengths = [s['length'] for s in selected]
    logger.info(f"\nLength distribution:")
    logger.info(f"  Mean: {sum(lengths)/len(lengths):.1f}")
    logger.info(f"  Min: {min(lengths)}")
    logger.info(f"  Max: {max(lengths)}")

    # Organism distribution
    organism_counts = Counter([s.get('organism', 'unknown') for s in selected])
    logger.info(f"\nTop 10 organisms in selected samples:")
    for org, count in organism_counts.most_common(10):
        logger.info(f"  {org}: {count}")

    # Save stats
    stats = {
        'source': 'UniProt Swiss-Prot',
        'total_downloaded': len(all_sequences),
        'filter_prokaryote': filter_prokaryote,
        'use_taxoniq': use_taxoniq,
        'resistance_related_excluded': filter_stats['resistance_related'],
        'not_prokaryote_excluded': filter_stats['not_prokaryote'],
        'too_short_excluded': filter_stats['too_short'],
        'too_long_excluded': filter_stats['too_long'],
        'retained_before_sampling': len(filtered),
        'final_count': len(selected),
        'length_stats': {
            'mean': sum(lengths) / len(lengths),
            'min': min(lengths),
            'max': max(lengths)
        },
        'organism_distribution': dict(organism_counts.most_common(20))
    }

    stats_path = os.path.join(output_dir, 'negative_samples_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(stats, f, indent=2)

    logger.info(f"\nSaved statistics to {stats_path}")
    logger.info("Negative sample preparation complete!")

    return selected


if __name__ == "__main__":
    main(
        output_dir='./data_now',
        target_count=None,  # Keep all filtered sequences
        seed=42,
        filter_prokaryote=True,
        use_taxoniq=True
    )
