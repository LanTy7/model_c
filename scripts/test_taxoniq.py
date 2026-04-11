#!/usr/bin/env python3
"""
Taxoniq Validation Test Script
Complete reproducible test for taxoniq-based prokaryote classification
"""

import sys
import gzip
from collections import Counter


def check_taxoniq_installation():
    """Step 1: Check if taxoniq is installed"""
    print("=" * 80)
    print("STEP 1: Checking taxoniq installation")
    print("=" * 80)

    try:
        import taxoniq
        print(f"✓ taxoniq is installed (version: {taxoniq.__version__ if hasattr(taxoniq, '__version__') else 'unknown'})")
        return True
    except ImportError:
        print("✗ taxoniq is NOT installed")
        print("  To install: pip install taxoniq")
        return False


def test_single_taxid(taxid, expected_superkingdom):
    """Test a single TaxID and return results"""
    import taxoniq

    try:
        t = taxoniq.Taxon(taxid)

        # Get superkingdom
        superkingdom = None
        for parent in t.ranked_lineage:
            if parent.rank.name == "superkingdom":
                superkingdom = parent.scientific_name
                break

        # Check if matches expected
        is_correct = superkingdom == expected_superkingdom

        return {
            'taxid': taxid,
            'name': t.scientific_name,
            'superkingdom': superkingdom,
            'expected': expected_superkingdom,
            'correct': is_correct,
            'lineage': [(p.rank.name, p.scientific_name) for p in t.ranked_lineage]
        }
    except Exception as e:
        return {
            'taxid': taxid,
            'error': str(e)
        }


def run_taxid_tests():
    """Step 2: Run test cases with known TaxIDs"""
    print("\n" + "=" * 80)
    print("STEP 2: Testing known TaxIDs")
    print("=" * 80)

    import taxoniq

    # Test cases: (TaxID, Expected Superkingdom, Description)
    test_cases = [
        (9606, "Eukaryota", "Homo sapiens (Human)"),
        (10090, "Eukaryota", "Mus musculus (Mouse)"),
        (3702, "Eukaryota", "Arabidopsis thaliana (Plant)"),
        (559292, "Eukaryota", "Saccharomyces cerevisiae (Yeast)"),
        (83333, "Bacteria", "Escherichia coli K-12 (Bacteria)"),
        (224308, "Bacteria", "Bacillus subtilis (Bacteria)"),
        (99287, "Bacteria", "Salmonella typhimurium (Bacteria)"),
        (243277, "Bacteria", "Vibrio cholerae (Bacteria)"),
        (224324, "Bacteria", "Aquifex aeolicus (Bacteria - deep branching)"),
        (64091, "Archaea", "Halobacterium salinarum (Archaea)"),
    ]

    print(f"\nRunning {len(test_cases)} test cases...\n")
    print(f"{'TaxID':<10} {'Expected':<12} {'Actual':<12} {'Status':<10} {'Organism'}")
    print("-" * 100)

    passed = 0
    failed = 0

    for taxid, expected, description in test_cases:
        result = test_single_taxid(taxid, expected)

        if 'error' in result:
            print(f"{taxid:<10} {expected:<12} {'ERROR':<12} {'✗ FAIL':<10} {result['error'][:40]}")
            failed += 1
        else:
            status = "✓ PASS" if result['correct'] else "✗ FAIL"
            print(f"{taxid:<10} {expected:<12} {result['superkingdom']:<12} {status:<10} {result['name'][:50]}")

            if result['correct']:
                passed += 1
            else:
                failed += 1

    print("-" * 100)
    print(f"\nResults: {passed} passed, {failed} failed out of {len(test_cases)} tests")

    return failed == 0


def extract_taxid_from_header(header):
    """Extract TaxID from UniProt FASTA header"""
    if 'OX=' in header:
        try:
            return int(header.split('OX=')[1].split()[0])
        except (ValueError, IndexError):
            pass
    return None


def extract_organism(header):
    """Extract organism name from UniProt FASTA header"""
    if 'OS=' in header:
        start = header.find('OS=') + 3
        end = len(header)
        for marker in [' OX=', ' GN=', ' PE=', ' SV=']:
            pos = header.find(marker, start)
            if pos != -1 and pos < end:
                end = pos
        return header[start:end].strip()
    return 'unknown'


def is_prokaryote_with_taxoniq(header):
    """Check if organism is prokaryote using taxoniq"""
    import taxoniq

    taxid = extract_taxid_from_header(header)
    if not taxid:
        return None, "No TaxID found"

    try:
        t = taxoniq.Taxon(taxid)
        for parent in t.ranked_lineage:
            if parent.scientific_name == "Bacteria" or parent.scientific_name == "Archaea":
                return True, parent.scientific_name
        return False, "Eukaryota or other"
    except Exception as e:
        return None, f"Error: {str(e)[:50]}"


def test_uniprot_headers():
    """Step 3: Test with actual UniProt headers"""
    print("\n" + "=" * 80)
    print("STEP 3: Testing with sample UniProt headers")
    print("=" * 80)

    # Sample headers from UniProt
    test_headers = [
        ">sp|P69905|HBA_HUMAN Hemoglobin subunit alpha OS=Homo sapiens OX=9606 GN=HBA1 PE=1 SV=2",
        ">sp|P63316|TNFA_MOUSE Tumor necrosis factor OS=Mus musculus OX=10090 GN=Tnf PE=1 SV=1",
        ">sp|P0A6X8|DNAA_ECOLI Chromosomal replication initiator protein DnaA OS=Escherichia coli (strain K12) OX=83333 GN=dnaA PE=1 SV=2",
        ">sp|P12689|FSHB_HUMAN Follitropin subunit beta OS=Homo sapiens OX=9606 GN=FSHB PE=1 SV=2",
        ">sp|P15000|IL2_MOUSE Interleukin-2 OS=Mus musculus OX=10090 GN=Il2 PE=1 SV=1",
        ">sp|P0A7V3|RL22_ECOLI 50S ribosomal protein L22 OS=Escherichia coli (strain K12) OX=83333 GN=rplV PE=1 SV=1",
        ">sp|P9WIU6|KATG_MYCTU Catalase-peroxidase OS=Mycobacterium tuberculosis (strain ATCC 25618 / H37Rv) OX=83332 GN=katG PE=1 SV=3",
        ">sp|P0A3R9|CYSK_BACSU Cysteine synthase OS=Bacillus subtilis (strain 168) OX=224308 GN=cysK PE=1 SV=1",
        ">sp|P0C2T4|H2A1_ARATH Histone H2A 1 OS=Arabidopsis thaliana OX=3702 GN=HTA1 PE=1 SV=1",
        ">sp|P00330|ADH1_YEAST Alcohol dehydrogenase 1 OS=Saccharomyces cerevisiae (strain ATCC 204508 / S288c) OX=559292 GN=ADH1 PE=1 SV=4",
    ]

    print(f"\nTesting {len(test_headers)} sample headers:\n")
    print(f"{'Expected':<12} {'Result':<12} {'TaxID':<8} {'Organism (truncated)'}")
    print("-" * 100)

    results = {'prokaryote': 0, 'eukaryote': 0, 'error': 0}

    for header in test_headers:
        taxid = extract_taxid_from_header(header)
        organism = extract_organism(header)

        is_prok, reason = is_prokaryote_with_taxoniq(header)

        if is_prok is True:
            result_str = f"PROKARYOTE ({reason})"
            results['prokaryote'] += 1
        elif is_prok is False:
            result_str = f"NOT PROK ({reason})"
            results['eukaryote'] += 1
        else:
            result_str = f"ERROR ({reason})"
            results['error'] += 1

        # Determine expected result based on organism name
        expected = "PROKARYOTE" if any(x in organism.lower() for x in ['coli', 'bacillus', 'mycobacterium', 'salmonella']) else "EUKARYOTE"

        print(f"{expected:<12} {result_str:<30} {str(taxid):<8} {organism[:45]}")

    print("-" * 100)
    print(f"\nSummary: {results['prokaryote']} prokaryotes, {results['eukaryote']} eukaryotes, {results['error']} errors")

    return results


def analyze_uniprot_sample(max_samples=1000):
    """Step 4: Analyze sample from UniProt file"""
    print("\n" + "=" * 80)
    print(f"STEP 4: Analyzing {max_samples} samples from UniProt Swiss-Prot")
    print("=" * 80)

    uniprot_path = "/home/mayue/model_c/data_now/uniprot_sprot.fasta.gz"

    try:
        with gzip.open(uniprot_path, 'rt', encoding='utf-8', errors='ignore') as f:
            counter = Counter()
            prok_count = 0
            euk_count = 0
            error_count = 0
            sample_count = 0

            for line in f:
                if line.startswith('>'):
                    is_prok, reason = is_prokaryote_with_taxoniq(line)

                    if is_prok is True:
                        counter['Prokaryote'] += 1
                        prok_count += 1
                    elif is_prok is False:
                        counter['Eukaryote'] += 1
                        euk_count += 1
                    else:
                        counter['Error/Unknown'] += 1
                        error_count += 1

                    sample_count += 1
                    if sample_count >= max_samples:
                        break

            print(f"\nClassification results from {max_samples} UniProt entries:")
            print("-" * 50)
            for category, count in counter.most_common():
                pct = count / max_samples * 100
                print(f"  {category:<20}: {count:5d} ({pct:5.1f}%)")

            print(f"\nProkaryote ratio: {prok_count}/{max_samples} ({prok_count/max_samples*100:.1f}%)")

            return counter

    except FileNotFoundError:
        print(f"✗ UniProt file not found: {uniprot_path}")
        print("  Run download_negative_samples.py first to download UniProt")
        return None


def main():
    """Main test flow"""
    print("\n" + "=" * 80)
    print("TAXONIQ VALIDATION TEST")
    print("Reproducible test for taxoniq-based prokaryote classification")
    print("=" * 80)

    # Step 1: Check installation
    if not check_taxoniq_installation():
        print("\n✗ ABORTED: taxoniq is required for testing")
        sys.exit(1)

    # Step 2: Run TaxID tests
    taxid_passed = run_taxid_tests()

    # Step 3: Test with UniProt headers
    header_results = test_uniprot_headers()

    # Step 4: Analyze UniProt sample (optional, requires file)
    sample_results = analyze_uniprot_sample(max_samples=1000)

    # Final summary
    print("\n" + "=" * 80)
    print("FINAL SUMMARY")
    print("=" * 80)
    print(f"TaxID tests:          {'✓ PASSED' if taxid_passed else '✗ FAILED'}")
    print(f"Header extraction:    ✓ TESTED ({header_results['prokaryote']} prok, {header_results['eukaryote']} euk)")
    if sample_results:
        prok_ratio = sample_results.get('Prokaryote', 0) / 1000 * 100
        print(f"UniProt sample:       ✓ ANALYZED ({prok_ratio:.1f}% prokaryotes)")

    print("\n" + "=" * 80)
    if taxid_passed:
        print("✓ ALL TESTS PASSED - taxoniq is working correctly")
    else:
        print("✗ SOME TESTS FAILED - check taxoniq installation")
    print("=" * 80)


if __name__ == "__main__":
    main()
