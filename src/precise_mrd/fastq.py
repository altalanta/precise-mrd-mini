"""FASTQ file processing for real sequencing data integration."""

from __future__ import annotations

import gzip
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterator, Any
import pandas as pd
import numpy as np

from .config import PipelineConfig


class FASTQReader:
    """Efficient FASTQ file reader with UMI extraction."""

    def __init__(self, file_path: str | Path):
        """Initialize FASTQ reader.

        Args:
            file_path: Path to FASTQ file (can be gzipped)
        """
        self.file_path = Path(file_path)
        self.is_gzipped = self.file_path.suffix == '.gz'

        # UMI extraction patterns
        self.umi_patterns = [
            r'UMI:([A-Z]+)',  # UMI:SEQUENCE format
            r'_([A-Z]{8,12})_',  # _UMI_ format (common in Illumina)
            r':([A-Z]{8,12})$',  # :UMI format (common in read names)
        ]

    def _open_file(self):
        """Open file with appropriate compression handling."""
        if self.is_gzipped:
            return gzip.open(self.file_path, 'rt')
        else:
            return open(self.file_path, 'r')

    def extract_umi(self, read_header: str) -> str | None:
        """Extract UMI sequence from read header.

        Args:
            read_header: FASTQ read header line

        Returns:
            UMI sequence if found, None otherwise
        """
        for pattern in self.umi_patterns:
            match = re.search(pattern, read_header)
            if match:
                return match.group(1)
        return None

    def parse_quality_string(self, quality: str) -> List[int]:
        """Convert quality string to numeric scores.

        Args:
            quality: ASCII quality string

        Returns:
            List of numeric quality scores
        """
        # Sanger/Illumina 1.8+ format (Phred+33)
        return [ord(char) - 33 for char in quality]

    def read_fastq(self, max_reads: Optional[int] = None) -> Iterator[Dict[str, Any]]:
        """Read FASTQ file and yield parsed reads.

        Args:
            max_reads: Maximum number of reads to process (None for all)

        Yields:
            Dictionary with read information
        """
        read_count = 0

        with self._open_file() as f:
            while True:
                if max_reads and read_count >= max_reads:
                    break

                # Read 4 lines of FASTQ record
                header = f.readline().strip()
                if not header:
                    break  # EOF

                sequence = f.readline().strip()
                plus = f.readline().strip()  # Should be '+'
                quality = f.readline().strip()

                # Skip malformed records
                if not (header and sequence and plus and quality):
                    continue

                # Extract UMI from header
                umi = self.extract_umi(header)

                # Parse quality scores
                quality_scores = self.parse_quality_string(quality)

                yield {
                    'read_id': header[1:],  # Remove '@' prefix
                    'sequence': sequence,
                    'umi': umi,
                    'quality_scores': quality_scores,
                    'mean_quality': np.mean(quality_scores),
                    'sequence_length': len(sequence),
                }

                read_count += 1

    def validate_fastq(self) -> Dict[str, Any]:
        """Validate FASTQ file format and extract basic statistics.

        Returns:
            Dictionary with validation results and file statistics
        """
        stats = {
            'total_reads': 0,
            'valid_reads': 0,
            'umi_reads': 0,
            'mean_read_length': 0.0,
            'mean_quality': 0.0,
            'is_valid': True,
            'errors': []
        }

        try:
            # Sample first 1000 reads for validation
            for i, read in enumerate(self.read_fastq(max_reads=1000)):
                stats['total_reads'] = i + 1
                stats['valid_reads'] += 1

                if read['umi']:
                    stats['umi_reads'] += 1

                # Accumulate length and quality stats
                if i == 0:
                    stats['mean_read_length'] = read['sequence_length']
                    stats['mean_quality'] = read['mean_quality']
                else:
                    # Running average
                    stats['mean_read_length'] = (
                        (stats['mean_read_length'] * i + read['sequence_length']) / (i + 1)
                    )
                    stats['mean_quality'] = (
                        (stats['mean_quality'] * i + read['mean_quality']) / (i + 1)
                    )

        except Exception as e:
            stats['is_valid'] = False
            stats['errors'].append(str(e))

        return stats


def process_fastq_to_dataframe(
    fastq_path: str | Path,
    config: PipelineConfig,
    rng: np.random.Generator,
    output_path: Optional[str] = None,
    max_reads: Optional[int] = None
) -> pd.DataFrame:
    """Process FASTQ file and convert to pipeline-compatible DataFrame.

    Args:
        fastq_path: Path to FASTQ file
        config: Pipeline configuration
        rng: Random number generator for deterministic processing
        output_path: Optional path to save processed data
        max_reads: Maximum reads to process

    Returns:
        DataFrame with processed read data compatible with pipeline
    """

    reader = FASTQReader(fastq_path)

    # Validate FASTQ file first
    validation = reader.validate_fastq()
    if not validation['is_valid']:
        raise ValueError(f"Invalid FASTQ file: {validation['errors']}")

    print(f"Processing FASTQ file: {fastq_path}")
    print(f"Validation: {validation['total_reads']} reads, "
          f"{validation['umi_reads']} with UMIs, "
          f"mean length {validation['mean_read_length']:.1f}bp")

    # Process reads and group by UMI
    umi_groups = {}

    for read in reader.read_fastq(max_reads):
        umi = read['umi']
        if not umi:
            continue

        if umi not in umi_groups:
            umi_groups[umi] = []

        umi_groups[umi].append(read)

    # Convert to pipeline format
    processed_data = []

    for sample_id, umi in enumerate(umi_groups.keys()):
        reads = umi_groups[umi]

        # Calculate family statistics
        family_size = len(reads)
        quality_scores = [q for read in reads for q in read['quality_scores']]
        sequences = [read['sequence'] for read in reads]

        # Consensus calling (simplified for now)
        # Use highest quality read as consensus
        best_read_idx = np.argmax([read['mean_quality'] for read in reads])
        consensus_sequence = sequences[best_read_idx]
        consensus_quality = quality_scores

        processed_data.append({
            'sample_id': sample_id,
            'umi': umi,
            'family_size': family_size,
            'consensus_sequence': consensus_sequence,
            'quality_scores': consensus_quality,
            'mean_quality': np.mean(quality_scores),
            'consensus_agreement': 1.0,  # Simplified for now
            'passes_quality': np.mean(quality_scores) >= 20,  # Quality threshold
            'passes_consensus': True,  # Simplified for now
            'config_hash': config.config_hash(),
        })

    df = pd.DataFrame(processed_data)

    if output_path:
        df.to_parquet(output_path, index=False)

    return df


def detect_umi_format(fastq_path: str | Path, sample_size: int = 100) -> str:
    """Detect UMI format in FASTQ file.

    Args:
        fastq_path: Path to FASTQ file
        sample_size: Number of reads to sample for detection

    Returns:
        Detected UMI format description
    """
    reader = FASTQReader(fastq_path)

    umi_formats = {
        'UMI:SEQUENCE': 0,
        '_UMI_': 0,
        ':UMI': 0,
        'none': 0
    }

    for i, read in enumerate(reader.read_fastq(max_reads=sample_size)):
        umi = read['umi']
        if not umi:
            umi_formats['none'] += 1
            continue

        # Check which pattern matched
        header = read['read_id']
        if 'UMI:' in header:
            umi_formats['UMI:SEQUENCE'] += 1
        elif '_' in header and len(umi) >= 8:
            umi_formats['_UMI_'] += 1
        elif header.endswith(umi):
            umi_formats[':UMI'] += 1

    # Return most common format
    if max(umi_formats.values()) == 0:
        return "No UMI detected in sample"

    return max(umi_formats.items(), key=lambda x: x[1])[0]
