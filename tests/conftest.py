"""
Test configuration and fixtures for precise-mrd tests.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil

# Import available modules
from precise_mrd.simulate import simulate_reads
from precise_mrd.collapse import collapse_umis
from precise_mrd.config import load_config
from precise_mrd.rng import choose_rng


@pytest.fixture
def seed():
    """Fixed random seed for reproducible tests."""
    return 42


@pytest.fixture
def temp_dir():
    """Temporary directory for test outputs."""
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_target_sites():
    """Sample target sites for testing."""
    return [
        TargetSite("chr1", 1000000, "A", "T", "ACG", "GENE1"),
        TargetSite("chr2", 2000000, "C", "G", "CCG", "GENE2"),
        TargetSite("chr3", 3000000, "G", "A", "GCG", "GENE3"),
        TargetSite("chr4", 4000000, "T", "C", "TCG", "GENE4"),
    ]


@pytest.fixture
def read_generator(seed):
    """Synthetic read generator with fixed seed."""
    return SyntheticReadGenerator(seed=seed)


@pytest.fixture
def sample_reads(read_generator, sample_target_sites):
    """Sample reads for testing."""
    reads = []
    for site in sample_target_sites[:2]:  # Use first 2 sites
        site_reads = read_generator.generate_reads_for_site(
            site=site, n_umi_families=10, allele_fraction=0.01, mean_family_size=3.0
        )
        reads.extend(site_reads)
    return reads


@pytest.fixture
def umi_processor():
    """UMI processor with test settings."""
    return UMIProcessor(
        min_family_size=2, max_family_size=50, quality_threshold=20, consensus_threshold=0.6
    )


@pytest.fixture
def statistical_tester():
    """Statistical tester with test settings."""
    return StatisticalTester(test_type="poisson", alpha=0.05, fdr_method="benjamini_hochberg")


@pytest.fixture
def lod_estimator():
    """LoD estimator with test settings."""
    return LODEstimator(
        detection_threshold=0.95,
        confidence_level=0.95,
        n_bootstrap=50,  # Small for fast tests
        alpha=0.05,
    )


@pytest.fixture
def mock_simulation_results():
    """Mock simulation results for testing."""
    np.random.seed(42)
    results = []

    allele_fractions = [0.0001, 0.0005, 0.001, 0.005, 0.01]
    umi_depths = [5000, 10000, 20000]

    for af in allele_fractions:
        for depth in umi_depths:
            for rep in range(20):  # 20 replicates each
                # Simulate detection probability
                base_prob = min(0.95, af * depth / 10000)
                detected = np.random.random() < base_prob

                results.append(
                    {
                        "allele_fraction": af,
                        "umi_depth": depth,
                        "replicate_id": rep,
                        "detected": detected,
                        "pvalue": np.random.uniform(0.001, 0.1)
                        if detected
                        else np.random.uniform(0.1, 1.0),
                        "consensus_alt_count": np.random.poisson(af * depth) if detected else 0,
                        "consensus_total_depth": depth,
                    }
                )

    return pd.DataFrame(results)


@pytest.fixture
def mock_consensus_data():
    """Mock consensus data for testing."""
    data = []
    sites = ["chr1:1000:A", "chr2:2000:C", "chr3:3000:G"]
    alleles = ["T", "G", "A"]

    for i, (site_key, alt) in enumerate(zip(sites, alleles)):
        # Reference allele counts
        data.append(
            {
                "site_key": site_key,
                "chrom": site_key.split(":")[0],
                "pos": int(site_key.split(":")[1]),
                "ref": site_key.split(":")[2],
                "allele": site_key.split(":")[2],  # Reference
                "consensus_count": np.random.poisson(1000),
                "context": f"{'ACG'[i]}{site_key.split(':')[2]}{'CGT'[i]}",
            }
        )

        # Alternative allele counts
        data.append(
            {
                "site_key": site_key,
                "chrom": site_key.split(":")[0],
                "pos": int(site_key.split(":")[1]),
                "ref": site_key.split(":")[2],
                "allele": alt,
                "consensus_count": np.random.poisson(10),
                "context": f"{'ACG'[i]}{site_key.split(':')[2]}{'CGT'[i]}",
            }
        )

    return pd.DataFrame(data)


@pytest.fixture
def mock_error_rates():
    """Mock error rates by context."""
    return {
        "ACG": {"A>T": 1e-4, "A>C": 5e-5, "A>G": 2e-5},
        "CCG": {"C>T": 2e-4, "C>A": 8e-5, "C>G": 3e-5},
        "GCG": {"G>A": 1.5e-4, "G>T": 6e-5, "G>C": 2e-5},
        "TCG": {"T>C": 3e-4, "T>A": 1e-4, "T>G": 4e-5},
    }


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "run_id": "test_run",
        "seed": 42,
        "simulation": {
            "allele_fractions": [0.01, 0.001],
            "umi_depths": [5000, 10000],
            "n_replicates": 10,
            "n_bootstrap": 10,
        },
        "umi": {"min_family_size": 2, "max_family_size": 100, "quality_threshold": 20},
        "stats": {"test_type": "poisson", "alpha": 0.05},
    }


# Utility functions for tests
def create_test_read(
    chrom="chr1",
    pos=1000,
    ref="A",
    alt="T",
    umi="AAGGCCTT",
    quality=30,
    strand="+",
    read_position=50,
    family_id=None,
):
    """Create a test read with specified parameters."""
    return Read(chrom, pos, ref, alt, umi, quality, strand, read_position, family_id)


def create_test_family(umi="AAGGCCTT", size=3, consensus_allele="T"):
    """Create a test UMI family with specified parameters."""
    from precise_mrd.umi import UMIFamily

    reads = []
    for i in range(size):
        read = create_test_read(umi=umi, alt=consensus_allele)
        reads.append(read)

    family = UMIFamily(umi=umi, reads=reads)
    family.consensus_allele = consensus_allele
    family.consensus_quality = 30.0

    return family


# Test data validation functions
def validate_detection_matrix(df):
    """Validate detection matrix format."""
    required_columns = ["allele_fraction", "umi_depth", "detection_rate"]
    assert all(col in df.columns for col in required_columns)
    assert (df["detection_rate"] >= 0).all()
    assert (df["detection_rate"] <= 1).all()
    assert (df["allele_fraction"] > 0).all()
    assert (df["umi_depth"] > 0).all()


def validate_consensus_data(df):
    """Validate consensus data format."""
    required_columns = ["site_key", "chrom", "pos", "ref", "allele", "consensus_count"]
    assert all(col in df.columns for col in required_columns)
    assert (df["consensus_count"] >= 0).all()
    assert (df["pos"] > 0).all()


def validate_statistical_results(df):
    """Validate statistical results format."""
    required_columns = ["pvalue", "qvalue", "significant"]
    assert all(col in df.columns for col in required_columns)
    assert (df["pvalue"] >= 0).all()
    assert (df["pvalue"] <= 1).all()
    assert (df["qvalue"] >= 0).all()
    assert df["significant"].dtype == bool


# Performance testing utilities
def time_operation(func, *args, **kwargs):
    """Time an operation and return duration."""
    import time

    start = time.time()
    result = func(*args, **kwargs)
    duration = time.time() - start
    return result, duration


def memory_usage():
    """Get current memory usage in MB."""
    import psutil

    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024


# Mock data generators for specific test scenarios
def generate_calibrated_pvalues(n=1000, seed=42):
    """Generate well-calibrated p-values (uniform distribution)."""
    np.random.seed(seed)
    return np.random.uniform(0, 1, n)


def generate_biased_pvalues(n=1000, bias_factor=0.5, seed=42):
    """Generate biased p-values (skewed distribution)."""
    np.random.seed(seed)
    return np.random.beta(bias_factor, 2, n)


def generate_monotonic_detection_curve(af_range=(1e-4, 1e-2), n_points=10):
    """Generate monotonic detection curve data."""
    afs = np.logspace(np.log10(af_range[0]), np.log10(af_range[1]), n_points)
    # Sigmoid-like detection curve
    detection_rates = 1 / (1 + np.exp(-10 * (afs - af_range[1] / 10)))

    return pd.DataFrame(
        {
            "allele_fraction": afs,
            "detection_rate": detection_rates,
            "ci_lower": np.maximum(0, detection_rates - 0.1),
            "ci_upper": np.minimum(1, detection_rates + 0.1),
        }
    )
