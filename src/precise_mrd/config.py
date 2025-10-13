"""Configuration management for precise MRD pipeline."""

from __future__ import annotations

import hashlib
import json
import yaml
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SimulationConfig:
    """Configuration for simulation parameters."""
    allele_fractions: List[float]
    umi_depths: List[int]
    n_replicates: int
    n_bootstrap: int


@dataclass
class UMIConfig:
    """Configuration for UMI processing."""
    min_family_size: int
    max_family_size: int
    quality_threshold: int
    consensus_threshold: float


@dataclass
class StatsConfig:
    """Configuration for statistical testing."""
    test_type: str
    alpha: float
    fdr_method: str


@dataclass
class LODConfig:
    """Configuration for LoD estimation."""
    detection_threshold: float
    confidence_level: float


@dataclass
class FASTQConfig:
    """Configuration for FASTQ file processing."""
    input_path: str
    max_reads: Optional[int] = None
    umi_pattern: Optional[str] = None
    quality_threshold: int = 20
    min_family_size: int = 3


@dataclass
class PipelineConfig:
    """Main pipeline configuration."""
    run_id: str
    seed: int
    umi: UMIConfig
    stats: StatsConfig
    lod: LODConfig
    simulation: Optional[SimulationConfig] = None
    fastq: Optional[FASTQConfig] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return asdict(self)
    
    def config_hash(self) -> str:
        """Compute deterministic hash of configuration."""
        config_str = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def load_config(path: str | Path) -> PipelineConfig:
    """Load configuration from YAML file."""
    with open(path, 'r') as f:
        data = yaml.safe_load(f)
    
    return PipelineConfig(
        run_id=data['run_id'],
        seed=data['seed'],
        simulation=SimulationConfig(**data['simulation']),
        umi=UMIConfig(**data['umi']),
        stats=StatsConfig(**data['stats']),
        lod=LODConfig(**data['lod'])
    )


def dump_config(config: PipelineConfig, path: str | Path) -> None:
    """Save configuration to YAML file."""
    with open(path, 'w') as f:
        yaml.safe_dump(config.to_dict(), f, default_flow_style=False)