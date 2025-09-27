
"""Deterministic random utilities."""

from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


@dataclass(slots=True)
class RandomState:
    """Wrapper for deterministic random number generation."""

    seed: int
    generator: np.random.Generator

    @classmethod
    def create(cls, seed: int) -> "RandomState":
        generator = np.random.default_rng(seed)
        random.seed(seed)
        np.random.seed(seed)
        return cls(seed=seed, generator=generator)

    def spawn(self, offset: int) -> "RandomState":
        """Derive a child generator with deterministic offset."""

        bit_generator = self.generator.bit_generator.jumped(offset)
        return RandomState(seed=self.seed + offset, generator=np.random.Generator(bit_generator))


def choose_rng(seed: int) -> RandomState:
    """Convenience helper to create a ``RandomState``."""

    return RandomState.create(seed)

