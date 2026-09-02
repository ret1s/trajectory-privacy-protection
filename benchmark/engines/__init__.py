"""Dependency-light algorithm engines used by benchmark method adapters.

The engines implement deterministic local adaptations.  Reporting status and
paper-to-code evidence live one layer above in :mod:`benchmark.methods`.
"""

from .paper_adaptations import (
    AnotherMeEngine,
    CandidateSetRelease,
    SemanticDummyEngine,
    TransProtectEngine,
)
from .proposed import AnchoredDummyBatch, GeoIAnchoredDummyEngine

__all__ = [
    "AnchoredDummyBatch",
    "AnotherMeEngine",
    "CandidateSetRelease",
    "GeoIAnchoredDummyEngine",
    "SemanticDummyEngine",
    "TransProtectEngine",
]
