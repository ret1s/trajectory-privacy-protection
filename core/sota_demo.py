"""Deprecated import compatibility for pre-benchmark ``*Lite`` names.

New code must import source-mapped methods from :mod:`benchmark.methods`.
These aliases keep historical scripts readable without making the legacy names
part of the canonical benchmark API.
"""

from benchmark.engines.paper_adaptations import (
    AnotherMeEngine,
    CandidateSetRelease,
    SemanticDummyEngine,
    TransProtectEngine,
)


class TransProtectLite(TransProtectEngine):
    name = "transprotect_lite"
    demo_only = True


class AnotherMeLite(AnotherMeEngine):
    name = "anotherme_lite"
    demo_only = True


class SemanticDummyLite(SemanticDummyEngine):
    name = "semantic_dummy_lite"
    demo_only = True

__all__ = [
    "AnotherMeLite",
    "CandidateSetRelease",
    "SemanticDummyLite",
    "TransProtectLite",
]
