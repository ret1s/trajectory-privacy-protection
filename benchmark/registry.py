"""Single inventory used by runners, reports and future web front ends."""

from __future__ import annotations

from typing import Type

from benchmark.contracts import BenchmarkMethod, MethodCard
from benchmark.methods import (
    AnotherMeAdaptation,
    GeoIAnchoredDummyTrajectories,
    SemanticCorrelationComparator,
    TransProtectAdaptation,
)


METHOD_CLASSES: tuple[Type[BenchmarkMethod], ...] = (
    TransProtectAdaptation,
    AnotherMeAdaptation,
    SemanticCorrelationComparator,
    GeoIAnchoredDummyTrajectories,
)
METHOD_CARDS: tuple[MethodCard, ...] = tuple(
    method.method_card for method in METHOD_CLASSES
)
_BY_ID = {card.method_id: card for card in METHOD_CARDS}

if len(_BY_ID) != len(METHOD_CARDS):  # pragma: no cover - import-time invariant
    raise RuntimeError("benchmark method IDs must be unique")


def method_card(method_id: str) -> MethodCard:
    """Return a registered method card or raise a precise key error."""

    try:
        return _BY_ID[method_id]
    except KeyError:
        known = ", ".join(sorted(_BY_ID))
        raise KeyError(f"unknown benchmark method {method_id!r}; known: {known}") from None


def method_inventory() -> list[dict]:
    """Return JSON-ready evidence metadata in stable registry order."""

    return [card.to_dict() for card in METHOD_CARDS]


def require_faithful_sota(method_ids: tuple[str, ...] | list[str]) -> None:
    """Fail before a paper-adaptation run is called a SOTA reproduction."""

    for method_id in method_ids:
        method_card(method_id).require_faithful()


# Compatibility alias for the initial benchmark-v3 API spelling.
require_reproduced_sota = require_faithful_sota


__all__ = [
    "METHOD_CARDS",
    "METHOD_CLASSES",
    "method_card",
    "method_inventory",
    "require_faithful_sota",
    "require_reproduced_sota",
]
