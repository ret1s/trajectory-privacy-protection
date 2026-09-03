"""Dummy-generation benchmark contracts and executable method adapters.

The package separates two concerns that were previously mixed in
the earlier demo modules:

* the public-output protocol used by every benchmark method; and
* the fidelity/status of a particular implementation relative to its paper.

An executable method is not automatically a faithful reproduction.  Callers
must inspect its :class:`~benchmark.contracts.MethodCard` (or call
``require_faithful``) before presenting a result as a reproduced SOTA result.
"""

from .contracts import (
    BenchmarkMethod,
    ComponentMapping,
    ComponentStatus,
    ImplementationLevel,
    MethodCard,
    MethodUnavailableError,
    SourceReference,
)

__all__ = [
    "BenchmarkMethod",
    "ComponentMapping",
    "ComponentStatus",
    "ImplementationLevel",
    "MethodCard",
    "MethodUnavailableError",
    "SourceReference",
]
