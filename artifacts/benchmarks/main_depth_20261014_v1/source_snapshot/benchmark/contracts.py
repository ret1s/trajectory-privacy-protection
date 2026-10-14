"""Auditable contracts for trajectory-protection benchmark methods.

``ProtectedRun`` defines *what* a method publishes.  The classes here define
*what implementation was actually run*.  Keeping those statements separate
prevents a runnable approximation from silently becoming a claimed paper
reproduction in an experiment table.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Protocol, runtime_checkable

from core.demo_protocol import OutputKind, ProtectedRun, PublicTranscript


class ImplementationLevel(str, Enum):
    """Evidence level of executable code relative to the named method."""

    OFFICIAL = "official"
    FAITHFUL_REIMPLEMENTATION = "faithful_reimplementation"
    PAPER_ADAPTATION = "paper_adaptation"
    THESIS_CANDIDATE = "thesis_candidate"

    @property
    def reportable_as_reproduced_sota(self) -> bool:
        return self in {
            ImplementationLevel.OFFICIAL,
            ImplementationLevel.FAITHFUL_REIMPLEMENTATION,
        }


class ComponentStatus(str, Enum):
    """Relationship between one executable component and the source method."""

    IMPLEMENTED = "implemented"
    ADAPTED = "adapted"
    MISSING = "missing"


@dataclass(frozen=True)
class SourceReference:
    """Stable bibliographic/code pointer for a benchmark method."""

    citation: str
    doi: str | None = None
    repository_url: str | None = None
    repository_revision: str | None = None

    def __post_init__(self) -> None:
        if not self.citation.strip():
            raise ValueError("source citation must be non-empty")
        if self.doi is not None and not self.doi.strip():
            raise ValueError("doi must be non-empty when provided")
        if self.repository_url is not None and not self.repository_url.startswith(
            ("https://", "http://")
        ):
            raise ValueError("repository_url must be an HTTP(S) URL")
        if self.repository_revision is not None and not self.repository_revision.strip():
            raise ValueError("repository_revision must be non-empty when provided")

    def to_dict(self) -> dict[str, str | None]:
        return {
            "citation": self.citation,
            "doi": self.doi,
            "repository_url": self.repository_url,
            "repository_revision": self.repository_revision,
        }


@dataclass(frozen=True)
class ComponentMapping:
    """Source-to-code trace for one meaningful algorithm component."""

    component: str
    status: ComponentStatus
    implementation: str
    evidence: str

    def __post_init__(self) -> None:
        if not isinstance(self.status, ComponentStatus):
            raise TypeError("component status must be a ComponentStatus")
        for value, name in (
            (self.component, "component"),
            (self.implementation, "implementation"),
            (self.evidence, "evidence"),
        ):
            if not value.strip():
                raise ValueError(f"{name} must be non-empty")

    def to_dict(self) -> dict[str, str]:
        return {
            "component": self.component,
            "status": self.status.value,
            "implementation": self.implementation,
            "evidence": self.evidence,
        }


class MethodUnavailableError(RuntimeError):
    """Raised when a requested evidence level is unavailable locally."""


@dataclass(frozen=True)
class MethodCard:
    """Machine-readable implementation and reporting status for one method."""

    method_id: str
    display_name: str
    output_kind: OutputKind
    implementation_level: ImplementationLevel
    source: SourceReference
    source_mapping: tuple[ComponentMapping, ...]
    adaptation_summary: str
    validation_evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.method_id.strip() or not self.display_name.strip():
            raise ValueError("method_id and display_name must be non-empty")
        if not isinstance(self.output_kind, OutputKind):
            raise TypeError("output_kind must be an OutputKind")
        if not isinstance(self.implementation_level, ImplementationLevel):
            raise TypeError("implementation_level must be an ImplementationLevel")
        if not self.source_mapping:
            raise ValueError("source_mapping must describe at least one component")
        if not self.adaptation_summary.strip():
            raise ValueError("adaptation_summary must be non-empty")
        validation_evidence = tuple(self.validation_evidence)
        if any(
            not isinstance(item, str) or not item.strip()
            for item in validation_evidence
        ):
            raise ValueError("validation_evidence entries must be non-empty strings")
        if len(validation_evidence) != len(set(validation_evidence)):
            raise ValueError("validation_evidence entries must be unique")
        components = [item.component for item in self.source_mapping]
        if len(components) != len(set(components)):
            raise ValueError("source_mapping component names must be unique")
        object.__setattr__(self, "validation_evidence", validation_evidence)

    @property
    def missing_components(self) -> tuple[str, ...]:
        return tuple(
            item.component
            for item in self.source_mapping
            if item.status is ComponentStatus.MISSING
        )

    @property
    def has_reproduction_source_evidence(self) -> bool:
        """Whether the claimed implementation level has a verifiable source.

        Official code must identify a pinned upstream repository revision.
        A faithful clean-room reimplementation must identify either the source
        publication by DOI or a pinned repository revision.  Lower evidence
        levels are intentionally ineligible regardless of their references.
        """

        if self.implementation_level is ImplementationLevel.OFFICIAL:
            return bool(
                self.source.repository_url and self.source.repository_revision
            )
        if self.implementation_level is ImplementationLevel.FAITHFUL_REIMPLEMENTATION:
            return bool(
                self.source.doi
                or (
                    self.source.repository_url
                    and self.source.repository_revision
                )
            )
        return False

    @property
    def reproduction_reporting_blockers(self) -> tuple[str, ...]:
        """Return every reason this card cannot support a reproduced-SOTA label."""

        blockers: list[str] = []
        eligible_level = self.implementation_level.reportable_as_reproduced_sota
        if not eligible_level:
            blockers.append(
                "implementation level is neither official nor faithful reimplementation"
            )
        if self.missing_components:
            blockers.append(
                "missing components: " + ", ".join(self.missing_components)
            )
        if eligible_level and not self.has_reproduction_source_evidence:
            blockers.append("source evidence is not sufficient for the claimed level")
        if not self.validation_evidence:
            blockers.append("paper-equivalent validation evidence is absent")
        return tuple(blockers)

    @property
    def reportable_as_reproduced_sota(self) -> bool:
        return not self.reproduction_reporting_blockers

    def require_faithful(self) -> None:
        """Fail closed before a run is labelled as a reproduced SOTA result."""

        if self.reportable_as_reproduced_sota:
            return
        blockers = "; ".join(self.reproduction_reporting_blockers)
        raise MethodUnavailableError(
            f"{self.display_name} is available only as "
            f"{self.implementation_level.value}; a reproduced-SOTA run requires "
            f"complete source-mapped code and validation. Missing: {blockers}."
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "method_id": self.method_id,
            "display_name": self.display_name,
            "output_kind": self.output_kind.value,
            "implementation_level": self.implementation_level.value,
            "reportable_as_reproduced_sota": self.reportable_as_reproduced_sota,
            "source": self.source.to_dict(),
            "source_mapping": [item.to_dict() for item in self.source_mapping],
            "adaptation_summary": self.adaptation_summary,
            "missing_components": list(self.missing_components),
            "has_reproduction_source_evidence": (
                self.has_reproduction_source_evidence
            ),
            "validation_evidence": list(self.validation_evidence),
            "reproduction_reporting_blockers": list(
                self.reproduction_reporting_blockers
            ),
        }

    def public_parameters(self) -> dict[str, str | bool]:
        """Compact JSON-scalar status embedded in attacker-visible provenance."""

        return {
            "implementation_level": self.implementation_level.value,
            "reportable_as_reproduced_sota": self.reportable_as_reproduced_sota,
            "source_citation": self.source.citation,
            "source_doi": self.source.doi or "not_available",
        }


@runtime_checkable
class BenchmarkMethod(Protocol):
    """Runtime-checkable minimum interface used by benchmark orchestration."""

    name: str
    method_card: MethodCard

    def protect_run(self, real_trajectory: Any) -> ProtectedRun:
        ...


class MethodCardMixin:
    """Attach a :class:`MethodCard` to an existing mechanism implementation."""

    method_card: MethodCard

    @property
    def source_method(self) -> str:
        return self.method_card.source.citation

    @classmethod
    def require_faithful(cls) -> None:
        cls.method_card.require_faithful()

    def implementation_metadata(self) -> dict[str, Any]:
        return self.method_card.to_dict()

    def _attach_method_card(self, run: ProtectedRun) -> ProtectedRun:
        """Replace legacy demo labels with explicit implementation evidence."""

        parameters = dict(run.transcript.public_parameters)
        parameters.pop("demo_only", None)
        parameters.pop("source_method", None)
        parameters.update(self.method_card.public_parameters())
        transcript = replace(
            run.transcript,
            mechanism=self.method_card.method_id,
            public_parameters=tuple(parameters.items()),
        )
        if transcript.output_kind is not self.method_card.output_kind:
            raise ValueError(
                f"{self.method_card.method_id} emitted {transcript.output_kind.value}, "
                f"expected {self.method_card.output_kind.value}"
            )
        return ProtectedRun(transcript, run.truth)


__all__ = [
    "BenchmarkMethod",
    "ComponentMapping",
    "ComponentStatus",
    "ImplementationLevel",
    "MethodCard",
    "MethodCardMixin",
    "MethodUnavailableError",
    "SourceReference",
]
