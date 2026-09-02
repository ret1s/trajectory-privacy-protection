"""Evidence/provenance tests for the reorganized benchmark package."""

from dataclasses import replace
import json

import networkx as nx
import numpy as np

from benchmark.contracts import (
    BenchmarkMethod,
    ComponentMapping,
    ComponentStatus,
    ImplementationLevel,
    MethodCard,
    MethodUnavailableError,
    SourceReference,
)
from benchmark.methods import (
    AnotherMeAdaptation,
    GeoIAnchoredDummyTrajectories,
    SemanticCorrelationComparator,
    TransProtectAdaptation,
)
from benchmark.registry import METHOD_CARDS, method_card, method_inventory
from core.demo_protocol import OutputKind, TrajectoryPoint
from core.road_network import RoadNetwork
from core.sota_demo import AnotherMeLite, SemanticDummyLite, TransProtectLite
from core.thesis_demo import GeoIAnchoredDummyTrajectoriesLite


def _line_network(n=7):
    graph = nx.DiGraph()
    for idx in range(n):
        graph.add_node(idx, y=39.9, x=116.3 + idx * 0.001)
        if idx:
            graph.add_edge(idx - 1, idx, length=100.0)
            graph.add_edge(idx, idx - 1, length=100.0)
    return RoadNetwork(graph)


def _trajectory(rn):
    return tuple(
        TrajectoryPoint(step * 30, *rn.latlon(node))
        for step, node in enumerate((2, 3, 4))
    )


def _reproduction_card(
    *,
    level=ImplementationLevel.OFFICIAL,
    source=None,
    component_status=ComponentStatus.IMPLEMENTED,
    validation_evidence=(),
):
    return MethodCard(
        method_id="evidence_fixture",
        display_name="Evidence fixture",
        output_kind=OutputKind.REPLACEMENT_TRAJECTORY,
        implementation_level=level,
        source=source
        or SourceReference(
            citation="Primary method paper",
            repository_url="https://example.test/official-method",
            repository_revision="0123456789abcdef",
        ),
        source_mapping=(
            ComponentMapping(
                component="complete algorithm",
                status=component_status,
                implementation="benchmark.fixture.protect_run",
                evidence="Source-to-code mapping checked in the fixture.",
            ),
        ),
        adaptation_summary="Fixture used to exercise the evidence gate.",
        validation_evidence=validation_evidence,
    )


def test_registry_has_unique_json_serializable_method_cards():
    inventory = method_inventory()
    ids = [item["method_id"] for item in inventory]

    assert len(inventory) == 4
    assert len(ids) == len(set(ids))
    assert json.loads(json.dumps(inventory)) == inventory
    assert tuple(ids) == tuple(card.method_id for card in METHOD_CARDS)
    for card in METHOD_CARDS:
        assert method_card(card.method_id) is card
        statuses = {mapping.status for mapping in card.source_mapping}
        assert ComponentStatus.IMPLEMENTED in statuses
        assert card.missing_components


def test_all_current_paper_comparators_fail_closed_as_faithful_reproductions():
    for cls in (
        TransProtectAdaptation,
        AnotherMeAdaptation,
        SemanticCorrelationComparator,
    ):
        assert cls.method_card.implementation_level is ImplementationLevel.PAPER_ADAPTATION
        try:
            cls.require_faithful()
        except MethodUnavailableError as exc:
            message = str(exc)
        else:
            raise AssertionError("paper adaptations must not pass a faithful gate")
        assert "Missing:" in message
        assert cls.method_card.display_name in message


def test_reproduced_sota_gate_requires_complete_code_source_and_validation():
    complete = _reproduction_card(
        validation_evidence=("tests/evidence.json: official parity passed",)
    )
    assert complete.reportable_as_reproduced_sota is True
    assert complete.reproduction_reporting_blockers == ()
    complete.require_faithful()

    missing_component = replace(
        complete,
        source_mapping=(
            ComponentMapping(
                component="complete algorithm",
                status=ComponentStatus.MISSING,
                implementation="not implemented",
                evidence="The required component is absent.",
            ),
        ),
    )
    assert missing_component.reportable_as_reproduced_sota is False
    assert "missing components: complete algorithm" in (
        missing_component.reproduction_reporting_blockers
    )

    no_validation = replace(complete, validation_evidence=())
    assert no_validation.reportable_as_reproduced_sota is False
    assert "paper-equivalent validation evidence is absent" in (
        no_validation.reproduction_reporting_blockers
    )

    unpinned_official_source = replace(
        complete,
        source=SourceReference(
            citation="Primary method paper",
            repository_url="https://example.test/official-method",
        ),
    )
    assert unpinned_official_source.reportable_as_reproduced_sota is False
    assert unpinned_official_source.has_reproduction_source_evidence is False


def test_faithful_reimplementation_accepts_primary_source_plus_validation():
    card = _reproduction_card(
        level=ImplementationLevel.FAITHFUL_REIMPLEMENTATION,
        source=SourceReference(
            citation="Primary method paper",
            doi="10.1000/example-doi",
        ),
        validation_evidence=("artifacts/parity.json: reported-table parity passed",),
    )

    assert card.has_reproduction_source_evidence is True
    assert card.reportable_as_reproduced_sota is True
    serialized = card.to_dict()
    assert serialized["validation_evidence"] == [
        "artifacts/parity.json: reported-table parity passed"
    ]
    assert serialized["reproduction_reporting_blockers"] == []


def test_validation_evidence_field_is_backward_compatible_and_fail_closed():
    legacy_shape = _reproduction_card()

    assert legacy_shape.validation_evidence == ()
    assert legacy_shape.reportable_as_reproduced_sota is False
    try:
        legacy_shape.require_faithful()
    except MethodUnavailableError as exc:
        assert "validation evidence is absent" in str(exc)
    else:
        raise AssertionError("a card without validation evidence must fail closed")


def test_executable_methods_match_runtime_contract_and_embed_status():
    rn = _line_network()
    real = _trajectory(rn)
    models = (
        TransProtectAdaptation.from_road_network(
            rn,
            candidate_k=3,
            target_count=3,
            rng=np.random.default_rng(1),
        ),
        AnotherMeAdaptation(
            rn,
            anchor_min_m=50,
            anchor_max_m=500,
            minimum_raw_samples=1,
            rng=np.random.default_rng(2),
        ),
        SemanticCorrelationComparator(rn, k=3, rng=np.random.default_rng(3)),
        GeoIAnchoredDummyTrajectories(
            0.02, rn, k=2, rng=np.random.default_rng(4)
        ),
    )

    for model in models:
        assert isinstance(model, BenchmarkMethod)
        run = model.protect_run(real)
        assert run.transcript.mechanism == model.method_card.method_id
        assert run.transcript.output_kind is model.method_card.output_kind
        parameters = dict(run.transcript.public_parameters)
        assert parameters["implementation_level"] == (
            model.method_card.implementation_level.value
        )
        assert parameters["reportable_as_reproduced_sota"] is False
        assert parameters["source_citation"] == model.method_card.source.citation
        assert parameters["implementation_origin"].startswith("benchmark.")
        assert "demo_only" not in parameters


def test_legacy_lite_imports_remain_available_during_migration():
    # Existing scripts do not break immediately; new orchestration imports the
    # source-mapped classes from benchmark.methods instead.
    assert TransProtectLite.name == "transprotect_lite"
    assert AnotherMeLite.name == "anotherme_lite"
    assert SemanticDummyLite.name == "semantic_dummy_lite"
    assert GeoIAnchoredDummyTrajectoriesLite.name == "geo_i_anchored_dummy_lite"
