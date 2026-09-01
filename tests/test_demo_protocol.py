"""Contract tests for the generalized dummy-generation demo protocol."""

import json

from core.demo_protocol import (
    EvaluationTruth,
    OutputKind,
    ProtectedRun,
    PublicCandidate,
    PublicEvent,
    PublicTranscript,
    TrajectoryPoint,
    make_dummy_only_run,
    make_real_plus_dummies_run,
    make_replacement_run,
)


def _assert_raises(exc_type, message_fragment, call):
    """Dependency-light equivalent of ``pytest.raises(..., match=...)``."""
    try:
        call()
    except exc_type as exc:
        assert message_fragment in str(exc), str(exc)
        return
    except Exception as exc:  # pragma: no cover - produces a clearer failure
        raise AssertionError(
            f"expected {exc_type.__name__}, got {type(exc).__name__}: {exc}"
        ) from exc
    raise AssertionError(f"expected {exc_type.__name__}")


def _track(offset=0.0, n=3):
    return tuple(
        TrajectoryPoint(timestamp_s=float(i * 20), lat=39.9 + offset, lon=116.4 + i * 0.001)
        for i in range(n)
    )


def test_replacement_supports_variable_length_and_hides_truth():
    real = _track(n=3)
    replacement = _track(offset=0.01, n=2)
    run = make_replacement_run("anotherme_demo", real, replacement)

    assert run.transcript.output_kind is OutputKind.REPLACEMENT_TRAJECTORY
    assert len(run.transcript.events) == 2
    assert run.truth.real_trajectory == real
    public = run.to_attacker_dict()
    encoded = json.dumps(public, sort_keys=True)
    assert "truth" not in encoded
    assert "real_candidate" not in encoded
    assert public == run.attacker_view().to_dict()


def test_real_plus_k_minus_one_supports_k_one_and_keeps_label_private():
    real = _track()
    run = make_real_plus_dummies_run(
        "candidate_demo",
        real,
        {"candidate_7": real},
        real_candidate_id="candidate_7",
        public_parameters={"k": 1, "seed": 42},
    )

    assert all(len(event.candidates) == 1 for event in run.transcript.events)
    assert "candidate_7" in json.dumps(run.to_attacker_dict())
    assert "real_candidate_ids" not in json.dumps(run.to_attacker_dict())
    assert run.to_evaluator_dict()["truth"]["real_candidate_ids"] == [
        "candidate_7"
    ] * len(real)


def test_real_plus_dummies_rejects_wrong_truth_label_or_coordinates():
    real = _track()
    dummy = _track(offset=0.02)
    _assert_raises(
        ValueError,
        "absent",
        lambda: make_real_plus_dummies_run(
            "set_demo", real, {"candidate_0": real}, "candidate_missing"
        ),
    )
    _assert_raises(
        ValueError,
        "does not match",
        lambda: make_real_plus_dummies_run(
            "set_demo",
            real,
            {"candidate_0": dummy, "candidate_1": real},
            "candidate_0",
        ),
    )
    _assert_raises(
        ValueError,
        "non-empty",
        lambda: make_real_plus_dummies_run(
            "set_demo", real, {"candidate_0": real}, ""
        ),
    )


def test_dummy_only_has_no_real_member_and_is_deterministic():
    real = _track()
    dummies = {"candidate_0": _track(0.01), "candidate_1": _track(-0.01)}
    run_a = make_dummy_only_run("ours_demo", real, dummies, public_parameters={"seed": 9})
    run_b = make_dummy_only_run("ours_demo", real, dummies, public_parameters={"seed": 9})

    assert run_a.transcript.output_kind is OutputKind.DUMMY_ONLY
    assert run_a.truth.real_candidate_ids == ()
    assert run_a.to_attacker_dict() == run_b.to_attacker_dict()
    assert len(run_a.transcript.events[0].candidates) == 2


def test_public_schema_rejects_private_parameter_and_duplicate_candidate_ids():
    point = PublicCandidate("candidate_0", 39.9, 116.4)
    _assert_raises(
        ValueError,
        "private field",
        lambda: PublicTranscript(
            mechanism="bad",
            output_kind=OutputKind.DUMMY_ONLY,
            events=(PublicEvent("e0", 0.0, (point,)),),
            public_parameters=(("real_candidate_id", "candidate_0"),),
        ),
    )
    _assert_raises(
        ValueError,
        "unique",
        lambda: PublicEvent("e0", 0.0, (point, point)),
    )


def test_output_kind_validation_prevents_truth_labels_on_non_set_outputs():
    event = PublicEvent("e0", 0.0, (PublicCandidate("candidate_0", 39.9, 116.4),))
    transcript = PublicTranscript("ours", OutputKind.DUMMY_ONLY, (event,))
    truth = EvaluationTruth(
        (TrajectoryPoint(0.0, 39.9, 116.4),), ("candidate_0",)
    )
    _assert_raises(
        ValueError,
        "must not designate",
        lambda: ProtectedRun(transcript, truth),
    )


def test_tracks_must_be_time_aligned():
    real = _track()
    shifted_time = tuple(
        TrajectoryPoint(point.timestamp_s + 1.0, point.lat + 0.01, point.lon)
        for point in real
    )
    _assert_raises(
        ValueError,
        "timestamps do not align",
        lambda: make_dummy_only_run(
            "ours", real, {"candidate_0": real, "candidate_1": shifted_time}
        ),
    )
