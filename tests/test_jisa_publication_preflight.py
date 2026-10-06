from copy import deepcopy
import json

import pytest

from experiments.jisa_publication_preflight import (
    content_digest, file_digest, main, read_json, spec_fingerprint, validate_protocol,
)


def write_json(path, data, *, pretty=False):
    path.write_text(json.dumps(data, indent=2 if pretty else None), encoding="utf-8")
    return dict(path=path.name, sha256=file_digest(path), content_sha256=content_digest(data))


def protocol_fixture(tmp_path):
    """Actual independent source/data files, not fabricated digest strings."""
    source = tmp_path / "mechanism.py"
    source.write_text("def protect(x):\n    return x\n", encoding="utf-8")
    attack = tmp_path / "attacker.py"
    attack.write_text("def infer(q):\n    return q[0]\n", encoding="utf-8")
    pins = [dict(path=p.name, sha256=file_digest(p)) for p in (source, attack)]
    records = [dict(record_id=f"trip-{i}", split=split, family_id=f"family-{i}",
                    person_id=f"person-{i}", vehicle_id=f"vehicle-{i}",
                    trace=[[i, 0], [i, 1]])
               for i, split in enumerate(("train", "selection", "confirmation"))]
    fresh = dict(id="fresh-actual", purpose="confirmation", inspection=False,
                 records_path=["records"], record_id_field="record_id", split_field="split",
                 group_fields=["family_id", "person_id", "vehicle_id"], expected_records=3,
                 expected_splits=["train", "selection", "confirmation"],
                 **write_json(tmp_path / "fresh.json", dict(records=records)))
    old = write_json(tmp_path / "old.json", dict(records=[dict(record_id="old-trip", trace=[[100, 0]])]))
    unit = .23 / (6 * 15)
    methods = [dict(id="geoi", fidelity="proposed", output_contract="q_set",
                    configuration=dict(K=5, L=20), source_pins=[pins[0]]),
               dict(id="paper-adapted", fidelity="adaptation", output_contract="point",
                    paper_url="https://www.sciencedirect.com/science/article/pii/example",
                    configuration=dict(K=1), source_pins=[pins[0]])]
    attackers = [dict(id="shape-attacker", configuration=dict(seed_policy="independent", neighbors=5),
                      source_pins=[pins[1]])]
    metrics = [dict(id="Hit100", kind="privacy", unit="1", direction="lower",
                    definition="Fraction of attacker estimates within 100 m, family-macro averaged.",
                    supported_contracts=["q_set", "point"],
                    methods={m["id"]: dict(status="supported") for m in methods})]
    return dict(schema_version=1, protocol_id="unit-test-study", stage="frozen",
                primary_claim="Predeclared coordinate privacy/utility tradeoff.", analysis_unit="family",
                split_unit=["family_id", "person_id", "vehicle_id"], source_pins=pins,
                privacy_budget=dict(distance_unit="m", time_unit="s", epsilon_unit="m^-1",
                                    epoch_cap=.23, public_slots=6, public_horizon=8,
                                    epsilon_unit_value=unit, nominal_session_cap=16*unit,
                                    effective_session_cap=15*unit), methods=methods, attackers=attackers,
                metrics=metrics, confirmation=dict(dataset_registrations=[fresh], historical_datasets=[old],
                    frozen_methods={m["id"]: spec_fingerprint(m) for m in methods},
                    frozen_attackers={a["id"]: spec_fingerprint(a) for a in attackers}))


def codes(result, key="errors"):
    return {finding["code"] for finding in result[key]}


def refresh_dataset(tmp_path, registration, data):
    registration.update(write_json(tmp_path / registration["path"], data))


def test_preflight_passes_only_actual_bound_uninspected_registration(tmp_path):
    protocol = protocol_fixture(tmp_path)
    result = validate_protocol({"publication_preflight": protocol, "narrative": "ignored"}, root=tmp_path)
    assert result["integrity_valid"] and result["confirmation_eligible"]
    assert result["status"] == "confirmation_eligible"
    assert result["protocol_content_sha256"] == content_digest(protocol)
    assert result["registered_datasets"][0]["split_counts"] == dict(train=1, selection=1, confirmation=1)
    assert "scores" not in result
    assert len(result["verified_source_pins"]) == 4
    assert any("not publication readiness" in line for line in result["limitations"])


def test_draft_missing_fresh_holdout_is_pending_and_confirmation_cli_fails(tmp_path, capsys):
    protocol = protocol_fixture(tmp_path)
    protocol["stage"] = "draft"
    protocol["confirmation"]["dataset_registrations"] = []
    protocol["confirmation"]["frozen_methods"] = {}
    path = tmp_path / "protocol.json"
    write_json(path, protocol)
    assert main(["--protocol", str(path), "--root", str(tmp_path)]) == 0
    draft = json.loads(capsys.readouterr().out)
    assert draft["status"] == "blocked_confirmation" and draft["integrity_valid"]
    assert "missing_fresh_holdout" in codes(draft, "confirmation_blockers")
    assert main(["--protocol", str(path), "--root", str(tmp_path), "--check-confirmation"]) == 2
    assert not json.loads(capsys.readouterr().out)["confirmation_eligible"]


def test_source_byte_tampering_and_attacker_config_changes_are_detected(tmp_path):
    protocol = protocol_fixture(tmp_path)
    (tmp_path / "mechanism.py").write_text("def protect(x):\n    return 0\n", encoding="utf-8")
    result = validate_protocol(protocol, root=tmp_path)
    assert not result["confirmation_eligible"] and "hash_mismatch" in codes(result)
    protocol = protocol_fixture(tmp_path)
    protocol["attackers"][0]["configuration"]["neighbors"] = 1
    result = validate_protocol(protocol, root=tmp_path)
    assert "confirmation_fingerprint_mismatch" in codes(result)
    assert not result["confirmation_eligible"]


def test_whitespace_reserialized_old_content_is_not_a_fresh_cohort(tmp_path):
    protocol = protocol_fixture(tmp_path)
    fresh = protocol["confirmation"]["dataset_registrations"][0]
    data = read_json(tmp_path / fresh["path"])
    history = write_json(tmp_path / "old_copy.json", data, pretty=True)
    assert history["sha256"] != fresh["sha256"]
    assert history["content_sha256"] == fresh["content_sha256"]
    protocol["confirmation"]["historical_datasets"].append(history)
    result = validate_protocol(protocol, root=tmp_path)
    assert result["integrity_valid"]
    assert not result["confirmation_eligible"]
    assert "historical_content_reuse" in codes(result, "confirmation_blockers")


def test_actual_vehicle_overlap_rejects_a_declared_disjoint_cohort(tmp_path):
    protocol = protocol_fixture(tmp_path)
    registration = protocol["confirmation"]["dataset_registrations"][0]
    data = read_json(tmp_path / registration["path"])
    # Family/person IDs differ, but the same physical vehicle links training and confirmation.
    data["records"][2]["vehicle_id"] = data["records"][0]["vehicle_id"]
    refresh_dataset(tmp_path, registration, data)
    result = validate_protocol(protocol, root=tmp_path)
    assert "cross_split_group_leakage" in codes(result, "confirmation_blockers")
    assert not result["confirmation_eligible"]
    registration["group_fields"].remove("vehicle_id")
    result = validate_protocol(protocol, root=tmp_path)
    assert "missing_disjointness_units" in codes(result)


@pytest.mark.parametrize("change,expected", [
    (lambda r: r.update(inspection=True), "already_inspected_or_undeclared"),
    (lambda r: r.pop("inspection"), "already_inspected_or_undeclared"),
    (lambda r: r.update(path="not-generated.json"), "unregistered_file"),
])
def test_inspection_or_unmaterialized_dataset_blocks_confirmation(tmp_path, change, expected):
    protocol = protocol_fixture(tmp_path)
    change(protocol["confirmation"]["dataset_registrations"][0])
    result = validate_protocol(protocol, root=tmp_path)
    assert result["integrity_valid"] and not result["confirmation_eligible"]
    assert expected in codes(result, "confirmation_blockers")


def test_native_faithful_baseline_label_requires_bound_review_not_renaming(tmp_path):
    protocol = protocol_fixture(tmp_path)
    method = protocol["methods"][1]
    assert method["fidelity"] == "adaptation"
    method["fidelity"] = "native_faithful"
    protocol["confirmation"]["frozen_methods"][method["id"]] = spec_fingerprint(method)
    result = validate_protocol(protocol, root=tmp_path)
    assert "unsupported_native_fidelity" in codes(result)
    assert not result["confirmation_eligible"]
    review = dict(schema_version=1, method_id=method["id"], status="verified_native",
                  paper_url=method["paper_url"], output_contract=method["output_contract"],
                  method_fingerprint=spec_fingerprint(method), reviewed_by="test-reviewer",
                  matched_dimensions=["algorithm", "parameters", "output_contract", "threat_model"])
    pin = write_json(tmp_path / "fidelity.json", review)
    method["fidelity_audit"] = {key: pin[key] for key in ("path", "sha256")}
    assert validate_protocol(protocol, root=tmp_path)["confirmation_eligible"]
    # A parameter change cannot inherit the old fidelity review even if the freeze is updated.
    method["configuration"]["K"] = 5
    protocol["confirmation"]["frozen_methods"][method["id"]] = spec_fingerprint(method)
    result = validate_protocol(protocol, root=tmp_path)
    assert "unsupported_native_fidelity" in codes(result)


def test_output_contract_prevents_forced_native_metric_and_requires_na_reason(tmp_path):
    protocol = protocol_fixture(tmp_path)
    eie = dict(id="EIE", kind="privacy", unit="m", direction="higher",
               definition="Posterior expected estimation loss with declared estimator/prior.",
               supported_contracts=["posterior_distribution"],
               methods={m["id"]: dict(status="supported") for m in protocol["methods"]})
    protocol["metrics"].append(eie)
    result = validate_protocol(protocol, root=tmp_path)
    assert "incompatible_metric_contract" in codes(result)
    eie["methods"] = {m["id"]: dict(status="n/a", reason="No posterior distribution is emitted.")
                      for m in protocol["methods"]}
    assert validate_protocol(protocol, root=tmp_path)["confirmation_eligible"]
    eie["methods"]["geoi"].pop("reason")
    assert "unexplained_na" in codes(validate_protocol(protocol, root=tmp_path))


@pytest.mark.parametrize("key,value,expected", [
    ("epsilon_unit", "km^-1", "budget_unit_mismatch"),
    ("epoch_cap", .0575, "budget_composition_mismatch"),
    ("public_slots", True, "invalid_public_budget_schedule"),
    ("effective_session_cap", .24, "budget_composition_mismatch"),
])
def test_budget_unit_and_linked_session_composition_are_not_interchangeable(tmp_path, key, value, expected):
    protocol = protocol_fixture(tmp_path)
    protocol["privacy_budget"][key] = value
    result = validate_protocol(protocol, root=tmp_path)
    assert expected in codes(result) and not result["confirmation_eligible"]


def test_dataset_byte_and_canonical_pins_and_actual_denominators_are_checked(tmp_path):
    protocol = protocol_fixture(tmp_path)
    registration = protocol["confirmation"]["dataset_registrations"][0]
    data = read_json(tmp_path / registration["path"])
    data["records"][2]["trace"].append([9, 9])
    write_json(tmp_path / registration["path"], data)
    assert "hash_mismatch" in codes(validate_protocol(protocol, root=tmp_path))
    registration["sha256"] = file_digest(tmp_path / registration["path"])
    assert "dataset_content_mismatch" in codes(validate_protocol(protocol, root=tmp_path))
    registration["content_sha256"] = content_digest(data)
    registration["expected_records"] = 999
    assert "record_count_mismatch" in codes(validate_protocol(protocol, root=tmp_path))


def test_null_or_ambiguous_json_cannot_pass_as_actual_dataset(tmp_path):
    protocol = protocol_fixture(tmp_path)
    registration = protocol["confirmation"]["dataset_registrations"][0]
    refresh_dataset(tmp_path, registration, None)
    assert "invalid_dataset_json" in codes(validate_protocol(protocol, root=tmp_path))
    path = tmp_path / registration["path"]
    path.write_text('{"records": [], "records": [{"x": 1}]}', encoding="utf-8")
    registration["sha256"] = file_digest(path)
    assert "invalid_json" in codes(validate_protocol(protocol, root=tmp_path))


def test_cli_is_read_only_and_creates_only_a_new_optional_audit(tmp_path, capsys):
    protocol = protocol_fixture(tmp_path)
    path = tmp_path / "protocol.json"
    write_json(path, protocol)
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    output = tmp_path / "audit.json"
    assert main(["--protocol", str(path), "--root", str(tmp_path), "--output", str(output),
                 "--check-confirmation"]) == 0
    assert json.loads(output.read_text())["confirmation_eligible"]
    assert json.loads(capsys.readouterr().out)["confirmation_eligible"]
    assert all((tmp_path / name).read_bytes() == data for name, data in before.items())
    saved_audit = output.read_bytes()
    with pytest.raises(SystemExit) as exc:
        main(["--protocol", str(path), "--root", str(tmp_path), "--output", str(output)])
    assert exc.value.code == 2 and output.read_bytes() == saved_audit


def test_draft_schema_requires_claim_source_implementation_and_metric_definitions(tmp_path):
    protocol = protocol_fixture(tmp_path)
    protocol.pop("primary_claim")
    protocol["methods"][0]["source_pins"] = []
    protocol["metrics"][0]["unit"] = "%"
    protocol["metrics"][0].pop("definition")
    result = validate_protocol(protocol, root=tmp_path)
    assert {"required_protocol_field", "unbound_implementation", "undefined_metric", "rate_unit_mismatch"}.issubset(codes(result))
    assert not result["integrity_valid"]
