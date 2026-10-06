"""Read-only integrity and confirmation checks for a publication protocol.

Accept either the preflight object itself or a study-design JSON containing a
``publication_preflight`` object. No experiment, scoring, selection or dataset
mutation is performed. Hashes establish artifact identity, not a DP theorem,
semantic implementation fidelity, or proof that a human has not inspected data.

Schema version 1 (all paths resolve against --root, default repository root):
  protocol_id, stage=draft|frozen, primary_claim, analysis_unit, split_unit;
  source_pins=[{path, sha256}];
  privacy_budget={distance_unit:"m", time_unit:"s", epsilon_unit:"m^-1",
    epoch_cap, public_slots, public_horizon, epsilon_unit_value,
    nominal_session_cap, effective_session_cap};
  methods=[{id, fidelity:proposed|control|adaptation|native_faithful,
    output_contract, configuration:{}, source_pins:[...],
    paper_url?:"https://...", frozen_sha256?:...,
    fidelity_audit?:{path,sha256}}];
  attackers=[{id, configuration:{}, source_pins:[...], frozen_sha256?:...}];
  metrics=[{id, kind:privacy|utility|cost, unit, direction:lower|higher,
    definition, supported_contracts:[...],
    methods:{method_id:{status:supported|n/a, reason?:...}}}];
  confirmation={dataset_registrations:[], historical_datasets:[...],
    frozen_methods:{id:spec_fingerprint(method)},
    frozen_attackers:{id:spec_fingerprint(attacker)}}.

Actual registration: {id, purpose:"confirmation", path, sha256,
  content_sha256:content_digest(decoded_json), inspection:false,
  records_path:["records"], record_id_field:"record_id", split_field:"split",
  group_fields:["family_id","person_id","vehicle_id"], expected_records:N,
  expected_splits:["train","selection","confirmation"]}.
records_path selects an actual JSON list; each record must contain those fields.
split_unit names the required group field(s), with each group disjoint across
declared splits. Historical datasets use {path,sha256,content_sha256}, recomputed
from the files. No historical inventory means freshness remains unconfirmed.

native_faithful requires a separately pinned JSON audit: {schema_version:1,
  method_id, status:"verified_native", paper_url, output_contract,
  method_fingerprint:spec_fingerprint(method), reviewed_by:nonempty_string,
  matched_dimensions:["algorithm","parameters","output_contract","threat_model"]}.
It records a bound human review; this checker cannot itself reproduce that review.
"""

import argparse
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
SHA256 = re.compile(r"^[0-9a-f]{64}$")
FIDELITIES = {"proposed", "control", "adaptation", "native_faithful"}
NATIVE_DIMENSIONS = {"algorithm", "parameters", "output_contract", "threat_model"}
RATE_METRICS = {"hit100", "recall@5", "auc", "balanced_accuracy"}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def content_digest(value):
    """Digest parsed JSON; ignore formatting/key ordering, retain list ordering."""
    return hashlib.sha256(_canonical(value)).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def spec_fingerprint(spec):
    """Bind source/config/contract/label, excluding the digest and its audit pin."""
    return content_digest({k: v for k, v in spec.items()
                           if k not in {"frozen_sha256", "fidelity_audit"}})


def _unique_object(pairs):
    obj = {}
    for key, value in pairs:
        if key in obj:
            raise ValueError(f"Duplicate JSON key: {key}")
        obj[key] = value
    return obj


def read_json(path):
    raw = Path(path).read_bytes()
    if str(path).endswith(".gz"):
        raw = gzip.decompress(raw)
    def invalid_constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}")
    return json.loads(raw.decode("utf-8"), object_pairs_hook=_unique_object,
                      parse_constant=invalid_constant)


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _digest(value):
    return isinstance(value, str) and bool(SHA256.fullmatch(value))


def _url(value):
    return _text(value) and urlparse(value).scheme == "https" and bool(urlparse(value).netloc)


class _Audit:
    def __init__(self, root):
        self.root = Path(root)
        self.errors = []
        self.blockers = []
        self.verified_sources = {}
        self.datasets = []

    def error(self, code, message, location):
        self.errors.append(dict(code=code, message=message, location=location))

    def block(self, code, message, location):
        self.blockers.append(dict(code=code, message=message, location=location))

    def resolve(self, path):
        candidate = Path(path)
        return candidate if candidate.is_absolute() else self.root / candidate

    def pin(self, pin, location, *, pending_missing=False):
        if not isinstance(pin, dict) or not _text(pin.get("path")) or not _digest(pin.get("sha256")):
            self.error("invalid_pin", "A pin requires path and lowercase SHA-256.", location)
            return None
        path = self.resolve(pin["path"])
        try:
            actual = file_digest(path)
        except OSError:
            if pending_missing:
                self.block("unregistered_file", "A declared dataset file does not yet exist.", location)
            else:
                self.error("missing_source", "A pinned source file is not readable.", location)
            return None
        if actual != pin["sha256"]:
            self.error("hash_mismatch", "Actual file bytes differ from the pinned SHA-256.", location)
            return None
        self.verified_sources[pin["path"]] = actual
        return path

    def parsed(self, path, location):
        try:
            return read_json(path)
        except (OSError, ValueError, UnicodeError, EOFError) as exc:
            self.error("invalid_json", f"Artifact is not unambiguous finite JSON ({type(exc).__name__}).", location)
            return None


def _specs(audit, protocol, key, global_pins):
    specs = protocol.get(key)
    if not isinstance(specs, list) or not specs:
        audit.error("required_specs", f"{key} must be a nonempty list.", key)
        return {}
    result = {}
    for index, spec in enumerate(specs):
        location = f"{key}[{index}]"
        if not isinstance(spec, dict) or not _text(spec.get("id")):
            audit.error("invalid_spec", "Each specification needs a nonempty id.", location)
            continue
        identity = spec["id"]
        if identity in result:
            audit.error("duplicate_spec", "Specification IDs must be unique.", location)
            continue
        result[identity] = spec
        if not isinstance(spec.get("configuration"), dict):
            audit.error("missing_configuration", "Configuration must be an explicit JSON object.", location)
        pins = spec.get("source_pins")
        if not isinstance(pins, list) or not pins:
            audit.error("unbound_implementation", "Each implementation requires actual source pins.", location)
        else:
            for j, pin in enumerate(pins):
                if isinstance(pin, dict) and global_pins.get(pin.get("path")) == pin.get("sha256"):
                    continue
                audit.error("unregistered_source", "Implementation pins must occur in the verified source inventory.",
                            f"{location}.source_pins[{j}]")
        if "frozen_sha256" in spec:
            try:
                matches = _digest(spec["frozen_sha256"]) and spec["frozen_sha256"] == spec_fingerprint(spec)
            except (ValueError, TypeError):
                matches = False
            if not matches:
                audit.error("spec_fingerprint_mismatch", "Frozen implementation/configuration fingerprint differs.", location)
        if key != "methods":
            continue
        if spec.get("fidelity") not in FIDELITIES or not _text(spec.get("output_contract")):
            audit.error("invalid_method_contract", "Each method requires fidelity and output_contract.", location)
        if spec.get("fidelity") in {"adaptation", "native_faithful"} and not _url(spec.get("paper_url")):
            audit.error("missing_paper_source", "Paper adaptations/native methods require an HTTPS paper source.", location)
        if spec.get("fidelity") == "native_faithful":
            path = audit.pin(spec.get("fidelity_audit"), f"{location}.fidelity_audit")
            review = audit.parsed(path, location) if path else None
            try:
                bound = (isinstance(review, dict) and review.get("schema_version") == 1
                         and review.get("method_id") == identity
                         and review.get("status") == "verified_native"
                         and review.get("paper_url") == spec.get("paper_url")
                         and review.get("output_contract") == spec.get("output_contract")
                         and review.get("method_fingerprint") == spec_fingerprint(spec)
                         and _text(review.get("reviewed_by"))
                         and isinstance(review.get("matched_dimensions"), list)
                         and NATIVE_DIMENSIONS.issubset(set(review["matched_dimensions"])))
            except (TypeError, ValueError):
                bound = False
            if not bound:
                audit.error("unsupported_native_fidelity", "Native-faithful label lacks a source/config-bound fidelity review.", location)
    return result


def _budget(audit, protocol):
    budget = protocol.get("privacy_budget")
    if not isinstance(budget, dict):
        audit.error("missing_budget", "An explicit common privacy-budget policy is required.", "privacy_budget")
        return
    for key, unit in (("distance_unit", "m"), ("time_unit", "s"), ("epsilon_unit", "m^-1")):
        if budget.get(key) != unit:
            audit.error("budget_unit_mismatch", f"This REM/Geo-I protocol declares {key} as {unit}.", f"privacy_budget.{key}")
    for key in ("public_slots", "public_horizon"):
        if type(budget.get(key)) is not int or budget[key] < 1:
            audit.error("invalid_public_budget_schedule", "Slot count/horizon must be positive public integers.", f"privacy_budget.{key}")
    for key in ("epoch_cap", "epsilon_unit_value", "nominal_session_cap", "effective_session_cap"):
        if not _number(budget.get(key)) or budget[key] <= 0:
            audit.error("invalid_budget", "Budget values must be positive finite numbers.", f"privacy_budget.{key}")
    if any(e["location"].startswith("privacy_budget") for e in audit.errors):
        return
    h, n, u = budget["public_horizon"], budget["public_slots"], budget["epsilon_unit_value"]
    expected = {"nominal_session_cap": 2*h*u, "effective_session_cap": (2*h-1)*u,
                "epoch_cap": n*(2*h-1)*u}
    for key, value in expected.items():
        if not math.isclose(budget[key], value, rel_tol=1e-10, abs_tol=1e-12):
            audit.error("budget_composition_mismatch", f"{key} does not match the declared public REM schedule.", f"privacy_budget.{key}")


def _metrics(audit, protocol, methods):
    metrics = protocol.get("metrics")
    if not isinstance(metrics, list) or not metrics:
        audit.error("missing_metrics", "Predeclare metric definitions, units and applicability.", "metrics")
        return
    seen = set()
    for index, metric in enumerate(metrics):
        location = f"metrics[{index}]"
        if not isinstance(metric, dict) or not _text(metric.get("id")):
            audit.error("invalid_metric", "Each metric requires a unique id.", location)
            continue
        if metric["id"] in seen:
            audit.error("duplicate_metric", "Metric IDs must be unique.", location)
        seen.add(metric["id"])
        if (metric.get("kind") not in {"privacy", "utility", "cost"}
                or metric.get("direction") not in {"lower", "higher"}
                or not _text(metric.get("unit")) or not _text(metric.get("definition"))):
            audit.error("undefined_metric", "Specify metric kind, direction, unit and definition.", location)
        name = metric["id"].lower()
        if name in RATE_METRICS and metric.get("unit") != "1":
            audit.error("rate_unit_mismatch", "Rates use unit 1 (fractions), not percentages or distance.", location)
        if name in {"mae", "mae_m", "eie", "eie_m"} and metric.get("unit") != "m":
            audit.error("distance_metric_unit_mismatch", "This protocol expresses coordinate-error metrics in meters.", location)
        contracts = metric.get("supported_contracts")
        applicability = metric.get("methods")
        if not isinstance(contracts, list) or not contracts or not all(_text(x) for x in contracts):
            audit.error("missing_metric_contracts", "Declare the actual output contracts supported by this metric.", location)
            contracts = []
        if not isinstance(applicability, dict) or set(applicability) != set(methods):
            audit.error("missing_metric_applicability", "Every registered method needs supported or n/a applicability.", location)
            continue
        for identity, spec in methods.items():
            entry = applicability[identity]
            if not isinstance(entry, dict) or entry.get("status") not in {"supported", "n/a"}:
                audit.error("invalid_metric_applicability", "Applicability must be supported or n/a.", f"{location}.{identity}")
            elif entry["status"] == "supported" and spec.get("output_contract") not in contracts:
                audit.error("incompatible_metric_contract", "A supported score cannot coerce an incompatible output contract.", f"{location}.{identity}")
            elif entry["status"] == "n/a" and not _text(entry.get("reason")):
                audit.error("unexplained_na", "N/A requires a concrete contract or unavailable-evidence reason.", f"{location}.{identity}")


def _freeze(audit, protocol, confirmation, key, specs):
    frozen = confirmation.get(f"frozen_{key}")
    if not isinstance(frozen, dict) or set(frozen) != set(specs):
        audit.block("unfrozen_implementations", f"Freeze every {key} source/configuration fingerprint before confirmation.", f"confirmation.frozen_{key}")
        return
    for identity, spec in specs.items():
        try:
            matches = _digest(frozen[identity]) and frozen[identity] == spec_fingerprint(spec)
        except (ValueError, TypeError):
            matches = False
        if not matches:
            audit.error("confirmation_fingerprint_mismatch", "Confirmation freeze differs from the actual method/attacker specification.", f"confirmation.frozen_{key}.{identity}")
    if protocol.get("stage") != "frozen":
        audit.block("draft_not_frozen", "The protocol stage is draft, not frozen for confirmation.", "stage")


def _dataset(audit, registration, location, required_groups, *, historical=False):
    if not isinstance(registration, dict):
        audit.error("invalid_dataset_registration", "Dataset registration must be an object.", location)
        return None
    path = audit.pin(registration, location, pending_missing=not historical)
    data = audit.parsed(path, location) if path else None
    if data is None:
        if path and not any(e["location"] == location and e["code"] == "invalid_json" for e in audit.errors):
            audit.error("invalid_dataset_json", "A dataset cannot be JSON null.", location)
        return None
    if not isinstance(data, (dict, list)) or not data:
        audit.error("invalid_dataset_json", "Dataset content must be a nonempty JSON object or list.", location)
        return None
    actual_digest = content_digest(data)
    if not _digest(registration.get("content_sha256")) or registration["content_sha256"] != actual_digest:
        audit.error("dataset_content_mismatch", "Content SHA-256 must match actual decoded canonical JSON.", location)
        return None
    if historical:
        return actual_digest
    if not _text(registration.get("id")) or registration.get("purpose") != "confirmation":
        audit.error("invalid_confirmation_role", "A fresh dataset needs an id and explicit confirmation purpose.", location)
    if registration.get("inspection") is not False:
        audit.block("already_inspected_or_undeclared", "Fresh confirmation requires an explicit inspection=false declaration.", location)
    selector = registration.get("records_path")
    records = data
    if not isinstance(selector, list) or not all(_text(key) for key in selector):
        audit.error("invalid_records_path", "records_path must explicitly select the actual list of records.", location)
        return actual_digest
    try:
        for key in selector:
            records = records[key]
    except (KeyError, TypeError):
        records = None
    if not isinstance(records, list) or not records:
        audit.error("missing_actual_records", "Registration must reference a nonempty actual record list.", location)
        return actual_digest
    if type(registration.get("expected_records")) is not int or registration["expected_records"] != len(records):
        audit.error("record_count_mismatch", "The declared record denominator differs from actual records.", location)
    group_fields = registration.get("group_fields")
    if (not isinstance(group_fields, list) or not group_fields or not all(_text(x) for x in group_fields)
            or not set(required_groups).issubset(set(group_fields))):
        audit.error("missing_disjointness_units", "Registration must check all protocol split-unit fields.", location)
        group_fields = []
    id_field, split_field = registration.get("record_id_field"), registration.get("split_field")
    if not _text(id_field) or not _text(split_field):
        audit.error("missing_record_identifiers", "Declare actual record ID and split fields.", location)
        return actual_digest
    expected_splits = registration.get("expected_splits")
    if (not isinstance(expected_splits, list) or not expected_splits or "confirmation" not in expected_splits
            or not all(_text(x) for x in expected_splits) or len(set(expected_splits)) != len(expected_splits)):
        audit.error("invalid_declared_splits", "Declare unique nonempty splits including confirmation.", location)
        expected_splits = []
    identifiers, group_splits, split_counts = set(), {}, {}
    for index, record in enumerate(records):
        at = f"{location}.records[{index}]"
        if not isinstance(record, dict) or id_field not in record or split_field not in record:
            audit.error("missing_record_fields", "Each actual record needs an ID and split.", at)
            continue
        identity, split = record[id_field], record[split_field]
        if not isinstance(identity, (str, int)) or isinstance(identity, bool) or not _text(str(identity)):
            audit.error("invalid_record_identifier", "Record ID must be a nonempty string or integer.", at)
            continue
        if str(identity) in identifiers:
            audit.block("reused_record_identifier", "An actual record ID occurs more than once.", at)
        identifiers.add(str(identity))
        if not _text(split) or split not in expected_splits:
            audit.error("undeclared_actual_split", "An actual record has an undeclared split.", at)
            continue
        split_counts[split] = split_counts.get(split, 0) + 1
        for field in group_fields:
            group = record.get(field)
            if not isinstance(group, (str, int)) or isinstance(group, bool) or not _text(str(group)):
                audit.error("missing_group_identifier", "Each record needs an actual group ID for every split unit.", at)
                continue
            key = field, str(group)
            if key in group_splits and group_splits[key] != split:
                audit.block("cross_split_group_leakage", "The same declared person/vehicle/family group occurs across splits.", at)
            group_splits[key] = split
    if set(split_counts) != set(expected_splits):
        audit.error("empty_or_missing_split", "Every declared split must contain actual records.", location)
    audit.datasets.append(dict(id=registration.get("id"), path=registration.get("path"),
                               sha256=registration.get("sha256"), content_sha256=actual_digest,
                               records=len(records), split_counts=split_counts,
                               disjointness_fields=group_fields, inspection_declared=registration.get("inspection")))
    return actual_digest


def validate_protocol(value, *, root=ROOT):
    """Return integrity errors separately from incomplete confirmation gates."""
    audit = _Audit(root)
    protocol = value.get("publication_preflight", value) if isinstance(value, dict) else value
    if not isinstance(protocol, dict):
        audit.error("invalid_protocol", "Protocol must be a JSON object.", "protocol")
        protocol = {}
    if type(protocol.get("schema_version")) is not int or protocol.get("schema_version") != 1:
        audit.error("unsupported_schema", "Use publication_preflight schema_version=1.", "schema_version")
    for key in ("protocol_id", "primary_claim", "analysis_unit"):
        if not _text(protocol.get(key)):
            audit.error("required_protocol_field", f"A nonempty {key} is required.", key)
    if protocol.get("stage") not in {"draft", "frozen"}:
        audit.error("invalid_stage", "Stage must be draft or frozen.", "stage")
    split_unit = protocol.get("split_unit")
    groups = [split_unit] if _text(split_unit) else split_unit
    if not isinstance(groups, list) or not groups or not all(_text(x) for x in groups):
        audit.error("missing_split_unit", "Declare group field(s) that must be disjoint across splits.", "split_unit")
        groups = []
    global_pins = {}
    pins = protocol.get("source_pins")
    if not isinstance(pins, list) or not pins:
        audit.error("missing_source_inventory", "A nonempty actual source-pin inventory is required.", "source_pins")
        pins = []
    for index, pin in enumerate(pins):
        path = audit.pin(pin, f"source_pins[{index}]")
        if path:
            if pin["path"] in global_pins:
                audit.error("duplicate_source_pin", "Each source path must be registered once.", f"source_pins[{index}]")
            global_pins[pin["path"]] = pin["sha256"]
    _budget(audit, protocol)
    methods = _specs(audit, protocol, "methods", global_pins)
    attackers = _specs(audit, protocol, "attackers", global_pins)
    _metrics(audit, protocol, methods)
    confirmation = protocol.get("confirmation")
    if not isinstance(confirmation, dict):
        audit.error("missing_confirmation_plan", "An explicit confirmation object is required even when pending.", "confirmation")
        confirmation = {}
    for key, specs in (("methods", methods), ("attackers", attackers)):
        _freeze(audit, protocol, confirmation, key, specs)
    histories = confirmation.get("historical_datasets")
    history_digests = set()
    if not isinstance(histories, list) or not histories:
        audit.block("missing_historical_inventory", "Register actual previously inspected/development datasets for freshness comparison.", "confirmation.historical_datasets")
    else:
        for index, history in enumerate(histories):
            digest = _dataset(audit, history, f"confirmation.historical_datasets[{index}]", groups, historical=True)
            if digest:
                history_digests.add(digest)
    registrations = confirmation.get("dataset_registrations")
    if not isinstance(registrations, list) or not registrations:
        audit.block("missing_fresh_holdout", "No actual uninspected confirmation dataset is registered.", "confirmation.dataset_registrations")
    else:
        fresh_digests, ids = set(), set()
        for index, registration in enumerate(registrations):
            location = f"confirmation.dataset_registrations[{index}]"
            digest = _dataset(audit, registration, location, groups)
            if isinstance(registration, dict):
                identity = registration.get("id")
                if _text(identity) and identity in ids:
                    audit.error("duplicate_dataset_id", "Fresh registration IDs must be unique.", location)
                if _text(identity):
                    ids.add(identity)
            if digest:
                if digest in history_digests:
                    audit.block("historical_content_reuse", "Actual canonical content is identical to a registered old dataset.", location)
                if digest in fresh_digests:
                    audit.block("duplicate_fresh_content", "Two fresh registrations contain the same dataset content.", location)
                fresh_digests.add(digest)
    try:
        protocol_digest = content_digest(protocol)
    except (ValueError, TypeError):
        protocol_digest = None
        audit.error("invalid_protocol_content", "Protocol content must be finite JSON.", "protocol")
    valid = not audit.errors
    eligible = valid and not audit.blockers
    return dict(schema_version=1, protocol_id=protocol.get("protocol_id"),
                protocol_content_sha256=protocol_digest,
                status="confirmation_eligible" if eligible else "invalid_protocol" if not valid else "blocked_confirmation",
                integrity_valid=valid, confirmation_eligible=eligible,
                errors=audit.errors, confirmation_blockers=audit.blockers,
                verified_source_pins=audit.verified_sources, registered_datasets=audit.datasets,
                limitations=["No dataset mutation, scoring, model fitting or experiment execution performed.",
                             "Hashes bind code/configuration and files; they do not prove DP or semantic method fidelity.",
                             "inspection=false is an explicit human declaration, not independently provable by this CLI.",
                             "Freshness checks cover registered historical files and canonical JSON equality only; they do not detect every overlapping trajectory.",
                             "Confirmation eligibility is a protocol gate, not publication readiness or acceptance."])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, help="Optional NEW JSON audit file; existing files are never overwritten.")
    parser.add_argument("--check-confirmation", action="store_true", help="Exit nonzero until actual confirmation gates pass.")
    args = parser.parse_args(argv)
    try:
        result = validate_protocol(read_json(args.protocol), root=args.root)
    except (OSError, ValueError, UnicodeError, EOFError) as exc:
        parser.error(f"Cannot read protocol ({type(exc).__name__}).")
    rendered = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if args.output:
        try:
            with args.output.open("x", encoding="utf-8") as stream:
                stream.write(rendered)
        except OSError as exc:
            parser.error(f"Cannot create NEW audit file ({type(exc).__name__}).")
    sys.stdout.write(rendered)
    return 1 if not result["integrity_valid"] else 2 if args.check_confirmation and not result["confirmation_eligible"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
