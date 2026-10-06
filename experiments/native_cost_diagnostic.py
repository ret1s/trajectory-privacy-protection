"""Post-hoc Eq.13-family diagnostic of frozen no-delay Geo-I Q sets.

All accepted public OSM POIs have a uniform public destination prior. Compute
exact directed graph distances, preserve infinities, and leave undefined events
N/A. A mean over Q is an explicitly labeled extension, not a singleton
TransProtect score or the utility of locally selected POIs.
"""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.sparse.csgraph import dijkstra

from evaluation.lane_travel import matrix
from evaluation.native_comparator_metrics import expected_travel_cost_distortion
from experiments.public_research_resources import ROOT, load_public_research_resources


ENDPOINT = ROOT / "artifacts/benchmarks/endpoint_noise_20261005/round2"
OUT = ROOT / "artifacts/benchmarks/native_cost_diagnostic_20261005"
CACHE = Path("/private/tmp/trajectory-research-20261005-public-map")


def read(path):
    path = Path(path)
    return json.loads(gzip.decompress(path.read_bytes()) if path.suffix == ".gz" else path.read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + "\n")


def strict_query_cost(real_costs, query_costs, prior):
    """Full-public-prior Eq.13 per Q; an event is eligible only if ALL Q are.

    Never give inf-inf a zero error, renormalize on reachable destinations,
    or silently drop an expensive/inaccessible Q. Coverage accompanies N/A.
    """
    real, queries, q = np.asarray(real_costs, float), np.asarray(query_costs, float), np.asarray(prior, float)
    if real.ndim != 1 or queries.ndim != 2 or not len(queries) or queries.shape[1:] != real.shape or q.shape != real.shape:
        raise ValueError("one true cost row, nonempty Q rows, aligned prior required")
    if not np.isfinite(q).all() or np.any(q < 0) or not np.isclose(q.sum(), 1, rtol=1e-8, atol=1e-10):
        raise ValueError("normalized finite public prior required")
    if np.any(np.isnan(real)) or np.any(np.isnan(queries)) or np.any(real < 0) or np.any(queries < 0):
        raise ValueError("nonnegative cost or explicit positive infinity required")
    active = q > 0
    real, queries, q = real[active], queries[:, active], q[active]
    finite = np.isfinite(queries) & np.isfinite(real)[None, :]
    mass = finite @ q
    values = [expected_travel_cost_distortion([real.tolist()], [row.tolist()], q.tolist())
              if np.all(mask) else None for row, mask in zip(queries, finite)]
    eligible = all(value is not None for value in values)
    return {"status": "computed" if eligible else "not_available", "query_count": len(queries),
            "per_query_delta_cost_m": values,
            "mean_of_per_query_delta_cost_m": float(np.mean(values)) if eligible else None,
            "finite_public_prior_mass_per_query": mass.tolist(),
            "reason": "" if eligible else "Positive-prior destination unreachable from truth or at least one Q; no penalty/subsetting."}


def average_by_family(rows, field):
    sessions = defaultdict(list)
    for row in rows:
        if row[field] is not None:
            sessions[row["family_id"], row["session_id"], row["seed"]].append(row[field])
    session_reps = defaultdict(list)
    for (family, session, _), values in sessions.items():
        session_reps[family, session].append(float(np.mean(values)))
    families = defaultdict(list)
    for (family, _), values in session_reps.items():
        families[family].append(float(np.mean(values)))
    by_family = {family: float(np.mean(values)) for family, values in sorted(families.items())}
    return {"mean_m": float(np.mean(list(by_family.values()))) if by_family else None,
            "eligible_event_rows": sum(row[field] is not None for row in rows),
            "attempted_event_rows": len(rows), "eligible_families": len(by_family),
            "attempted_families": len({row["family_id"] for row in rows}), "by_family_m": by_family,
            "aggregation": "equal events within session/seed -> equal seeds -> sessions -> families"}


def run(out=OUT, cache=CACHE):
    out = Path(out)
    if out.exists():
        raise FileExistsError("Refusing to overwrite cost evidence; choose a fresh output directory")
    selection_path = ENDPOINT / "selection.json"
    selection = read(selection_path)
    method = selection["selected_defense"]
    source_paths = [selection_path, ENDPOINT / "protocol.json", ENDPOINT / "resources.json",
                    ENDPOINT / f"test-{method}.json.gz", ENDPOINT / "test-raw.json.gz",
                    ROOT / "artifacts/benchmarks/research_loop/resources.json",
                    ROOT / "artifacts/benchmarks/endpoint_noise_20261005/public_resources.json",
                    ROOT / "artifacts/benchmarks/endpoint_noise_20261005/public_reconstructed.net.xml.gz"]
    source_hashes = {str(path.relative_to(ROOT)): sha(path) for path in source_paths}
    # Destination rule fixed before opening any private transcript or cost score.
    rn, _, _, _, _, ranking, resources = load_public_research_resources(cache)
    expected = read(ENDPOINT / "resources.json")
    if resources["catalogue"]["sha256"] != expected["catalogue"]["sha256"]:
        raise ValueError("Source transcripts and cost graph use different catalogues")
    targets = list(ranking.pois)  # all public accepted POI IDs, lexical order
    prior = np.full(len(targets), 1. / len(targets))
    target_states = np.asarray([poi["vertex"] for poi in targets], dtype=int)
    # Reverse edges to obtain forward origin->destination costs; no undirected fallback.
    full_cost = dijkstra(matrix(rn).transpose().tocsr(), directed=True, indices=target_states).T
    raw, protected = read(ENDPOINT / "test-raw.json.gz"), read(ENDPOINT / f"test-{method}.json.gz")
    key = lambda row: (row["family_id"], row["session_id"], row["seed"])
    raw_by_key = {key(row): row for row in raw}
    if len(raw_by_key) != len(raw) or set(raw_by_key) != {key(row) for row in protected}:
        raise ValueError("Frozen raw/protected cohorts do not align")
    event_rows, states, sample_checks = [], set(), []
    for protected_run in protected:
        raw_run = raw_by_key[key(protected_run)]
        if protected_run["config"].get("delay") or len(protected_run["events"]) != len(raw_run["events"]):
            raise ValueError("This diagnostic requires matched no-delay event clocks")
        for i, (control_event, protected_event) in enumerate(zip(raw_run["events"], protected_run["events"])):
            if control_event["timestamp_s"] != protected_event["timestamp_s"] or len(control_event["coordinates"]) != 1:
                raise ValueError("Raw/protected event times or truth contract differ")
            truth_state = int(rn.nearest(*control_event["coordinates"][0])[0])
            query_states = [int(rn.nearest(*coordinate)[0]) for coordinate in protected_event["coordinates"]]
            if len(query_states) != 5:
                raise ValueError("Selected source configuration must publish exactly five Q")
            states.update([truth_state, *query_states])
            protected_score = strict_query_cost(full_cost[truth_state], full_cost[query_states], prior)
            control_score = strict_query_cost(full_cost[truth_state], full_cost[[truth_state]], prior)
            for label, ids, score in ((method, query_states, protected_score), ("raw", [truth_state], control_score)):
                event_rows.append({"family_id": protected_run["family_id"], "session_id": protected_run["session_id"],
                    "seed": protected_run["seed"], "ordinal": i, "timestamp_s": control_event["timestamp_s"],
                    "method": label, "true_state_evaluator_only": truth_state, "query_states": ids,
                    "metric_contract": "native_singleton_Eq13" if label == "raw" else "Q_only_mean_per_query_Eq13_extension",
                    **score})
            if control_score["status"] == "computed" and control_score["mean_of_per_query_delta_cost_m"] != 0:
                raise ValueError("Raw identical-origin cost must be zero")
    state_ids = np.asarray(sorted(states), dtype=int)
    # Independent NetworkX Dijkstra parity on a deterministic spread of public destinations.
    import networkx as nx
    for target in target_states[np.linspace(0, len(target_states)-1, 8, dtype=int)]:
        reference = nx.single_source_dijkstra_path_length(rn.graph.reverse(copy=False), int(target), weight="length")
        column = int(np.flatnonzero(target_states == target)[0])
        for state in state_ids:
            expected_cost = reference.get(int(state), np.inf)
            if not np.isclose(full_cost[state, column], expected_cost, rtol=1e-10, atol=1e-7):
                raise ValueError("Sparse/native directed cost differs from NetworkX reference")
        sample_checks.append({"destination_state": int(target), "used_state_checks": len(state_ids)})
    out.mkdir(parents=True)
    np.savez_compressed(out / "target_cost_tables.npz", state_ids=state_ids, target_states=target_states,
                        target_prior=prior, cost_m=full_cost[state_ids])
    write(out / "public_targets.json", {"rule": "ALL accepted public OSM POI IDs, uniformly weighted; no private visit/score prior",
        "unit": "m", "targets": targets, "prior": prior.tolist(), "table_row_key": "state_ids",
        "access_policy": "same nearest-lane-state snap as source service; no connector offset added"})
    write(out / "protocol.json", {"schema": "native-public-cost-diagnostic-v1", "date": "2026-10-05",
        "selected_source_method": method, "fixed_destination_prior": "uniform over all accepted public POI IDs",
        "cost": "exact directed shortest-path length on shared reconstructed map",
        "unreachable": "preserve +inf in NPZ; entire Q/event metric stays N/A; no secret target renormalization",
        "source_sha256": source_hashes, "resources": resources,
        "code_sha256": {str(path.relative_to(ROOT)): sha(path) for path in
            (Path(__file__), ROOT / "evaluation/native_comparator_metrics.py", ROOT / "evaluation/lane_travel.py")},
        "scope": "Post-hoc metric-family diagnostic of already selected frozen public Q; previously used SUMO development data. "
                 "Not an untouched confirmation, original-cache rerun, or head-to-head with TransProtect.",
        "truth_evaluator_only": "raw frozen GPS control used ONLY to score costs; no private input to query generation here",
        "table_scope": "all states occurring in paired raw/protected streams, all418public targets; evaluator-only state membership"})
    (out / "event_rows.json.gz").write_bytes(gzip.compress(json.dumps(event_rows, allow_nan=False,
        separators=(",", ":")).encode(), mtime=0))
    summaries = {label: average_by_family([row for row in event_rows if row["method"] == label],
                                        "mean_of_per_query_delta_cost_m") for label in ("raw", method)}
    protected_rows = [row for row in event_rows if row["method"] == method]
    paired_count = sum(row["status"] == "computed" for row in protected_rows)
    values = [value for row in protected_rows if row["status"] == "computed" for value in row["per_query_delta_cost_m"]]
    readout = {"schema": "native-public-cost-diagnostic-readout-v1", "summaries": summaries,
        "paired_eligible_raw_zero_events": paired_count,
        "protected_eligible_Q_distribution_m": {"count": len(values), "median": float(np.median(values)) if values else None,
                                                 "p95": float(np.quantile(values, .95)) if values else None},
        "unreachable_event_rows": sum(row["status"] != "computed" for row in event_rows),
        "target_count": len(targets), "used_states_preserved": len(state_ids),
        "interpretation": "Larger value is MORE release-origin travel-cost distortion, not privacy improvement. "
                          "Raw singleton and five-Q extension have different service contracts. "
                          "Local merge/ranking POI utility must be reported alongside; this is not final service loss.",
        "transprotect_comparator_executed": False,
        "original_comparative_Eq13_still_NA": True, "no_additional_fits": True}
    write(out / "readout.json", readout)
    if not all(sha(ROOT / name) == digest for name, digest in source_hashes.items()):
        raise ValueError("Source artifact changed during diagnostic")
    write(out / "verification.json", {"status": "passed", "sources_unchanged": True,
        "independent_directed_cost_checks": sample_checks, "full_public_prior_mass": float(prior.sum()),
        "raw_finite_cost_checks_zero": True, "undefined_events_retained": True,
        "outputs": {p.name: sha(p) for p in out.iterdir() if p.is_file()}})
    print(json.dumps(readout, ensure_ascii=False), flush=True)
    return readout


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--resources", type=Path, default=CACHE)
    args = parser.parse_args()
    run(args.out, args.resources)
