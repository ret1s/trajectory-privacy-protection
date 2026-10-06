"""Read-only independent recomputation from preserved destination-cost tables."""
import argparse
from collections import defaultdict
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from experiments.native_cost_diagnostic import OUT, ROOT


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(out=OUT):
    out = Path(out)
    protocol = json.loads((out / "protocol.json").read_text())
    verification = json.loads((out / "verification.json").read_text())
    for name, digest in protocol["source_sha256"].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in protocol["code_sha256"].items():
        assert sha(ROOT / name) == digest, name
    for name, digest in verification["outputs"].items():
        assert sha(out / name) == digest, name
    tables = np.load(out / "target_cost_tables.npz", allow_pickle=False)
    rows = json.loads(gzip.decompress((out / "event_rows.json.gz").read_bytes()))
    prior, costs = tables["target_prior"], tables["cost_m"]
    assert np.isclose(prior.sum(), 1.) and np.all(prior >= 0)
    assert costs.shape == (len(tables["state_ids"]), len(tables["target_states"]))
    assert np.all(prior == 1. / len(prior)), "This diagnostic specifies uniform public prior"
    states = {int(state): i for i, state in enumerate(tables["state_ids"])}
    assert len(states) == len(tables["state_ids"])
    values_by_run, attempts, checked_queries = defaultdict(list), defaultdict(int), 0
    active = prior > 0
    for row in rows:
        true = costs[states[row["true_state_evaluator_only"]], active]
        queries = costs[[states[state] for state in row["query_states"]]][:, active]
        finite = np.isfinite(queries) & np.isfinite(true)[None, :]
        expected_values = []
        for query, valid in zip(queries, finite):
            # Independent NumPy expression, not the production Eq.13 helper.
            expected_values.append(float((np.abs(query - true) * prior[active]).sum()) if valid.all() else None)
        assert len(expected_values) == len(row["per_query_delta_cost_m"])
        for expected, saved in zip(expected_values, row["per_query_delta_cost_m"]):
            assert (expected is None and saved is None) or (
                expected is not None and saved is not None and np.isclose(expected, saved, rtol=1e-10, atol=1e-7))
        assert np.allclose(finite @ prior[active], row["finite_public_prior_mass_per_query"])
        complete = all(value is not None for value in expected_values)
        assert (row["status"] == "computed") == complete
        method = row["method"]
        attempts[method] += 1
        if complete:
            value = float(np.mean(expected_values))
            assert np.isclose(value, row["mean_of_per_query_delta_cost_m"], rtol=1e-10, atol=1e-7)
            values_by_run[method, row["family_id"], row["session_id"], row["seed"]].append(value)
            if method == "raw":
                assert value == 0
        else:
            assert row["mean_of_per_query_delta_cost_m"] is None
        checked_queries += len(expected_values)
    sessions = defaultdict(list)
    for (method, family, session, _), values in values_by_run.items():
        sessions[method, family, session].append(float(np.mean(values)))
    families = defaultdict(list)
    for (method, family, _), values in sessions.items():
        families[method, family].append(float(np.mean(values)))
    methods = defaultdict(list)
    for (method, _), values in families.items():
        methods[method].append(float(np.mean(values)))
    readout = json.loads((out / "readout.json").read_text())
    for method, values in methods.items():
        assert np.isclose(float(np.mean(values)), readout["summaries"][method]["mean_m"], rtol=1e-10, atol=1e-7)
        assert attempts[method] == readout["summaries"][method]["attempted_event_rows"]
    result = {"status": "passed", "event_rows_recomputed": len(rows), "query_rows_recomputed": checked_queries,
              "sources_and_output_hashes_valid": True, "summary_formula_recomputed": True}
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=OUT)
    args = parser.parse_args()
    verify(args.out)
