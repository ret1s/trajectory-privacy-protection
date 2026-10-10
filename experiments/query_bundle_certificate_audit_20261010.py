"""Static exact audit of tighter PQB privacy and actual-utility certificates.

No source of the frozen v2 audit or active trajectory engine is modified.
All public floors, infeasible cases and parameter rules are retained. Belief
mismatch is explicit; public-floor utility does not depend on calibration.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp

from benchmark.probabilistic_query_bundle import ProbabilisticQueryBundle, bundle_coverage
from benchmark.query_bundle_bounds import (public_privacy_certificate, calibrated_beta,
                                          public_floor_indices, utility_certificate)
from experiments import query_bundle_audit_20261010 as base

OUT = base.ROOT/'artifacts/benchmarks/query_bundle_certificates_20261010_v1'
CONFIG = dict(minimum_public_coverage=[0., .75, .8, .9], modes=['fixed5', 'dynamic_3_5_7'],
              beta_rules=['literal_beta2', 'public_epsilon1_calibration'],
              cost_weight=.25, public_epsilon_target=1.,
              belief_cases=['matched_uniform', 'matched_skewed80', 'uniform_client_skewed80_truth'],
              scope='one fresh REM, static native map, 36 public prior states; nearest-reference proxy',
              floor_policy='all public proxy states; strict >= comparison, no current-GPS/Z pruning',
              design_status='development after inspection of PUBLIC coverage geometry; no private holdout selection')


def pins():
    names = list(base.source_pins()) + ['benchmark/query_bundle_bounds.py',
              'experiments/query_bundle_certificate_audit_20261010.py', 'tests/test_query_bundle_bounds.py',
              'tests/test_probabilistic_query_bundle.py']
    return {name: base.sha(base.ROOT/name) for name in names}


def integrate(rem, prior, kernel, belief, posterior):
    cc = kernel.coverage
    model_cover = np.einsum('ij,jk->ik', belief, cc, optimize=False)
    true_cover = np.einsum('ij,jk->ik', posterior, cc, optimize=False)
    scores = (model_cover + kernel.cost_weight*(1-kernel.cardinalities/kernel.cardinalities.max()))/(1+kernel.cost_weight)
    logs = kernel.log_base_weights + .5*kernel.beta*scores
    logs -= logsumexp(logs, axis=1)[:, None]
    channel = np.exp(logs)
    assert np.all(channel > 0)
    cert = public_privacy_certificate(kernel)
    assert float(np.ptp(logs, axis=0).max()) <= cert['epsilon_Q']+1e-11
    metrics = base.readout(rem, prior, channel, cc, kernel.cardinalities)
    assert metrics['maximum_pairwise_log_likelihood_ratio'] <= cert['epsilon_Q']+1e-11
    pz = np.einsum('i,ij->j', prior, rem, optimize=False)
    tv = .5*np.abs(belief-posterior).sum(axis=1)
    predicted = np.sum(channel*model_cover, axis=1)
    true_conditional = np.sum(channel*true_cover, axis=1)
    local_lower = np.maximum(cert['public_min_coverage'], np.maximum(0., predicted-tv))
    assert np.all(true_conditional >= local_lower-1e-11)
    assert np.sum(pz*true_conditional) == np.float64(metrics['expected_recall5']) or abs(np.sum(pz*true_conditional)-metrics['expected_recall5']) < 1e-12
    cap = np.exp(cert['epsilon_Q'])*prior.max()/(np.exp(cert['epsilon_Q'])*prior.max()+1-prior.max())
    assert metrics['max_posterior'] <= cap+1e-11
    # Inspect 36 fixed, public output IDs plus map endpoints: local certificates
    # are validated independently against true conditional utility.
    sample_ids = np.unique(np.r_[0, len(rem.T)//2, len(rem.T)-1,
                                np.linspace(0, len(rem.T)-1, 36, dtype=int)])
    inspections = []
    for i in sample_ids:
        local = utility_certificate(kernel, belief[i], tv_bound=float(tv[i]))
        assert true_conditional[i] >= local['lower_expected_actual_utility']-1e-11
        assert true_cover[i].max()-true_conditional[i] <= local['additive_shortfall_to_best_actual_library_utility']+1e-11
        assert local['exact_surrogate_score_regret'] <= local['variational_score_regret_bound']+1e-11
        np.testing.assert_allclose(logs[i], kernel.log_probabilities(belief[i]), atol=1e-12)
        inspections.append(dict(public_output_state=int(i), **local))
    return dict(**metrics, public_certificate=cert, certified_prior_posterior_cap=float(cap),
                weighted_actual_belief_TV=float(np.sum(pz*tv)),
                weighted_predicted_recall=float(np.sum(pz*predicted)),
                lower_expected_actual_recall_from_bridge=float(np.sum(pz*local_lower)),
                maximum_anchor_channel_log_ratio=float(np.ptp(logs, axis=0).max()),
                inspected_local_certificates=inspections)


def run(output, work):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve evidence; choose a new output directory')
    output.mkdir(parents=True)
    protocol = dict(schema='query-bundle-certificate-audit-v1', configuration=CONFIG,
                    source_sha256=pins(), baseline_audit_protocol_sha256=base.sha(base.OUT/'protocol.json'),
                    limits='float diagnostic of ideal-real proofs; no moving-trajectory/all-purpose/native-identity claim')
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    for name in protocol['source_sha256']:
        target = output/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((base.ROOT/name).read_bytes())
    print('Public protocol frozen; building same map, POIs and bundle library', flush=True)
    rn, reference, reply, ids = base.resources(Path(work))
    actions = base.library(reference, reply, ids)
    cover = bundle_coverage(reference, reply, ids, actions)
    logits = -.5*base.CONFIG['epsilon_release_per_m']*cdist(rn.xy[ids], rn.xy)
    rem = np.exp(logits-logsumexp(logits, axis=1)[:, None])
    uniform = np.full(len(ids), 1./len(ids))
    skewed = np.r_[.8, np.full(len(ids)-1, .2/(len(ids)-1))]
    cases = [('matched_uniform', uniform, uniform), ('matched_skewed80', skewed, skewed),
             ('uniform_client_skewed80_truth', uniform, skewed)]
    rows, infeasible = [], []
    for mode in CONFIG['modes']:
        subset = [j for j,a in enumerate(actions) if mode != 'fixed5' or len(a.states) == 5]
        original = ProbabilisticQueryBundle([actions[j] for j in subset], cover[:, subset], cost_weight=.25)
        for floor in CONFIG['minimum_public_coverage']:
            try:
                kept = public_floor_indices(original, minimum_coverage=floor)
            except ValueError as exc:
                infeasible.append(dict(mode=mode, requested_floor=floor, reason=str(exc)))
                continue
            aa = [original.bundles[j] for j in kept]
            cc = original.coverage[:, kept]
            model = ProbabilisticQueryBundle(aa, cc, beta=2., cost_weight=.25)
            for rule in CONFIG['beta_rules']:
                beta = 2. if rule == 'literal_beta2' else calibrated_beta(model, epsilon_target=1.)
                kernel = ProbabilisticQueryBundle(aa, cc, beta=beta, cost_weight=.25)
                for case, client_prior, true_prior in cases:
                    pzc = np.einsum('i,ij->j', client_prior, rem, optimize=False)
                    pzt = np.einsum('i,ij->j', true_prior, rem, optimize=False)
                    belief = (client_prior[:, None]*rem/pzc).T
                    posterior = (true_prior[:, None]*rem/pzt).T
                    row = integrate(rem, true_prior, kernel, belief, posterior)
                    rows.append(dict(mode=mode, floor=floor, beta_rule=rule, beta=beta,
                                     belief_case=case, public_library_size=len(aa),
                                     public_K_values=sorted(set(map(int, kernel.cardinalities))), **row))
            print(f'Integrated {mode}, floor {floor:g}, {len(aa)} public bundles', flush=True)
    payload = dict(schema='query-bundle-certificate-results-v1',
                   protocol_sha256=base.sha(output/'protocol.json'), native_map_states=len(rn),
                   public_prior_states=ids.tolist(), public_bundles=[list(a.states) for a in actions],
                   catalogue_sha256=rn.catalogue_sha256, rows=rows, infeasible_public_floors=infeasible)
    (output/'results.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    verify(output)


def verify(output):
    output = Path(output)
    p = json.loads((output/'protocol.json').read_text())
    assert p['configuration'] == CONFIG
    for name,digest in p['source_sha256'].items():
        assert base.sha(base.ROOT/name) == digest and base.sha(output/'source_snapshot'/name) == digest, name
    r = json.loads((output/'results.json').read_text())
    assert r['protocol_sha256'] == base.sha(output/'protocol.json')
    assert len(r['rows'])+len(r['infeasible_public_floors'])*6 == 48
    for row in r['rows']:
        cert = row['public_certificate']
        assert row['expected_recall5'] >= row['lower_expected_actual_recall_from_bridge']-1e-11
        assert row['expected_recall5'] >= row['floor']-1e-11
        assert row['maximum_anchor_channel_log_ratio'] <= cert['epsilon_Q']+1e-11
        assert row['maximum_pairwise_log_likelihood_ratio'] <= cert['epsilon_Q']+1e-11
        assert row['max_posterior'] <= row['certified_prior_posterior_cap']+1e-11
        if row['beta_rule'] == 'public_epsilon1_calibration':
            assert cert['epsilon_Q'] <= 1.+1e-11
    print(f'PASS: {len(r["rows"])} exact integrations, retained infeasibility and privacy/utility certificates')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--work', type=Path, default=Path('/private/tmp/query-bundle-audit-20261010'))
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    verify(args.output) if args.verify else run(args.output, args.work)
