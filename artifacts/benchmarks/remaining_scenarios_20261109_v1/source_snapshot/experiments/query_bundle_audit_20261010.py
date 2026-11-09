"""Exact one-release development audit on the existing PUBLIC native map/POIs.

No trajectory, private key, split, attacker fitting or protected test transcript.
Enumerates the full REM output domain and every bundle: reproducible expectations
and optimal finite-prior Bayes inference, rather than a lucky Monte Carlo draw.
This is NOT a replacement for the frozen moving-trajectory benchmark.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp

from benchmark.probabilistic_query_bundle import QueryBundle, ProbabilisticQueryBundle, bundle_coverage
from benchmark.public_poi_context import PublicPoiContext
from data.lane_states import build_lane_states
from evaluation.lane_travel import LanePoiService

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/benchmarks/query_bundle_kernel_20261010_v2'
NETWORK = 'artifacts/datasets/future_controlled_20261005_v1/public_native.net.xml.gz'
RESOURCES = 'artifacts/benchmarks/research_loop/resources.json'
CONFIG = dict(grid_quantiles=[.1, .26, .42, .58, .74, .9], K=[3, 5, 7],
              beta=[0., 1., 2., 4., 8.], illustrative_beta=2., cost_weight=.25,
              epsilon_release_per_m=.01, reference_k=5, reply_L=30, spacing_m=40.,
              priors=['uniform_public_grid', 'skewed_80_percent_first_grid_state'],
              library='public greedy reference-cover bundles from each grid state; ordered canonical unique states',
              score='(expected public nearest Recall@5 + cost_weight*(1-K/Kmax))/(1+cost_weight)',
              selection='none: publish entire prespecified beta frontier; no production promotion')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_pins():
    names = [NETWORK, RESOURCES, str(Path(__file__).relative_to(ROOT)),
             'benchmark/probabilistic_query_bundle.py', 'benchmark/public_poi_context.py',
             'data/lane_states.py', 'core/road_network.py', 'data/sumo_demo.py',
             'evaluation/lane_travel.py', 'evaluation/scenario_metrics.py']
    return {name: sha(ROOT/name) for name in names}


def resources(work):
    work.mkdir(parents=True, exist_ok=True)
    net = work/'native.net.xml'
    content = gzip.decompress((ROOT/NETWORK).read_bytes())
    if not net.exists() or net.read_bytes() != content:
        net.write_bytes(content)
    rn = build_lane_states(net, spacing_m=CONFIG['spacing_m'])
    pois = [{k: v for k, v in p.items() if k not in ('vertex', 'access_offset_m')}
            for p in json.loads((ROOT/RESOURCES).read_text())['pois_used']]
    reference = PublicPoiContext(LanePoiService(rn, pois, k=5), work/'reference5.npz')
    reply = PublicPoiContext(LanePoiService(rn, pois, k=30), work/'reply30.npz')
    # Public POI bounding box, not private trajectory bounding box.
    sites = np.array([rn.xy[rn.nearest(p['lat'], p['lon'])[0]] for p in pois])
    lo, hi = sites.min(axis=0), sites.max(axis=0)
    grid = [lo+(hi-lo)*[a, b] for a in CONFIG['grid_quantiles'] for b in CONFIG['grid_quantiles']]
    ids = np.unique(rn.tree.query(grid)[1]).astype(int)
    ids = np.array([i for i in ids if np.any(reference.query_indices(i) >= 0)])
    if len(ids) < 8:
        raise RuntimeError('Insufficient public states with defined POI service')
    return rn, reference, reply, ids


def library(reference, reply, ids):
    single = [QueryBundle((int(i),)) for i in ids]
    cover = bundle_coverage(reference, reply, ids, single)
    actions = set()
    # Greedy public coverage for each public prototype, deterministic state-ID
    # ties. This is offline library construction, NEVER nearest-Z pruning.
    for row in range(len(ids)):
        selected = []
        for k in range(1, max(CONFIG['K'])+1):
            candidates = [int(i) for i in ids if i not in selected]
            proposals = [QueryBundle(tuple(sorted(selected+[i]))) for i in candidates]
            if selected:
                values = bundle_coverage(reference, reply, [ids[row]], proposals)[0]
            else:
                values = cover[row]
            selected.append(candidates[int(np.argmax(values))])
            if k in CONFIG['K']:
                actions.add(QueryBundle(tuple(sorted(selected))))
    return tuple(sorted(actions, key=lambda a: (len(a.states), a.states)))


def readout(rem, prior, channel, cover, cardinalities):
    observation = np.einsum('ij,jk->ik', rem, channel, optimize=False)
    joint = prior[:, None]*observation
    pxq = joint/joint.sum(axis=0)
    return dict(expected_recall5=float(np.sum(joint*cover)),
                expected_K=float(np.sum(joint.sum(axis=0)*cardinalities)),
                bayes_exact_grid_accuracy=float(joint.max(axis=0).sum()),
                max_posterior=float(pxq.max()),
                maximum_pairwise_log_likelihood_ratio=float(np.max(np.log(observation.max(axis=0)/observation.min(axis=0)))))


def run(output, work):
    output = Path(output)
    if output.exists():
        raise FileExistsError('Preserve completed evidence; use --verify or a fresh output directory')
    output.mkdir(parents=True)
    protocol = dict(schema='query-bundle-kernel-audit-v1', configuration=CONFIG, source_sha256=source_pins(),
                    scope='exact ideal one-fresh-REM finite-public-prior static nearest-POI diagnostic; no moving-family benchmark',
                    control='deterministic fixed-K5 maximizer in SAME public library, not the full legacy engine',
                    guarantee='ideal-real arithmetic; float audit is numerical evidence, not certified pure-DP sampling')
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    for name in protocol['source_sha256']:
        target = output/'source_snapshot'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((ROOT/name).read_bytes())
    print('Protocol frozen; constructing public map and L30 replies', flush=True)
    rn, reference, reply, ids = resources(Path(work))
    actions = library(reference, reply, ids)
    cover = bundle_coverage(reference, reply, ids, actions)
    # Full native REM output domain, including duplicate lane coordinates. Belief
    # depends on coordinates: identical coordinates give identical posterior.
    logits = -.5*CONFIG['epsilon_release_per_m']*cdist(rn.xy[ids], rn.xy)
    rem = np.exp(logits-logsumexp(logits, axis=1)[:, None])
    if np.any(rem <= 0):
        raise RuntimeError('Numerical zero REM support; do not silently certify')
    priors = [np.full(len(ids), 1./len(ids)), np.r_[.8, np.full(len(ids)-1, .2/(len(ids)-1))]]
    results = []
    for prior_name, prior in zip(CONFIG['priors'], priors):
        pz = np.einsum('i,ij->j', prior, rem, optimize=False)
        belief = (prior[:, None]*rem/pz).T
        vulnerability_z = float((prior[:, None]*rem).max(axis=0).sum())
        for dynamic in (False, True):
            selected = np.array([i for i, a in enumerate(actions) if dynamic or len(a.states) == 5])
            aa, cc = tuple(actions[i] for i in selected), cover[:, selected]
            if not dynamic:
                # Deterministic fixed-K5 utility control, same candidate support.
                choices = np.argmax(np.einsum('ij,jk->ik', belief, cc, optimize=False), axis=1)
                control = np.zeros((len(rn), len(aa)))
                control[np.arange(len(rn)), choices] = 1.
                # Empty observed actions are removed, without claiming beta-DP.
                used = np.flatnonzero(np.any(control > 0, axis=0))
                control_metrics = readout(rem, prior, control[:, used], cc[:, used], np.full(len(used), 5))
                results.append(dict(prior=prior_name, mode='deterministic_fixed5_library_control', beta=None,
                                    bayes_observing_Z=vulnerability_z, **control_metrics))
            for beta in CONFIG['beta']:
                kernel = ProbabilisticQueryBundle(aa, cc, beta=beta, cost_weight=CONFIG['cost_weight'])
                # Evaluate all Z without Python-loop sampling; same module scores
                # checked against a few posterior rows before exact integration.
                scores = (np.einsum('ij,jk->ik', belief, cc, optimize=False)
                          + CONFIG['cost_weight']*(1-kernel.cardinalities/kernel.cardinalities.max()))/(1+CONFIG['cost_weight'])
                logs = kernel.log_base_weights + .5*beta*scores
                logs -= logsumexp(logs, axis=1)[:, None]
                for index in (0, len(rn)//2, len(rn)-1):
                    np.testing.assert_allclose(logs[index], kernel.log_probabilities(belief[index]), atol=1e-12)
                channel = np.exp(logs)
                metrics = readout(rem, prior, channel, cc, kernel.cardinalities)
                range_log = float(np.max(logs.max(axis=0)-logs.min(axis=0)))
                assert range_log <= beta+1e-12
                assert metrics['maximum_pairwise_log_likelihood_ratio'] <= beta+1e-12
                assert metrics['bayes_exact_grid_accuracy'] <= vulnerability_z+1e-12
                prior_cap = float(np.exp(beta)*prior.max()/(np.exp(beta)*prior.max()+1-prior.max()))
                assert metrics['max_posterior'] <= prior_cap+1e-12
                ks = {str(k): float((pz[:, None]*channel[:, kernel.cardinalities == k]).sum())
                      for k in np.unique(kernel.cardinalities)}
                conditional_mean_k = np.einsum('ij,j->i', channel, kernel.cardinalities, optimize=False)
                results.append(dict(prior=prior_name, mode='dynamic_3_5_7' if dynamic else 'randomized_fixed5', beta=beta,
                                    beta_posterior_cap=prior_cap, bayes_observing_Z=vulnerability_z,
                                    max_anchor_channel_log_ratio=range_log, K_mass=ks,
                                    min_expected_K_given_Z=float(conditional_mean_k.min()),
                                    max_expected_K_given_Z=float(conditional_mean_k.max()), **metrics))
        print(f'Integrated all outputs: {prior_name}', flush=True)
    payload = dict(schema='query-bundle-kernel-results-v1', protocol_sha256=sha(output/'protocol.json'),
                   native_map_states=len(rn), public_prior_states=ids.tolist(),
                   public_prior_latlon=[rn.latlon(int(i)) for i in ids],
                   public_bundles=[list(a.states) for a in actions],
                   reference_sha256=reference.sha256, reply_sha256=reply.sha256,
                   catalogue_sha256=rn.catalogue_sha256, results=results)
    (output/'results.json').write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n')
    verify(output)


def verify(output):
    output = Path(output)
    protocol = json.loads((output/'protocol.json').read_text())
    assert protocol['configuration'] == CONFIG
    for name, digest in protocol['source_sha256'].items():
        assert sha(ROOT/name) == digest and sha(output/'source_snapshot'/name) == digest, name
    result = json.loads((output/'results.json').read_text())
    assert result['protocol_sha256'] == sha(output/'protocol.json')
    assert len(result['results']) == 22
    for row in result['results']:
        assert 0 <= row['expected_recall5'] <= 1
        assert row['bayes_exact_grid_accuracy'] <= row['bayes_observing_Z']+1e-12
        if row['beta'] is not None:
            assert row['max_anchor_channel_log_ratio'] <= row['beta']+1e-12
            assert row['maximum_pairwise_log_likelihood_ratio'] <= row['beta']+1e-12
            assert row['max_posterior'] <= row['beta_posterior_cap']+1e-12
            assert abs(sum(row['K_mass'].values())-1) < 1e-12
    print('PASS: source pins, exact integration, posterior and joint-K likelihood bounds')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--work', type=Path, default=Path('/private/tmp/query-bundle-audit-20261010'))
    parser.add_argument('--verify', action='store_true')
    args = parser.parse_args()
    verify(args.output) if args.verify else run(args.output, args.work)
