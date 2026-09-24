"""Generate resumable whole-session shadow transcripts on auxiliary SUMO data."""
from functools import partial
import gzip
import json
from pathlib import Path
import numpy as np
from benchmark.anchor_belief import PublicAnchorModel
from benchmark.engines.matched_filter import MatchedFilteredProgressCoverLaneDummy
from benchmark.engines.origin_guard import OriginGuardProgressLaneDummy
from benchmark.engines.paced_guard import PacedProgressLaneDummy, PacedOriginGuardLaneDummy
from experiments.research_loop_resources import ROOT, CACHE, load, sha
from experiments.rng_util import rng_from_key

DATA = ROOT/'artifacts/datasets/research_loop_shadow_v1/dataset.json'
OUT = ROOT/'artifacts/benchmarks/research_loop/sequential_shadows'
SOURCES = ('benchmark/engines/quotient_cover.py', 'benchmark/engines/filtered_cover.py',
           'benchmark/engines/progress_cover.py', 'benchmark/engines/matched_filter.py',
           'benchmark/engines/origin_guard.py', 'benchmark/engines/paced_guard.py',
           'benchmark/anchor_belief.py', 'core/mechanisms.py')


def main():
    OUT.mkdir(exist_ok=True)
    if (OUT/'manifest.json').exists():
        raise FileExistsError('Complete evidence already exists')
    rn, service, context, belief, metadata = load()
    _, inv, counts = np.unique(np.floor(rn.xy/120.).astype(np.int64), axis=0,
                               return_inverse=True, return_counts=True)
    prior = 1./counts[inv]; prior /= prior.sum()
    early = PublicAnchorModel(rn, context, prior, epsilon_release=.0025,
                              epsilon_test=.0025, cache_path=CACHE/'belief_origin_quarter.npz')
    opts = dict(belief_model=belief, k=5, horizon=12, budget=.24)
    factories = {'filter_progress': MatchedFilteredProgressCoverLaneDummy,
                 'origin_first': partial(OriginGuardProgressLaneDummy, early_belief_model=early, guard_seconds=0.),
                 'paced': PacedProgressLaneDummy,
                 'paced_origin': partial(PacedOriginGuardLaneDummy, early_belief_model=early, guard_seconds=0.)}
    models = {name: cls(rn, **opts, rng=np.random.default_rng(0)) for name, cls in factories.items()}
    data = json.loads(DATA.read_text())
    provenance = {'dataset_sha256': sha(DATA), 'code_sha256': sha(Path(__file__)),
                  'source_sha256': {p: sha(ROOT/p) for p in SOURCES},
                  'resources_sha256': metadata['resource_sha256'], 'early_belief_sha256': early.sha256}
    manifest = []
    for family in data['families']:
        path = OUT/(family['family_id']+'.json.gz')
        if path.exists():
            shard = json.loads(gzip.decompress(path.read_bytes()))
            assert shard['provenance'] == provenance
            assert len(shard['runs']) == len(family['sessions'])*len(models)
        else:
            runs = []
            for session in family['sessions']:
                sid = session['session_id']; trace = data['traces'][sid]
                points = trace[::20]
                seeds = rng_from_key(sid, schema='research-loop-sequential-shadow-v1').integers(0, 2**63, 2, dtype=np.int64)
                for method, model in models.items():
                    model.anchor_rng, model.dummy_rng = (np.random.default_rng(int(s)) for s in seeds)
                    model.reset(); events = []
                    for i, p in enumerate(points):
                        t = p['time_s']-points[0]['time_s']
                        coords = model.protect_step(p['lat'], p['lon'], t)
                        events.append({'event_id': f'e{i:04d}', 'timestamp_s': t,
                                       'candidates': [{'candidate_id': f'candidate_{j:04d}', 'lat': lat, 'lon': lon}
                                                      for j, (lat, lon) in enumerate(coords)]})
                    assert model.spent_bound <= .23+1e-12
                    runs.append({'method': method, 'session_id_evaluator_only': sid, 'role': session['role'],
                                 'public': {'events': events},
                                 'evaluation_truth': {'current_xy': [rn.point_xy(p['lat'], p['lon']) for p in points],
                                                      'origin_xy': rn.point_xy(trace[0]['lat'], trace[0]['lon']),
                                                      'endpoint_xy': rn.point_xy(trace[-1]['lat'], trace[-1]['lon']),
                                                      'last_time_s': trace[-1]['time_s']-trace[0]['time_s']},
                                 'budget_bound': model.spent_bound,
                                 'evaluator_ledger': model.evaluator_ledger})
            shard = {'schema': 'auxiliary-whole-session-shadow-v1', 'provenance': provenance,
                     'family_id': family['family_id'], 'split': family['split'], 'runs': runs}
            payload = json.dumps(shard, separators=(',', ':'), allow_nan=False).encode()
            temporary = path.with_suffix('.pending')
            temporary.write_bytes(gzip.compress(payload, compresslevel=6, mtime=0))
            temporary.replace(path)
        manifest.append({'file': path.name, 'sha256': sha(path), 'family_id': family['family_id'],
                         'split': family['split'], 'runs': len(shard['runs'])})
        print('Completed shadow family', family['family_id'], len(manifest), '/', len(data['families']), flush=True)
    (OUT/'manifest.json').write_text(json.dumps({'schema': 'sequential-shadow-generation-v1',
        'provenance': provenance, 'scope': 'attacker training/selection only; never defender confirmation',
        'methods': list(models), 'shards': manifest}, indent=2)+'\n')


if __name__ == '__main__':
    main()
