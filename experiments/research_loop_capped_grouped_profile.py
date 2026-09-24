"""Engineering profile on an auxiliary training prefix, not a utility screen."""
from functools import partial
import json
import hashlib
from pathlib import Path
import platform
import resource
import time
import numpy as np
from benchmark.engines.paced_slack import PacedSlackProgressLaneDummy
from benchmark.engines.capped_service import CappedServiceLaneDummy
from benchmark.grouped_capped_service import GroupedCappedServiceLaneDummy
from benchmark.public_poi_context import PublicPoiContext
from benchmark.response_aware_belief import ResponseAwareAnchorModel
from evaluation.lane_travel import LanePoiService
from experiments.research_loop_resources import ROOT, CACHE, load, sha

BASE = ROOT/'artifacts/benchmarks/research_loop'
OUT = BASE/'iteration25_capped_grouped_profile.json'
DATA = ROOT/'artifacts/datasets/research_loop_shadow_v1/dataset.json'


def main():
    if OUT.exists():
        raise FileExistsError('Preserve completed evidence')
    rn, _, context, belief, metadata = load()
    reply = PublicPoiContext(LanePoiService(rn, list(context.pois), k=10), CACHE/'poi10.npz')
    belief = ResponseAwareAnchorModel(belief, reply)
    data = json.loads(DATA.read_text()); family = data['families'][0]
    assert family['split'] == 'auxiliary_train'
    sid = family['sessions'][0]['session_id']; points = data['traces'][sid][::20][:12]
    rows = []
    factories = {'capped90_sparse': partial(CappedServiceLaneDummy, utility_slack=0.),
        'capped90_grouped': partial(GroupedCappedServiceLaneDummy, utility_slack=0.),
        'capped90_grouped_slack03': partial(GroupedCappedServiceLaneDummy, utility_slack=.03)}
    for method, cls in factories.items():
        start = time.perf_counter()
        model = cls(rn, belief_model=belief, k=5, horizon=12, budget=.24, rng=np.random.default_rng(2525))
        setup_ms = (time.perf_counter()-start)*1000
        print('Constructed', method, round(setup_ms, 1), 'ms', flush=True)
        for i, p in enumerate(points):
            model.protect_step(p['lat'], p['lon'], p['time_s']-points[0]['time_s'])
            print(method, 'step', i, round(model.step_ms[-1], 1), 'ms', flush=True)
        rows.append({'method': method, 'setup_ms': setup_ms,
            'capped_index_build_ms': getattr(model, 'capped_index_build_ms', None),
            'steps': len(points), 'states': model.evaluator_states,
            'anchors_sha256': hashlib.sha256(json.dumps(model.evaluator_anchors).encode()).hexdigest(),
            'reference_rows': model.capped_index.reference.shape[0], 'step_ms': model.step_ms, 'mean_ms': float(np.mean(model.step_ms)),
            'p95_ms': float(np.percentile(model.step_ms, 95)),
            'max_rss_bytes': resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*(1 if platform.system() == 'Darwin' else 1024)})
    OUT.write_text(json.dumps({'scope': '12-event auxiliary-training prefix; serial warm-cache engineering profile, no utility selection or controlled comparative speed claim',
        'family_id': family['family_id'], 'session_id': sid, 'resources': metadata,
        'source_sha256': {str(p.relative_to(ROOT)): sha(p) for p in (Path(__file__), DATA,
            ROOT/'benchmark/capped_service_objective.py', ROOT/'benchmark/engines/capped_service.py',
            ROOT/'benchmark/grouped_capped_service.py')},
        'rows': rows}, indent=2, allow_nan=False)+'\n')


if __name__ == '__main__':
    main()
