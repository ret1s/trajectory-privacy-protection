from itertools import combinations, product
import json

import numpy as np
import pytest

from benchmark.engines.fair_cover import CoverageObjective, FairCoverLaneDummy, exchange_refine
from benchmark.engines.service_cover import ServiceCoverLaneDummy, greedy_cover
from core.demo_protocol import TrajectoryPoint
from tests.test_belief_lane import fixture


@pytest.mark.parametrize('cap', [None, .9])
def test_exhaustive_objective_gains_submodularity_and_half_bound(cap):
    rng = np.random.default_rng(551)
    for _ in range(12):
        signatures = np.full((6, 2, 3), -1)
        for i, c in product(range(6), range(2)):
            ids = np.arange(c*3, (c+1)*3)[rng.random(3) < .6]
            signatures[i, c, :len(ids)] = ids
        weights = np.r_[rng.random(6), 0.]
        objective = CoverageObjective(signatures, np.arange(6), weights, cap)
        def oracle(selected):
            values = []
            for c in range(2):
                all_ids = set(signatures[:, c].ravel()) - {-1}
                ids = set(signatures[list(selected), c].ravel()) - {-1} if selected else set()
                mass = sum(weights[i] for i in all_ids)
                if cap is None: values.append(sum(weights[i] for i in ids))
                elif mass: values.append(min(cap, sum(weights[i] for i in ids)/mass))
            return sum(values) if cap is None else np.mean(values) if values else 0.
        subsets = [list(s) for n in range(4) for s in combinations(range(6), n)]
        for chosen in subsets:
            assert objective.value(chosen) == pytest.approx(oracle(chosen))
            expected = [oracle(chosen+[i])-oracle(chosen) for i in range(6)]
            assert objective.marginal(np.arange(6), chosen) == pytest.approx(expected)
            for j in range(6):
                assert np.all(objective.marginal(np.arange(6), chosen) + 1e-12 >=
                              objective.marginal(np.arange(6), chosen+[j]))
        groups = [np.arange(3), np.arange(3, 6)]
        ties = lambda j, ids: (ids*0., ids*0.)
        greedy, _ = greedy_cover(groups, objective.marginal, ties)
        selected, history = exchange_refine(groups, greedy, objective, ties)
        assert history[0] == pytest.approx(oracle(greedy))
        assert all(b > a+1e-12 for a, b in zip(history, history[1:]))
        assert objective.value(selected)+1e-12 >= .5*max(oracle(s) for s in product(*groups))


def test_single_exchange_actually_improves_greedy_counterexample():
    # Greedy picks A={0,1,2}, then D={3}; replace A by B={3,4} after
    # choosing C={0,1,2}, a feasible assignment supplied to the refiner.
    s = np.array([[[0,1,2]], [[3,4,-1]], [[0,1,2]], [[3,-1,-1]]])
    objective = CoverageObjective(s, np.arange(4), np.r_[np.ones(5),0.])
    selected, history = exchange_refine([[0,1],[2,3]], [0,2], objective,
                                       lambda j, ids: (ids*0.,ids*0.))
    assert selected == [1,2] and history == [3.,5.]


def test_empty_category_duplicate_slots_and_bad_input():
    s = np.array([[[0],[-1]], [[0],[-1]]])
    objective = CoverageObjective(s, np.arange(2), [1.,0.], .9)
    assert objective.value([0,1]) == pytest.approx(.9)
    assert objective.marginal([1],[0]).tolist() == [0.]
    assert CoverageObjective(s,np.arange(2),[0.,0.],.9).value([0]) == 0
    for cap in (0.,1.1,float('nan')):
        with pytest.raises(ValueError): CoverageObjective(s,np.arange(2),[1.,0.],cap)
    with pytest.raises(ValueError): exchange_refine([[0]], [1], objective, None)


@pytest.mark.parametrize('cap', [None,.9])
def test_causal_prefix_pairing_and_information_boundary(cap):
    rn, _, belief = fixture()
    points = tuple(TrajectoryPoint(i*5.,0.,.0001+i*.0001) for i in range(15))
    def make(exchanges=3):
        return FairCoverLaneDummy(rn,belief_model=belief,category_cap=cap,
                                  max_exchanges=exchanges,rng=np.random.default_rng(19))
    model = make(); public = model.protect_run(points).to_attacker_dict()
    assert make().protect_run(points[:5]).to_attacker_dict()['events'] == public['events'][:5]
    old = ServiceCoverLaneDummy(rn,belief_model=belief,rng=np.random.default_rng(19))
    old_public = old.protect_run(points).to_attacker_dict()
    assert model.evaluator_anchors == old.evaluator_anchors
    if cap is None:
        assert make(0).protect_run(points).to_attacker_dict()['events'] == old_public['events']
    assert model.spent_bound == pytest.approx(.23)
    assert not any(key in json.dumps(public) for key in ('evaluator','category_values','objective_history'))
    for a,b in zip(model.evaluator_states,model.evaluator_states[1:]):
        assert all(y in model.travel.reachable(x,5.) for x,y in zip(a,b))
    a,b=make(),make()
    for m in (a,b): m.anchor.perturb=lambda *args,**kwargs:rn.latlon(15)
    for i in range(15):
        assert a.protect_step(0.,0.,i*5.) == b.protect_step(20.,30.,i*5.)
    assert a.protect_step(float('nan'),float('nan'),75) == b.protect_step(80,90,75)


@pytest.mark.parametrize('method',['geometric','mean_greedy','mean_exchange','capped_exchange'])
def test_worker_resource_reuse_matches_fresh_instances_across_sessions(method):
    from experiments.run_coverage_frontier import generate,make_model
    rn,_,belief=fixture()
    pool={method:make_model(method,rn,belief,.24,0)}
    for seed,n in ((17,5),(99,15),(17,5)):
        points=tuple(TrajectoryPoint(i*5.,0.,.0001+i*.0001) for i in range(n))
        reused=generate(method,points,rn,belief,.24,seed,pool=pool)
        fresh=generate(method,points,rn,belief,.24,seed)
        for key in ('public','evaluator_states','evaluator_anchors','evaluator_objective','spent_bound'):
            assert reused[key]==fresh[key]
