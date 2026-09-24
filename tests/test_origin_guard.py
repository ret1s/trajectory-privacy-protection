import math
import numpy as np
import pytest
from scipy.stats import laplace
from benchmark.anchor_belief import PublicAnchorModel
from benchmark.engines.origin_guard import OriginGuardProgressLaneDummy
from core.demo_protocol import TrajectoryPoint
from tests.test_belief_lane import fixture


def make(seed=70, guard=60.):
    rn, context, belief = fixture()
    early = PublicAnchorModel(rn, context, np.linspace(1, 2, len(rn)), spacing_m=2.,
                              epsilon_release=.0025, epsilon_test=.0025)
    model = OriginGuardProgressLaneDummy(rn, belief_model=belief, early_belief_model=early,
                                         guard_seconds=guard, k=3, horizon=12, budget=.24,
                                         rng=np.random.default_rng(seed))
    return rn, context, model


def test_public_phase_and_budget_do_not_read_after_stop():
    _, _, model = make()
    for i in range(70):
        model.protect_step(0., .001, 17.+20*i)
    assert [x['phase_unit_cost'] for x in model.evaluator_ledger[:5]] == [1, 1, 1, 4, 4]
    assert sum(x['cost_units'] for x in model.evaluator_ledger) == model.spent_units <= 92
    assert model.spent_bound <= .23
    assert model.filter_stopped
    model.protect_step(float('nan'), float('nan'), 1417.)
    assert not model.privacy_read_this_step


def test_first_and_regular_emissions_match_actual_primitive():
    _, _, model = make(guard=0)
    model.protect_step(0., .001, 0.)
    assert model.anchor.epsilon == model.anchor.eps_test == .0025
    assert model.belief.model.emission(model.last_anchor).tolist() == model.early_belief_model.emission(model.last_anchor).tolist()
    model.protect_step(0., .001, 20.)
    assert model.anchor.epsilon == model.anchor.eps_test == .01
    assert model.belief.model.emission(model.last_anchor).tolist() == model.belief_model.emission(model.last_anchor).tolist()


def test_causal_output_and_no_ledger_leak():
    points = tuple(TrajectoryPoint(20.*i, 0., .001) for i in range(25))
    rn, context, model = make()
    full = model.protect_run(points).to_attacker_dict()
    prefix = make()[2].protect_run(points[:4]).to_attacker_dict()
    assert full['events'][:4] == prefix['events']
    assert 'spent_units' not in str(full) and 'evaluator_ledger' not in str(full)
    for record in model.evaluator_objective[1:]:
        a, b = record['pre_progress_states'], record['progress_states']
        assert np.array_equal(context.signatures[context.access[a]], context.signatures[context.access[b]])


def test_variable_unit_extended_path_likelihood_bound():
    u, cap = .04, 15
    steps = [1, 1, 4, 4, 4, 4]
    def distribution(secret):
        paths = {((), None, 0, False): 1.}
        for x, units in zip(secret, steps):
            new = {}
            for (history, last, spent, stopped), mass in paths.items():
                if stopped or spent+units*(1 if last is None else 2) > cap:
                    new[history+(('stop', last),), last, spent, True] = mass
                    continue
                eps = units*u
                rem = np.exp(-.5*eps*np.abs(np.array([0., 1.])-x)); rem /= rem.sum()
                q = 0. if last is None else float(laplace.cdf(.5-abs(x-last), scale=1/eps))
                if last is not None:
                    new[history+(('reuse', last),), last, spent+units, False] = mass*q
                for z in (0, 1):
                    new[history+(('fresh', z),), z, spent+units*(1 if last is None else 2), False] = mass*(1-q)*rem[z]
            paths = new
        return {key[0]: value for key, value in paths.items()}
    p, q = distribution([0.]*6), distribution([1.]*6)
    assert p.keys() == q.keys()
    assert sum(p.values()) == pytest.approx(1.)
    assert max(abs(math.log(p[k]/q[k])) for k in p) <= cap*u+1e-12
