import numpy as np
from types import SimpleNamespace
from experiments.research_loop_cases import public_view, recall_pair, mean_optional, passes_recall_gate


def test_exact_view_does_not_disclose_hidden_prefix_or_event_index():
    c = [{'candidate_id': 'candidate_0000', 'lat': 0., 'lon': 0.}]
    events = {1: {'timestamp_s': 40., 'candidates': c, 'truth': 'not public'},
              7: {'timestamp_s': 90., 'candidates': c, 'private_read': True},
              8: {'timestamp_s': 110., 'candidates': c}}
    view = public_view(events, [7, 8])
    assert [e['timestamp_s'] for e in view['events']] == [0., 20.]
    assert [e['event_id'] for e in view['events']] == ['e0000', 'e0001']
    assert set(view['events'][0]) == {'event_id', 'timestamp_s', 'candidates'}


def test_empty_reference_is_null_independent_of_candidate_reply():
    context = SimpleNamespace(categories=['cafe'], signatures=np.array([[[-1]], [[0]]]))
    assert recall_pair(context, context, [1], 0) == {'5': None, '10': None}
    assert recall_pair(context, context, [0], 0) == {'5': None, '10': None}
    assert recall_pair(context, context, [0], 1) == {'5': 0., '10': 0.}
    assert mean_optional([None, .5, 1.]) == .75
    assert mean_optional([None]) is None


def test_gate_tolerates_float_roundoff_only():
    assert passes_recall_gate(.8999999999999999)
    assert not passes_recall_gate(.899999)
    assert not passes_recall_gate(None)
