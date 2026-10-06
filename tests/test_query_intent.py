import numpy as np

from evaluation.query_intent import wire_features, selected_attack


def test_attacker_reads_wire_fields_and_not_external_truth():
    request={'coordinate':[39.99,116.32],'timestamp_s':0.,'categories':['cafe','fuel']}
    a=wire_features([request],[[[0],[1]]],['cafe','fuel'],['nearest','detour'])
    b=wire_features([dict(request,purpose='detour')],[[[0],[1]]],['cafe','fuel'],['nearest','detour'])
    assert not np.array_equal(a,b)
    assert np.isfinite(a).all()
    # An evaluator label outside these arguments cannot affect the features.
    assert np.array_equal(a,wire_features([request],[[[0],[1]]],['cafe','fuel'],['nearest','detour']))


def test_strong_attacker_positive_control_and_noninformative_control():
    y=np.tile([0,1],12)
    x=np.c_[y,np.ones(len(y))]
    result=selected_attack(x,y,x,y,x,y,[0,1])
    assert result['test']['balanced_accuracy']==1.
    hidden=np.ones_like(x)
    result=selected_attack(hidden,y,hidden,y,hidden,y,[0,1])
    assert result['test']['balanced_accuracy']==.5
