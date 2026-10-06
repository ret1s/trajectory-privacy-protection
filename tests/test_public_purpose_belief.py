from types import SimpleNamespace

import numpy as np
import pytest
from scipy.sparse import csr_matrix

from benchmark.public_purpose_belief import (build_public_purpose_weights,
                                            PurposeCoverAnchorModel)
from tests.test_query_purpose import fixture


def test_public_mixture_distinguishes_speed_and_detour_from_nearest():
    ranking = fixture()
    public = build_public_purpose_weights(ranking, [0], public_destination_states=[3],
                                         public_radius_m=150., k=1)
    # Nearest/radius need a, fastest/detour need b; unreachable c gets no mass.
    assert public.weights.toarray().tolist() == [[.5,.5,0.,0.]]
    assert public.metadata['public_destination_prior'].startswith('uniform')


def test_utility_wrapper_preserves_emission_and_zero_alpha_baseline():
    ranking = fixture()
    public = build_public_purpose_weights(ranking, [0], public_destination_states=[3],
                                         public_radius_m=150., k=1)
    calls = []
    def emission(anchor, previous=None):
        calls.append((anchor,previous))
        return np.array([.0123])
    original = csr_matrix([[1.,0.,0.,0.]])
    prior = np.array([1.])
    base = SimpleNamespace(poi_weights=original,state_ids=np.array([0]),
                           prior=prior,epsilon_release=.0025,sha256='base',emission=emission)
    baseline = PurposeCoverAnchorModel(base,public,alpha=0.)
    changed = PurposeCoverAnchorModel(base,public,alpha=1.)
    assert baseline.poi_weights is original
    assert changed.prior is prior and changed.epsilon_release == .0025
    assert changed.emission((1.,2.)).tolist() == [.0123]
    assert calls == [((1.,2.),None)]
    assert changed.poi_weights.toarray().tolist() == [[.5,.5,0.,0.]]
    assert original.toarray().tolist() == [[1.,0.,0.,0.]]


def test_undefined_public_destination_does_not_create_infinite_weights():
    public = build_public_purpose_weights(fixture(), [4], public_destination_states=[3], k=1)
    assert public.weights.toarray().tolist() == [[0.,0.,1.,0.]]
    assert public.weights[:, -1].sum() == 0


def test_public_builder_rejects_invalid_or_secret_query_arguments():
    ranking = fixture()
    for options in [{'public_destination_states':[]}, {'public_destination_states':[3,3]},
                    {'public_destination_states':[5]},
                    {'public_destination_states':[3],'public_radius_m':float('nan')}]:
        with pytest.raises(ValueError):build_public_purpose_weights(ranking,[0],**options)
    with pytest.raises(TypeError):
        build_public_purpose_weights(ranking,[0],public_destination_states=[3],private_query='cafe')
