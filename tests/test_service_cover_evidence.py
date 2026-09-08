"""v3 publication guards, independently of score-dependent method selection."""
from copy import deepcopy
import json
from pathlib import Path

import pytest

from experiments.verify_scenario_suite_v3 import verify_new_gates

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('mutation', ['old_confirmation', 'rare_label', 'future_history', 'duplicate'])
def test_v3_gate_rejects_corruption(mutation):
    original = json.loads((ROOT/'artifacts/datasets/urban_scenarios_v3/dataset.json').read_text())
    verify_new_gates(original)
    data = deepcopy(original)
    if mutation == 'old_confirmation':
        next(f for f in data['families'] if f['seed'] == 201)['seed'] = 105
    elif mutation == 'duplicate':
        data['records'][1]['record_id'] = data['records'][0]['record_id']
    elif mutation == 'rare_label':
        next(r for r in data['records'] if r['case_id'] == 'S1.C')['evidence']['category'] = 'restaurant'
    else:
        next(r for r in data['records'] if r['case_id'] == 'S6.C')['evidence']['query_day'] = 2
    with pytest.raises(AssertionError):
        verify_new_gates(data)
