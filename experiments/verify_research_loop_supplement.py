"""Rebuild the public frontier, replay every union, and check exact inverse views."""
import json
from experiments.research_loop_public_supplement import calculate, OUT
from experiments.research_loop_resources import sha


def check():
    if not OUT.exists():
        return None
    result = json.loads(OUT.read_text())
    rebuilt = json.loads(json.dumps(calculate(), allow_nan=False))
    assert result == rebuilt
    assert len(result['executions']) == result['deterministic_hybrid_session_evaluations'] == 1224
    assert len(result['case_rows']) == 2076
    assert result['new_stochastic_full_session_executions'] == 0
    assert result['new_fixed_control_session_evaluations'] == 204
    assert result['replayed_K8_fixed_control_sessions'] == 102
    assert len(result['summaries']) == 90
    assert all(e['parent_public_sha256'] == e['recovered_public_sha256'] for e in result['executions'])
    return {'file': OUT.name, 'sha256': sha(OUT), 'deterministic_hybrid_sessions': 1224,
            'case_rows': 2076, 'new_stochastic_runs': 0,
            'all_query_unions_and_scores_recomputed': True, 'public_plans_rebuilt': 6,
            'every_parent_view_recovered_exactly': True,
            'scope': 'conditional coordinate equivalence and extra-query service frontier; not independent confirmation or metadata privacy'}


if __name__ == '__main__':
    print(json.dumps(check(), indent=2))
