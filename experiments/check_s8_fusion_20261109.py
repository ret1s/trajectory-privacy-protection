"""Independent V2 replay, distinguishing available Raw control from selection.

The frozen V2 verifier incorrectly required a SELECTION-chosen estimator to
generalize perfectly on TEST. Its failure is retained; no score is replaced.
"""
import argparse
from pathlib import Path

import numpy as np

from experiments import s8_fusion_20261109 as study
from experiments.verify_remaining_scenarios_20261109 import summary


def verify(write=False):
    study.validate_sources();base=study.base;out=study.OUT
    result=base.read(out/'results.json');assert result['full_coverage'] and len(result['tasks'])==24
    for n,h in base.read(out/'output_manifest.json')['files_sha256'].items():assert base.sha(out/n)==h,n
    old={t['task']:t for t in base.read(base.OUT/'results.json')['tasks']}
    controls={};selection_checks=0
    for task in result['tasks']:
        name=task['task'];sel=base.read(out/'selection_predictions'/(name+'.json.gz'))
        candidates={n:np.array([r['estimates'][n] for r in sel]) for n in sel[0]['estimates']}
        chosen,scores=base.regression_selection(sel,candidates)
        assert chosen==task['selected']
        assert base.read(out/'selections'/(name+'.json'))==dict(selected=chosen,scores=scores)
        rows=base.read(out/'test_predictions'/(name+'.json.gz'));assert len(rows)==120
        target_task=next(t for t in result['tasks'] if t['task']=='S8--target_only--'+task['method'])
        target_rows=base.read(out/'test_predictions'/(target_task['task']+'.json.gz'))
        assert [r['session_id'] for r in rows]==[r['session_id'] for r in target_rows]
        for row in sel+rows:
            for n,estimate in row['estimates'].items():
                error=np.linalg.norm(np.array(estimate)-row['target_xy'])
                assert np.isclose(error,row['errors'][n],rtol=0,atol=1e-9)
        for metric,key in [('MAE_m','mae'),('Hit100','hit100')]:
            error=np.array([r['errors'][chosen[key]] for r in rows])
            before=np.array([r['errors'][target_task['selected'][key]] for r in target_rows])
            assert summary(rows,error if key=='mae' else error<=100)==task['metrics'][metric]
            diff=error-before if key=='mae' else (error<=100).astype(float)-(before<=100).astype(float)
            assert summary(rows,diff)==task['paired_vs_target_only'][metric]
            if task['view']=='target_only':assert task['metrics'][metric]==old[name]['metrics'][metric]
        if task['view']=='raw_partner':
            available=np.array([r['errors']['partner_centroid'] for r in rows])
            assert np.max(available)<1e-6,'Declared exact Raw side channel must be available'
            controls[task['method']]=dict(available_raw_estimator_hit100=1.,
                selected_hit100=task['metrics']['Hit100']['mean'],
                selected_MAE_m=task['metrics']['MAE_m']['mean'])
        selection_checks+=1
    failure=dict(schema='s8-v2-retained-verification-failure',
        checker='frozen experiments/s8_fusion_20261109.py verify --write',
        reason='Overstrict selected TEST Hit100==1; V2 fusion tied on SELECTION but TransProtect TEST Hit100=0.9833333333333332',
        available_raw_control_perfect=True,scores_or_defense_changed=False,
        resolution='Independent replay verifies selected scores and the separately available Raw estimator')
    receipt=dict(schema='s8-fusion-corrected-verification-v2',status='pass',tasks_recomputed=24,
        selection_replayed=True,selection_checks=selection_checks,raw_control_availability=controls,
        prior_verifier_failure_retained=True,core_outputs_changed=False,new_defense_samples=False,
        checker_sha256=base.sha(Path(__file__)),results_sha256=base.sha(out/'results.json'),
        protocol_sha256=base.sha(out/'protocol.json'))
    if write:
        base.save(out/'verification_failures/initial_gate.json',failure)
        base.save(out/'validation.json',receipt)
    else:
        assert base.read(out/'validation.json')==receipt
        assert base.read(out/'verification_failures/initial_gate.json')==failure
    print('PASS:24 V2 selectors/metrics; available Raw control; initial false gate retained',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--write',action='store_true')
    verify(parser.parse_args().write)
