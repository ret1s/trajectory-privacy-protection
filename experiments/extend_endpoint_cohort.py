"""Predeclared larger cohort requested after the seven-family pilot.

Reuse the sealed attacker/model selection byte-for-byte. This extension is
reported separately from the pilot, with no optional stopping or seed swaps.
The original runner and evidence remain immutable.
"""
import argparse
import importlib
import json
from pathlib import Path
import shutil

from experiments.research_loop_resources import ROOT,sha

BASE=ROOT/'artifacts/benchmarks/endpoint_calendar_v1'
OUT=ROOT/'artifacts/benchmarks/endpoint_calendar_expanded_v1'
DATA=ROOT/'artifacts/datasets/endpoint_holdout_expanded_v1'


def prepare():
    if (OUT/'protocol.json').exists():raise FileExistsError('Larger cohort already declared')
    original=json.loads((BASE/'protocol.json').read_text())
    source=dict(original['source_sha256']);source['experiments/build_endpoint_holdout.py']=sha(ROOT/'experiments/build_endpoint_holdout.py')
    source[str(Path(__file__).resolve().relative_to(ROOT))]=sha(Path(__file__))
    p={**original,'schema':'endpoint-calendar-expanded-protocol-v1','source_sha256':source,
       'holdout_builder_sha256':sha(ROOT/'experiments/build_endpoint_holdout.py'),
       'holdout_seeds':list(range(1201,1233)),'world_seeds':[5101,5102,5103],
       'parent_protocol_sha256':sha(BASE/'protocol.json'),'frozen_selection_sha256':sha(BASE/'selection.json'),
       'pilot_readout_sha256':sha(BASE/'readout.json'),
       'reason':'User requested larger samples; seven-family pilot already observed. No attack/defense/metric choices changed; report new cohort separately.',
       'stopping_rule':'Attempt all 32 seeds; retain all construction failures without replacement; no stopping based on benchmark outcomes'}
    OUT.mkdir(parents=True);(OUT/'protocol.json').write_text(json.dumps(p,indent=2)+'\n')
    (OUT/'protocol.sha256').write_text(sha(OUT/'protocol.json')+'\n')
    # Builder/validator compatibility: its already audited failure-handling
    # version is frozen from the start here, so this is not a source amendment.
    amendment={'protocol_sha256':sha(OUT/'protocol.json'),'original_builder_sha256':p['holdout_builder_sha256'],
        'revised_builder_sha256':p['holdout_builder_sha256'],'source_changed':False,
        'reason':'Reuse already audited failure accounting and resume behavior from the pilot'}
    (OUT/'construction_amendment.json').write_text(json.dumps(amendment,indent=2)+'\n')
    for f in ('selection.json','selection_rows.json.gz'):shutil.copyfile(BASE/f,OUT/f)
    print('Sealed 32 new seeds and reused unchanged selection',flush=True)


def run(stage):
    p=json.loads((OUT/'protocol.json').read_text())
    assert sha(Path(__file__))==p['source_sha256'][str(Path(__file__).resolve().relative_to(ROOT))]
    assert sha(OUT/'selection.json')==p['frozen_selection_sha256']
    import experiments.build_endpoint_holdout as builder
    # The same builder uses its CACHE variable only for resource files and its
    # work subdirectory. Keep inputs shared, isolate simulation output by using
    # a dedicated cache directory with verified symlinks to public resources.
    cache=ROOT/'cache/research_loop_20260924/endpoint_expanded_resources';cache.mkdir(exist_ok=True)
    for f in ('resources.json','beijing.net.xml'):
        target=cache/f
        if not target.exists():target.symlink_to(ROOT/'cache/research_loop_20260924'/f)
    builder.OUT,builder.DATA,builder.CACHE=OUT,DATA,cache
    import experiments.endpoint_calendar_study as study
    study.OUT,study.DATA=OUT,DATA
    if stage=='build':builder.build();return
    if stage in ('generate','attacks','service'):
        {'generate':study.generate_holdout,'attacks':study.evaluate_attacks,'service':study.evaluate_service}[stage]();return
    modules={'readout':'read_endpoint_calendar','verify':'verify_endpoint_calendar',
             'flow':'check_calendar_information_flow','ablation':'measure_calendar_ablation'}
    module=importlib.import_module('experiments.'+modules[stage])
    (module.verify if stage=='verify' else module.main)()


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['prepare','build','generate','attacks','service','readout','verify','flow','ablation']);a=p.parse_args()
    prepare() if a.stage=='prepare' else run(a.stage)
