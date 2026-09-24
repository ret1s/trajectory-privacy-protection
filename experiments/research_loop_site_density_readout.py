"""Attribute density attack performance to prior, pooling or product fusion."""
import json
from pathlib import Path
import numpy as np
from experiments.research_loop_resources import ROOT, sha
from experiments.research_loop_sequence_attack import select

BASE = ROOT/'artifacts/benchmarks/research_loop'
SOURCE = BASE/'iteration24_site_density.json'
OUT = BASE/'iteration24_site_density_components.json'


def main():
    if OUT.exists():
        raise FileExistsError('Preserve completed evidence')
    source = json.loads(SOURCE.read_text()); selections, rows = {}, []
    for method in sorted({r['method'] for r in source['rows']}):
        for task in ('S9', 'S10'):
            auxiliary = [r for r in source['auxiliary_selection_rows'] if r['method'] == method and r['task'] == task+'_joint']
            cases = [r for r in source['rows'] if r['method'] == method and r['case_id'] == task+'.C']
            for component in ('prior', 'pool', 'product'):
                selected = select([{**r, 'errors': {a: v for a, v in r['errors'].items() if '_'+component+'_' in a}} for r in auxiliary])
                selections[method+'/'+task+'/'+component] = selected
                family = {}
                for f in sorted({r['family_id'] for r in cases}):
                    group = [r for r in cases if r['family_id'] == f]
                    family[f] = {'mae_m': float(np.mean([np.mean(r['errors'][selected['mae']]) for r in group])),
                        **{f'hit{radius}': float(np.mean([np.mean(np.array(r['errors'][selected[f'hit{radius}']]) <= radius) for r in group])) for radius in (50, 100, 200, 500)}}
                rows.append({'method': method, 'case_id': task+'.C', 'component': component,
                    'selected_attack': selected, 'family_metrics': family,
                    'mean': {metric: float(np.mean([v[metric] for v in family.values()])) for metric in next(iter(family.values()))}})
    OUT.write_text(json.dumps({'scope': 'components selected only on auxiliary families; exposed expanded development',
        'source_sha256': sha(SOURCE), 'code_sha256': sha(Path(__file__)), 'selection': selections, 'rows': rows},
        indent=2, allow_nan=False)+'\n')
    for r in rows:
        print(r['method'], r['case_id'], r['component'], {k: round(v, 4) for k, v in r['mean'].items()})


if __name__ == '__main__':
    main()
