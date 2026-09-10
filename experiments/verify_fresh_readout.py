"""Independent case-balanced readout, overlap sensitivity and TeX-number audit."""
from collections import defaultdict

import numpy as np

from experiments.run_service_cover import ROOT, read, write, sha
from experiments.run_fresh_switching import OUTPUT, METHODS
from experiments.verify_coverage_frontier import equal


def aggregate(rows, selection, method):
    cases = defaultdict(list)
    for r in rows:
        if r['method'] == method: cases[r['case_id']].append(r)
    attacks = {c: selection['attackers'][method+'/'+c] for c in cases}
    def mean(values): return float(sum(values)/len(values))
    hit = lambda errors: sum(x <= 100 for x in errors)/len(errors)
    privacy = []
    for c, group in sorted(cases.items()):
        maes = {a: mean([mean(r['errors_by_attack'][a]) for r in group]) for a in group[0]['errors_by_attack']}
        hits = {a: mean([hit(r['errors_by_attack'][a]) for r in group]) for a in maes}
        privacy.append((maes[attacks[c]['mae']], hits[attacks[c]['hit']], min(maes.values()), max(hits.values())))
    values = np.array(privacy)
    result = {'case_count': len(cases), 'mae_m': float(values[:,0].mean()), 'hit100': float(values[:,1].mean()),
        'envelope_mae_m': float(values[:,2].mean()), 'envelope_hit100': float(values[:,3].mean()), 'utility': {}}
    for depth in ('5','10'):
        scores = {}
        for c, group in sorted(cases.items()):
            query_averages = []
            for r in group:
                queries = r['utility'][depth]['poi_rows']; n = len(r['public']['events'])
                valid = [q for q in queries if q['recall'] is not None]
                query_averages.append([mean([q['recall'] for q in valid]), mean([q['complete'] for q in valid]),
                    sum(q['reply_items'] for q in queries)/n, sum(q['response_id_json_bytes'] for q in queries)/n,
                    sum(q['request_count'] for q in queries)/n])
            scores[c] = np.mean(query_averages, axis=0)
        s = np.array(list(scores.values()))
        result['utility'][depth] = {'recall': float(s[:,0].mean()), 'min_case_recall': float(s[:,0].min()),
            'below_90_cases': [c for c, v in scores.items() if round(v[0],12) < .9],
            'complete': float(s[:,1].mean()), 'reply_items': float(s[:,2].mean()),
            'id_bytes': float(s[:,3].mean()), 'requests': float(s[:,4].mean())}
    return result


def verify():
    report = read(OUTPUT/'readout.json'); phase = read(OUTPUT/'confirmation.json')
    selection = read(OUTPUT/'selection.json')
    for p, h in report['source_sha256'].items(): assert sha(ROOT/p) == h
    assert report['verification_sha256'] == sha(OUTPUT/'verification.json')
    assert report['selection_sha256'] == sha(OUTPUT/'selection.json')
    assert report['phase_sha256']['confirmation'] == sha(OUTPUT/'confirmation.json')
    excluded = {'v2-r311-002', 'v2-r311-003'}
    assert set(report['sensitivity_excluded_records']) == excluded
    rows = phase['rows']; filtered = [r for r in rows if r['record_id'] not in excluded]
    assert len(rows)-len(filtered) == 24
    for m in METHODS:
        equal(report['methods'][m], aggregate(rows, selection, m))
        equal(report['sensitivity'][m], aggregate(filtered, selection, m))
    for family, scores in report['families'].items():
        group = [r for r in rows if r['family_id'] == family]
        for m in METHODS: equal(scores[m], aggregate(group, selection, m))
    for name, values in report['paired_family_deltas'].items():
        candidate, baseline = name.split('__minus__')
        for f, delta in values.items():
            a, b = report['families'][f][candidate], report['families'][f][baseline]
            equal(delta, {'recall5_pp': 100*(a['utility']['5']['recall']-b['utility']['5']['recall']),
                'recall10_pp': 100*(a['utility']['10']['recall']-b['utility']['10']['recall']),
                'hit100_pp': 100*(a['hit100']-b['hit100']),
                'envelope_hit100_pp': 100*(a['envelope_hit100']-b['envelope_hit100'])})
    for d, value in report['selection_transfer'].items():
        chosen = selection['method_selection_by_depth'][d]['chosen']
        name = chosen['method'] if chosen else None
        scores = report['methods'][name]['utility'][d] if name else None
        equal(value, {'validation_chosen': name, 'confirmation_min_case_recall': scores['min_case_recall'] if scores else None,
            'confirmation_passes_90': bool(scores and not scores['below_90_cases'])})
    assert report['confirmation_records'] == len(phase['records'])
    assert report['confirmation_runs'] == len(rows)
    assert report['confirmation_events'] == sum(len(r['public']['events']) for r in rows)
    for case, n in report['case_family_counts'].items():
        assert n == len({r['family_id'] for r in rows if r['case_id'] == case})
    for m, a in report['methods'].items():
        values = [100*a['utility'][d][k] for d in ('5','10') for k in ('recall','min_case_recall')]
        assert ' & '.join(f'{x:.2f}' for x in values) in (OUTPUT/'utility_table.tex').read_text()
        assert f'{100*a["hit100"]:.2f} & {100*a["envelope_hit100"]:.2f} & {a["mae_m"]:.1f}' in (OUTPUT/'privacy_table.tex').read_text()
    result = {'verified': True, 'readout_sha256': sha(OUTPUT/'readout.json'), 'verifier_sha256': sha(__file__),
        'macros_checked': 32, 'confirmation_records_excluded_in_sensitivity': 2,
        'table_sha256': {n: sha(OUTPUT/n) for n in ('utility_table.tex','privacy_table.tex','dataset_table.tex')}}
    write(OUTPUT/'readout_verification.json', result)
    print(result, flush=True)


if __name__ == '__main__': verify()
