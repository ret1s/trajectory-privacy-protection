"""Small report tables from sealed, independently verified fresh results."""
from collections import defaultdict
from pathlib import Path

import numpy as np

from experiments.run_fresh_switching import OUTPUT, METHODS, choose
from experiments.run_coverage_frontier import summarize
from experiments.run_service_cover import ROOT, read, write, sha
from experiments.publish_fresh_scenarios import FOLDER

EXCLUDED = ('v2-r311-002', 'v2-r311-003')
LABELS = {'geometric': 'Hình học', 'mean_greedy': 'Phủ tham lam',
          'mean_exchange': 'Phủ + thay điểm', 'switching_exchange': 'Hai chế độ + thay điểm'}


def macro(summaries, method):
    group = [s for s in summaries if s['method'] == method]
    return {'case_count': len(group), 'mae_m': float(np.mean([s['selected_mae_m'] for s in group])),
        'hit100': float(np.mean([s['selected_hit100'] for s in group])),
        'envelope_hit100': float(np.mean([s['envelope_hit100'] for s in group])),
        'envelope_mae_m': float(np.mean([s['envelope_mae_m'] for s in group])),
        'utility': {d: {'recall': float(np.mean([s['utility'][d]['poi_recall_at_5'] for s in group])),
            'min_case_recall': min(s['utility'][d]['poi_recall_at_5'] for s in group),
            'below_90_cases': [s['case_id'] for s in group if round(s['utility'][d]['poi_recall_at_5'],12) < .9],
            'complete': float(np.mean([s['utility'][d]['poi_complete_rate'] for s in group])),
            'reply_items': float(np.mean([s['utility'][d]['reply_items_per_event'] for s in group])),
            'id_bytes': float(np.mean([s['utility'][d]['response_id_bytes_per_event'] for s in group])),
            'requests': float(np.mean([s['utility'][d]['requests_per_event'] for s in group]))} for d in ('5', '10')}}


def build(output=OUTPUT):
    output = Path(output)
    phases = {p: read(output/f'{p}.json') for p in ('training', 'validation', 'confirmation')}
    verified = read(output/'verification.json'); selection = read(output/'selection.json')
    assert verified['verified'] and verified['phase_sha256'] == {p: sha(output/f'{p}.json') for p in phases}
    confirmed = phases['confirmation']
    methods = {m: macro(confirmed['summaries'], m) for m in METHODS}
    reduced = [r for r in confirmed['rows'] if r['record_id'] not in EXCLUDED]
    reduced_summaries = summarize(reduced, selection)
    sensitivity = {m: macro(reduced_summaries, m) for m in METHODS}
    # Each family is a cluster; repeated events/seeds are not independent users.
    family_results = {}
    for family in sorted({r['family_id'] for r in confirmed['rows']}):
        rows = [r for r in confirmed['rows'] if r['family_id'] == family]
        family_results[family] = {m: macro(summarize(rows, selection), m) for m in METHODS}
    paired = {}
    for baseline, candidate in (('mean_greedy', 'mean_exchange'), ('mean_exchange', 'switching_exchange')):
        name = candidate+'__minus__'+baseline
        paired[name] = {f: {'recall5_pp': 100*(g[candidate]['utility']['5']['recall']-g[baseline]['utility']['5']['recall']),
            'recall10_pp': 100*(g[candidate]['utility']['10']['recall']-g[baseline]['utility']['10']['recall']),
            'hit100_pp': 100*(g[candidate]['hit100']-g[baseline]['hit100']),
            'envelope_hit100_pp': 100*(g[candidate]['envelope_hit100']-g[baseline]['envelope_hit100'])}
            for f, g in family_results.items()}
    gates = {}
    for depth, value in selection['method_selection_by_depth'].items():
        name = value['chosen']['method'] if value['chosen'] else None
        gates[depth] = {'validation_chosen': name,
            'confirmation_min_case_recall': methods[name]['utility'][depth]['min_case_recall'] if name else None,
            'confirmation_passes_90': bool(name and not methods[name]['utility'][depth]['below_90_cases'])}
    dataset = read(FOLDER/'verification.json')
    report = {'schema': 'fresh-switching-readout-v1', 'methods': methods, 'selection_transfer': gates,
        'sensitivity_excluded_records': EXCLUDED, 'sensitivity': sensitivity,
        'families': family_results, 'paired_family_deltas': paired,
        'confirmation_records': len(confirmed['records']), 'confirmation_runs': len(confirmed['rows']),
        'confirmation_events': sum(len(r['public']['events']) for r in confirmed['rows']),
        'case_family_counts': {s['case_id']: s['families'] for s in confirmed['summaries'] if s['method'] == METHODS[0]},
        'verification_sha256': sha(output/'verification.json'), 'phase_sha256': verified['phase_sha256'],
        'selection_sha256': sha(output/'selection.json'), 'dataset_verification_sha256': sha(FOLDER/'verification.json'),
        'source_sha256': {p: sha(ROOT/p) for p in ('experiments/fresh_switching_readout.py', 'thesis/notes/fresh_overlap_audit.md')},
        'scope': 'six same-city confirmation families; no retuning; internal controls; not all-target or SOTA proof'}
    write(output/'readout.json', report)
    def tex(name, lines):
        with (output/name).open('x') as stream: stream.write('\n'.join(lines)+'\n')
    lines = [r'\begin{table}[htbp]', r'\centering\small',
        r'\caption{Số bản ghi từng trường hợp con ở tập mới. Mỗi ô A/B/C là ba số đếm, không phải điểm bảo vệ.}',
        r'\label{tab:fresh-dataset}', r'\begin{tabular}{lrr}', r'\toprule',
        r'Kịch bản & Tập chọn (A/B/C) & Tập xác nhận (A/B/C) \\', r'\midrule']
    for scenario in range(1, 11):
        cells = [' / '.join(str(dataset['by_split'][s].get(f'S{scenario}.{c}',0)) for c in 'ABC')
                 for s in ('fresh_validation','fresh_confirmation')]
        lines.append(f'S{scenario} & {cells[0]} & {cells[1]} '+r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    tex('dataset_table.tex', lines)
    lines = [r'\begin{table}[htbp]', r'\centering\footnotesize',
        r'\caption{Recall trên xác nhận mới; cùng $B=0{,}24$, $K=5$. Đơn vị phần trăm; $L$ là độ sâu phản hồi.}',
        r'\label{tab:fresh-utility}', r'\begin{tabular}{lrrrr}', r'\toprule',
        r'Bộ chọn & TB ($L=5$) & Ca thấp nhất & TB ($L=10$) & Ca thấp nhất \\', r'\midrule']
    for m, a in methods.items():
        values = [100*a['utility'][d][k] for d in ('5','10') for k in ('recall','min_case_recall')]
        lines.append(LABELS[m]+' & '+' & '.join(f'{x:.2f}' for x in values)+r' \\')
    tex('utility_table.tex', lines+[r'\bottomrule', r'\end{tabular}', r'\end{table}'])
    lines = [r'\begin{table}[htbp]', r'\centering\small',
        r'\caption{Rủi ro suy luận trên xác nhận. Hit thấp hơn và MAE cao hơn là tốt hơn trước đối thủ tương ứng.}',
        r'\label{tab:fresh-privacy}', r'\begin{tabular}{lrrr}', r'\toprule',
        r'Bộ chọn & Hit đã chọn (\%) & Hit cực trị (\%) & MAE đã chọn (m) \\', r'\midrule']
    for m, a in methods.items():
        lines.append(f'{LABELS[m]} & {100*a["hit100"]:.2f} & {100*a["envelope_hit100"]:.2f} & {a["mae_m"]:.1f} '+r'\\')
    tex('privacy_table.tex', lines+[r'\bottomrule', r'\end{tabular}', r'\end{table}'])
    print({'methods': methods, 'selection_transfer': gates, 'paired_family_deltas': paired}, flush=True)


if __name__ == '__main__': build()
