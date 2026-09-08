"""Exact thesis lookup tables from the frozen development ablation."""
import json

import numpy as np

from experiments.run_recovery_cover import OUTPUT, METHODS, read, sha

NAMES = {'baseline': 'BR-làn', 'belief24': 'BR-phân bố 24', 'service_cover': 'Phủ-học',
         'uniform_cover': 'Phủ-đều', 'learned_corridor': 'Phủ-học + miền',
         'uniform_corridor': 'Phủ-đều + miền'}


def export():
    p = read(OUTPUT/'development.json'); v = read(OUTPUT/'validation.json')
    selection = read(OUTPUT/'selection.json'); diagnostic = read(OUTPUT/'diagnostic.json')
    entries = []
    for name in METHODS:
        group = [s for s in p['summaries'] if s['method'] == name]
        val = [s for s in v['summaries'] if s['method'] == name]
        rows = [r for r in p['rows'] if r['method'] == name]
        categories = {c: float(np.mean([s['by_category'][c]['recall'] for s in group])) for c in group[0]['by_category']}
        families = []
        for fid in sorted({r['family_id'] for r in rows}):
            a = [r['utility']['poi_recall_at_5'] for r in rows if r['family_id'] == fid]
            b = [r['utility']['poi_recall_at_5'] for r in p['rows'] if r['family_id'] == fid and r['method'] == 'baseline']
            families.append({'family_id': fid, 'recall': float(np.mean(a)), 'recall_delta_vs_baseline': float(np.mean(a)-np.mean(b))})
        times = np.concatenate([r['step_ms'] for r in rows])
        entries.append({'method': name, 'validation_recall': float(np.mean([s['poi_recall'] for s in val])),
            'validation_min_case_recall': min(s['poi_recall'] for s in val),
            'recall': float(np.mean([s['poi_recall'] for s in group])), 'min_case_recall': min(s['poi_recall'] for s in group),
            'selected_hit100': float(np.mean([s['selected_hit100'] for s in group])),
            'selected_mae_m': float(np.mean([s['selected_mae_m'] for s in group])),
            'exploratory_hit100': float(np.mean([s['audit_max_hit100'] for s in group])),
            'completion': float(np.mean([s['poi_complete'] for s in group])),
            'by_category': categories, 'dense_category_recall': float(np.mean([categories[c] for c in ('cafe', 'restaurant')])),
            'valid_query_n': sum(r['utility']['poi_evaluable_n'] for r in rows),
            'empty_reference_n': sum(sum(q['recall'] is None for q in r['utility']['poi_rows']) for r in rows),
            'extra_distance_conditional_mean_m': float(np.mean([r['utility']['poi_extra_distance_m'] for r in rows if r['utility']['poi_extra_distance_m'] is not None])),
            'extra_distance_n': sum(r['utility']['poi_extra_distance_n'] for r in rows),
            'step_p50_ms': float(np.median(times)), 'step_p95_ms': float(np.percentile(times, 95)),
            'init_mean_ms': float(np.mean([r['init_ms'] for r in rows])),
            'min_distinct_coordinates': min(len({(c['lat'], c['lon']) for c in e['candidates']}) for r in rows for e in r['public']['events']),
            'logical_category_location_queries': sum(len(r['public']['events'])*6*5 for r in rows),
            'paired_families': families})
    result = {'scope': 'development_only_not_independent_confirmation', 'summary': entries,
        'selection': selection['method_selection'], 'diagnostic': diagnostic['summary'],
        'sources': {s: sha(OUTPUT/f'{s}.json') for s in ('training', 'validation', 'development', 'selection', 'diagnostic')},
        'exporter_sha256': sha(__file__),
        'definitions': {'aggregation': 'eligible category-events per run, then equal case/family/replicate',
            'min_case': 'minimum of nine case means, not a per-event bound',
            'selected_attacks': 'separate validation selections for MAE and Hit per method/case',
            'timing': 'shared development machine; old baseline times reused, not a controlled cross-method speed test',
            'independence': 'four previously inspected SUMO families; three RNG repetitions do not create more independent trajectories',
            'logical_queries': 'six counterfactual category workloads, not measured HTTP packets'}}
    (OUTPUT/'readout.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    lines = [r'\begin{table}[htbp]', r'\centering\small',
        r'\caption{Phân tích phát triển trên nhóm 201--204 đã được xem trước: $K=5$, chín ca và ba hạt ngẫu nhiên. Không phải xác nhận độc lập.}',
        r'\label{tab:recovery-main}', r'\setlength{\tabcolsep}{4pt}', r'\begin{tabular}{lrrrrr}', r'\toprule',
        r'Cấu hình & Recall (\%) & Ca thấp (\%) & Hit$_{100}$ (\%) & MAE (m) & POI dày (\%) \\', r'\midrule']
    for e in entries:
        lines.append(f"{NAMES[e['method']]} & {e['recall']*100:.2f} & {e['min_case_recall']*100:.2f} & {e['selected_hit100']*100:.2f} & {e['selected_mae_m']:.1f} & {e['dense_category_recall']*100:.2f} " + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (OUTPUT/'results_tables.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__': export()
