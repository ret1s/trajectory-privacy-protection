"""Traceable exact comparison tables; old/new attack banks share outputs."""
import json
import numpy as np

from experiments.run_prior_factors import OUTPUT, METHODS, CONTROLS, read, sha

LABEL = {'baseline': 'BR-làn', 'service_cover': 'L/L', 'uniform_cover': 'U/U',
         'learned_initial_uniform_motion': 'L/U', 'uniform_initial_learned_motion': 'U/L',
         'balanced_cover': 'C/C', 'mixed_cover': 'M/M'}


def export():
    train = read(OUTPUT/'training.json'); val = read(OUTPUT/'validation.json')
    dev = read(OUTPUT/'development.json'); selection = read(OUTPUT/'selection.json')
    entries = []
    for name in METHODS:
        summaries = [s for s in dev['summaries'] if s['method'] == name]
        validation = {s['case_id']: s for s in val['summaries'] if s['method'] == name}
        rows = [r for r in dev['rows'] if r['method'] == name]
        old_selected_hit, old_selected_mae, old_envelope, cases = [], [], [], []
        for s in summaries:
            v = validation[s['case_id']]
            old_attacks = [a for a in s['hit_by_attack'] if not a.startswith('shadow_loss_') and a != 'shadow_mean_45']
            ah = min(old_attacks, key=lambda a: (-v['hit_by_attack'][a], a))
            am = min(old_attacks, key=lambda a: (v['mae_by_attack'][a], a))
            old_selected_hit.append(s['hit_by_attack'][ah]); old_selected_mae.append(s['mae_by_attack'][am])
            old_envelope.append(max(s['hit_by_attack'][a] for a in old_attacks))
            assert s['audit_max_hit100']+1e-14 >= old_envelope[-1]
            cases.append({'case_id': s['case_id'], 'recall': s['poi_recall'],
                          'old_selected_hit': s['hit_by_attack'][ah], 'new_selected_hit': s['selected_hit100'],
                          'old_selected_attack': ah, 'new_selected_attack': s['selected_hit_attack'],
                          'old_envelope': old_envelope[-1], 'new_envelope': s['audit_max_hit100']})
        categories = {c: float(np.mean([s['by_category'][c]['recall'] for s in summaries])) for c in summaries[0]['by_category']}
        families = []
        for fid in sorted({r['family_id'] for r in rows}):
            scores = [r['utility']['poi_recall_at_5'] for r in rows if r['family_id'] == fid]
            control = [r['utility']['poi_recall_at_5'] for r in dev['rows'] if r['family_id'] == fid and r['method'] == 'uniform_cover']
            families.append({'family_id': fid, 'recall': float(np.mean(scores)),
                             'recall_delta_vs_uniform': float(np.mean(scores)-np.mean(control))})
        entries.append({'method': name, 'label': LABEL[name],
            'validation_recall': float(np.mean([s['poi_recall'] for s in validation.values()])),
            'validation_min_case': min(s['poi_recall'] for s in validation.values()),
            'recall': float(np.mean([s['poi_recall'] for s in summaries])),
            'min_case_recall': min(s['poi_recall'] for s in summaries),
            'old_selected_hit100': float(np.mean(old_selected_hit)),
            'selected_hit100': float(np.mean([s['selected_hit100'] for s in summaries])),
            'old_selected_mae_m': float(np.mean(old_selected_mae)),
            'selected_mae_m': float(np.mean([s['selected_mae_m'] for s in summaries])),
            'old_exploratory_hit100': float(np.mean(old_envelope)),
            'exploratory_hit100': float(np.mean([s['audit_max_hit100'] for s in summaries])),
            'by_category': categories, 'dense_category_recall': (categories['cafe']+categories['restaurant'])/2,
            'valid_query_n': sum(r['utility']['poi_evaluable_n'] for r in rows),
            'empty_reference_n': sum(q['recall'] is None for r in rows for q in r['utility']['poi_rows']),
            'min_distinct_coordinates': min(len({(c['lat'], c['lon']) for c in e['candidates']}) for r in rows for e in r['public']['events']),
            'step_p95_ms': float(np.percentile(np.concatenate([r['step_ms'] for r in rows]), 95)),
            'cases': cases, 'families': families})
    index = {e['method']: e for e in entries}
    result = {'scope': 'reused_development_only', 'summary': entries, 'selection': selection['method_selection'],
        'factorial_recall_contrasts': {
            'initial_change_given_learned_motion': index['uniform_initial_learned_motion']['recall']-index['service_cover']['recall'],
            'initial_change_given_uniform_motion': index['uniform_cover']['recall']-index['learned_initial_uniform_motion']['recall'],
            'motion_change_given_learned_initial': index['learned_initial_uniform_motion']['recall']-index['service_cover']['recall'],
            'motion_change_given_uniform_initial': index['uniform_cover']['recall']-index['uniform_initial_learned_motion']['recall']},
        'sources': {p: sha(OUTPUT/f'{p}.json') for p in ('training', 'validation', 'selection', 'development')},
        'exporter_sha256': sha(__file__),
        'training_support': {n: {'observations': len(train['shadow_models'][f'{n}/5']['y']),
            'unique_xy': len(np.unique(np.array(train['shadow_models'][f'{n}/5']['y']), axis=0))} for n in METHODS},
        'definitions': {'aggregation': 'eligible category-event mean per run, equal case/family/replicate averages',
            'old_bank': 'all original attacks incl kNN 1/5/15; validation-selected independently per metric/case',
            'new_bank': 'old bank plus 5 loss-aware/mean45 attacks, same train data, validation selection',
            'independence': 'four reused families; three RNG draws are not additional independent users',
            'privacy': 'approximate empirical attacks, not calibrated posterior or bound on all attacks',
            'timing': 'shared machine; controls reuse old timings, not a controlled speed comparison'}}
    (OUTPUT/'readout.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    lines = [r'\begin{table}[htbp]', r'\centering\small',
             r'\caption{Phân rã prior trên bốn nhóm phát triển 201--204: $K=5$, chín ca, ba lần lặp. $A/B$: khởi tạo/chuyển động. Đơn vị Recall là \%.}',
             r'\label{tab:prior-factor-utility}', r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Cấu hình & Recall & Ca thấp nhất & Cà phê & Nhà hàng\\', r'\midrule']
    for e in entries:
        lines.append(f"{e['label']} & {100*e['recall']:.2f} & {100*e['min_case_recall']:.2f} & {100*e['by_category']['cafe']:.2f} & {100*e['by_category']['restaurant']:.2f} " + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', r'\begin{table}[htbp]', r'\centering\small',
             r'\caption{Đánh giá lại cùng đầu ra với bộ đối thủ mở rộng. Hit (\%) càng cao, MAE (m) càng thấp thì đối thủ càng hiệu quả. Bao trên chỉ xét các đối thủ đã thử.}',
             r'\label{tab:prior-factor-attacks}', r'\begin{tabular}{lrrrr}', r'\toprule',
             r'Cấu hình & Hit chọn cũ & Hit chọn mới & Bao trên mới & MAE chọn mới\\', r'\midrule']
    for e in entries:
        lines.append(f"{e['label']} & {100*e['old_selected_hit100']:.2f} & {100*e['selected_hit100']:.2f} & {100*e['exploratory_hit100']:.2f} & {e['selected_mae_m']:.1f} " + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (OUTPUT/'results_tables.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps({**{k: v for k, v in result.items() if k != 'summary'},
                      'summary': [{k: v for k, v in e.items() if k not in ('cases', 'families')} for e in entries]}, indent=2))


if __name__ == '__main__': export()
