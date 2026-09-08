"""Exact lookup tables; macro cases/families/repeats, never event pseudo-N."""
import json
from pathlib import Path

import numpy as np

from experiments.run_service_cover import ROOT, OUTPUT, METHODS, read
from experiments.run_contextual_lane import sha

NAMES = {'baseline':'BR-làn', 'belief24':'BR-phân bố 24',
         'service_cover':'BR-phủ dịch vụ', 'prior_cover':'Phủ-prior'}


def export(output=OUTPUT):
    output = Path(output)
    p = read(output/'confirmation.json'); validation = read(output/'validation.json')
    selection = read(output/'selection.json')
    entries = []
    for k in (3,5):
        for name in METHODS:
            group = [s for s in p['summaries'] if (s['method'],s['k']) == (name,k)]
            rows = [r for r in p['rows'] if (r['method'],r['k']) == (name,k)]
            latency = np.concatenate([r['step_ms'] for r in rows])
            categories = {c: float(np.mean([s['by_category'][c]['recall'] for s in group])) for c in group[0]['by_category']}
            per_family = []
            for family in sorted({r['family_id'] for r in rows}):
                fg = [r for r in rows if r['family_id'] == family]
                base = [r for r in p['rows'] if r['family_id'] == family and r['k'] == k and r['method'] == 'baseline']
                per_family.append({'family_id':family,
                    'recall_delta':float(np.mean([r['utility']['poi_recall_at_5'] for r in fg])-np.mean([r['utility']['poi_recall_at_5'] for r in base]))})
            entries.append({'method':name,'k':k,'families':4,'cases':9,'replicates':3,
                'recall':float(np.mean([s['poi_recall'] for s in group])),
                'min_case_recall':min(s['poi_recall'] for s in group),
                'selected_hit100':float(np.mean([s['selected_hit100'] for s in group])),
                'selected_mae_m':float(np.mean([s['selected_mae_m'] for s in group])),
                'exploratory_envelope_hit100':float(np.mean([s['audit_max_hit100'] for s in group])),
                'poi_complete':float(np.mean([s['poi_complete'] for s in group])),
                'extra_distance_conditional_mean_m':float(np.mean([r['utility']['poi_extra_distance_m'] for r in rows if r['utility']['poi_extra_distance_m'] is not None])),
                'extra_distance_n':sum(r['utility']['poi_extra_distance_n'] for r in rows),
                'by_category':categories,'dense_category_recall':float(np.mean([categories[c] for c in ('cafe','restaurant')])),
                'valid_query_n':sum(r['utility']['poi_evaluable_n'] for r in rows),
                'empty_reference_n':sum(sum(q['recall'] is None for q in r['utility']['poi_rows']) for r in rows),
                'min_distinct_coordinates':min(len({(c['lat'],c['lon']) for c in e['candidates']}) for r in rows for e in r['public']['events']),
                'logical_category_location_queries':sum(len(r['public']['events'])*6*k for r in rows),
                'step_p50_ms':float(np.median(latency)), 'step_p95_ms':float(np.percentile(latency,95)),
                'mean_init_ms':float(np.mean([r['init_ms'] for r in rows])),
                'paired_family_recall_deltas':per_family})
    result = {'confirmation_sha256':sha(output/'confirmation.json'),
        'validation_sha256':sha(output/'validation.json'), 'selection_sha256':sha(output/'selection.json'),
        'method_selection':selection['method_selection'], 'summary':entries,
        'definitions':{'recall':'equal-weight nine cases; within case equal family/replicate; within run eligible category-event observations',
            'min_case_recall':'minimum of nine case means, not an event-wise utility bound',
            'selected_attacks':'MAE and Hit attacks independently selected per method/K/case on validation',
            'exploratory_envelope':'strongest fixed attacker on confirmation, not held-out selection',
            'dense_category_recall':'equal cafe/restaurant mean diagnostic, not a new primary metric',
            'family_deltas':'descriptive paired effects after RNG averaging; no significance/CI claim',
            'latency':'shared development-machine measurements; excludes LSP network and initialization; not mobile real-time certification'},
        'exporter_sha256':sha(Path(__file__))}
    (output/'readout.json').write_text(json.dumps(result,indent=2,ensure_ascii=False)+'\n')
    lines = [r'\begin{table}[htbp]',r'\centering\small',
        r'\caption{Xác nhận vòng phủ dịch vụ: bốn nhóm mới, chín ca S1--S3, ba lần lặp ngẫu nhiên. Hit thấp hơn và MAE cao hơn chỉ có ý nghĩa đối với tập đối thủ đã khai báo.}',
        r'\label{tab:service-cover-main}',r'\setlength{\tabcolsep}{4pt}',
        r'\begin{tabular}{rlrrrr}',r'\toprule',
        r'$K$ & Cấu hình & Recall (\%) & Ca thấp nhất (\%) & Hit$_{100}$ (\%) & MAE (m) \\',r'\midrule']
    for e in entries:
        lines.append(f"{e['k']} & {NAMES[e['method']]} & {e['recall']*100:.2f} & {e['min_case_recall']*100:.2f} & {e['selected_hit100']*100:.2f} & {e['selected_mae_m']:.1f} " + r'\\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}',
        r'\begin{table}[htbp]',r'\centering\footnotesize',
        r'\caption{Recall@5 theo loại POI (\%), cùng dữ liệu xác nhận. Trung bình theo ca, nhóm và lần lặp; không giả định tần suất truy vấn thực bằng nhau.}',
        r'\label{tab:service-cover-categories}',r'\setlength{\tabcolsep}{3pt}',
        r'\begin{tabular}{rlrrrrrr}',r'\toprule',
        r'$K$ & Cấu hình & Cà phê & Nhà hàng & Hiệu thuốc & Bệnh viện & Phòng khám & Nhiên liệu \\',r'\midrule']
    for e in entries:
        lines.append(f"{e['k']} & {NAMES[e['method']]} & " + ' & '.join(f"{e['by_category'][c]*100:.2f}" for c in ('cafe','restaurant','pharmacy','hospital','clinic','fuel')) + r' \\')
    lines += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    (output/'results_tables.tex').write_text('\n'.join(lines)+'\n')
    print(json.dumps(result,indent=2,ensure_ascii=False))


if __name__ == '__main__': export()
