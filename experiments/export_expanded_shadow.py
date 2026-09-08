"""Exact, source-backed readout of old versus expanded attack banks."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from experiments.run_expanded_shadow import ROOT, OUTPUT, OLD, DATA, METHODS, setup
from experiments.run_service_cover import read, write, sha
from experiments.export_prior_factors import LABEL


def export(output=OUTPUT):
    OUTPUT = Path(output)
    receipt = read(OUTPUT / 'verification.json')
    assert receipt['verified'] and receipt['source_sha256'] == sha(ROOT / 'experiments/verify_expanded_shadow.py')
    data = {p: read(OUTPUT / f'{p}.json') for p in ('training', 'validation', 'development', 'holdout')}
    selection = read(OUTPUT / 'selection.json')
    old_selection, old_dev = read(OLD / 'selection.json'), read(OLD / 'development.json')
    for p, h in receipt['artifact_sha256'].items():
        assert sha(OUTPUT / f'{p}.json') == h
    summaries, per_attack, family_changes = [], [], []
    for method in METHODS:
        ss = [s for s in data['development']['summaries'] if s['method'] == method]
        previous = {s['case_id']: s for s in old_dev['summaries'] if s['method'] == method}
        summaries.append({'method': method, 'label': LABEL[method],
            'old_selected_hit100': float(np.mean([previous[s['case_id']]['selected_hit100'] for s in ss])),
            'selected_hit100': float(np.mean([s['selected_hit100'] for s in ss])),
            'old_selected_mae_m': float(np.mean([previous[s['case_id']]['selected_mae_m'] for s in ss])),
            'selected_mae_m': float(np.mean([s['selected_mae_m'] for s in ss])),
            'old_envelope_hit100': float(np.mean([previous[s['case_id']]['audit_max_hit100'] for s in ss])),
            'envelope_hit100': float(np.mean([s['envelope_hit100'] for s in ss])),
            'old_envelope_mae_m': float(np.mean([previous[s['case_id']]['audit_min_mae_m'] for s in ss])),
            'envelope_mae_m': float(np.mean([s['envelope_mae_m'] for s in ss])),
            'poi_recall': float(np.mean([s['poi_recall'] for s in previous.values()])),
            'min_case_recall': min(s['poi_recall'] for s in previous.values()),
            'cases': [{**s, 'old_selected_hit100': previous[s['case_id']]['selected_hit100'],
                       'old_selected_mae_m': previous[s['case_id']]['selected_mae_m']} for s in ss]})
        common = set.intersection(*(set(s['mae_by_attack']) for s in ss))
        for a in sorted(common):
            per_attack.append({'method': method, 'attack': a,
                'mae_m': float(np.mean([s['mae_by_attack'][a] for s in ss])),
                'hit100': float(np.mean([s['hit_by_attack'][a] for s in ss]))})
        rows = [r for r in data['development']['rows'] if r['method'] == method]
        for family in sorted({r['family_id'] for r in rows}):
            group = [r for r in rows if r['family_id'] == family]
            values = {}
            for tag, choices in (('old', old_selection), ('new', selection)):
                hit, mae = [], []
                for r in group:
                    a = choices['attackers'][method + '/' + r['case_id']]
                    hit.append(np.mean(np.asarray(r['errors_by_attack'][a['hit']]) <= 100))
                    mae.append(np.mean(r['errors_by_attack'][a['mae']]))
                values.update({tag + '_hit100': float(np.mean(hit)), tag + '_mae_m': float(np.mean(mae))})
            family_changes.append({'method': method, 'family': family, **values,
                'delta_hit100': values['new_hit100'] - values['old_hit100'],
                'delta_mae_m': values['new_mae_m'] - values['old_mae_m']})
    # Auxiliary transfer: a single per-metric attacker selected on original
    # validation across all nine cases. No per-profile holdout selection.
    holdout = []
    old_val = read(OLD / 'validation.json')
    for s in data['holdout']['summaries']:
        vv = [v for v in old_val['summaries'] if v['method'] == s['method']]
        common = set.intersection(*(set(v['mae_by_attack']) for v in vv))
        old_mae = min(common, key=lambda a: (np.mean([v['mae_by_attack'][a] for v in vv]), a))
        old_hit = min(common, key=lambda a: (-np.mean([v['hit_by_attack'][a] for v in vv]), a))
        holdout.append({**s, 'old_global_mae_attack': old_mae, 'old_global_hit_attack': old_hit,
            'old_selected_mae_m': s['mae_by_attack'][old_mae],
            'old_selected_hit100': s['hit_by_attack'][old_hit]})
    # Ground-truth locations below are evaluator-only coverage diagnostics.
    _, rn, _, _, _ = setup('training')
    old_train = read(OLD / 'training.json')['shadow_models']['uniform_cover/5']
    new_train = data['training']['shadow_models']['uniform_cover']
    coverage = {}
    for phase in ('validation', 'development', 'holdout'):
        xy = np.asarray([rn.point_xy(p['lat'], p['lon']) for r in data[phase]['records'] for p in r['points']])
        coverage[phase] = {}
        for tag, model in (('old', old_train), ('expanded', new_train)):
            distances = cKDTree(np.asarray(model['y'])).query(xy)[0]
            coverage[phase][tag] = {'median_m': float(np.median(distances)),
                'p95_m': float(np.percentile(distances, 95)), 'observations': len(xy)}
    result = {'schema': 'expanded-shadow-readout-v1', 'summary': summaries,
        'per_attack_development': per_attack, 'family_changes': family_changes, 'holdout': holdout,
        'coverage_diagnostics': coverage,
        'training_support': {'old_observations': len(old_train['y']), 'old_families': 2,
            'old_unique_xy': len(np.unique(old_train['y'], axis=0)),
            'expanded_observations': len(new_train['y']), 'expanded_families': 66,
            'expanded_unique_xy': len(np.unique(new_train['y'], axis=0))},
        'defender_selection_unchanged': old_selection['method_selection'],
        'sources': {p: sha(OUTPUT / f'{p}.json') for p in (*data, 'selection', 'verification')},
        'exporter_sha256': sha(__file__),
        'definitions': {'core': 'equal nine cases, four reused families, three paired RNG draws; NOT fresh confirmation',
            'selected': 'per metric and case chosen only on original 103/104; defender never reselected',
            'holdout': '16 unseen auxiliary families; global original-validation-selected attacker; four different AUX profiles',
            'envelope': 'per-case exploratory best within observed bank, not an upper bound on arbitrary attackers',
            'distance': 'raw GPS metric XY; Hit iff distance<=100 m; no geometric tolerance in scoring',
            'training': 'observation weighted, old360 plus2873 auxiliary; correlated windows are not independent users'}}
    write(OUTPUT / 'readout.json', result)
    lines = [r'\begin{table}[htbp]', r'\centering\small',
        r'\caption{Suy luận trên cùng đầu ra của bốn nhóm phát triển 201--204; chín ca, $K=5$. Đối thủ chọn trên 103/104 riêng cho từng ca và phép đo. Hit tính bằng \%, MAE bằng m.}',
        r'\label{tab:expanded-shadow-selected}', r'\begin{tabular}{lrrrr}', r'\toprule',
        r'Cấu hình & Hit chọn cũ & Hit chọn mới & MAE chọn cũ & MAE chọn mới\\', r'\midrule']
    for s in summaries:
        lines.append(f"{s['label']} & {100*s['old_selected_hit100']:.2f} & {100*s['selected_hit100']:.2f} & {s['old_selected_mae_m']:.1f} & {s['selected_mae_m']:.1f} " + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', r'\begin{table}[htbp]', r'\centering\small',
        r'\caption{Bao đối thủ thăm dò trên cùng tập phát triển: cực đại Hit (\%) và cực tiểu MAE (m) trong từng ca, rồi lấy trung bình chín ca. Đây không phải điểm xác nhận của đối thủ đã chọn.}',
        r'\label{tab:expanded-shadow-envelope}', r'\begin{tabular}{lrrrr}', r'\toprule',
        r'Cấu hình & Hit bao cũ & Hit bao mới & MAE bao cũ & MAE bao mới\\', r'\midrule']
    for s in summaries:
        lines.append(f"{s['label']} & {100*s['old_envelope_hit100']:.2f} & {100*s['envelope_hit100']:.2f} & {s['old_envelope_mae_m']:.1f} & {s['envelope_mae_m']:.1f} " + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', r'\begin{table}[htbp]', r'\centering\small',
        r'\caption{Chuyển giao đối thủ sang 16 nhóm phụ trợ giữ lại, cấu hình $U/U$. Một đối thủ toàn cục cho mỗi phép đo được chọn trên chín ca validation cũ, không chọn lại theo cửa sổ giữ lại.}',
        r'\label{tab:expanded-shadow-holdout}', r'\begin{tabular}{lrrrr}', r'\toprule',
        r'Cửa sổ & Hit cũ (\%) & Hit mới (\%) & MAE cũ (m) & MAE mới (m)\\', r'\midrule']
    labels = {'AUX.cruise20': 'Di chuyển 20 s', 'AUX.cruise60': 'Di chuyển 60 s', 'AUX.stop5': 'Dừng 5 s', 'AUX.return5': 'Quay lại 5 s'}
    for s in holdout:
        if s['method'] == 'uniform_cover':
            lines.append(f"{labels[s['case_id']]} & {100*s['old_selected_hit100']:.2f} & {100*s['selected_hit100']:.2f} & {s['old_selected_mae_m']:.1f} & {s['selected_mae_m']:.1f} " + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    lines += [r'\begin{table}[htbp]', r'\centering\small',
        r'\caption{Tách ảnh hưởng mở rộng bộ học ở $U/U$: từng đối thủ cố định trên chín ca phát triển. Các hàng này là phân tích thành phần, không thay điểm của đối thủ đã chọn bằng validation.}',
        r'\label{tab:expanded-shadow-data-effect}', r'\begin{tabular}{lrrrr}', r'\toprule',
        r'Đối thủ & MAE cũ (m) & MAE mới (m) & Hit cũ (\%) & Hit mới (\%)\\', r'\midrule']
    attack_index = {p['attack']: p for p in per_attack if p['method'] == 'uniform_cover'}
    pairs = [('1-NN', 'shadow_knn_1', 'expanded_knn_1'),
             ('Trung bình 5', 'shadow_knn_5', 'expanded_knn_5'),
             ('Trung bình 15', 'shadow_knn_15', 'expanded_knn_15'),
             ('MAE, 15 nhãn', 'shadow_loss_mae_action_15', 'expanded_shadow_loss_mae_action_15'),
             ('Hit, 15 nhãn', 'shadow_loss_hit_action_15', 'expanded_shadow_loss_hit_action_15'),
             ('MAE, 45 nhãn', 'shadow_loss_mae_action_45', 'expanded_shadow_loss_mae_action_45'),
             ('Hit, 45 nhãn', 'shadow_loss_hit_action_45', 'expanded_shadow_loss_hit_action_45'),
             ('Trung bình 45', 'shadow_mean_45', 'expanded_shadow_mean_45')]
    for label, old_name, new_name in pairs:
        a, b = attack_index[old_name], attack_index[new_name]
        lines.append(f"{label} & {a['mae_m']:.1f} & {b['mae_m']:.1f} & {100*a['hit100']:.2f} & {100*b['hit100']:.2f} " + r'\\')
    for label, name in (('Cây: tọa độ', 'expanded_tree_direct'), ('Cây: phần lệch', 'expanded_tree_residual')):
        a = attack_index[name]
        lines.append(f"{label} & --- & {a['mae_m']:.1f} & --- & {100*a['hit100']:.2f} " + r'\\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    (OUTPUT / 'results_tables.tex').write_text('\n'.join(lines) + '\n')
    print(json.dumps({**{k: v for k, v in result.items() if k not in ('summary', 'holdout', 'per_attack_development')},
        'summary': [{k: v for k, v in s.items() if k != 'cases'} for s in summaries]}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    export(parser.parse_args().output)
