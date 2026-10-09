"""Frozen historical S8 localization sensitivity, not current-model protection.

The target and associated partner are explicitly linkable in this observer
model. Only their causal public prefixes are supplied to an attacker. Actual
proximity and synthetic relationship labels remain evaluator-only. A Raw
partner is an explicitly granted public positive/limit control, not hidden GPS.
"""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import shutil

import numpy as np

from evaluation.identity_future import (RegressorBank, prefix_features,
    public_arrays, xy_from_latlon)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'artifacts/benchmarks/s8_companion_inference_20261007_v1'
DATA = 'artifacts/datasets/research_loop_expanded_v1/dataset.json'
ANCESTOR = 'artifacts/benchmarks/identity_future_20261005'
SPLITS = {'train': [f'family-{i}' for i in range(701, 707)],
          'selection': [f'family-{i}' for i in range(707, 710)],
          'test': [f'family-{i}' for i in range(710, 713)]}
VIEWS = ('target_only', 'joint_protected_partner', 'joint_public_raw_partner',
         'joint_unrelated_protected_partner', 'raw_target_positive_control')
SEED = 2026100723
FEATURE_SIZE = 32


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def write(path, value):
    path = Path(path)
    if path.exists():
        raise FileExistsError(f'Preserve existing evidence: {path}')
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def causal_prefix(public, offset, cut):
    """The common clock uses first visible events, not private departure/route.

    Stop before reading future candidate coordinates. Full saved tapes are
    evaluator inputs to this adapter, never inputs to the fitted attacker.
    """
    if set(public) != {'events'} or not np.isfinite([offset, cut]).all():
        raise ValueError('Strict public tape and finite public clock required')
    allowed = []
    previous = -np.inf
    for event in public['events']:
        t = float(event['timestamp_s']) + float(offset)
        if t > cut:
            break
        if t <= previous:
            raise ValueError('Public event order must be strictly increasing')
        previous = t
        allowed.append({**event, 'timestamp_s': t})
    if not allowed:
        return None
    result = {'events': allowed}
    public_arrays(result)  # Reject private fields in every supplied event.
    return result


def features(target, partner, cut, *, joint):
    """No labels, case IDs, family IDs or actual co-location enter this API."""
    a = prefix_features(target)
    assert len(a) == FEATURE_SIZE
    if target['events'][-1]['timestamp_s'] > cut:
        raise ValueError('Future target observation')
    if not joint:
        return a
    if partner is None:
        return np.r_[a, np.zeros(FEATURE_SIZE), 0., 0., 0.]
    if partner['events'][-1]['timestamp_s'] > cut:
        raise ValueError('Future partner observation')
    b = prefix_features(partner)
    return np.r_[a, b, (cut-target['events'][-1]['timestamp_s'])/60.,
                 (cut-partner['events'][-1]['timestamp_s'])/60., 1.]


def position_from_public(prefix, cut, *, velocity=False):
    xy, _, times = public_arrays(prefix)
    latest = xy[-1].copy()
    if velocity and len(times) > 1:
        # Mean prefix velocity is a declared public heuristic, no future GPS.
        speed = (xy[-1]-xy[0])/max(1., times[-1])
        latest += (cut-prefix['events'][-1]['timestamp_s'])*speed
    return latest


def inventory():
    """Authenticate exact tape/record/clock ancestry without fitting/scoring."""
    data = json.loads((ROOT/DATA).read_text())
    old = json.loads((ROOT/ANCESTOR/'linkage_results.json').read_text())
    public_file = ROOT/ANCESTOR/'linkage_public_transcripts.json'
    tapes = json.loads(public_file.read_text())['transcripts']
    assert old['dataset_sha256'] == sha(ROOT/DATA)
    assert old['public_transcripts_sha256'] == sha(public_file)
    assert old['protocol_sha256'] == sha(ROOT/ANCESTOR/'protocol_v2.json')
    assert len(tapes) == len(old['evaluator_sessions']) == 72
    sessions = {}
    for index, (meta, tape) in enumerate(zip(old['evaluator_sessions'], tapes)):
        assert tape['public_index'] == index
        assert meta['family_id'] in SPLITS[meta['split']]
        sid = meta['session_id']
        source = data['traces'][sid]
        for method in ('raw', 'geoi_slack_reconstructed'):
            assert tape[method] == meta[method]
            public_arrays(tape[method])
        assert len(tape['raw']['events']) == len(range(0, len(source), 60))
        for event, point in zip(tape['raw']['events'], source[::60]):
            assert event['timestamp_s'] == point['time_s']-source[0]['time_s']
            c = event['candidates'][0]
            assert len(event['candidates']) == 1
            assert (c['lat'], c['lon']) == (point['lat'], point['lon'])
        assert [e['timestamp_s'] for e in tape['raw']['events']] == [
            e['timestamp_s'] for e in tape['geoi_slack_reconstructed']['events']]
        sessions[sid] = dict(family_id=meta['family_id'], split=meta['split'],
                            role=meta['role'], tape=tape,
                            first_visible_time_s=source[0]['time_s'])
    by_role = {(s['family_id'], s['role']): sid for sid, s in sessions.items()}
    pairs = []
    for record in data['records']:
        if record['case_id'] not in ('S8.A', 'S8.B'):
            continue
        a, b = record['session_ids']
        assert a in sessions and b in sessions
        assert record['observation_policy']['clock'] == 'common_pair_epoch'
        assert sessions[a]['family_id'] == sessions[b]['family_id'] == record['family_id']
        assert sessions[a]['role'] == 'base'
        assert sessions[b]['role'] == ('partial' if record['case_id'] == 'S8.A' else 'companion')
        assert record['labels']['declared_companions'] is True
        epoch = min(sessions[a]['first_visible_time_s'], sessions[b]['first_visible_time_s'])
        split = sessions[a]['split']
        groups = SPLITS[split]
        other = groups[(groups.index(record['family_id'])+1) % len(groups)]
        # Fixed within-split rotation, never choose an unrelated partner by score.
        unrelated = by_role[(other, sessions[b]['role'])]
        pairs.append(dict(record_id=record['record_id'], family_id=record['family_id'],
            split=split, case_id=record['case_id'], target_sid=a, partner_sid=b,
            unrelated_sid=unrelated,
            target_offset_s=sessions[a]['first_visible_time_s']-epoch,
            partner_offset_s=sessions[b]['first_visible_time_s']-epoch,
            public_role=sessions[b]['role'],
            target_events=len(sessions[a]['tape']['raw']['events']),
            association='Observer is explicitly told which account is the associated partner; '
                        'relationship/proximity labels themselves are not features'))
    assert len(pairs) == 22
    return sessions, pairs


def frames(sessions, pairs, view):
    out = []
    for pair in pairs:
        target = sessions[pair['target_sid']]['tape']
        raw = view == 'raw_target_positive_control'
        for index, truth_event in enumerate(target['raw']['events']):
            cut = truth_event['timestamp_s']+pair['target_offset_s']
            a = causal_prefix(target['raw' if raw else 'geoi_slack_reconstructed'],
                              pair['target_offset_s'], cut)
            partner = None
            if view.startswith('joint_'):
                sid = pair['unrelated_sid'] if 'unrelated' in view else pair['partner_sid']
                method = 'raw' if view == 'joint_public_raw_partner' else 'geoi_slack_reconstructed'
                # The negative control replaces the associated stream with an
                # unrelated stream at the same granted public first-visible offset.
                partner = causal_prefix(sessions[sid]['tape'][method], pair['partner_offset_s'], cut)
            c = truth_event['candidates'][0]
            truth = xy_from_latlon([[c['lat'], c['lon']]])[0]
            out.append(dict(record_id=pair['record_id'], family_id=pair['family_id'],
                split=pair['split'], case_id=pair['case_id'], target_event_index=index,
                public_cut_s=cut, partner_visible=partner is not None,
                target=a, partner=partner, truth_xy=truth,
                x=features(a, partner, cut, joint=view.startswith('joint_'))))
    return out


def aggregate(rows, prediction):
    errors = np.linalg.norm(np.asarray([r['truth_xy'] for r in rows])-prediction, axis=1)
    cells = defaultdict(list)
    for row, error in zip(rows, errors):
        cells[(row['family_id'], row['case_id'])].append(float(error))
    by_family = defaultdict(list)
    for (family, case), values in cells.items():
        by_family[family].append(dict(case_id=case, mae_m=float(np.mean(values)),
            hit100=float(np.mean(np.asarray(values) <= 100.)), events=len(values)))
    families = {f: dict(mae_m=float(np.mean([v['mae_m'] for v in vals])),
                       hit100=float(np.mean([v['hit100'] for v in vals])),
                       cases=vals) for f, vals in sorted(by_family.items())}
    return dict(family_macro_mae_m=float(np.mean([v['mae_m'] for v in families.values()])),
        family_macro_hit100=float(np.mean([v['hit100'] for v in families.values()])),
        family_values=families, events=len(rows), case_family_cells=len(cells),
        partner_missing_events=sum(not r['partner_visible'] for r in rows),
        event_pooled_mae_m=float(errors.mean()),
        event_pooled_hit100=float(np.mean(errors <= 100.)))


def predictions(bank, rows, view, target_bank=None):
    x = np.asarray([r['x'] for r in rows])
    candidates = bank.predict(x)
    candidates['public_target_last'] = np.array([
        position_from_public(r['target'], r['public_cut_s']) for r in rows])
    candidates['public_target_velocity'] = np.array([
        position_from_public(r['target'], r['public_cut_s'], velocity=True) for r in rows])
    if view.startswith('joint_'):
        # A joint bank may ignore the partner, rather than forcing fusion.
        prior = target_bank.predict(np.asarray([prefix_features(r['target']) for r in rows]))
        candidates.update({f'ignore_partner_{name}': value for name, value in prior.items()})
        for velocity in (False, True):
            name = 'public_partner_velocity' if velocity else 'public_partner_last'
            candidates[name] = np.array([position_from_public(r['partner'] or r['target'],
                r['public_cut_s'], velocity=velocity) for r in rows])
    return candidates


def declare(out=OUT):
    out = Path(out)
    if (out/'protocol.json').exists():
        raise FileExistsError('Protocol is write-once')
    _, pairs = inventory()
    files = [DATA, f'{ANCESTOR}/linkage_results.json', f'{ANCESTOR}/linkage_public_transcripts.json',
             f'{ANCESTOR}/protocol_v2.json', f'{ANCESTOR}/validation_recheck.json',
             'experiments/s8_companion_inference_20261007.py',
             'experiments/verify_s8_companion_inference_20261007.py',
             'tests/test_s8_companion_inference.py', 'evaluation/identity_future.py',
             'requirements.txt', 'requirements-dev.txt']
    p = dict(schema='historical-frozen-S8-location-diagnostic-v1', seed=SEED,
        splits=SPLITS, views=list(VIEWS), source_sha256={f: sha(ROOT/f) for f in files},
        pairs=pairs, inventory_sha256=canonical(pairs),
        observer='Target and partner account association + first-visible common-clock offsets '
                 'explicitly granted; only causal public events enter features. No route/labels/proximity oracle.',
        Raw_auxiliary='Saved single-coordinate partner Raw transcript explicitly granted to observer; '
                      'same60s clock and causal cut, not arbitrary evaluator GPS',
        original_mechanism=dict(name='historical GeoI-Slack on reconstructed public map',
            K=5, nominal_B=.24, H=12, effective_per_session_cap=.23,
            read_interval_s=60., theta_m=200., utility_slack=.03,
            unchanged=True, current_Epoch8_or_L30=False,
            private_stream_limit='Historical session-ID-derived synthetic seeds were public/reproducible; '
                'this finite-bank observer does not exploit sampler-seed inversion. No secret-key-aware protection claim.'),
        target='True target GPS at EACH saved60s target emission, available from Raw positive-control source',
        pairing='Only22 realized S8.A/B records; no incidental protected tape exists. Unrelated negative '
                'uses next family within same split, same partner role/offset, fixed before scoring.',
        bank='ExtraTrees96 leaf2 depth16 + standardized distance-weighted kNN1/5/15 + train prior; '
             'public target last/velocity; joint bank includes target-only alternatives + public partner last/velocity',
        selection='Minimize equal-family MAE (equal defined A/B cases within each family) on selection only; '
                  'save attacker_selection.json BEFORE predicting/scoring test. Keep finite-bank test scores descriptive.',
        unit='Family group; all pair windows/session roles remain in one split. Test3 families only; no tick CI.',
        primary='Absolute target-only risk and change under joint protected partner; Raw auxiliary limit and '
                'unrelated negative separate. No success threshold or claimed improvement prescribed.',
        scope='Previously inspected12-family6/3/3 synthetic development; legacy per-session cap, '
              'reconstructed map, one retained draw; NOT current-model/fullS8/group-privacy/real-person confirmation')
    out.mkdir(parents=True, exist_ok=True)
    for f in files:
        target = out/'source_snapshot'/f
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT/f, target)
    write(out/'protocol.json', p)
    write(out/'protocol_preview.json', dict(schema=p['schema'], pair_count=len(pairs),
        split_pair_counts={s: sum(r['split']==s for r in pairs) for s in SPLITS},
        split_target_event_counts={s: sum(r['target_events'] for r in pairs if r['split']==s) for s in SPLITS},
        views=p['views'], observer=p['observer'], scope=p['scope']))
    return p


def contract(out=OUT):
    out = Path(out)
    p = json.loads((out/'protocol.json').read_text())
    assert p['schema'] == 'historical-frozen-S8-location-diagnostic-v1'
    for file, digest in p['source_sha256'].items():
        assert sha(ROOT/file) == sha(out/'source_snapshot'/file) == digest
    sessions, pairs = inventory()
    assert pairs == p['pairs'] and canonical(pairs) == p['inventory_sha256']
    return p, sessions, pairs


def run(out=OUT):
    out = Path(out)
    p, sessions, pairs = contract(out)
    if (out/'attacker_selection.json').exists() or (out/'readout.json').exists():
        raise FileExistsError('Use the independently verified frozen readout; no overwrite/refit')
    all_rows = {view: frames(sessions, pairs, view) for view in VIEWS}
    banks, selection = {}, {}
    for view in VIEWS:
        train = [r for r in all_rows[view] if r['split']=='train']
        select = [r for r in all_rows[view] if r['split']=='selection']
        bank = RegressorBank([r['x'] for r in train], [r['truth_xy'] for r in train], SEED)
        banks[view] = bank
        estimates = predictions(bank, select, view, banks.get('target_only'))
        values = {name: aggregate(select, pred) for name, pred in estimates.items()}
        chosen = min(values, key=lambda name: (values[name]['family_macro_mae_m'], name))
        selection[view] = dict(selected=chosen, selection_bank=values)
    write(out/'attacker_selection.json', dict(protocol_sha256=sha(out/'protocol.json'),
        test_predictions_not_opened=True, selection=selection))
    results, saved = {}, []
    for view in VIEWS:
        test = [r for r in all_rows[view] if r['split']=='test']
        pred = predictions(banks[view], test, view, banks.get('target_only'))
        name = selection[view]['selected']
        results[view] = dict(selected_attacker=name, test=aggregate(test, pred[name]),
                            bank_test_descriptive_only={n:aggregate(test, y) for n, y in pred.items()})
        for row, y in zip(test, pred[name]):
            saved.append({k:row[k] for k in ('record_id','family_id','case_id','target_event_index',
                'public_cut_s','partner_visible')} | dict(view=view,
                    evaluator_truth_xy_m=row['truth_xy'].tolist(), prediction_xy_m=y.tolist()))
    assert results['raw_target_positive_control']['test']['family_macro_mae_m'] <= 1e-9
    write(out/'predictions.json', dict(evaluator_only=True, rows=saved))
    write(out/'readout.json', dict(schema='historical-frozen-S8-location-readout-v1',
        protocol_sha256=sha(out/'protocol.json'), selection_sha256=sha(out/'attacker_selection.json'),
        predictions_sha256=sha(out/'predictions.json'), results=results,
        no_new_GPS_sampler_or_protection_calls=True, scope=p['scope']))
    contract(out)
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('declare', 'contract', 'run'))
    parser.add_argument('--output', type=Path, default=OUT)
    args = parser.parse_args()
    value = declare(args.output) if args.action=='declare' else run(args.output) if args.action=='run' else contract(args.output)[0]
    print(json.dumps({'action':args.action, 'output':str(args.output), 'status':'complete',
                      'schema':value.get('schema')}, indent=2))
