"""Independently recalculate saved endpoint metrics, clock and service scores."""
import argparse
import gzip
import hashlib
import json
import pickle
from pathlib import Path
import tarfile

import numpy as np

from evaluation.endpoint_noise_attacks import family_mean
from evaluation.live_poi import AvailabilityWorld, EpochResponseCache, LivePointService, score_returned
from experiments import endpoint_noise_loop as common
from experiments.public_research_resources import load_public_research_resources, sha


def validate(out, cache, *, depth=False):
    p, selection, heldout = [common.read(out/name) for name in
                             ('protocol.json', 'selection.json', 'heldout.json')]
    assert sha(out/'protocol.json') == (out/'protocol.sha256').read_text().strip()
    assert heldout['selection_sha256'] == sha(out/'selection.json')
    assert heldout['protocol_sha256'] == sha(out/'protocol.json')
    assert heldout['selected_defense'] == selection['selected_defense']
    assert sha(common.DATA) == p['dataset_sha256']
    archived_sources = {}
    if (out/'sources.tar.gz').exists():
        archive_metadata = common.read(out/'source_archive.json')
        assert sha(out/'sources.tar.gz') == archive_metadata['sha256']
        with tarfile.open(out/'sources.tar.gz', 'r:gz') as bundle:
            archived_sources = {member.name: bundle.extractfile(member).read()
                                for member in bundle.getmembers() if member.isfile()}
    for file, digest in p['source_sha256'].items():
        if sha(common.ROOT/file) != digest:
            assert file in archived_sources, file
            assert hashlib.sha256(archived_sources[file]).hexdigest() == digest, file
    if depth:
        from experiments.endpoint_noise_depth_loop import FRESH
        assert sha(FRESH) == p['fresh_dataset_sha256']
        dataset = common.read(FRESH)
    else:
        dataset = common.read(common.DATA)
    fit, dev, test = (set(p[k]) for k in ('fit_families', 'selection_families', 'test_families'))
    assert not fit & dev and not fit & test and not dev & test
    saved_models = out/'attackers.pkl'
    if saved_models.exists():
        assert sha(saved_models) == selection['attackers_sha256']
        model_bytes = saved_models.read_bytes()
    else:
        model_bytes = gzip.decompress((out/'attackers.pkl.gz').read_bytes())
        assert hashlib.sha256(model_bytes).hexdigest() == selection['attackers_sha256']
    saved = pickle.loads(model_bytes)  # trusted local artifact, verified above
    rn, _, _, _, _, ranking, metadata = load_public_research_resources(cache)
    assert metadata['catalogue'] == common.read(out/'resources.json')['catalogue']
    expected = {s['session_id']: s for s in common.sessions(dataset, p['test_families'])}
    attack_rows = common.read(out/'heldout_attack_rows.json.gz')
    checks = metric_count = services = attack_predictions = 0
    for method, summary in heldout['rows'].items():
        executions = common.read(out/f'test-{method}.json.gz')
        assert {r['family_id'] for r in executions} == test
        assert len(executions) == len(expected)*len(p['rep_seeds'])
        for item in executions:
            session = expected[item['session_id']]
            is_delay = method in ('delay60', 'delay60_L10')
            if not is_delay:
                assert [e['timestamp_s'] for e in item['events']] == [pt['t'] for pt in session['points']]
                assert item['accounting']['released_events'] == item['accounting']['input_events']
                assert item['publication_mean_delay_s'] == 0.
            else:
                assert item['accounting']['released_events'] < item['accounting']['input_events']
                assert item['publication_mean_delay_s'] >= 60.
            for scenario, point in [('S9', session['points'][0]), ('S10', session['points'][-1])]:
                assert np.array_equal(item['target_xy_evaluator_only'][scenario],
                                      rn.point_xy(point['lat'], point['lon']))
                bank = saved['banks' if depth else 'bank'][method, scenario]
                recomputed = common.bank_rows([item], scenario, bank, rn, saved['history'])[0]
                recorded = next(r for r in attack_rows if r['method']==method and r['scenario']==scenario
                    and r['session_id']==item['session_id'] and r['seed']==item['seed'])
                assert set(recomputed['errors']) == set(recorded['errors'])
                for name, value in recomputed['errors'].items():
                    assert np.isclose(value, recorded['errors'][name], atol=1e-9), name
                    attack_predictions += 1
            if method != 'raw':
                if depth:
                    phase, scale = item['config']['phase'], item['config']['scale']
                    unit = .0025 if phase else .01*scale
                    bound = .23 if phase else .23*scale
                else:
                    unit = .01*common.SCALE[method]
                    bound = .23*common.SCALE[method]
                assert item['budget_spent_per_m'] <= bound+1e-12
                assert np.isclose(item['budget_spent_per_m']/unit,
                                  round(item['budget_spent_per_m']/unit), atol=1e-12)
            for event in item['events']:
                assert set(event) == {'timestamp_s', 'coordinates'}
                assert len(event['coordinates']) == (1 if method=='raw' else 5)
            world = AvailabilityWorld(ranking.n, p['world_seed'], probability=.8, epoch_seconds=60)
            L = item['config']['L'] if depth else 10
            server, client = LivePointService(ranking, world, response_l=L), EpochResponseCache(ranking.n)
            by_time = {}
            for event in item['events']:
                by_time.setdefault(event['timestamp_s'], []).append(event)
            scores, bytes_total = [], 0
            for point in session['points']:
                epoch, replies = world.epoch(point['t']), []
                for event in by_time.get(point['t'], []):
                    reply = [server.query(rn.nearest(*coordinate)[0], epoch) for coordinate in event['coordinates']]
                    replies.extend(reply)
                    if depth:
                        bytes_total += len(json.dumps(dict(event, L=L), separators=(',', ':')).encode())
                        bytes_total += len(json.dumps([[[ranking.pois[i]['id'] for i in category]
                            for category in r] for r in reply], separators=(',', ':')).encode())
                _, known = client.receive(epoch, replies)
                state = rn.nearest(point['lat'], point['lon'])[0]
                available = world.at_epoch(epoch)
                score = score_returned(ranking.top(state, available, 5), ranking.top(state, known, 5), available)['recall']
                if score is not None:
                    scores.append(score)
                services += 1
            assert np.isclose(item['recall'], np.mean(scores), atol=1e-14)
            if depth:
                assert item['request_bytes']+item['response_bytes'] == bytes_total
            checks += 1
        for scenario in ('S9', 'S10'):
            rows = [r for r in attack_rows if r['method']==method and r['scenario']==scenario]
            assert len(rows) == len(executions)
            chosen = selection['selected_attackers'][method][scenario]
            for key in ('mae_m', 'hit50', 'hit100', 'hit200', 'hit500'):
                attacker = chosen['mae' if key=='mae_m' else key]
                values = [dict(r, v=r['errors'][attacker] if key=='mae_m'
                               else float(r['errors'][attacker] <= int(key[3:]))) for r in rows]
                assert np.isclose(summary[scenario][key], family_mean(values, 'v'), atol=1e-14)
                metric_count += 1
            assert chosen == summary[scenario]['selected_attackers']
        assert np.isclose(summary['recall'], family_mean(executions, 'recall'), atol=1e-14)
        if depth:
            assert np.isclose(summary['bytes_per_input'], family_mean(executions, 'bytes_per_input'), atol=1e-14)
    assert all(heldout['raw_positive_control'].values())
    result = dict(status='pass', sessions_recomputed=checks, endpoint_metrics_recomputed=metric_count,
        input_service_times_recomputed=services, raw_endpoint_hit100=1.,
        individual_attack_predictions_recomputed=attack_predictions,
        protocol_and_selection_hashes_verified=True, disjoint_family_splits=True,
        public_transcript_field_check=True, all_input_times_preserved_for_no_delay=True,
        privacy_cap_and_integer_unit_accounting=True,
        reconstructed_network_imported=True, original_network_turns_verified=False,
        native_sumo_network_import='netconvert checked externally; original turns still unknown',
        reviewer_scope='Small reconstructed-map development only')
    common.write(out/'verification.json', result)
    print(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--depth', action='store_true')
    parser.add_argument('--cache', type=Path, default=Path('/private/tmp/trajectory-research-20261005-public-map'))
    args = parser.parse_args()
    validate(args.out, args.cache, depth=args.depth)


if __name__ == '__main__':
    main()
