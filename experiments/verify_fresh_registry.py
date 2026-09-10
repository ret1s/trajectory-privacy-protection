"""Independently reverse the metadata adapter and compare all original data."""
from copy import deepcopy

from data.scenario_store import ScenarioStore
from data.scenario_store.store import content_hash
from experiments.publish_fresh_scenarios import DB, RELEASE, FOLDER
from experiments.run_service_cover import ROOT, read, write, sha


def verify():
    receipt = read(FOLDER/'registry.json'); original = read(FOLDER/'dataset.json')
    assert receipt['source_sha256'] == sha(ROOT/'experiments/publish_fresh_scenarios.py')
    assert receipt['generator_dataset_sha256'] == sha(FOLDER/'dataset.json')
    assert receipt['database_byte_sha256_at_publish'] == sha(DB)
    with ScenarioStore(DB) as store:
        assert store.verify()['status'] == 'passed'
        exported = store.export_bundle(RELEASE)
        assert content_hash(exported) == receipt['content_sha256']
        assert store.releases() == receipt['update_log']
        total = 0
        for r in exported['records']:
            native = next(x for x in original['records'] if x['record_id'] == r['record_id'])
            for slot, (sid, indices) in enumerate(zip(native['session_ids'], native['observed_indices'])):
                data = store.device_view(RELEASE, r['record_id'], slot=slot)
                traces = [original['traces'][s] for s in native['session_ids']]
                epoch = (min(t[ii[0]]['time_s'] for t, ii in zip(traces, native['observed_indices']))
                    if native['observation_policy']['clock'] == 'common_pair_epoch' else traces[slot][indices[0]]['time_s'])
                queries = native['labels'].get('true_queries', [])
                assert data == [{'time_s': traces[slot][i]['time_s']-epoch,
                    'lat': traces[slot][i]['lat'], 'lon': traces[slot][i]['lon'],
                    'query_category': queries[j] if j < len(queries) else 'cafe'} for j, i in enumerate(indices)]
                total += len(data)
    restored = deepcopy(exported)
    adapter = restored.pop('registry_adapter')
    for row in restored['families']+restored['records']:
        role = row.pop('evaluation_role')
        assert row['split'] == adapter['split_mapping'][role]
        row['split'] = role
    assert restored == original
    historical = ROOT/'artifacts/datasets/scenarios.sqlite3'
    assert sha(historical) == receipt['historical_database_sha256']
    with ScenarioStore(historical) as store:
        assert store.verify()['status'] == 'passed'
        assert store.releases() == receipt['historical_release_logs']
    result = {'verified': True, 'registry_sha256': sha(FOLDER/'registry.json'),
        'original_dataset_sha256': sha(FOLDER/'dataset.json'), 'device_observations_checked': total,
        'lossless_reverse_adapter': True, 'historical_database_unchanged': True, 'verifier_sha256': sha(__file__)}
    write(FOLDER/'registry_verification.json', result)
    print(result, flush=True)


if __name__ == '__main__': verify()
