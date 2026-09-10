"""Publish the verified fresh shard; preserve the historical DB byte-for-byte."""
from data.scenario_store import ScenarioStore
from copy import deepcopy
from data.scenario_store.store import content_hash
from experiments.run_service_cover import ROOT, read, write, sha

FOLDER = ROOT/'artifacts/datasets/urban_fresh_v2'
DB = ROOT/'artifacts/datasets/evaluation_v1.sqlite3'
RELEASE = 'urban-fresh-v2'


def publish():
    if (FOLDER/'registry.json').exists(): raise FileExistsError('Immutable publication receipt')
    d, verified = read(FOLDER/'dataset.json'), read(FOLDER/'verification.json')
    assert verified['verified'] and verified['dataset_sha256'] == sha(FOLDER/'dataset.json')
    assert verified['dataset_content_sha256'] == content_hash(d)
    assert verified['source_sha256'] == sha(ROOT/'experiments/verify_fresh_scenarios.py')
    old_db = ROOT/'artifacts/datasets/scenarios.sqlite3'; old_hash = sha(old_db)
    ancestor = d['external_ancestor']
    assert sha(ROOT/ancestor['path']) == ancestor['sha256']
    with ScenarioStore(old_db) as old:
        assert old.verify()['status'] == 'passed'
        previous = old.releases()
        assert any(r['source_sha256'] == ancestor['sha256'] for r in previous)
    # Explicit metadata adapter, not a schema/guard bypass. Keep the generator's
    # role as evaluation_role and use the store's established split vocabulary.
    adapted = deepcopy(d)
    mapping = {'fresh_validation': 'development_validation', 'fresh_confirmation': 'confirmation'}
    for row in adapted['families']+adapted['records']:
        row['evaluation_role'] = row['split']
        row['split'] = mapping[row['split']]
    adapted['registry_adapter'] = {'schema': 'fresh-role-to-store-v1',
        'generator_source_sha256': sha(FOLDER/'dataset.json'), 'split_mapping': mapping}
    source = ROOT/'cache/fresh_registry_v1/dataset.json'
    if source.exists(): assert read(source) == adapted
    else: write(source, adapted)
    if not DB.exists(): ScenarioStore.create(DB)
    with ScenarioStore(DB, writable=True) as store:
        result = store.import_revision(source, RELEASE, expected_parent=None,
            actor='codex', reason='Fresh 12 SUMO families with continuous lane changes; six validation and six confirmation; separate immutable evaluation shard')
        assert store.export_bundle(RELEASE) == adapted
        assert store.verify()['status'] == 'passed'
        for r in d['records']:
            for slot, indices in enumerate(r['observed_indices']):
                assert len(store.device_view(RELEASE, r['record_id'], slot=slot)) == len(indices)
        logs = store.releases()
    assert sha(old_db) == old_hash
    write(FOLDER/'registry.json', {**result, 'verified': True, 'database': str(DB.relative_to(ROOT)),
        'database_byte_sha256_at_publish': sha(DB), 'historical_database_sha256': old_hash,
        'historical_release_logs': previous, 'update_log': logs, 'device_records_checked': len(d['records']),
        'generator_dataset_sha256': sha(FOLDER/'dataset.json'), 'split_adapter': adapted['registry_adapter'],
        'source_sha256': sha(__file__)})
    print(result, flush=True)


if __name__ == '__main__': publish()
