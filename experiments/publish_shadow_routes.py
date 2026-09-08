"""Import only the verified auxiliary release; preserve all old snapshots."""
from pathlib import Path

from data.scenario_store import ScenarioStore
from data.scenario_store.store import content_hash
from experiments.run_service_cover import read, write, sha

ROOT = Path(__file__).resolve().parents[1]
FOLDER = ROOT / 'artifacts/datasets/urban_shadow_v1'
DB = ROOT / 'artifacts/datasets/scenarios.sqlite3'
RELEASE = 'urban-shadow-v1'


def publish():
    if (FOLDER / 'registry.json').exists():
        raise FileExistsError('Preserve the original publication receipt')
    data = read(FOLDER / 'dataset.json')
    verified = read(FOLDER / 'verification.json')
    assert verified['verified'] and verified['dataset_sha256'] == sha(FOLDER / 'dataset.json')
    assert verified['source_sha256'] == sha(ROOT / 'experiments/verify_shadow_routes.py')
    assert verified['dataset_content_sha256'] == content_hash(data)
    with ScenarioStore(DB, writable=True) as store:
        before = store.releases()
        assert store.head() == 'urban-scenarios-v3' and len(before) == 3
        assert store.verify()['status'] == 'passed'
        result = store.import_revision(FOLDER / 'dataset.json', RELEASE,
            expected_parent='urban-scenarios-v3', actor='codex',
            reason='Add 64 SUMO auxiliary shadow-training and 16 holdout route families; preserve v1/v2/v3 benchmark snapshots')
        assert store.releases()[:3] == before
        assert store.export_bundle(RELEASE) == data
        check = store.verify()
        assert check['status'] == 'passed'
        for r in data['records']:
            p = store.device_view(RELEASE, r['record_id'])
            assert len(p) == len(r['observed_indices'][0])
        log = store.releases()[-1]
    write(FOLDER / 'registry.json', {**result, 'verified': True,
        'database': str(DB.relative_to(ROOT)), 'database_byte_sha256_at_publish': sha(DB),
        'old_release_logs_preserved': before, 'update_log': log,
        'device_views_checked': len(data['records']), 'source_sha256': sha(__file__)})
    print(result)


if __name__ == '__main__':
    publish()
