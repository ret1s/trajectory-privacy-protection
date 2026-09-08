"""Storage failure injection and lossless benchmark-input regression tests."""
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import sqlite3
from threading import Barrier

import pytest

from data.scenario_store import ScenarioStore, StoreError
from data.scenario_store.store import TABLES, content_hash
from data.scenario_suite.records import device_view
from experiments.scenario_db import main

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def bundle():
    families, traces, records = [], {}, []
    for n, split in enumerate(('development_train', 'confirmation')):
        fid, sid = f'f{n}', f'u{n}'
        families.append({'family_id': fid, 'split': split, 'seed': n,
                         'sessions': [{'session_id': sid, 'person_id': f'p{n}',
                                       'device_id': f'd{n}', 'physical_vehicle_id': f'c{n}', 'role': 'base'}]})
        traces[sid] = [{'time_s': float(i), 'lat': 40., 'lon': 116., 'speed_m_s': 8.,
                        'lane_id': 'e_0', 'edge_id': 'e', 'lane_pos_m': float(i), 'angle_deg': 0.}
                       for i in range(3)]
        records.append({'record_id': f'r{n}', 'family_id': fid, 'split': split,
                        'case_id': 'S1.A', 'scenario': 'S1', 'session_ids': [sid],
                        'observed_indices': [[0, 2]], 'labels': {'target_index': 0}, 'evidence': {},
                        'observation_policy': {'clock': 'relative_to_first_allowed_sample_per_session'}})
    return {'schema': 'urban-scenario-suite-v2', 'families': families, 'traces': traces,
            'records': records, 'catalogue': [{'case_id': 'S1.A', 'scenario': 'S1', 'generated_records': 2}],
            'source_sha256': {'example.py': 'a'*64}, 'summary': {
                'families': 2, 'sessions': 2, 'raw_fcd_samples': 6, 'scenario_records': 2,
                'generated_subcases': 1, 'declared_subcases': 1, 'generated_scenarios': 1,
                'by_case': {'S1.A': 2}, 'by_scenario': {'S1': 2}}}


def save(tmp_path, bundle, name='source.json'):
    path = tmp_path / name
    path.write_text(json.dumps(bundle))
    return path


@pytest.fixture
def database(tmp_path, bundle):
    path = tmp_path / 'scenarios.sqlite3'
    ScenarioStore.create(path)
    with ScenarioStore(path, writable=True) as store:
        store.import_revision(save(tmp_path, bundle), 'v1', expected_parent=None, actor='test', reason='initial')
    return path


def test_roundtrip_and_pin(database, bundle):
    with ScenarioStore(database) as store:
        assert store.export_bundle('v1') == bundle
        assert store.verify()['status'] == 'passed'
        assert store.connection.execute('PRAGMA foreign_keys').fetchone()[0] == 1
        assert store.device_view('v1', 'r0') == list(device_view(bundle['records'][0], bundle['traces']))
        assert store.releases()[0]['content_sha256'] == content_hash(bundle)
        with pytest.raises(sqlite3.OperationalError):
            store.connection.execute("UPDATE releases SET state='sealed'")


def test_no_overwrite_or_silent_creation(database, tmp_path):
    with pytest.raises(FileExistsError):
        ScenarioStore.create(database)
    with pytest.raises(sqlite3.OperationalError):
        ScenarioStore(tmp_path / 'missing.sqlite3')


def test_idempotence_and_conflict(database, bundle, tmp_path):
    with ScenarioStore(database, writable=True) as store:
        result = store.import_revision(save(tmp_path, bundle), 'v1', expected_parent=None, actor='test', reason='retry')
        assert result['status'] == 'already_present' and len(store.releases()) == 1
        bundle['purpose'] = 'revised'
        with pytest.raises(StoreError, match='different content'):
            store.import_revision(save(tmp_path, bundle), 'v1', expected_parent=None, actor='test', reason='conflict')
        assert store.head() == 'v1'


def test_new_revision_preserves_old(database, bundle, tmp_path):
    original = copy.deepcopy(bundle)
    bundle['purpose'] = 'new revision'
    with ScenarioStore(database, writable=True) as store:
        store.import_revision(save(tmp_path, bundle), 'v2', expected_parent='v1', actor='alice', reason='clarify scope')
        assert store.export_bundle('v1') == original
        assert store.export_bundle('v2') == bundle
        assert store.head() == 'v2' and len(store.verify()['releases']) == 2
        assert store.releases()[1]['reason'] == 'clarify scope'
        bundle['purpose'] = 'stale'
        with pytest.raises(StoreError, match='stale'):
            store.import_revision(save(tmp_path, bundle), 'v3', expected_parent='v1', actor='bob', reason='stale writer')
        assert len(store.releases()) == 2


@pytest.mark.parametrize('table', (*TABLES, 'releases', 'update_log'))
def test_immutable_rows_and_log(database, table):
    with ScenarioStore(database, writable=True) as store:
        for statement in (f"UPDATE {table} SET release_id='v1'", f'DELETE FROM {table}'):
            with pytest.raises(sqlite3.IntegrityError):
                store.connection.execute(statement)
        assert store.verify()['status'] == 'passed'


@pytest.mark.parametrize('fault', ('latitude', 'time_order', 'missing_trace', 'observation_index',
                                  'observation_order', 'split', 'identity', 'summary', 'nan', 'slot'))
def test_reject_bad_import_atomically(database, bundle, tmp_path, fault):
    bundle['purpose'] = fault
    if fault == 'latitude': bundle['traces']['u0'][0]['lat'] = 91
    if fault == 'time_order': bundle['traces']['u0'][1]['time_s'] = 0.
    if fault == 'missing_trace': del bundle['traces']['u1']
    if fault == 'observation_index': bundle['records'][0]['observed_indices'] = [[5]]
    if fault == 'observation_order': bundle['records'][0]['observed_indices'] = [[2, 0]]
    if fault == 'split': bundle['records'][0]['split'] = 'confirmation'
    if fault == 'identity': bundle['families'][1]['sessions'][0]['person_id'] = 'p0'
    if fault == 'summary': bundle['summary']['raw_fcd_samples'] = 999
    if fault == 'nan': bundle['traces']['u0'][0]['lat'] = float('nan')
    if fault == 'slot': bundle['records'][0]['labels']['target_slot'] = 3
    with ScenarioStore(database, writable=True) as store:
        with pytest.raises((StoreError, sqlite3.IntegrityError, ValueError, KeyError)):
            store.import_revision(save(tmp_path, bundle), 'bad', expected_parent='v1', actor='test', reason=fault)
        assert store.head() == 'v1' and len(store.releases()) == 1
        assert store.connection.execute('SELECT count(*) FROM releases').fetchone()[0] == 1
        assert store.verify()['status'] == 'passed'


def test_duplicate_json_key(database, tmp_path):
    source = tmp_path / 'duplicate.json'
    source.write_text('{"schema":"a","schema":"b"}')
    with ScenarioStore(database, writable=True) as store:
        with pytest.raises(StoreError, match='duplicate JSON key'):
            store.import_revision(source, 'v2', expected_parent='v1', actor='test', reason='invalid')


def test_schema_guard_detection(database):
    connection = sqlite3.connect(database)
    connection.execute('DROP TRIGGER points_update')
    connection.close()
    with pytest.raises(StoreError, match='schema/guards'):
        ScenarioStore(database)


def test_foreign_key_enforcement_inside_transaction(database):
    with ScenarioStore(database, writable=True) as store:
        con = store.connection
        con.execute('BEGIN IMMEDIATE')
        try:
            con.execute('''INSERT INTO releases SELECT 'staging','v1',?, ?,source_path,
                           created_at,actor,reason,metadata_json,'staging' FROM releases WHERE release_id='v1' ''',
                        ('b'*64, 'c'*64))
            # The old release's u0 cannot satisfy a new release's point FK.
            with pytest.raises(sqlite3.IntegrityError, match='FOREIGN KEY'):
                con.execute("INSERT INTO points VALUES('staging','u0',0,0.,40.,116.,0.,'e_0','e',0.,0.)")
            con.execute("INSERT INTO families VALUES('staging','f0',0,'development_train','{}')")
            con.execute("INSERT INTO cases VALUES('staging','S1.A','S1',0,'{}')")
            with pytest.raises(sqlite3.IntegrityError, match='FOREIGN KEY'):
                con.execute("INSERT INTO records VALUES('staging','r0',0,'S1.A','S1','f0','confirmation','{}')")
        finally:
            con.rollback()
        assert len(store.verify()['releases']) == 1


def test_sealed_release_rejects_new_points_and_replace(database):
    with ScenarioStore(database, writable=True) as store:
        with pytest.raises(sqlite3.IntegrityError):
            store.connection.execute("INSERT INTO points VALUES('v1','u0',3,3.,40.,116.,0.,'e_0','e',0.,0.)")
        with pytest.raises(sqlite3.IntegrityError):
            store.connection.execute("INSERT OR REPLACE INTO points VALUES('v1','u0',0,0.,40.,116.,0.,'e_0','e',0.,0.)")
        assert store.verify()['status'] == 'passed'


def test_competing_writers(database, bundle, tmp_path):
    sources = []
    for i in range(2):
        revised = dict(bundle, purpose=f'change {i}')
        sources.append(save(tmp_path, revised, f'{i}.json'))
    barrier = Barrier(2)
    def write(i):
        with ScenarioStore(database, writable=True) as store:
            barrier.wait(timeout=10)
            try:
                store.import_revision(sources[i], f'new{i}', expected_parent='v1', actor=f'agent{i}', reason='concurrent')
                return 'committed'
            except StoreError as exc:
                assert 'stale' in str(exc)
                return 'stale'
    with ThreadPoolExecutor(max_workers=2) as pool:
        assert sorted(pool.map(write, range(2))) == ['committed', 'stale']
    with ScenarioStore(database) as store:
        assert len(store.verify()['releases']) == 2


def test_cli_pin_and_no_overwrite(database, bundle, tmp_path):
    destination = tmp_path / 'output.json'
    prefix = ['--db', str(database), 'export', '--release', 'v1', '--output', str(destination), '--sha256']
    with pytest.raises(SystemExit) as err:
        main(prefix + ['0'*64])
    assert err.value.code == 2 and not destination.exists()
    assert main(prefix + [content_hash(bundle)]) == 0
    assert json.loads(destination.read_text()) == bundle
    with pytest.raises(SystemExit):
        main(prefix + [content_hash(bundle)])


def test_current_store_all_releases_and_private_input_parity():
    with ScenarioStore(ROOT / 'artifacts/datasets/scenarios.sqlite3') as store:
        releases = store.verify()['releases']
        assert [r['release_id'] for r in releases][:2] == ['urban-scenarios-v1', 'urban-scenarios-v2']
        for version in range(1, len(releases) + 1):
            source = json.loads((ROOT / f'artifacts/datasets/urban_scenarios_v{version}/dataset.json').read_text())
            release = f'urban-scenarios-v{version}'
            assert store.export_bundle(release) == source
            for record in source['records']:
                for slot in range(len(record['session_ids'])):
                    assert store.device_view(release, record['record_id'], slot=slot) == list(device_view(record, source['traces'], slot))
        missing = [row for row in store.coverage('urban-scenarios-v2') if row['case_id'] == 'S5.C']
        assert {r['split']: r['records'] for r in missing} == {
            'development_train': 0, 'development_validation': 0, 'confirmation': 1}
