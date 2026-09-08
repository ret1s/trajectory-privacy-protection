"""Independent migration audit against frozen JSON and every allowed device view."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import sqlite3

from data.scenario_store import ScenarioStore
from data.scenario_suite.records import device_view
from experiments.scenario_db import DEFAULT_DB, ROOT

FROZEN = {
    'artifacts/datasets/urban_scenarios_v1/dataset.json': '826bf236588448e5e59b5e4e0304ddef3e3cf85751df8dddf901b72ae88aabb9',
    'artifacts/datasets/urban_scenarios_v2/dataset.json': 'e5ee6dad69074f870f2ef4cd2c1ada1a33d3fd71b846c8ef771a3b7b72856ef8',
    'artifacts/benchmarks/paper_benchmark/results.json': '2df043bfb73c3b2f630046381daf5b4bc9f6941fb5be925b0967d6de98f27538',
    'artifacts/benchmarks/lane_comparison/results.json': '20f93757ba2e2dde28a3b63db01363c8689aa7692b0703c5212412f9e3b7b65c',
    'artifacts/benchmarks/contextual_lane/results.json': '61274220fd89b2ec06ebda6be67cc7f01f305b4e610307fe990486258ef85d0a',
    'artifacts/benchmarks/belief_suite/validation.json': '009d5c0a73ca43d439bd938e374a507f39660d25f3a94e5943e0449a5dc6a5c3',
    'artifacts/benchmarks/belief_suite/selection.json': 'a5d80de1aa1299ecce57c5a49a4f744409a1b5f603c6a17f6850e6efe3d854b6',
    'artifacts/benchmarks/belief_suite/confirmation.json': '0319caba8029aa59f7069ccf58500ab355bb180cda6ec9595eda7b5bdf8382cc',
    'artifacts/benchmarks/belief_suite/results_tables.tex': '5c2a72a3f5d48aa4c0a89f0354700f2ff9e4b0bfed7d1cee9a526df947128a19',
}


def file_hash(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def verify(path=DEFAULT_DB):
    for name, expected in FROZEN.items():
        assert file_hash(ROOT / name) == expected, name
    snapshots = [json.loads((ROOT / f'artifacts/datasets/urban_scenarios_v{v}/dataset.json').read_text()) for v in (1, 2)]
    pinned = json.loads((ROOT / 'artifacts/benchmarks/belief_suite/confirmation.json').read_text())['source_sha256']
    for bundle in snapshots:
        pinned = {**pinned, **bundle['source_sha256']}
    for name, expected in pinned.items():
        assert file_hash(ROOT / name) == expected, name
    with ScenarioStore(path) as store:
        result = store.verify()
        assert store.head() == 'urban-scenarios-v2' and len(store.releases()) == 2
        counts = {'raw_points': 0, 'records': 0, 'device_slots': 0, 'allowed_observations': 0}
        for v, original in enumerate(snapshots, 1):
            release = f'urban-scenarios-v{v}'
            reconstructed = store.export_bundle(release)
            assert reconstructed == original
            counts['raw_points'] += sum(map(len, original['traces'].values()))
            for record in original['records']:
                counts['records'] += 1
                for slot in range(len(record['session_ids'])):
                    expected = list(device_view(record, original['traces'], slot))
                    actual = store.device_view(release, record['record_id'], slot=slot)
                    assert expected == actual
                    assert all(set(p) == {'time_s', 'lat', 'lon', 'query_category'} for p in actual)
                    counts['device_slots'] += 1
                    counts['allowed_observations'] += len(actual)
            # Check complete SQL ordinal sequences independently of bundle export.
            for table, group, ordinal in (
                ('points', 'session_id', 'point_index'), ('observations', 'record_id,slot', 'ordinal'),
                ('record_sessions', 'record_id', 'slot')):
                bad = store.connection.execute(f'''SELECT {group} FROM {table} WHERE release_id=?
                    GROUP BY {group} HAVING min({ordinal})!=0 OR max({ordinal})+1!=count(*)''', (release,)).fetchall()
                assert not bad, (table, bad)
        coverage = store.coverage('urban-scenarios-v2')
        assert len(coverage) == 90  # Catalogue cross all three actual splits, including zeros.
        logs = store.releases()
    sources = ('data/scenario_store/schema.sql', 'data/scenario_store/store.py',
               'data/scenario_store/__init__.py', 'experiments/scenario_db.py',
               'experiments/verify_scenario_db.py', 'tests/test_scenario_store.py')
    return {**result, 'verified_at': datetime.now(timezone.utc).isoformat(),
            'database_sha256': file_hash(path), 'sqlite_version': sqlite3.sqlite_version,
            'migration_comparisons': counts, 'update_log': logs, 'v2_case_split_coverage': coverage,
            'frozen_artifact_sha256': FROZEN, 'frozen_scientific_sources_checked': len(pinned),
            'storage_source_sha256': {p: file_hash(ROOT / p) for p in sources},
            'limitations': ['No new protection/attacker experiment is implied by migration.',
                            'Native FCD and scenario gates require their separate SUMO verifiers.',
                            'Database and device observations are evaluator/private input, not public transcripts.',
                            'Checks detect ordinary corruption, not a database administrator forging every artifact.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, default=DEFAULT_DB)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = verify(args.db)
    rendered = json.dumps(result, indent=2, ensure_ascii=False) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered)
