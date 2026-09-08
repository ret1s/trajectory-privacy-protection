"""Local scenario registry CLI. Example: python -m experiments.scenario_db verify."""
import argparse
import json
from pathlib import Path
import sqlite3

from data.scenario_store import ScenarioStore, StoreError

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB = ROOT / 'artifacts/datasets/scenarios.sqlite3'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--db', type=Path, default=DEFAULT_DB)
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('init')
    imp = commands.add_parser('import')
    imp.add_argument('source', type=Path)
    imp.add_argument('--release', required=True)
    imp.add_argument('--parent', required=True, help='Expected current head, or NONE for first release')
    imp.add_argument('--actor', required=True)
    imp.add_argument('--reason', required=True)
    commands.add_parser('log')
    commands.add_parser('verify')
    cov = commands.add_parser('coverage')
    cov.add_argument('--release', required=True)
    for name in ('export', 'device'):
        sub = commands.add_parser(name)
        sub.add_argument('--release', required=True)
        sub.add_argument('--sha256', required=True, help='Pinned semantic content SHA256 (not JSON byte SHA256)')
        sub.add_argument('--output', type=Path, required=True, help='New file; refuses overwrite')
        if name == 'device':
            sub.add_argument('--record', required=True)
            sub.add_argument('--slot', type=int, default=0)
    args = parser.parse_args(argv)
    try:
        if args.command == 'init':
            ScenarioStore.create(args.db)
            result = {'status': 'created', 'database': str(args.db)}
        else:
            with ScenarioStore(args.db, writable=args.command == 'import') as store:
                if args.command == 'import':
                    result = store.import_revision(args.source, args.release,
                        expected_parent=None if args.parent == 'NONE' else args.parent,
                        actor=args.actor, reason=args.reason)
                elif args.command == 'log':
                    result = store.releases()
                elif args.command == 'verify':
                    result = store.verify()
                elif args.command == 'coverage':
                    result = store.coverage(args.release)
                else:
                    # Pin before consumption; never silently substitute current head.
                    if store._release(args.release)['content_sha256'] != args.sha256:
                        raise StoreError('pinned content hash does not match release')
                    if args.command == 'export':
                        result = store.export_bundle(args.release)
                    else:
                        result = {'release_id': args.release, 'content_sha256': args.sha256,
                                  'record_id': args.record, 'slot': args.slot,
                                  'visibility': 'private_device_input_not_attacker_output',
                                  'observations': store.device_view(args.release, args.record, slot=args.slot)}
                    with args.output.open('x') as stream:
                        json.dump(result, stream, ensure_ascii=False, indent=2, allow_nan=False)
                        stream.write('\n')
                    result = {'status': 'exported', 'output': str(args.output), 'release_id': args.release,
                              'content_sha256': args.sha256}
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    except (StoreError, sqlite3.Error, OSError, KeyError, TypeError, ValueError) as exc:
        parser.exit(2, f'scenario database: {exc}\n')


if __name__ == '__main__':
    raise SystemExit(main())
