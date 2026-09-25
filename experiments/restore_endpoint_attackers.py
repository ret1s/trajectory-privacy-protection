"""Restore the sealed local attacker checkpoint without deserializing it."""
import gzip
import json

from experiments.research_loop_resources import ROOT, sha


def main():
    base = ROOT / 'artifacts/benchmarks/endpoint_calendar_v1'
    selection = json.loads((base / 'selection.json').read_text())
    manifest = json.loads((base / 'attacker_archive.json').read_text())
    archive = base / 'selected_attackers.pkl.gz'
    assert sha(archive) == manifest['archive_sha256']
    assert selection['model_sha256'] == manifest['model_sha256']
    target = ROOT / selection['model_path']
    if target.exists():
        assert sha(target) == selection['model_sha256'], 'Preserve and inspect the existing different checkpoint'
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix('.restoring')
        temporary.write_bytes(gzip.decompress(archive.read_bytes()))
        assert sha(temporary) == selection['model_sha256']
        temporary.rename(target)
    print('Frozen attacker checkpoint verified:', selection['model_sha256'])


if __name__ == '__main__': main()
