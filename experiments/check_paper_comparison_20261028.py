"""Replay the frozen verifier without replacing its write-once receipt."""
from experiments import verify_paper_comparison_20261028 as verifier


def main():
    study = verifier.study
    study.validate_sources()
    receipt_path = study.OUT / 'validation.json'
    retained = study.read(receipt_path)
    for name, digest in retained['files_sha256'].items():
        assert study.sha(study.OUT / name) == digest, name
    original_save = study.save

    def compare_receipt(path, value):
        if path != receipt_path:
            raise AssertionError('Read-only replay attempted an unexpected write')
        assert value == retained, 'Independent replay differs from retained receipt'

    study.save = compare_receipt
    try:
        verifier.verify()
    finally:
        study.save = original_save
    print('PASS: retained evidence and receipt unchanged; no generation or fitting')


if __name__ == '__main__':
    main()
