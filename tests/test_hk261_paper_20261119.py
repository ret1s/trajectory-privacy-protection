"""Guard scientific parity and supplied-template fidelity in the new paper version."""
import shutil

import pytest

from experiments.build_hk261_paper_20261119 import PAPER, source_check, validate


def test_hk261_version_preserves_science_template_and_portable_build():
    result = validate()
    assert result["original_version_unchanged"]
    assert result["pages"] <= 6
    assert result["theorem_numbers_preserved"]
    assert result["structural"]["references"] == 18


def test_hk261_version_rejects_a_changed_proof(tmp_path):
    shutil.copytree(PAPER, tmp_path / "paper")
    folder = tmp_path / "paper"
    path = folder / "analysis.tex"
    path.write_text(path.read_text().replace("2.437372", "1.000001"))
    with pytest.raises(AssertionError, match="Scientific input changed"):
        source_check(folder)


def test_hk261_version_rejects_changes_to_supplied_class(tmp_path):
    shutil.copytree(PAPER, tmp_path / "paper")
    folder = tmp_path / "paper"
    path = folder / "llncs.cls"
    path.write_text(path.read_text() + "\n% local modification\n")
    with pytest.raises(AssertionError, match="Supplied template changed"):
        source_check(folder)
