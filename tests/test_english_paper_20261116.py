"""Evidence-binding and negative checks for the standalone English paper package."""
from pathlib import Path
import json
import shutil

import pytest

from experiments.export_english_paper_20261116 import (
    B1, METHODS, SCENARIOS, PAPER, generate, pct, n,
)
from experiments.validate_english_paper_20261116 import source_check, validate


def test_every_primary_privacy_cell_binds_to_preserved_results():
    outputs, _ = generate()
    result = json.loads((B1 / "results.json").read_text())
    rows = {(t["method"], t["scenario"]): t for t in result["attackers"]}
    assert len(rows) == 30
    for method in METHODS:
        for scenario in SCENARIOS:
            t = rows[method, scenario]
            expected = pct(t["Hit100"]) + " / " + n(t["MAE_m"], 1)
            assert expected in outputs["table_privacy.tex"]
    assert "N/A" in outputs["table_native.tex"]
    # Favorable privacy values must not erase the stronger public-base control.
    assert "99.711" in outputs["table_ablation.tex"]
    assert "99.595" in outputs["table_ablation.tex"]
    assert r"\textbf{Failed}" in outputs["table_coverage.tex"]


def test_source_rejects_a_missing_citation(tmp_path):
    for path in PAPER.iterdir():
        if path.is_file():
            shutil.copyfile(path, tmp_path / path.name)
    entry = tmp_path / "introduction.tex"
    entry.write_text(entry.read_text() + "\n" + r"\cite{invented-paper}")
    with pytest.raises(AssertionError):
        source_check(tmp_path)


def test_source_rejects_nonportable_inputs(tmp_path):
    for path in PAPER.iterdir():
        if path.is_file():
            shutil.copyfile(path, tmp_path / path.name)
    (tmp_path / "main.tex").write_text(r"\input{../thesis/main}")
    with pytest.raises(AssertionError):
        source_check(tmp_path)


def test_completed_pdf_and_flat_archive_have_matching_provenance():
    result = validate()
    assert result["status"] == "pass"
    assert result["frozen_pins_verified"] == 108
    assert result["archive_flat_self_contained"]
    assert result["portable_compilation_text_matches"]
