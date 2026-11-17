"""Evidence-binding and negative checks for the standalone English paper package."""
from pathlib import Path
import json
import shutil

import pytest

from experiments.export_english_paper_20261116 import (
    B1, METHODS, SCENARIOS, PAPER, generate, pct, n,
)
from experiments.validate_english_paper_20261116 import source_check, validate
from experiments.build_english_paper_20261116 import check_main_page_limit
from experiments.export_paper_walkthrough_20261117 import (
    PRIMARY, UPSTREAM, read_gzip, sample_data,
)


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
    assert result["main_page_limit_pass"]
    assert result["pdfs"]["manuscript.pdf"]["pages"] <= 6


def test_walkthrough_uses_retained_wire_and_counts_reuse_as_a_private_read():
    import numpy as np
    from core.road_network import LocalProjection
    data = sample_data()
    trip = read_gzip(PRIMARY)["trips"][0]
    original = read_gzip(UPSTREAM)["trips"][0]
    assert data["native_session_id"] == "cache-029_day7"
    projection = LocalProjection(data["public_projection_lat0"])
    for row in data["records"]:
        i = row["index"]
        assert row["q_latlon"] == trip["events"]["geoi_bq"][i]["coordinates"]
        assert row["z_latlon"] == original["GeoI_anchors"][i]
        assert row["cache_unique"] == int(trip["masks"]["geoi_bq"][i]["session"], 16).bit_count()
        # The map may not move points independently to make separation look better.
        coordinates = np.asarray(trip["events"]["geoi_bq"][i]["coordinates"])
        expected = np.column_stack(projection.to_xy(coordinates[:, 0], coordinates[:, 1]))
        np.testing.assert_allclose(row["q_xy_m"], expected, rtol=0, atol=1e-8)
        np.testing.assert_allclose(row["z_xy_m"], projection.to_xy(*original["GeoI_anchors"][i]),
                                   rtol=0, atol=1e-8)
        assert row["gps_xy_m"] == trip["truth"][i]["xy"]
    for i in (1, 2):
        row = data["records"][i]
        assert row["belief_operation"] == "predict"
        assert row["geo_i_spent_units"] == 1 and not row["private_read"]
        assert row["gps_xy_m"] != data["records"][0]["gps_xy_m"]
    reuse = data["records"][9]
    assert reuse["z_latlon"] == data["records"][6]["z_latlon"]
    assert reuse["private_read"] and reuse["geo_i_cost_units"] == 1
    assert reuse["geo_i_spent_units"] == 6
    assert reuse["q_reserved_upper_exact"] == "4/5"


def test_main_page_limit_rejects_a_seventh_page():
    check_main_page_limit(6)
    with pytest.raises(AssertionError, match="limit is 6"):
        check_main_page_limit(7)
