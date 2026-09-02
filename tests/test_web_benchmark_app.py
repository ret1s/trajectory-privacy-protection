"""Contract and separation tests for the read-only benchmark dashboard."""

from __future__ import annotations

import json
import hashlib
import tempfile
from pathlib import Path

from web.benchmark_app import create_app


def _artifact(root: Path) -> Path:
    outputs = root / "outputs"
    outputs.mkdir()
    map_path = outputs / "map.html"
    preview_path = outputs / "preview.png"
    map_path.write_text("<html><body>map</body></html>", encoding="utf-8")
    preview_path.write_bytes(b"\x89PNG\r\n\x1a\n")
    payload = {
        "schema": "msc-dummy-benchmark-v4",
        "status": "READY",
        "disclaimer": "Test artifact",
        "method_inventory": [
            {
                "method_id": "method_a",
                "display_name": "Method A adaptation",
                "implementation_level": "paper_adaptation",
                "reportable_as_reproduced_sota": False,
                "missing_components": ["learned model"],
                "adaptation_summary": "Local source-mapped adaptation.",
                "source": {
                    "citation": "Method A",
                    "repository_revision": "abc123",
                },
            }
        ],
        "provenance": {
            "source_commit": "abc123",
            "source_dirty_before_run": False,
            "mobility_source": "SUMO",
            "n_points_per_record": 2,
        },
        "summaries": [
            {
                "mechanism": "method_a",
                "output_kind": "real_plus_dummies",
                "source_method": "Method A",
                "implementation_level": "paper_adaptation",
                "reportable_as_reproduced_sota": False,
                "records": 1,
                "metrics": {"request_multiplier": 4.0},
            }
        ],
        "runs": [
            {
                "record_id": "record-1",
                "mechanism": "method_a",
                "source_method": "Method A",
                "implementation": {"implementation_level": "paper_adaptation"},
                "output_kind": "real_plus_dummies",
                "runtime_ms": 12.5,
                "runtime_breakdown_ms": {
                    "setup_runtime_ms": 4.5,
                    "inference_runtime_ms": 8.0,
                    "end_to_end_runtime_ms": 12.5,
                },
                "paper_metrics": {
                    "paper_asr_percent": None,
                    "paper_asr_status": "not_available_without_calibrated_attack",
                },
                "metrics": {"privacy_score": 0.8},
                "attacker_view": {
                    "events": [
                        {
                            "event_id": "e0",
                            "timestamp_s": 0,
                            "candidates": [{"candidate_id": "c0", "lat": 1.0, "lon": 2.0}],
                        }
                    ]
                },
                "evaluator_truth": {
                    "real_trajectory": [{"timestamp_s": 0, "lat": 9.0, "lon": 8.0}],
                    "real_candidate_ids": ["c0"],
                },
            }
        ],
        "visual_artifacts": {
            "evaluator_only": True,
            "contains_ground_truth": True,
            "interactive_map": {
                "path": "outputs/map.html",
                "sha256": hashlib.sha256(map_path.read_bytes()).hexdigest(),
            },
            "static_preview": {
                "path": "outputs/preview.png",
                "sha256": hashlib.sha256(preview_path.read_bytes()).hexdigest(),
            },
        },
    }
    result = outputs / "results.json"
    result.write_text(json.dumps(payload), encoding="utf-8")
    return result


def _client(root: Path):
    result = _artifact(root)
    app = create_app(
        {
            "TESTING": True,
            "BENCHMARK_PROJECT_ROOT": str(root),
            "BENCHMARK_RESULTS_PATH": str(result),
            "BENCHMARK_EXPECTED_METHOD_IDS": ("method_a",),
            "BENCHMARK_ENABLE_EVALUATOR_VIEW": True,
        }
    )
    return app.test_client()


def test_dashboard_and_overview_load_generated_artifact():
    with tempfile.TemporaryDirectory() as directory:
        client = _client(Path(directory))
        page = client.get("/")
        assert page.status_code == 200
        page_text = page.get_data(as_text=True)
        assert "Bảo vệ tính riêng tư về quỹ đạo" in page_text
        assert 'id="paper-metric-grid"' in page_text
        assert 'id="runtime-grid"' in page_text

        response = client.get("/api/benchmark")
        assert response.status_code == 200
        payload = response.get_json()
        assert payload["mechanisms"][0]["id"] == "method_a"
        assert payload["mechanisms"][0]["label"] == "Method A adaptation"
        assert payload["mechanisms"][0]["implementation_level"] == "paper_adaptation"
        assert payload["mechanisms"][0]["reportable_as_reproduced_sota"] is False
        assert payload["mechanisms"][0]["missing_components"] == ["learned model"]
        assert payload["contracts"] == [
            {"id": "real_plus_dummies", "label": "Real Plus Dummies", "mechanisms": 1}
        ]
        assert payload["artifacts"]["map"]["available"] is True
        assert payload["artifacts"]["map"]["integrity_verified"] is True
        assert payload["provenance"]["point_count_label"] == "2 samples / trajectory"


def test_point_count_list_is_formatted_as_per_trajectory_data():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        result = _artifact(root)
        payload = json.loads(result.read_text(encoding="utf-8"))
        payload["provenance"]["n_points_per_record"] = [8, 10]
        result.write_text(json.dumps(payload), encoding="utf-8")
        app = create_app(
            {
                "TESTING": True,
                "BENCHMARK_PROJECT_ROOT": str(root),
                "BENCHMARK_RESULTS_PATH": str(result),
                "BENCHMARK_EXPECTED_METHOD_IDS": ("method_a",),
            }
        )

        overview = app.test_client().get("/api/benchmark").get_json()
        assert overview["provenance"]["n_points_per_record"] == [8, 10]
        assert overview["provenance"]["point_count_label"] == (
            "8, 10 samples / trajectory (2 trajectories)"
        )


def test_artifact_with_stale_registry_inventory_is_rejected():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        result = _artifact(root)
        # No expected-ID override: the app must compare this artifact with the
        # active benchmark registry, where the fixture's method_a cannot exist.
        app = create_app(
            {
                "TESTING": True,
                "BENCHMARK_PROJECT_ROOT": str(root),
                "BENCHMARK_RESULTS_PATH": str(result),
            }
        )

        response = app.test_client().get("/api/benchmark")
        assert response.status_code == 503
        error = response.get_json()["error"]
        assert "does not match the active benchmark registry" in error
        assert "regenerate the benchmark artifacts" in error


def test_artifact_run_ids_must_match_its_validated_inventory():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        result = _artifact(root)
        payload = json.loads(result.read_text(encoding="utf-8"))
        payload["runs"][0]["mechanism"] = "retired_method"
        result.write_text(json.dumps(payload), encoding="utf-8")
        app = create_app(
            {
                "TESTING": True,
                "BENCHMARK_PROJECT_ROOT": str(root),
                "BENCHMARK_RESULTS_PATH": str(result),
                "BENCHMARK_EXPECTED_METHOD_IDS": ("method_a",),
            }
        )

        response = app.test_client().get("/api/benchmark")
        assert response.status_code == 503
        assert "runs method IDs do not match method_inventory" in response.get_json()[
            "error"
        ]


def test_attacker_endpoint_never_contains_evaluator_truth():
    with tempfile.TemporaryDirectory() as directory:
        client = _client(Path(directory))
        response = client.get("/api/mechanisms/method_a/attacker-view")
        assert response.status_code == 200
        payload = response.get_json()
        serialized = json.dumps(payload)
        assert payload["visibility"] == "attacker_visible"
        assert "attacker_view" in serialized
        assert "evaluator_truth" not in serialized
        assert "paper_metrics" not in serialized
        assert "runtime_breakdown_ms" not in serialized
        assert "real_trajectory" not in serialized
        assert '"lat": 9.0' not in serialized


def test_evaluator_endpoint_is_explicit_and_contains_truth():
    with tempfile.TemporaryDirectory() as directory:
        client = _client(Path(directory))
        response = client.get("/api/mechanisms/method_a/evaluation")
        assert response.status_code == 200
        payload = response.get_json()
        assert payload["visibility"] == "evaluator_only"
        run = payload["runs"][0]
        assert run["evaluator_truth"]["real_candidate_ids"] == ["c0"]
        assert run["runtime_breakdown_ms"] == {
            "setup_runtime_ms": 4.5,
            "inference_runtime_ms": 8.0,
            "end_to_end_runtime_ms": 12.5,
        }
        assert run["paper_metrics"] == {
            "paper_asr_percent": None,
            "paper_asr_status": "not_available_without_calibrated_attack",
        }


def test_visual_routes_only_serve_declared_evaluator_artifacts():
    with tempfile.TemporaryDirectory() as directory:
        client = _client(Path(directory))
        map_response = client.get("/artifacts/evaluator/map")
        assert map_response.status_code == 200
        assert map_response.headers["X-Artifact-Integrity"] == "sha256-verified"
        assert client.get("/artifacts/evaluator/preview").status_code == 200
        assert client.get("/artifacts/evaluator/../../results.json").status_code == 404
        assert client.get("/artifacts/evaluator/arbitrary").status_code == 404


def test_visual_artifact_fails_closed_on_sha256_mismatch():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        client = _client(root)
        (root / "outputs" / "map.html").write_text("tampered", encoding="utf-8")
        overview = client.get("/api/benchmark").get_json()
        assert overview["artifacts"]["map"]["available"] is False
        assert overview["artifacts"]["map"]["integrity_verified"] is False
        assert "does not match" in overview["artifacts"]["map"]["error"]
        assert client.get("/artifacts/evaluator/map").status_code == 404


def test_evaluator_view_is_disabled_by_default_for_wsgi_deployment():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        result = _artifact(root)
        app = create_app(
            {
                "TESTING": True,
                "BENCHMARK_PROJECT_ROOT": str(root),
                "BENCHMARK_RESULTS_PATH": str(result),
                "BENCHMARK_EXPECTED_METHOD_IDS": ("method_a",),
            }
        )
        client = app.test_client()
        assert client.get("/api/mechanisms/method_a/attacker-view").status_code == 200
        assert client.get("/api/mechanisms/method_a/evaluation").status_code == 404
        assert client.get("/artifacts/evaluator/map").status_code == 404


def test_missing_artifact_degrades_without_server_error():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        app = create_app(
            {
                "TESTING": True,
                "BENCHMARK_PROJECT_ROOT": str(root),
                "BENCHMARK_RESULTS_PATH": str(root / "missing.json"),
            }
        )
        client = app.test_client()
        assert client.get("/").status_code == 200
        response = client.get("/api/benchmark")
        assert response.status_code == 503
        assert response.get_json()["available"] is False


def test_previous_benchmark_schema_is_rejected_instead_of_silently_displayed():
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        result = _artifact(root)
        payload = json.loads(result.read_text(encoding="utf-8"))
        payload["schema"] = "msc-dummy-benchmark-v3"
        result.write_text(json.dumps(payload), encoding="utf-8")
        app = create_app(
            {
                "TESTING": True,
                "BENCHMARK_PROJECT_ROOT": str(root),
                "BENCHMARK_RESULTS_PATH": str(result),
            }
        )
        response = app.test_client().get("/api/benchmark")
        assert response.status_code == 503
        assert "Unsupported artifact schema" in response.get_json()["error"]


def test_no_http_route_executes_a_benchmark():
    with tempfile.TemporaryDirectory() as directory:
        app = _client(Path(directory)).application
        methods = {rule.rule: sorted(rule.methods or []) for rule in app.url_map.iter_rules()}
        assert all("run" not in path and "execute" not in path for path in methods)
        assert all("POST" not in verbs for path, verbs in methods.items() if path != "/static/<path:filename>")
