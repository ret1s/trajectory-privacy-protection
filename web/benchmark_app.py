"""Read-only Flask dashboard for the trajectory-privacy benchmark artifacts.

The application intentionally does not execute experiments.  Benchmark runs stay
in the reproducible CLI pipeline; this process only presents artifacts that have
already been generated under ``outputs/``.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any

from flask import Flask, abort, jsonify, render_template, send_file

from benchmark.registry import METHOD_CARDS
from experiments.dummy_benchmark.constants import BENCHMARK_SCHEMA


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULTS_PATH = PROJECT_ROOT / "outputs" / "dummy_benchmark_results.json"
EXPECTED_SCHEMA = BENCHMARK_SCHEMA
CURRENT_METHOD_IDS = tuple(card.method_id for card in METHOD_CARDS)


def _humanize(value: str) -> str:
    return value.replace("_", " ").strip().title()


def _point_count_label(value: Any) -> str | None:
    """Format the runner's per-record point-count list without JS coercion."""

    raw_values = value if isinstance(value, (list, tuple)) else (value,)
    counts: list[int] = []
    for raw in raw_values:
        if isinstance(raw, bool):
            return None
        try:
            count = int(raw)
        except (TypeError, ValueError, OverflowError):
            return None
        if count < 0 or count != raw:
            return None
        counts.append(count)
    if not counts:
        return None
    if len(counts) == 1:
        return f"{counts[0]} samples / trajectory"
    if len(set(counts)) == 1:
        return f"{counts[0]} samples × {len(counts)} trajectories"
    values = ", ".join(str(count) for count in counts)
    return f"{values} samples / trajectory ({len(counts)} trajectories)"


def _method_ids(items: Any, field: str) -> set[str]:
    """Extract method IDs from a v4 artifact section or reject malformed rows."""

    if not isinstance(items, list):
        raise ValueError(f"The benchmark artifact field {field!r} must be a list")
    result: set[str] = set()
    for index, item in enumerate(items):
        if not isinstance(item, dict):
            raise ValueError(f"{field}[{index}] must be an object")
        method_id = item.get(
            "method_id" if field == "method_inventory" else "mechanism"
        )
        if not isinstance(method_id, str) or not method_id.strip():
            raise ValueError(f"{field}[{index}] has no valid method ID")
        if field == "method_inventory" and method_id in result:
            raise ValueError(f"method_inventory contains duplicate ID {method_id!r}")
        result.add(method_id)
    return result


def _format_id_difference(expected: set[str], observed: set[str]) -> str:
    pieces = []
    missing = sorted(expected - observed)
    unexpected = sorted(observed - expected)
    if missing:
        pieces.append("missing: " + ", ".join(missing))
    if unexpected:
        pieces.append("unexpected: " + ", ".join(unexpected))
    return "; ".join(pieces) or "unknown difference"


class BenchmarkArtifacts:
    """Load and expose one benchmark artifact without mutating it."""

    def __init__(
        self,
        results_path: Path,
        project_root: Path,
        expected_schema: str,
        expected_method_ids: tuple[str, ...],
    ) -> None:
        self.results_path = results_path.resolve()
        self.project_root = project_root.resolve()
        self.expected_schema = expected_schema
        self.expected_method_ids = frozenset(expected_method_ids)
        if not self.expected_method_ids:
            raise ValueError("expected_method_ids must not be empty")

    def load(self) -> dict[str, Any]:
        if not self.results_path.is_file():
            raise FileNotFoundError(self.results_path)
        with self.results_path.open(encoding="utf-8") as stream:
            payload = json.load(stream)
        if not isinstance(payload, dict):
            raise ValueError("The benchmark artifact must be a JSON object")
        if payload.get("schema") != self.expected_schema:
            raise ValueError(
                f"Unsupported artifact schema {payload.get('schema')!r}; "
                f"expected {self.expected_schema!r}"
            )
        if not isinstance(payload.get("runs", []), list):
            raise ValueError("The benchmark artifact field 'runs' must be a list")
        if not isinstance(payload.get("summaries", []), list):
            raise ValueError("The benchmark artifact field 'summaries' must be a list")

        inventory_ids = _method_ids(payload.get("method_inventory"), "method_inventory")
        expected_ids = set(self.expected_method_ids)
        if inventory_ids != expected_ids:
            difference = _format_id_difference(expected_ids, inventory_ids)
            raise ValueError(
                "Artifact method inventory does not match the active benchmark "
                f"registry ({difference}); regenerate the benchmark artifacts"
            )
        for field in ("summaries", "runs"):
            observed = _method_ids(payload.get(field), field)
            if observed != inventory_ids:
                difference = _format_id_difference(inventory_ids, observed)
                raise ValueError(
                    f"Artifact {field} method IDs do not match method_inventory "
                    f"({difference})"
                )
        return payload

    def state(self) -> tuple[dict[str, Any] | None, str | None]:
        try:
            return self.load(), None
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            return None, str(exc)

    def artifact_state(
        self, kind: str, payload: dict[str, Any]
    ) -> tuple[Path | None, bool, str | None]:
        """Resolve a declared visual and verify its recorded SHA-256."""

        visual = payload.get("visual_artifacts")
        if not isinstance(visual, dict):
            return None, False, "visual_artifacts is absent"
        key = {"map": "interactive_map", "preview": "static_preview"}.get(kind)
        entry = visual.get(key) if key else None
        if not isinstance(entry, dict) or not isinstance(entry.get("path"), str):
            return None, False, f"{kind} is not declared"

        candidate = (self.project_root / entry["path"]).resolve()
        try:
            candidate.relative_to(self.project_root)
        except ValueError:
            return None, False, "artifact path escapes project root"
        if not candidate.is_file():
            return None, False, "declared artifact file is missing"

        expected = entry.get("sha256")
        if not isinstance(expected, str) or len(expected) != 64:
            return None, False, "artifact SHA-256 is absent or invalid"
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if digest != expected.lower():
            return None, False, "artifact SHA-256 does not match the manifest"
        return candidate, True, None

    def declared_artifact(self, kind: str, payload: dict[str, Any]) -> Path | None:
        path, verified, _error = self.artifact_state(kind, payload)
        return path if verified else None


def _find_runs(payload: dict[str, Any], mechanism: str) -> list[dict[str, Any]]:
    return [
        run
        for run in payload.get("runs", [])
        if isinstance(run, dict) and run.get("mechanism") == mechanism
    ]


def _public_overview(payload: dict[str, Any], store: BenchmarkArtifacts) -> dict[str, Any]:
    provenance = payload.get("provenance")
    provenance = provenance if isinstance(provenance, dict) else {}
    summaries = [item for item in payload.get("summaries", []) if isinstance(item, dict)]
    inventory = {
        item.get("method_id"): item
        for item in payload.get("method_inventory", [])
        if isinstance(item, dict) and isinstance(item.get("method_id"), str)
    }

    mechanisms: list[dict[str, Any]] = []
    contracts: dict[str, int] = {}
    for item in summaries:
        mechanism = str(item.get("mechanism", "unknown"))
        output_kind = str(item.get("output_kind", "unspecified"))
        card = inventory.get(mechanism, {})
        contracts[output_kind] = contracts.get(output_kind, 0) + 1
        mechanisms.append(
            {
                "id": mechanism,
                "label": card.get("display_name") or _humanize(mechanism),
                "output_kind": output_kind,
                "source_method": item.get("source_method"),
                "records": item.get("records", 0),
                "implementation_level": card.get("implementation_level")
                or item.get("implementation_level"),
                "reportable_as_reproduced_sota": bool(
                    card.get(
                        "reportable_as_reproduced_sota",
                        item.get("reportable_as_reproduced_sota", False),
                    )
                ),
                "missing_components": card.get("missing_components", []),
                "source": card.get("source", {}),
                "adaptation_summary": card.get("adaptation_summary"),
            }
        )

    artifact_meta = payload.get("visual_artifacts")
    artifact_meta = artifact_meta if isinstance(artifact_meta, dict) else {}
    map_path, map_verified, map_error = store.artifact_state("map", payload)
    preview_path, preview_verified, preview_error = store.artifact_state(
        "preview", payload
    )
    return {
        "available": True,
        "schema": payload.get("schema"),
        "status": payload.get("status"),
        "disclaimer": payload.get("disclaimer"),
        "provenance": {
            "source_commit": provenance.get("source_commit"),
            "source_dirty_before_run": provenance.get("source_dirty_before_run"),
            "mode": provenance.get("mode"),
            "dataset": provenance.get("dataset"),
            "mobility_source": provenance.get("mobility_source"),
            "mobility_label": provenance.get("mobility_label"),
            "n_points_per_record": provenance.get("n_points_per_record"),
            "point_count_label": _point_count_label(
                provenance.get("n_points_per_record")
            ),
            "k": provenance.get("k"),
            "epsilons": provenance.get("epsilons"),
            "started_at_utc": provenance.get("started_at_utc"),
            "finished_at_utc": provenance.get("finished_at_utc"),
        },
        "mechanisms": mechanisms,
        "contracts": [
            {"id": name, "label": _humanize(name), "mechanisms": count}
            for name, count in sorted(contracts.items())
        ],
        "record_count": len(
            {
                run.get("record_id")
                for run in payload.get("runs", [])
                if isinstance(run, dict) and run.get("record_id") is not None
            }
        ),
        "artifacts": {
            "evaluator_only": bool(artifact_meta.get("evaluator_only", True)),
            "contains_ground_truth": bool(artifact_meta.get("contains_ground_truth", True)),
            "map": {
                "available": map_path is not None,
                "integrity_verified": map_verified,
                "error": map_error,
                "url": "/artifacts/evaluator/map",
            },
            "preview": {
                "available": preview_path is not None,
                "integrity_verified": preview_verified,
                "error": preview_error,
                "url": "/artifacts/evaluator/preview",
            },
        },
    }


def create_app(config: dict[str, Any] | None = None) -> Flask:
    """Create a configured, read-only benchmark dashboard."""

    app = Flask(
        __name__,
        template_folder="templates",
        static_folder="static",
    )
    app.config.from_mapping(
        BENCHMARK_PROJECT_ROOT=str(PROJECT_ROOT),
        BENCHMARK_RESULTS_PATH=str(DEFAULT_RESULTS_PATH),
        BENCHMARK_ENABLE_EVALUATOR_VIEW=False,
        BENCHMARK_EXPECTED_SCHEMA=EXPECTED_SCHEMA,
        BENCHMARK_EXPECTED_METHOD_IDS=CURRENT_METHOD_IDS,
        JSON_SORT_KEYS=False,
    )
    if config:
        app.config.update(config)

    store = BenchmarkArtifacts(
        Path(app.config["BENCHMARK_RESULTS_PATH"]),
        Path(app.config["BENCHMARK_PROJECT_ROOT"]),
        str(app.config["BENCHMARK_EXPECTED_SCHEMA"]),
        tuple(app.config["BENCHMARK_EXPECTED_METHOD_IDS"]),
    )
    app.extensions["benchmark_artifacts"] = store

    def require_payload() -> dict[str, Any]:
        payload, error = store.state()
        if payload is None:
            abort(503, description=error or "Benchmark artifact unavailable")
        return payload

    @app.after_request
    def set_security_headers(response):  # type: ignore[no-untyped-def]
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Cache-Control"] = "no-store"
        return response

    @app.get("/")
    def dashboard():
        payload, error = store.state()
        initial = _public_overview(payload, store) if payload is not None else {
            "available": False,
            "error": error,
        }
        return render_template("benchmark/index.html", initial_state=initial)

    @app.get("/api/benchmark")
    def benchmark_overview():
        payload, error = store.state()
        if payload is None:
            return jsonify({"available": False, "error": error}), 503
        return jsonify(_public_overview(payload, store))

    @app.get("/api/mechanisms/<mechanism>/attacker-view")
    def attacker_view(mechanism: str):
        runs = _find_runs(require_payload(), mechanism)
        if not runs:
            abort(404, description="Unknown benchmark mechanism")
        response = jsonify(
            {
                "visibility": "attacker_visible",
                "mechanism": mechanism,
                "runs": [
                    {
                        "record_id": run.get("record_id"),
                        "output_kind": run.get("output_kind"),
                        "source_method": run.get("source_method"),
                        "implementation": run.get("implementation", {}),
                        "attacker_view": run.get("attacker_view", {}),
                    }
                    for run in runs
                ],
            }
        )
        response.headers["X-Benchmark-Visibility"] = "attacker-visible"
        return response

    @app.get("/api/mechanisms/<mechanism>/evaluation")
    def evaluator_view(mechanism: str):
        if not app.config["BENCHMARK_ENABLE_EVALUATOR_VIEW"]:
            abort(404)
        runs = _find_runs(require_payload(), mechanism)
        if not runs:
            abort(404, description="Unknown benchmark mechanism")
        response = jsonify(
            {
                "visibility": "evaluator_only",
                "warning": "Contains ground truth. Never expose this payload to the attacker model.",
                "mechanism": mechanism,
                "runs": [
                    {
                        "record_id": run.get("record_id"),
                        "runtime_ms": run.get("runtime_ms"),
                        "runtime_breakdown_ms": run.get(
                            "runtime_breakdown_ms", {}
                        ),
                        "paper_metrics": run.get("paper_metrics", {}),
                        "metrics": run.get("metrics", {}),
                        "evaluator_truth": run.get("evaluator_truth", {}),
                    }
                    for run in runs
                ],
            }
        )
        response.headers["X-Benchmark-Visibility"] = "evaluator-only"
        return response

    @app.get("/artifacts/evaluator/<kind>")
    def evaluator_artifact(kind: str):
        if not app.config["BENCHMARK_ENABLE_EVALUATOR_VIEW"]:
            abort(404)
        if kind not in {"map", "preview"}:
            abort(404)
        path = store.declared_artifact(kind, require_payload())
        if path is None:
            abort(404, description="Requested visual artifact is unavailable")
        if kind == "map":
            response = send_file(path, mimetype="text/html")
        else:
            response = send_file(path, mimetype="image/png")
        response.headers["X-Benchmark-Visibility"] = "evaluator-only"
        response.headers["X-Artifact-Integrity"] = "sha256-verified"
        return response

    return app


if __name__ == "__main__":
    create_app({"BENCHMARK_ENABLE_EVALUATOR_VIEW": True}).run(
        host="127.0.0.1", port=5000, debug=False
    )
