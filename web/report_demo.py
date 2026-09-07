"""Read-only replay with separate public and evaluator interfaces."""
import hashlib
import json
from pathlib import Path

from flask import Blueprint, abort, current_app, jsonify, render_template

bp = Blueprint("report_demo", __name__)
DEFAULT_PATH = Path(__file__).resolve().parents[1] / "artifacts/benchmarks/report_demo/results.json"


def payload():
    path = Path(current_app.config.get("REPORT_DEMO_RESULTS_PATH", DEFAULT_PATH))
    try:
        raw = path.read_bytes()
        expected = path.with_suffix(".sha256").read_text().strip()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError("Artifact checksum mismatch")
        data = json.loads(raw)
        if data.get("schema") != "report-demo-v1":
            raise ValueError("Unsupported report artifact schema")
        return data
    except (OSError, ValueError) as exc:
        abort(503, description=str(exc))


@bp.get("/report-demo")
def page():
    return render_template("report_demo.html")


@bp.get("/api/report-demo")
def overview():
    data = payload()
    return jsonify({"catalogue": data["scenario_catalogue"], "limitations": data["limitations"],
                    "can_evaluate": bool(current_app.config["BENCHMARK_ENABLE_EVALUATOR_VIEW"]),
                    "runs": [{key: r[key] for key in ("id", "method", "scenario", "k", "status")}
                             for r in data["rows"]]})


@bp.get("/api/report-demo/roads")
def roads():
    return jsonify(payload()["roads"])


@bp.get("/api/report-demo/runs/<run_id>")
def public_run(run_id):
    row = next((r for r in payload()["rows"] if r["id"] == run_id), None)
    if row is None:
        abort(404)
    response = jsonify({key: row[key] for key in ("id", "method", "scenario", "k", "status", "public") if key in row})
    response.headers["X-Benchmark-Visibility"] = "public-transcript"
    return response


@bp.get("/api/report-demo/evaluation")
def evaluation():
    if not current_app.config["BENCHMARK_ENABLE_EVALUATOR_VIEW"]:
        abort(404)
    data = payload()
    response = jsonify({key: data[key] for key in ("summary", "rows", "causal_checks", "manifests")})
    response.headers["X-Benchmark-Visibility"] = "evaluator-only"
    return response
