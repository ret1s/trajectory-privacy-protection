"""Regression checks for repository ownership and artifact boundaries."""

import hashlib
import json
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ACTIVE_PYTHON_DIRS = (
    "benchmark",
    "core",
    "data",
    "evaluation",
    "experiments",
    "web",
)


def test_core_protocol_import_does_not_load_archived_gis_stack():
    code = (
        "import sys; "
        "from core.demo_protocol import TrajectoryPoint; "
        "blocked={'folium','geopandas','sklearn'} & set(sys.modules); "
        "assert not blocked, blocked"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_dashboard_import_does_not_load_experiment_or_sumo_runner():
    code = (
        "import sys; "
        "from web.benchmark_app import create_app; "
        "blocked={'experiments.run_dummy_benchmark','data.sumo_demo'} & set(sys.modules); "
        "assert not blocked, blocked"
    )
    subprocess.run(
        [sys.executable, "-c", code],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )


def test_active_python_does_not_import_archive():
    offenders = []
    for directory in ACTIVE_PYTHON_DIRS:
        for path in (PROJECT_ROOT / directory).rglob("*.py"):
            source = path.read_text(encoding="utf-8")
            if "from archive" in source or "import archive" in source:
                offenders.append(str(path.relative_to(PROJECT_ROOT)))
    assert not offenders, f"active modules import historical code: {offenders}"


def test_thesis_has_canonical_source_and_release_paths():
    assert (PROJECT_ROOT / "thesis" / "main.tex").is_file()
    assert (
        PROJECT_ROOT / "artifacts" / "reports" / "graduation_thesis.pdf"
    ).is_file()
    source = (PROJECT_ROOT / "thesis" / "main.tex").read_text(encoding="utf-8")
    assert "docs/supervisor_meeting" not in source
    gitignore = (PROJECT_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert "thesis/main.pdf" in gitignore
    build_help = (PROJECT_ROOT / "thesis" / "README.md").read_text(encoding="utf-8")
    assert "-outdir=../build/thesis" in build_help


def test_artifacts_has_one_canonical_hierarchy_and_valid_visual_manifest():
    assert not (PROJECT_ROOT / "output").exists()
    assert not (PROJECT_ROOT / "outputs").exists()

    artifact_root = PROJECT_ROOT / "artifacts"
    benchmark_dir = artifact_root / "benchmarks"
    report_dir = artifact_root / "reports"
    assert (artifact_root / "README.md").is_file()

    benchmark_names = {
        path.name
        for path in benchmark_dir.iterdir()
        if path.is_file()
    }
    required_benchmarks = {
        "README.md",
        "averaging_multi_results.json",
        "benchmark_results.json",
        "dummy_benchmark_map.html",
        "dummy_benchmark_preview.png",
        "dummy_benchmark_results.json",
    }
    assert required_benchmarks <= benchmark_names

    required_reports = {
        "README.md",
        "graduation_thesis.pdf",
        "location_trajectory_privacy_foundations.pdf",
        "research_improvements_report.pdf",
    }
    report_names = {path.name for path in report_dir.iterdir() if path.is_file()}
    assert required_reports <= report_names

    retired = {
        name
        for name in benchmark_names
        if name.startswith("trajectory_privacy_map") or name.startswith("sota_demo_")
    }
    assert not retired, f"retired artifacts belong under archive/: {sorted(retired)}"

    manifest_path = benchmark_dir / "dummy_benchmark_results.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for key in ("interactive_map", "static_preview"):
        entry = manifest["visual_artifacts"][key]
        artifact = PROJECT_ROOT / entry["path"]
        assert artifact.is_file()
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == entry["sha256"]
