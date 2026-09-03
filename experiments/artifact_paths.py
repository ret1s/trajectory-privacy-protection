"""Canonical repository-relative paths for generated research artifacts."""

from pathlib import Path


ARTIFACTS_DIR = Path("artifacts")
BENCHMARK_ARTIFACTS_DIR = ARTIFACTS_DIR / "benchmarks"
REPORT_ARTIFACTS_DIR = ARTIFACTS_DIR / "reports"

BENCHMARK_RESULTS_PATH = BENCHMARK_ARTIFACTS_DIR / "benchmark_results.json"
AVERAGING_RESULTS_PATH = BENCHMARK_ARTIFACTS_DIR / "averaging_multi_results.json"
DUMMY_BENCHMARK_RESULTS_PATH = (
    BENCHMARK_ARTIFACTS_DIR / "dummy_benchmark_results.json"
)
DUMMY_BENCHMARK_MAP_PATH = BENCHMARK_ARTIFACTS_DIR / "dummy_benchmark_map.html"
DUMMY_BENCHMARK_PREVIEW_PATH = (
    BENCHMARK_ARTIFACTS_DIR / "dummy_benchmark_preview.png"
)
