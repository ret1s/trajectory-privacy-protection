"""Shared constants for the dummy-generation benchmark."""

from __future__ import annotations

import os


DEFAULT_OUTPUT = os.path.join("outputs", "dummy_benchmark_results.json")
DEFAULT_MAP_OUTPUT = os.path.join("outputs", "dummy_benchmark_map.html")
DEFAULT_PREVIEW_OUTPUT = os.path.join("outputs", "dummy_benchmark_preview.png")

BENCHMARK_SCHEMA = "msc-dummy-benchmark-v4"
# Compatibility name retained for scripts/tests that imported the former
# constant. New code should use BENCHMARK_SCHEMA.
DEMO_SCHEMA = BENCHMARK_SCHEMA

SUMO_GRAPH_COMPATIBILITY_NOTE = (
    "SUMO mobility and protection candidates use the same passenger-only "
    ".net.xml converted from the Beijing OSM extract. FCD samples are continuous "
    "lane positions while mechanism candidates are vertices/polylines from that "
    "same network; no separately built multimodal graph is mixed into the run."
)
BASE_DISCLAIMER = (
    "DUMMY-GENERATION BENCHMARK HARNESS. Paper comparators are executable "
    "source-mapped clean-room adaptations, not official or paper-equivalent "
    "reproductions. Their numbers are local benchmark results, not reproduced "
    "SOTA results. "
    "Metrics stay within each output contract and must not form a cross-track "
    "leaderboard."
)
DISCLAIMER = BASE_DISCLAIMER + " " + SUMO_GRAPH_COMPATIBILITY_NOTE

MAJOR_HIGHWAYS = {
    "motorway",
    "motorway_link",
    "trunk",
    "trunk_link",
    "primary",
    "primary_link",
    "secondary",
    "secondary_link",
}

MAP_LABELS = {
    "transprotect_adaptation": "TransProtect adaptation — pseudolocation trajectory",
    "anotherme_adaptation": "AnotherMe adaptation — replacement trajectory",
    "semantic_correlation_local_adaptation": (
        "Semantic-correlation clean-room adaptation — candidate sets"
    ),
    "geo_i_anchored_dummy": "Proposed model — dummy-only trajectories",
}
