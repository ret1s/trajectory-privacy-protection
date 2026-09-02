"""Regression tests for copy-pasteable experiment provenance."""

import sys
from pathlib import Path

from experiments.provenance import _replay_command


def test_replay_command_converts_project_script_to_module_name():
    original = list(sys.argv)
    try:
        root = Path(__file__).resolve().parents[1]
        sys.argv[:] = [
            str(root / "experiments" / "run_dummy_benchmark.py"),
            "--quick",
        ]
        command = _replay_command()
    finally:
        sys.argv[:] = original

    assert command == [
        sys.executable,
        "-m",
        "experiments.run_dummy_benchmark",
        "--quick",
    ]


def test_replay_command_uses_direct_execution_for_external_script():
    original = list(sys.argv)
    try:
        sys.argv[:] = ["/private/tmp/external-runner.py", "--seed", "7"]
        command = _replay_command()
    finally:
        sys.argv[:] = original

    assert command == [
        sys.executable,
        "/private/tmp/external-runner.py",
        "--seed",
        "7",
    ]
