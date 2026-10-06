"""Canonical test entry point: pytest when available, simple-call fallback.

``python -m tests.run_all`` invokes pytest for the complete suite, including
fixtures and parametrization. Additional arguments use pytest's normal CLI.

Without pytest, the fallback can run only zero-argument ordinary functions.
Import errors and tests needing fixtures/marks are failures, with an instruction
to install requirements-dev.txt; they are never skipped to manufacture success.
The fallback is useful for old dependency-light suites, not a replacement for
the current full pytest suite.
"""
import importlib
import inspect
import pkgutil
from pathlib import Path
import sys
import traceback

import tests


def _run_without_pytest():
    passed = failed = 0
    failures = []
    for mod in pkgutil.iter_modules(tests.__path__):
        if not mod.name.startswith("test_"):
            continue
        try:
            m = importlib.import_module(f"tests.{mod.name}")
        except Exception:  # noqa: BLE001 - collect module failures too
            failed += 1
            failures.append((mod.name, "module import", traceback.format_exc()))
            continue
        fns = [v for k, v in sorted(vars(m).items())
               if k.startswith("test_") and callable(v)]
        for fn in fns:
            try:
                if inspect.signature(fn).parameters or getattr(fn, "pytestmark", None):
                    raise RuntimeError(
                        "This test needs pytest fixture/parametrization support. "
                        "Install requirements-dev.txt and rerun python -m tests.run_all."
                    )
                fn()
                passed += 1
            except Exception:  # noqa: BLE001 - report and continue
                failed += 1
                failures.append((mod.name, fn.__name__, traceback.format_exc()))
    print(f"\n{'='*60}\n{passed} passed, {failed} failed")
    for modname, fnname, tb in failures:
        print(f"\nFAILED {modname}.{fnname}\n{tb}")
    return 1 if failed else 0


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        pytest = importlib.import_module("pytest")
    except ModuleNotFoundError as exc:
        if exc.name != "pytest":
            raise
        print("pytest is unavailable; using the ordinary-function fallback.")
        if args:
            print("CLI selection needs pytest. Install requirements-dev.txt.")
            return 2
        return _run_without_pytest()
    return int(pytest.main(args or ["-q", str(Path(__file__).resolve().parent)]))


if __name__ == "__main__":
    raise SystemExit(main())
