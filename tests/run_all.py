"""Canonical, dependency-light test runner (verifier R3-014).

Runs every `test_*` function in every `tests/test_*.py` module with no pytest
required, so there is ONE green command that works in the project venv:

    python -m tests.run_all

(When pytest is installed, `python -m pytest -q tests` runs the same functions;
see pyproject.toml / requirements-dev.txt.)
"""
import importlib
import pkgutil
import traceback

import tests


def main():
    passed = failed = 0
    failures = []
    for mod in pkgutil.iter_modules(tests.__path__):
        if not mod.name.startswith("test_"):
            continue
        m = importlib.import_module(f"tests.{mod.name}")
        fns = [v for k, v in sorted(vars(m).items())
               if k.startswith("test_") and callable(v)]
        for fn in fns:
            try:
                fn()
                passed += 1
            except Exception:  # noqa: BLE001 - report and continue
                failed += 1
                failures.append((mod.name, fn.__name__, traceback.format_exc()))
    print(f"\n{'='*60}\n{passed} passed, {failed} failed")
    for modname, fnname, tb in failures:
        print(f"\nFAILED {modname}.{fnname}\n{tb}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
