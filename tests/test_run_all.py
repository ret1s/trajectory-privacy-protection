from types import SimpleNamespace

from tests import run_all


def test_main_delegates_to_pytest_and_preserves_failure_exit_code(monkeypatch):
    calls = []
    fake = SimpleNamespace(main=lambda args: calls.append(args) or 1)
    monkeypatch.setattr(run_all.importlib, "import_module", lambda name: fake)
    assert run_all.main(["-q", "tests/test_native_comparator_metrics.py"]) == 1
    assert calls == [["-q", "tests/test_native_comparator_metrics.py"]]


def test_fallback_reports_fixture_and_import_errors_as_failures(monkeypatch, capsys):
    def ordinary():
        pass
    def requires_fixture(tmp_path):
        pass
    modules = [SimpleNamespace(name="test_plain"), SimpleNamespace(name="test_missing")]
    monkeypatch.setattr(run_all.pkgutil, "iter_modules", lambda paths: modules)
    def import_module(name):
        if name.endswith("test_missing"):
            raise ImportError("dependency missing")
        return SimpleNamespace(test_ok=ordinary, test_fixture=requires_fixture)
    monkeypatch.setattr(run_all.importlib, "import_module", import_module)
    assert run_all._run_without_pytest() == 1
    output = capsys.readouterr().out
    assert "1 passed, 2 failed" in output
    assert "fixture/parametrization support" in output
    assert "dependency missing" in output


def test_fallback_is_used_only_when_pytest_itself_is_absent(monkeypatch):
    def missing(name):
        raise ModuleNotFoundError("pytest missing", name="pytest")
    monkeypatch.setattr(run_all.importlib, "import_module", missing)
    monkeypatch.setattr(run_all, "_run_without_pytest", lambda: 1)
    assert run_all.main([]) == 1
    assert run_all.main(["-q"]) == 2
