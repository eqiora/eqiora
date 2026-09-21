"""Exercise source cells through a real IPython shell and native compiler."""

from pathlib import Path

import pytest
from IPython.core.error import UsageError
from IPython.core.interactiveshell import InteractiveShell

import eqiora


@pytest.fixture
def shell(tmp_path):
    instance = InteractiveShell(ipython_dir=str(tmp_path))
    instance.run_line_magic("load_ext", "eqiora.jupyter")
    yield instance
    instance.run_line_magic("unload_ext", "eqiora.jupyter")
    instance.history_manager.end_session()


@pytest.fixture
def source():
    return (Path(__file__).resolve().parents[3] / "examples/decay.eqi").read_text()


def test_compiled_cell_hands_ordinary_model_to_python(shell, source):
    shell.run_cell_magic("eqiora", "model", source)
    model = shell.user_ns["model"]
    assert isinstance(model, eqiora.Model)
    assert model.digest == eqiora.compile(source=source).digest
    assert shell.run_cell("model.digest == model.digest").result is True


def test_failure_preserves_previous_binding_and_native_diagnostics(shell, source):
    shell.run_cell_magic("eqiora", "model", source)
    previous = shell.user_ns["model"]
    invalid = "// 日本語\nmodel Broken() {\n parameter value: m = missing;\n}\n"
    with pytest.raises(eqiora.EqioraError) as captured:
        shell.run_cell_magic("eqiora", "model", invalid)
    assert shell.user_ns["model"] is previous
    error = captured.value
    assert error.diagnostics
    assert any(".eqi:4:" in note for note in error.__notes__)
    assert all(d.source_span[0].startswith("In[") for d in error.diagnostics if d.source_span)
    shell.run_cell_magic("eqiora", "model", source)
    assert shell.user_ns["model"] is not previous


@pytest.mark.parametrize("line", ["", "class", "a.b", "a b", "{target}"])
def test_invalid_binding_syntax_is_rejected_without_interpolation(shell, source, line):
    shell.user_ns["target"] = "model"
    with pytest.raises(UsageError):
        shell.run_cell_magic("eqiora", line, source)
    assert "model" not in shell.user_ns


def test_entry_and_named_python_bindings_reuse_compile(shell):
    source = "model Bound(parameter rate: 1) { variable result: 1; relation law { result = 2 * rate; } }"
    shell.user_ns["inputs"] = {"rate": 3.0}
    shell.run_cell_magic("eqiora", "model --entry Bound --bindings inputs", source)
    expected = eqiora.compile(source=source, entry="Bound", bindings={"rate": 3.0})
    assert shell.user_ns["model"].digest == expected.digest
    previous = shell.user_ns["model"]
    with pytest.raises(UsageError, match="not defined"):
        shell.run_cell_magic("eqiora", "model --bindings absent", source)
    assert shell.user_ns["model"] is previous


def test_unicode_python_names_and_reload(shell, source):
    shell.run_cell_magic("eqiora", "ｍodel", source)
    assert isinstance(shell.user_ns["model"], eqiora.Model)
    previous = shell.user_ns["model"]
    shell.run_line_magic("reload_ext", "eqiora.jupyter")
    assert shell.user_ns["model"] is previous
    shell.run_cell_magic("eqiora", "other", source)
    shell.run_line_magic("unload_ext", "eqiora.jupyter")
    assert shell.find_cell_magic("eqiora") is None
    assert shell.user_ns["model"] is previous


def test_installed_wheel_contains_discoverable_notebook_frontend():
    import importlib.metadata
    import json

    distribution = importlib.metadata.distribution("eqiora")
    suffix = "share/jupyter/labextensions/@eqiora/jupyter/package.json"
    manifests = [file for file in distribution.files or () if str(file).endswith(suffix)]
    assert len(manifests) == 1
    manifest_path = Path(distribution.locate_file(manifests[0]))
    manifest = json.loads(manifest_path.read_text())
    assert manifest["name"] == "@eqiora/jupyter"
    assert manifest["jupyterlab"]["extension"] is True
    load = manifest["jupyterlab"]["_build"]["load"]
    asset = manifest_path.parent / load
    assert asset.resolve().is_relative_to(manifest_path.parent.resolve())
    assert asset.is_file() and asset.stat().st_size > 0
    assert (manifest_path.parent / "static/third-party-licenses.json").is_file()
