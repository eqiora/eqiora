"""Exercise source cells through a real IPython shell and native compiler."""

from pathlib import Path

import pytest
from IPython.core.error import UsageError
from IPython.core.completer import provisionalcompleter
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


def completions(shell, text, cursor=None):
    with provisionalcompleter():
        return list(shell.Completer.completions(text, len(text) if cursor is None else cursor))


def test_source_completion_uses_current_scope_without_executing(shell, monkeypatch):
    previous = object()
    shell.user_ns["model"] = previous
    shell.user_ns["speed_python_only"] = object()

    def forbidden_compile(**_kwargs):
        raise AssertionError("completion executed the cell")

    monkeypatch.setattr(eqiora, "compile", forbidden_compile)
    body = "model Cell(){ parameter speed:1=2; variable x:1; relation law{x=spe"
    text = "%%eqiora model --entry Cell\n" + body
    items = completions(shell, text)
    assert [item.text for item in items] == ["speed"]
    assert all((item.start, item.end) == (len(text) - 3, len(text)) for item in items)
    assert shell.user_ns["model"] is previous
    renamed = text.replace("speed:1", "specific:1")
    assert [item.text for item in completions(shell, renamed)] == ["specific"]
    assert "speed" not in [item.text for item in completions(shell, renamed)]


def test_completion_keeps_unicode_offsets_and_recovers_incomplete_source(shell):
    text = "%%eqiora model\r\n// 日本語 🦀\r\nmo"
    items = completions(shell, text)
    model = next(item for item in items if item.text == "model")
    assert (model.start, model.end) == (len(text) - 2, len(text))


def test_completion_does_not_cross_model_scopes_or_replace_token_suffixes(shell):
    from eqiora._eqiora import _complete_source_cell

    body = "model A(){parameter secret:1=2;} model B(){variable x:1; relation law{x=sec"
    result = _complete_source_cell(body, len(body), 50)
    assert result is not None and "secret" not in result[1]
    body = "model A(){parameter speed:1=2;variable x:1;relation law{x=speed;}}"
    cursor = body.rindex("speed") + 3
    assert _complete_source_cell(body, cursor, 50) is None
    comment = "model A(){ // spe"
    assert _complete_source_cell(comment, len(comment), 50) is None


def test_plain_python_and_extension_lifecycle_keep_their_completion_owner(shell):
    import eqiora.jupyter as extension

    shell.user_ns["notebook_python_value"] = 7
    assert "notebook_python_value" in [item.text for item in completions(shell, "notebook_python_v")]
    assert shell.Completer.custom_matchers.count(extension._complete_eqiora) == 1
    shell.run_line_magic("reload_ext", "eqiora.jupyter")
    assert shell.Completer.custom_matchers.count(extension._complete_eqiora) == 1
    assert "model" in [item.text for item in completions(shell, "%%eqiora model\nmo")]
    shell.run_line_magic("unload_ext", "eqiora.jupyter")
    assert extension._complete_eqiora not in shell.Completer.custom_matchers
    assert "notebook_python_value" in [item.text for item in completions(shell, "notebook_python_v")]


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
