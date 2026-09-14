from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest import mock

import pytest


MODULE = Path(__file__).parents[1] / "python" / "eqiora" / "colab.py"
SPEC = importlib.util.spec_from_file_location("_eqiora_colab_under_test", MODULE)
assert SPEC is not None and SPEC.loader is not None
colab = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = colab
SPEC.loader.exec_module(colab)


def identities(*, eqiora_version: str = "0.1.1") -> dict[str, object]:
    return {
        "eqiora": colab._Identity(eqiora_version, "/site/eqiora/__init__.py"),
        "numpy": colab._Identity("2.3.3", "/site/numpy/__init__.py"),
        "gmsh": colab._Identity("4.15.2", "/site/gmsh.py"),
        "anywidget": colab._Identity("0.11.0", "/site/anywidget/__init__.py"),
        "ipywidgets": colab._Identity("8.1.7", "/site/ipywidgets/__init__.py"),
        "traitlets": colab._Identity("5.14.3", "/site/traitlets/__init__.py"),
    }


def test_public_surface_is_only_prepare() -> None:
    assert colab.__all__ == ["prepare"]


def test_prepare_rejects_pypy_even_with_a_supported_language_version() -> None:
    implementation = SimpleNamespace(name="pypy")
    with (
        mock.patch.object(colab.sys, "implementation", implementation),
        pytest.raises(RuntimeError, match="supports CPython.*pypy"),
    ):
        colab.prepare()


def test_prepare_enables_colab_viewer_and_provides_missing_glu(capsys) -> None:
    installed = identities()
    loaded = {"eqiora": installed["eqiora"]}
    with (
        mock.patch.object(colab, "_is_colab", return_value=True),
        mock.patch.object(colab, "_installed_identities", return_value=installed),
        mock.patch.object(colab, "_loaded_identities", return_value=loaded),
        mock.patch.object(colab, "find_library", return_value=None),
        mock.patch.object(colab.subprocess, "run") as run,
        mock.patch.object(colab, "_enable_custom_widget_manager") as enable,
    ):
        assert colab.prepare() is None

    assert [call.args[0] for call in run.call_args_list] == [
        ["apt-get", "update", "-qq"],
        ["apt-get", "install", "-y", "-qq", "libglu1-mesa"],
    ]
    assert all(call.kwargs == {"check": True} for call in run.call_args_list)
    enable.assert_called_once_with()
    diagnostic = capsys.readouterr().out
    assert "'colab': True" in diagnostic
    assert "'version': '0.1.1'" in diagnostic
    assert "'anywidget': {'installed': {'version': '0.11.0'" in diagnostic
    assert "'gmsh': {'installed': {'version': '4.15.2'" in diagnostic


def test_prepare_outside_colab_only_validates_and_reports(capsys) -> None:
    installed = identities()
    with (
        mock.patch.object(colab, "_is_colab", return_value=False),
        mock.patch.object(colab, "_installed_identities", return_value=installed),
        mock.patch.object(
            colab, "_loaded_identities", return_value={"eqiora": installed["eqiora"]}
        ),
        mock.patch.object(colab, "_provide_glu") as provide_glu,
        mock.patch.object(colab, "_enable_custom_widget_manager") as enable,
    ):
        colab.prepare()

    provide_glu.assert_not_called()
    enable.assert_not_called()
    assert "'colab': False" in capsys.readouterr().out


def test_prepare_restarts_colab_when_loaded_eqiora_is_not_installed_eqiora(
    capsys,
) -> None:
    installed = identities()
    loaded = {
        "eqiora": colab._Identity("0.1.0", "/old/eqiora/__init__.py"),
    }
    with (
        mock.patch.object(colab, "_is_colab", return_value=True),
        mock.patch.object(colab, "_installed_identities", return_value=installed),
        mock.patch.object(colab, "_loaded_identities", return_value=loaded),
        mock.patch.object(colab, "_restart_runtime", side_effect=SystemExit) as restart,
        mock.patch.object(colab, "_provide_glu") as provide_glu,
        mock.patch.object(colab, "_enable_custom_widget_manager") as enable,
        pytest.raises(SystemExit),
    ):
        colab.prepare()

    restart.assert_called_once_with()
    provide_glu.assert_not_called()
    enable.assert_not_called()
    diagnostic = capsys.readouterr().out
    assert "'/site/eqiora/__init__.py'" in diagnostic
    assert "'/old/eqiora/__init__.py'" in diagnostic


def test_prepare_requires_manual_restart_outside_colab_on_identity_conflict() -> None:
    installed = identities()
    loaded = {
        "eqiora": colab._Identity("0.1.1", "/old/eqiora/__init__.py"),
        "anywidget": installed["anywidget"],
    }
    with (
        mock.patch.object(colab, "_is_colab", return_value=False),
        mock.patch.object(colab, "_installed_identities", return_value=installed),
        mock.patch.object(colab, "_loaded_identities", return_value=loaded),
        pytest.raises(RuntimeError, match=r"loaded modules \(eqiora\); restart Python"),
    ):
        colab.prepare()


def test_loaded_viewer_dependency_is_checked_against_its_installed_owner() -> None:
    installed = identities()
    loaded = {
        "eqiora": installed["eqiora"],
        "anywidget": colab._Identity("0.11.0", "/preinstall/anywidget/__init__.py"),
    }
    with (
        mock.patch.object(colab, "_is_colab", return_value=False),
        mock.patch.object(colab, "_installed_identities", return_value=installed),
        mock.patch.object(colab, "_loaded_identities", return_value=loaded),
        pytest.raises(RuntimeError, match=r"loaded modules \(anywidget\)"),
    ):
        colab.prepare()


def test_prepare_rejects_unsupported_anywidget() -> None:
    installed = identities()
    installed["anywidget"] = colab._Identity("0.10.0", "/site/anywidget/__init__.py")
    with (
        mock.patch.object(colab, "_is_colab", return_value=True),
        mock.patch.object(colab, "_installed_identities", return_value=installed),
        pytest.raises(RuntimeError, match="requires anywidget==0.11.0"),
    ):
        colab.prepare()


def test_prepare_rejects_unsupported_gmsh() -> None:
    installed = identities()
    installed["gmsh"] = colab._Identity("4.14.0", "/site/gmsh.py")
    with (
        mock.patch.object(colab, "_is_colab", return_value=True),
        mock.patch.object(colab, "_installed_identities", return_value=installed),
        pytest.raises(RuntimeError, match="requires gmsh==4.15.2"),
    ):
        colab.prepare()


def test_installed_identity_uses_each_modules_owning_distribution() -> None:
    packages = {
        name: SimpleNamespace(
            version={
                "eqiora": "0.1.1",
                "numpy": "2.3.3",
                "gmsh": "4.15.2",
                "anywidget": "0.11.0",
                "ipywidgets": "8.1.7",
                "traitlets": "5.14.3",
            }[name],
            locate_file=lambda relative: Path("/site") / relative,
        )
        for name in (
            "eqiora",
            "numpy",
            "gmsh",
            "anywidget",
            "ipywidgets",
            "traitlets",
        )
    }
    with mock.patch.object(
        colab, "_distribution", side_effect=packages.__getitem__
    ) as get:
        observed = colab._installed_identities()

    assert observed == identities()
    assert {call.args[0] for call in get.call_args_list} == set(packages)


def test_loaded_identity_reports_an_absent_module_version_honestly() -> None:
    module = ModuleType("traitlets")
    module.__file__ = "/site/traitlets/__init__.py"
    with mock.patch.dict(sys.modules, {"traitlets": module}):
        observed = colab._loaded_identities()

    assert observed["traitlets"] == colab._Identity(None, "/site/traitlets/__init__.py")
    assert (
        colab._identity_conflicts({"traitlets": observed["traitlets"]}, identities())
        == ()
    )


def test_custom_widget_manager_uses_the_colab_api() -> None:
    output = mock.Mock()
    with mock.patch.object(colab, "import_module", return_value=output) as load:
        colab._enable_custom_widget_manager()

    load.assert_called_once_with("google.colab.output")
    output.enable_custom_widget_manager.assert_called_once_with()


def test_restart_terminates_the_current_process() -> None:
    with (
        mock.patch.object(colab.os, "getpid", return_value=42),
        mock.patch.object(colab.os, "kill") as kill,
    ):
        colab._restart_runtime()

    kill.assert_called_once_with(42, colab.signal.SIGKILL)
