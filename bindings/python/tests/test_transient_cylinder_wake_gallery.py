#!/usr/bin/env python3
"""Product checks for the unverified transient-cylinder gallery sources."""

from __future__ import annotations

import ast
import json
import types
import unittest
from collections import Counter
from pathlib import Path
from unittest import mock


ROOT = Path(__file__).resolve().parents[3]
PLAIN = ROOT / "examples/python/transient_cylinder_wake.py"
COLAB = ROOT / "examples/python/transient_cylinder_wake_colab.ipynb"
PAGE = ROOT / "docs/site/src/content/docs/gallery/transient-cylinder-startup.mdx"
GALLERY_INDEX = ROOT / "docs/site/src/content/docs/gallery/index.mdx"
SITE_STYLES = ROOT / "docs/site/src/styles/site/components.css"
PRODUCER = ROOT / "tools/site/produce_transient_cylinder_startup_media.py"
MEDIA = {
    "poster": ROOT / "docs/site/src/assets/gallery/transient-cylinder-startup-poster.png",
    "reduced": ROOT
    / "docs/site/src/assets/gallery/transient-cylinder-startup-reduced-motion.png",
    "webm": ROOT / "docs/site/src/assets/gallery/transient-cylinder-startup.webm",
    "mp4": ROOT / "docs/site/src/assets/gallery/transient-cylinder-startup.mp4",
}

PRESENTATION_CALLS = {
    "eqiora.geometry.GeometryGraph": 1,
    "geometry_graph.rectangle": 1,
    "geometry_graph.circle": 1,
    "geometry_graph.subtract": 1,
    "geometry_graph.build": 1,
    "eqiora.meshing.GmshMesher": 1,
    "eqiora.meshing.resolve": 1,
    "eqiora.meshing.generate": 1,
    "eqiora.compile": 2,
    "eqiora.solve.Linear": 1,
    "eqiora.fem.MiniP1": 2,
    "eqiora.resolve": 2,
    "eqiora.run": 2,
    "eqiora.time.BackwardEuler": 1,
    "eqiora.solve.Newton": 1,
    "eqiora.fluid.IncompressibleScaling": 1,
    "steady_result.output": 2,
    "eqiora.State.initial": 1,
    "eqiora.InitialField": 2,
    "result.trajectory.state": 1,
    "accepted.curl": 1,
    # Three source call sites bind the volume and each boundary's exact parent;
    # the fourth selects the cylinder for the existing force observation.
    "geometry.selection": 4,
    "accepted.boundary_force": 1,
    "accepted.sample": 2,
    "vorticity.values": 1,
    "eqplot.plot_scalar_field": 1,
}


def notebook_source(notebook: dict[str, object]) -> str:
    cells = notebook["cells"]
    assert isinstance(cells, list)
    return "\n\n".join(
        "".join(cell["source"])
        for cell in cells
        if isinstance(cell, dict) and cell.get("cell_type") == "code"
    )


def qualified_name(node: ast.expr) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        owner = qualified_name(node.value)
        return f"{owner}.{node.attr}" if owner is not None else None
    return None


def call_inventory(source: str, filename: str) -> Counter[str]:
    tree = ast.parse(source, filename=filename)
    names = (
        qualified_name(node.func)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    )
    return Counter(name for name in names if name is not None)


def bootstrap_namespace(source: str) -> dict[str, object]:
    """Load only the notebook's stdlib bootstrap definitions."""

    tree = ast.parse(source, filename=COLAB.as_posix())
    definitions = []
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.value, ast.Call)
            and qualified_name(node.value.func) == "_prepare_environment"
        ):
            break
        definitions.append(node)
    namespace: dict[str, object] = {}
    exec(
        compile(ast.Module(body=definitions, type_ignores=[]), COLAB.as_posix(), "exec"),
        namespace,
    )
    return namespace


class TransientCylinderWakeGalleryProduct(unittest.TestCase):
    def setUp(self) -> None:
        if not PLAIN.is_file() or not COLAB.is_file():
            self.skipTest("consumer tree does not carry the checked-in wake sources")
        self.plain_source = PLAIN.read_text(encoding="utf-8")
        self.notebook = json.loads(COLAB.read_text(encoding="utf-8"))
        self.notebook_source = notebook_source(self.notebook)

    def test_notebook_is_clean_valid_python(self) -> None:
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertEqual(self.notebook["nbformat_minor"], 5)
        self.assertEqual(
            self.notebook["metadata"]["kernelspec"]["name"],  # type: ignore[index]
            "python3",
        )
        cell_ids = [cell["id"] for cell in self.notebook["cells"]]  # type: ignore[union-attr]
        self.assertEqual(len(cell_ids), len(set(cell_ids)))
        for cell in self.notebook["cells"]:  # type: ignore[union-attr]
            if cell["cell_type"] != "code":
                continue
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])
            compile("".join(cell["source"]), COLAB.as_posix(), "exec")

        self.assertIn('find_spec("google.colab") is not None', self.notebook_source)
        self.assertIn('find_library("GLU") is None', self.notebook_source)
        self.assertIn('"libglu1-mesa"', self.notebook_source)
        self.assertIn('f"eqiora=={EQIORA_VERSION}"', self.notebook_source)
        self.assertIn('f"gmsh=={GMSH_VERSION}"', self.notebook_source)
        self.assertIn('f"matplotlib{MATPLOTLIB_RANGE}"', self.notebook_source)
        self.assertIn('EQIORA_VERSION = "0.1.0"', self.notebook_source)
        self.assertIn('GMSH_VERSION = "4.15.2"', self.notebook_source)
        self.assertIn('MATPLOTLIB_RANGE = ">=3.10,<3.12"', self.notebook_source)
        self.assertNotIn('find_spec("eqiora")', self.notebook_source)
        self.assertIn("subprocess.run(", self.notebook_source)
        self.assertNotIn("drive.mount", self.notebook_source)

    def test_colab_bootstrap_preserves_a_coherent_preloaded_matplotlib(self) -> None:
        namespace = bootstrap_namespace(self.notebook_source)
        installed = {
            "eqiora": {"version": "0.1.0", "file": "/runtime/eqiora/__init__.py"},
            "matplotlib": {
                "version": "3.10.8",
                "file": "/runtime/matplotlib/__init__.py",
            },
            "matplotlib._api": {
                "version": "3.10.8",
                "file": "/runtime/matplotlib/_api/__init__.py",
            },
            "mpl_toolkits.axes_grid1.axes_divider": {
                "version": "3.10.8",
                "file": "/runtime/mpl_toolkits/axes_grid1/axes_divider.py",
            },
        }
        loaded = {
            "matplotlib": installed["matplotlib"],
            "matplotlib._api": {
                "version": None,
                "file": installed["matplotlib._api"]["file"],
            },
        }
        namespace["_module_state"] = mock.Mock(return_value=loaded)
        namespace["_distribution_state"] = mock.Mock(return_value=installed)
        namespace["find_spec"] = mock.Mock(return_value=None)
        namespace["version"] = mock.Mock(side_effect=("0.1.0", "4.15.2"))
        namespace["subprocess"] = types.SimpleNamespace(run=mock.Mock())

        observed = namespace["_prepare_environment"]()  # type: ignore[operator]

        self.assertEqual(observed, installed)
        command = namespace["subprocess"].run.call_args.args[0]  # type: ignore[union-attr]
        self.assertIn("eqiora==0.1.0", command)
        self.assertIn("gmsh==4.15.2", command)
        self.assertIn("matplotlib>=3.10,<3.12", command)
        self.assertEqual(command[command.index("--upgrade-strategy") + 1], "only-if-needed")

    def test_colab_bootstrap_restarts_after_loaded_distribution_changes(self) -> None:
        namespace = bootstrap_namespace(self.notebook_source)
        namespace["_module_state"] = mock.Mock(
            return_value={
                "matplotlib": {
                    "version": "3.9.4",
                    "file": "/preloaded/matplotlib/__init__.py",
                }
            }
        )
        namespace["_distribution_state"] = mock.Mock(
            return_value={
                "eqiora": {
                    "version": "0.1.0",
                    "file": "/installed/eqiora/__init__.py",
                },
                "matplotlib": {
                    "version": "3.11.1",
                    "file": "/installed/matplotlib/__init__.py",
                },
                "matplotlib._api": {
                    "version": "3.11.1",
                    "file": "/installed/matplotlib/_api/__init__.py",
                },
                "mpl_toolkits.axes_grid1.axes_divider": {
                    "version": "3.11.1",
                    "file": "/installed/mpl_toolkits/axes_grid1/axes_divider.py",
                },
            }
        )
        namespace["find_spec"] = mock.Mock(return_value=object())
        namespace["find_library"] = mock.Mock(return_value="libGLU.so.1")
        namespace["version"] = mock.Mock(side_effect=("0.1.0", "4.15.2"))
        namespace["subprocess"] = types.SimpleNamespace(run=mock.Mock())
        namespace["os"] = types.SimpleNamespace(getpid=mock.Mock(return_value=42), kill=mock.Mock())

        with self.assertRaisesRegex(RuntimeError, "restart did not terminate"):
            namespace["_prepare_environment"]()  # type: ignore[operator]

        namespace["os"].kill.assert_called_once_with(42, namespace["signal"].SIGKILL)  # type: ignore[union-attr]

    def test_plain_and_colab_use_only_the_current_public_route(self) -> None:
        for source in (
            self.plain_source,
            self.notebook_source,
        ):
            for retired in (
                "CadAuthoredGraph",
                "MeshRequest",
                "TransientNavierStokesReference2d",
                "fluid.resolve",
                "steady-flow-past-cylinder.model.json",
                "_repr_mimebundle_",
                "IPython.display",
                "from examples",
            ):
                self.assertNotIn(retired, source)
            self.assertEqual(source.count("steady-flow-past-cylinder.eqi"), 1)
            self.assertEqual(source.count("transient-flow-past-cylinder.eqi"), 1)
            self.assertEqual(source.count("files(eqiora)"), 1)
            self.assertIn("UNVERIFIED PRODUCT EXAMPLE", source)
            self.assertIn("BackwardEuler(0.01)", source)
            self.assertIn("steps=10", source)
            self.assertIn("output_steps=tuple(range(1, 11))", source)
            self.assertIn("profile=True", source)
            self.assertIn("result.profile.summary()", source)
            self.assertIn("trajectory.state(10)", source)

    def test_colab_has_the_public_composition(self) -> None:
        notebook_calls = call_inventory(self.notebook_source, COLAB.as_posix())
        for call, count in PRESENTATION_CALLS.items():
            self.assertEqual(notebook_calls[call], count, call)

        ordered_markers = (
            "GeometryGraph()",
            ".subtract(",
            ".build(",
            "GmshMesher(",
            "meshing.resolve(",
            "meshing.generate(",
            "steady_model = eqiora.compile(",
            "steady_plan = eqiora.resolve(",
            "steady_result = eqiora.run(",
            "model = eqiora.compile(",
            "plan = eqiora.resolve(",
            "State.initial(",
            "result = eqiora.run(",
            ".curl(",
            ".boundary_force(",
            ".sample(",
            "plot_scalar_field(",
        )
        self.assertIn("maximum_target_size=0.025", self.notebook_source)
        cursor = 0
        for marker in ordered_markers:
            cursor = self.notebook_source.index(marker, cursor) + len(marker)

    def test_plain_script_contains_the_same_computational_path(self) -> None:
        self.assertIn("maximum_target_size=0.025", self.plain_source)
        plain_calls = call_inventory(self.plain_source, PLAIN.as_posix())
        for call, count in PRESENTATION_CALLS.items():
            expected = 2 if call == "result.trajectory.state" else count
            self.assertEqual(plain_calls[call], expected, call)

    def test_static_site_publishes_accessible_product_media(self) -> None:
        page = PAGE.read_text(encoding="utf-8")
        gallery = GALLERY_INDEX.read_text(encoding="utf-8")
        styles = SITE_STYLES.read_text(encoding="utf-8")
        producer = PRODUCER.read_text(encoding="utf-8")

        self.assertIn("/gallery/transient-cylinder-startup/", gallery)
        self.assertIn(
            "The plotted values have not been validated against a cylinder-flow benchmark.",
            " ".join(page.split()),
        )
        self.assertIn("<video", page)
        self.assertIn('type="video/webm"', page)
        self.assertIn('type="video/mp4"', page)
        self.assertIn("controls", page)
        self.assertIn('preload="metadata"', page)
        self.assertNotIn("autoplay", page)
        self.assertIn("eq-gallery-motion__still", page)
        self.assertIn("startup-motion-description", page)
        self.assertIn("colab.research.google.com/github/nkiyohara/eqiora/blob/", page)
        self.assertIn(
            '<LinkButton href={colabUrl} variant="primary">Open in Colab</LinkButton>',
            page,
        )
        self.assertNotIn("colabMinimumSerial", page)
        self.assertNotIn("colabReady", page)
        self.assertIn("prefers-reduced-motion: reduce", styles)
        self.assertIn("trajectory.states", producer)
        self.assertIn("FRAME_RATE = 2", producer)
        self.assertIn("values - reference_values", producer)
        self.assertIn("startup vorticity change", producer)
        self.assertIn("Δω", page)
        self.assertNotIn("verify/", producer)

        self.assertTrue(MEDIA["poster"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertTrue(MEDIA["reduced"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertTrue(MEDIA["webm"].read_bytes().startswith(b"\x1aE\xdf\xa3"))
        self.assertIn(b"ftyp", MEDIA["mp4"].read_bytes()[:32])
        for path in MEDIA.values():
            self.assertLess(path.stat().st_size, 2 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
