#!/usr/bin/env python3
"""Product checks for the exact-cylinder plain-Python example."""

from __future__ import annotations

import ast
import unittest
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
PLAIN = ROOT / "examples/python/exact_cylinder_stokes.py"
PRESENTATION_PRODUCER = (
    ROOT / "tools/site/produce_exact_cylinder_pressure_presentation.py"
)
PRESENTATION_IMAGE = (
    ROOT
    / "docs/site/src/assets/gallery/exact-cylinder-pressure-presentation.png"
)
HISTORICAL_IMAGE = ROOT / "docs/site/src/assets/gallery/exact-cylinder-pressure.png"

PUBLIC_CALLS = {
    "eqiora.geometry.GeometryGraph": 1,
    "graph.rectangle": 1,
    "graph.circle": 1,
    "graph.subtract": 1,
    "graph.build": 1,
    "eqiora.meshing.GmshMesher": 1,
    "eqiora.meshing.generate": 1,
    "eqiora.compile": 1,
    "eqiora.solve.Linear": 1,
    "eqiora.fem.MiniP1": 1,
    "eqiora.resolve": 1,
    "eqiora.run": 1,
    "result.output": 1,
    "eqplot.plot_scalar_field": 1,
}


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


class ExactCylinderStokesExampleProduct(unittest.TestCase):
    def setUp(self) -> None:
        if not PLAIN.is_file():
            self.skipTest("consumer tree does not carry the checked-in example source")
        self.source = PLAIN.read_text(encoding="utf-8")

    def test_source_uses_only_the_current_public_route(self) -> None:
        compile(self.source, PLAIN.as_posix(), "exec")
        for retired in (
            "CadAuthoredGraph",
            "MeshRequest",
            "SteadyStokesPlan",
            "fluid.resolve",
            "fluid.SteadyStokes",
            "steady-flow-past-cylinder.model.json",
            "_repr_mimebundle_",
            "IPython.display",
            "from examples",
        ):
            self.assertNotIn(retired, self.source)

        self.assertEqual(self.source.count("steady-flow-past-cylinder.eqi"), 1)
        self.assertEqual(self.source.count("files(eqiora)"), 1)
        self.assertIn("return eqiora.run(plan)", self.source)

    def test_example_has_the_complete_public_composition(self) -> None:
        self.assertIn("maximum_target_size=0.025", self.source)
        self.assertIn("eqiora.meshing.generate(geometry, mesh_request)", self.source)
        calls = call_inventory(self.source, PLAIN.as_posix())
        for call, count in PUBLIC_CALLS.items():
            self.assertEqual(calls[call], count, call)

        ordered_markers = (
            "GeometryGraph()",
            ".subtract(",
            ".build(",
            "GmshMesher(",
            "meshing.generate(",
            "eqiora.compile(",
            "eqiora.resolve(",
            "eqiora.run(",
            "result.output(",
            "plot_scalar_field(",
        )
        positions = [self.source.index(marker) for marker in ordered_markers]
        self.assertEqual(positions, sorted(positions))

    def test_current_fine_mesh_image_is_presentation_only(self) -> None:
        presentation_files = (
            PRESENTATION_PRODUCER,
            PRESENTATION_IMAGE,
            HISTORICAL_IMAGE,
        )
        if not all(path.is_file() for path in presentation_files):
            self.skipTest("consumer tree does not carry repository presentation files")

        self.assertIn("maximum_target_size=0.025", PLAIN.read_text(encoding="utf-8"))
        producer = PRESENTATION_PRODUCER.read_text(encoding="utf-8")
        self.assertIn("from exact_cylinder_stokes import solve", producer)
        self.assertIn("fine-mesh steady-cylinder presentation", producer)
        self.assertNotIn("verify/", producer)
        self.assertTrue(PRESENTATION_IMAGE.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertTrue(HISTORICAL_IMAGE.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))


if __name__ == "__main__":
    unittest.main()
