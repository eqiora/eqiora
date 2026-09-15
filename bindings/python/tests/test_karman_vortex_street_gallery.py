#!/usr/bin/env python3
"""Product contracts for the unverified Reynolds-100 wake presentation."""

from __future__ import annotations

import ast
import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
PLAIN = ROOT / "examples/python/karman_vortex_street.py"
COLAB = ROOT / "examples/python/karman_vortex_street_colab.ipynb"
PAGE = ROOT / "docs/site/src/content/docs/gallery/karman-vortex-street.mdx"
GALLERY_INDEX = ROOT / "docs/site/src/content/docs/gallery/index.mdx"
HOME = ROOT / "docs/site/src/components/site/Home.astro"
PRODUCER = ROOT / "tools/site/produce_karman_vortex_street_media.py"
MEDIA = {
    "poster": ROOT / "docs/site/src/assets/gallery/karman-vortex-street-poster.png",
    "reduced": ROOT / "docs/site/src/assets/gallery/karman-vortex-street-reduced-motion.png",
    "webm": ROOT / "docs/site/src/assets/gallery/karman-vortex-street.webm",
    "mp4": ROOT / "docs/site/src/assets/gallery/karman-vortex-street.mp4",
}


def notebook_source(notebook: dict[str, object]) -> str:
    return "\n\n".join(
        "".join(cell["source"])
        for cell in notebook["cells"]  # type: ignore[index]
        if isinstance(cell, dict) and cell.get("cell_type") == "code"
    )


class KarmanVortexStreetGalleryProduct(unittest.TestCase):
    def setUp(self) -> None:
        self.plain = PLAIN.read_text(encoding="utf-8")
        self.notebook = json.loads(COLAB.read_text(encoding="utf-8"))
        self.colab = notebook_source(self.notebook)

    def test_python_sources_use_the_same_physical_and_numerical_boundary(self) -> None:
        ast.parse(self.plain, filename=PLAIN.as_posix())
        for source in (self.plain, self.colab):
            for token in (
                "x_bounds=(0.0, 2.2)",
                "y_bounds=(0.0, 0.41)",
                "center=(0.2, 0.2)",
                "MiniP1()",
                "maximum_target_size=0.02",
                "BackwardEuler(",
                "absolute_tolerance=1.0e-12",
                "boundary_force(",
                "on_selection",
            ):
                self.assertIn(token, source)
            self.assertNotIn("transient_cylinder_wake", source)
            self.assertIsNone(re.search(r"steps\s*=\s*10\b", source))
        self.assertIn("UNVERIFIED PRODUCT EXAMPLE", self.plain)
        self.assertIn("Unverified product example", COLAB.read_text(encoding="utf-8"))
        for token in (
            "INLET_MAXIMUM_M_PER_S = 1.5",
            "INLET_MEAN_M_PER_S = 1.0",
            "CYLINDER_DIAMETER_M = 0.1",
            "DEFAULT_SPIN_UP_STEPS = 700",
            "DEFAULT_SAMPLE_STEPS = 200",
            "SPIN_UP_CHUNK_STEPS = 100",
            "strouhal_number",
            "pressure_differences_pa",
        ):
            self.assertIn(token, self.plain)

    def test_notebook_is_clean_and_uses_the_colab_helper(self) -> None:
        self.assertEqual(self.notebook["nbformat"], 4)
        self.assertEqual(self.notebook["nbformat_minor"], 5)
        ids = [cell["id"] for cell in self.notebook["cells"]]  # type: ignore[index]
        self.assertEqual(len(ids), len(set(ids)))
        for cell in self.notebook["cells"]:  # type: ignore[index]
            if cell["cell_type"] != "code":
                continue
            self.assertIsNone(cell["execution_count"])
            self.assertEqual(cell["outputs"], [])
            source = "\n".join(
                line
                for line in "".join(cell["source"]).splitlines()
                if not line.startswith("%pip ")
            )
            compile(source, COLAB.as_posix(), "exec")
        prepare = self.notebook["cells"][1]  # type: ignore[index]
        self.assertEqual(prepare["id"], "wake-prepare")
        self.assertEqual(prepare["metadata"]["cellView"], "form")
        self.assertIn('eqiora[gmsh]==0.1.2', "".join(prepare["source"]))
        self.assertIn("from eqiora.colab import prepare", self.colab)
        self.assertNotIn("_prepare_environment", self.colab)
        self.assertIn("eqiora.View().add(geometry).add(mesh).add(vorticity)", self.colab)

    def test_site_publishes_accessible_successful_wake_media(self) -> None:
        page = PAGE.read_text(encoding="utf-8")
        gallery = GALLERY_INDEX.read_text(encoding="utf-8")
        home = HOME.read_text(encoding="utf-8")
        producer = PRODUCER.read_text(encoding="utf-8")
        self.assertIn("/gallery/karman-vortex-street/", gallery)
        self.assertIn("karman-vortex-street-poster.png", home)
        self.assertIn("Unverified product example", page)
        self.assertIn("MINI/P1", page)
        self.assertIn("Backward Euler", page)
        self.assertIn("not benchmark-validated", page)
        self.assertIn("<video", page)
        self.assertIn('type="video/webm"', page)
        self.assertIn('type="video/mp4"', page)
        self.assertIn("controls", page)
        self.assertIn('preload="metadata"', page)
        self.assertNotIn("autoplay", page)
        self.assertIn("eq-gallery-motion__still", page)
        self.assertIn("wake-motion-description", page)
        self.assertIn("FRAME_RATE = 8", producer)
        self.assertIn("wake.lift_coefficients", producer)
        self.assertIn("exactly 100 sampled output states", producer)
        self.assertNotIn("verify/", producer)

        self.assertTrue(MEDIA["poster"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertTrue(MEDIA["reduced"].read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertTrue(MEDIA["webm"].read_bytes().startswith(b"\x1aE\xdf\xa3"))
        self.assertIn(b"ftyp", MEDIA["mp4"].read_bytes()[:32])
        for path in MEDIA.values():
            self.assertLess(path.stat().st_size, 2 * 1024 * 1024)


if __name__ == "__main__":
    unittest.main()
