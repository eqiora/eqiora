from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from fixture import SOURCE_SHA, checker, make_fixture


def artifact_errors(root: Path, artifact: Path, identities):
    release, release_errors = checker.derive_release_identity(root)
    assert release is not None, release_errors
    return checker.check_artifact(artifact, SOURCE_SHA, release.python, identities)


class MathAndLinkMutantTests(unittest.TestCase):
    def test_math_node_kind_fallback_and_local_font_mutants_fail(self) -> None:
        mutations = (
            ("block wrapper", 'class="katex-display"', 'class="not-display"'),
            ("source fallback", "Eqiora source form", "Formula"),
            ("raw delimiter", "Eqiora source form", "Eqiora source form $$"),
        )
        for label, old, new in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                artifact, identities = make_fixture(root)
                page = artifact / "gallery/exact-cylinder-steady-stokes/index.html"
                page.write_text(
                    page.read_text(encoding="utf-8").replace(old, new), encoding="utf-8"
                )
                errors = artifact_errors(root, artifact, identities)
                self.assertTrue(errors)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact, identities = make_fixture(root)
            (artifact / "assets/KaTeX_Main-Regular.woff2").unlink()
            errors = artifact_errors(root, artifact, identities)
            self.assertTrue(any("CSS asset" in error for error in errors))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact, identities = make_fixture(root)
            stylesheet = artifact / "assets/site.css"
            stylesheet.write_text(
                stylesheet.read_text(encoding="utf-8").replace("KaTeX", "Site"),
                encoding="utf-8",
            )
            errors = artifact_errors(root, artifact, identities)
            self.assertTrue(any("KaTeX CSS" in error for error in errors))

    def test_exact_sha_link_and_external_runtime_request_mutants_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact, identities = make_fixture(root)
            page = artifact / "gallery/exact-cylinder-steady-stokes/index.html"
            page.write_text(
                page.read_text(encoding="utf-8").replace(
                    f"/blob/{SOURCE_SHA}/", "/blob/main/"
                ),
                encoding="utf-8",
            )
            errors = artifact_errors(root, artifact, identities)
            self.assertTrue(
                any("exact-head source link" in error for error in errors)
            )
            self.assertTrue(
                any("branch-relative source identity" in error for error in errors)
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact, identities = make_fixture(root)
            page = artifact / "gallery/exact-cylinder-steady-stokes/index.html"
            page.write_text(
                page.read_text(encoding="utf-8").replace(SOURCE_SHA, "b" * 40),
                encoding="utf-8",
            )
            errors = artifact_errors(root, artifact, identities)
            self.assertTrue(any("exact asserted SHA" in error for error in errors))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact, identities = make_fixture(root)
            page = artifact / "index.html"
            page.write_text(
                page.read_text(encoding="utf-8").replace(
                    "</main>",
                    '<img src="https://cdn.example/image.png" alt="external"></main>',
                ),
                encoding="utf-8",
            )
            errors = artifact_errors(root, artifact, identities)
            self.assertTrue(
                any("external runtime request" in error for error in errors)
            )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact, identities = make_fixture(root)
            page = artifact / "get-started/index.html"
            page.write_text(
                page.read_text(encoding="utf-8").replace(
                    "</main>", '<a href="/missing-target/">Broken</a></main>'
                ),
                encoding="utf-8",
            )
            errors = artifact_errors(root, artifact, identities)
            self.assertTrue(any("broken or escaping link" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
