from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fixture import (
    PRESSURE_ALT,
    REPOSITORY,
    SOURCE_SHA,
    checker,
    make_fixture,
)


class CompleteContractTests(unittest.TestCase):
    def test_release_identity_rejects_the_displaced_alpha_form(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            make_fixture(root, "0.1.0-alpha.14")
            release, errors = checker.derive_release_identity(root)
            self.assertIsNone(release)
            self.assertTrue(any("release SemVer identity" in error for error in errors))

    def test_source_command_reports_checker_result(self) -> None:
        for errors, status in (([], 0), (["invalid site source"], 1)):
            with self.subTest(errors=errors), mock.patch.object(
                checker, "check_source", return_value=errors
            ) as check, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(
                io.StringIO()
            ) as stderr:
                self.assertEqual(checker.main(["source", "--root", str(REPOSITORY)]), status)
                check.assert_called_once_with(REPOSITORY.resolve())
                if errors:
                    self.assertIn("invalid site source", stderr.getvalue())

    def test_00_synthetic_ordinary_site_passes_before_mutants(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact, identities = make_fixture(root)
            case = artifact / "gallery/exact-cylinder-steady-stokes/index.html"
            raw = case.read_text(encoding="utf-8")
            self.assertNotIn("104-triangle", raw)
            self.assertEqual(
                checker.check_site(root, artifact, SOURCE_SHA, identities), []
            )

    def test_public_information_architecture_mutants_fail(self) -> None:
        mutations = (
            (
                "missing primary Learn navigation",
                Path("index.html"),
                '<a href="/learn/">Learn</a>',
                '<a href="/learn/">Learning</a>',
                "public navigation omits",
            ),
            (
                "maintained guide replaced by a wrapper",
                Path("guides/modeling/index.html"),
                "Native declarations",
                "Read more on GitHub",
                "omits maintained content",
            ),
            (
                "missing Learn topics",
                Path("learn/index.html"),
                "Browse by topic",
                "Subjects",
                "Learn landing omits",
            ),
            (
                "missing committed Learn lesson map",
                Path("learn/mathematical-modeling/index.html"),
                "Build a model",
                "Contents pending",
                "learning path omits",
            ),
            (
                "missing published Learn lesson anatomy",
                Path("learn/mathematical-modeling/models-not-simulations/index.html"),
                "Deliberate failure",
                "Example",
                "Learn lesson 'Models are not simulations' omits",
            ),
            (
                "missing ODE lesson exercises",
                Path("learn/mathematical-modeling/ordinary-differential-equations/index.html"),
                "Exercises",
                "Practice omitted",
                "Learn lesson 'Ordinary differential equations' omits",
            ),
            (
                "missing verification guide",
                Path("evidence/index.html"),
                "Run a selected check",
                "Overview",
                "verification guide omits",
            ),
            (
                "broken learning route",
                Path("gallery/exact-cylinder-steady-stokes/index.html"),
                "/capabilities/#exact-cylinder-steady-stokes",
                "/capabilities/#missing",
                "omits the static learning-to-evidence route",
            ),
        )
        for label, relative, accepted, mutant, expected in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                artifact, identities = make_fixture(root)
                self._replace(artifact / relative, accepted, mutant)
                errors = checker.check_site(root, artifact, SOURCE_SHA, identities)
                self.assertTrue(any(expected in error for error in errors), errors)

    def test_01_exact_gmsh_publication_boundary_mutants_fail(self) -> None:
        mutations = {
            "fixed-mesh figure alt": (
                PRESSURE_ALT,
                PRESSURE_ALT.replace("current mesh", "fixed mesh"),
            ),
        }
        for label, (accepted, mutant) in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                artifact, identities = make_fixture(root)
                self.assertEqual(
                    checker.check_site(root, artifact, SOURCE_SHA, identities), []
                )
                case = artifact / "gallery/exact-cylinder-steady-stokes/index.html"
                self._replace(case, accepted, mutant)
                self.assertTrue(
                    checker.check_site(root, artifact, SOURCE_SHA, identities)
                )

    def test_01_foundation_shape_is_red_for_only_missing_downstream_inputs(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, identities = make_fixture(root)
            old = root / "docs/site/assets/social-card.svg"
            old.parent.mkdir(parents=True, exist_ok=True)
            old.write_text(checker.OLD_SOCIAL_LINE, encoding="utf-8")
            for relative in (
                "docs/site/src/assets/gallery/exact-cylinder-pressure-presentation.png",
                "docs/site/public/social-card.svg",
            ):
                (root / relative).unlink()
            errors = checker.check_source(root, identities)
            joined = "\n".join(errors)
            self.assertIn(
                "obsolete successor source remains: docs/site/assets/social-card.svg",
                joined,
            )
            self.assertIn("missing exact admitted pressure media", joined)
            self.assertIn("missing exact timeless social card", joined)
            self.assertNotIn("site package must pin", joined)
            self.assertNotIn("Pages path filters", joined)

    def test_02_execution_control_visible_and_accessible_labels_agree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact, identities = make_fixture(root)
            case = artifact / "gallery/exact-cylinder-steady-stokes/index.html"
            self._replace(
                case,
                "</main>",
                "<p>Documentation may discuss Start computation, Evaluate model, "
                "and Begin processing without making prose interactive.</p></main>",
            )
            self.assertEqual(
                checker.check_site(root, artifact, SOURCE_SHA, identities), []
            )

        pairs = {
            "native rendered value with benign ARIA override": (
                '<input type="button" value="Documentation" '
                'aria-label="Documentation">',
                '<input type="button" value="Start computation" '
                'aria-label="Documentation">',
            ),
            "descendant image alternative": (
                '<button><img src="data:image/gif;base64,R0lGODlhAQABAAAAACw=" '
                'alt="Documentation"></button>',
                '<button><img src="data:image/gif;base64,R0lGODlhAQABAAAAACw=" '
                'alt="Start computation"></button>',
            ),
            "ASCII-case-insensitive native type": (
                '<input type="text" value="Start computation">',
                '<input type="BUTTON" value="Start computation">',
            ),
        }
        for label, (ordinary, mutant) in pairs.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                artifact, identities = make_fixture(root)
                case = artifact / "gallery/exact-cylinder-steady-stokes/index.html"
                self._replace(case, "</main>", f"{ordinary}</main>")
                self.assertEqual(
                    checker.check_site(root, artifact, SOURCE_SHA, identities), []
                )
                self._replace(case, ordinary, mutant)
                errors = checker.check_site(root, artifact, SOURCE_SHA, identities)
                self.assertTrue(
                    any("uncontracted execution control" in error for error in errors),
                    errors,
                )

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact, identities = make_fixture(root)
            case = artifact / "gallery/exact-cylinder-steady-stokes/index.html"
            self._replace(
                case, "</main>",
                '<h2 id="before-you-begin">Before you begin</h2>'
                '<a href="#before-you-begin">Before you begin</a></main>',
            )
            self.assertEqual(checker.check_site(root, artifact, SOURCE_SHA, identities), [])
            self._replace(
                case, '<a href="#before-you-begin">',
                '<a href="#before-you-begin" role="button">',
            )
            self.assertTrue(any(
                "uncontracted execution control" in error
                for error in checker.check_site(root, artifact, SOURCE_SHA, identities)
            ))

        controls = {
            "accessible anchor label": (
                '<a href="/get-started/" aria-label="Run simulation">Documentation</a>'
            ),
            "visible anchor label overriding benign accessibility label": (
                '<a href="/get-started/" aria-label="Documentation">Run simulation</a>'
            ),
            "explicit button role": (
                '<div role="button" aria-label="Execute calculation">Details</div>'
            ),
            "native input button": '<input type="button" value="Launch computation">',
            "aria-labelledby anchor": (
                '<a href="/get-started/" aria-labelledby="execution-label">Docs</a>'
                '<span id="execution-label">Start computation</span>'
            ),
            "evaluate synonym": '<a href="/get-started/">Evaluate model</a>',
            "processing synonym": '<a href="/get-started/">Begin processing</a>',
            "generate synonym": "<button>Generate result</button>",
            "analyse synonym": "<button>Analyse case</button>",
            "predict synonym": "<button>Predict flow</button>",
        }
        for label, control in controls.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                artifact, identities = make_fixture(root)
                case = artifact / "gallery/exact-cylinder-steady-stokes/index.html"
                self._replace(case, "</main>", f"{control}</main>")
                errors = checker.check_site(root, artifact, SOURCE_SHA, identities)
                self.assertTrue(
                    any("uncontracted execution control" in error for error in errors),
                    errors,
                )

    def test_route_canonical_media_and_claim_mutants_fail(self) -> None:
        mutations = {
            "missing route": lambda root, artifact: (
                artifact / "reference/cli/index.html"
            ).unlink(),
            "duplicate canonical": lambda root, artifact: self._replace(
                artifact / "index.html",
                "</head>",
                '<link rel="canonical" href="https://eqiora.org/"></head>',
            ),
            "wrong pressure alt": lambda root, artifact: self._replace(
                artifact / "gallery/exact-cylinder-steady-stokes/index.html",
                checker.PRESSURE_ALT,
                "Pressure plot",
            ),
            "missing featured wake": lambda root, artifact: self._replace(
                artifact / "index.html",
                checker.WAKE_ALT,
                "Wake plot",
            ),
            "unlinked brand": lambda root, artifact: self._replace(
                artifact / "index.html",
                '<a class="site-title" href="/"><img src="/assets/brand.svg" alt=""><span>Eqiora</span></a>',
                '<img src="/assets/brand.svg" alt=""><span>Eqiora</span>',
            ),
            "widened featured claim": lambda root, artifact: self._replace(
                artifact / "index.html",
                "Featured walkthrough",
                "Featured walkthrough production ready",
            ),
            "fake run button": lambda root, artifact: self._replace(
                artifact / "gallery/exact-cylinder-steady-stokes/index.html",
                "</main>",
                "<button>Run now</button></main>",
            ),
            "fake run link": lambda root, artifact: self._replace(
                artifact / "gallery/exact-cylinder-steady-stokes/index.html",
                "</main>",
                '<a href="/get-started/">Run now</a></main>',
            ),
            "fake simulation link": lambda root, artifact: self._replace(
                artifact / "gallery/exact-cylinder-steady-stokes/index.html",
                "</main>",
                '<a href="/get-started/">Run simulation</a></main>',
            ),
            "fake calculation link": lambda root, artifact: self._replace(
                artifact / "gallery/exact-cylinder-steady-stokes/index.html",
                "</main>",
                '<a href="/get-started/">Execute calculation</a></main>',
            ),
            "fake stage-label link": lambda root, artifact: self._replace(
                artifact / "gallery/exact-cylinder-steady-stokes/index.html",
                "</main>",
                '<a href="/get-started/">Submit and result</a></main>',
            ),
            "missing social": lambda root, artifact: (
                artifact / "social-card.svg"
            ).unlink(),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                artifact, identities = make_fixture(root)
                mutate(root, artifact)
                self.assertTrue(
                    checker.check_site(root, artifact, SOURCE_SHA, identities)
                )

    def test_source_identity_and_stale_social_mutants_fail(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, identities = make_fixture(root)
            old = root / "docs/site/assets/social-card.svg"
            old.parent.mkdir(parents=True, exist_ok=True)
            old.write_text(checker.OLD_SOCIAL_LINE, encoding="utf-8")
            errors = checker.check_source(root, identities)
            self.assertTrue(
                any("obsolete successor source remains" in error for error in errors)
            )

    def test_provider_dependency_and_release_identity_mutants_fail(self) -> None:
        for relative in checker.PROVIDER_PATHS:
            with (
                self.subTest(provider=relative),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                _, identities = make_fixture(root)
                (root / relative).unlink()
                errors = checker.check_source(root, identities)
                self.assertTrue(any("accepted provider" in error for error in errors))

        provider_mutations = (
            (
                "docs/site/src/components/site/ExactSourceLink.astro",
                "EQIORA_SITE_SOURCE_SHA",
            ),
            (
                "docs/site/src/components/site/ReleaseIdentity.astro",
                "EQIORA_SITE_PYTHON_VERSION",
            ),
            (
                "docs/site/astro.config.mjs",
                "src/components/site/ExactSourceLink.astro",
            ),
        )
        for relative, token in provider_mutations:
            with (
                self.subTest(provider_token=f"{relative}:{token}"),
                tempfile.TemporaryDirectory() as temporary,
            ):
                root = Path(temporary)
                _, identities = make_fixture(root)
                source = root / relative
                source.write_text(
                    source.read_text(encoding="utf-8").replace(token, "removed"),
                    encoding="utf-8",
                )
                errors = checker.check_source(root, identities)
                self.assertTrue(any("provider" in error for error in errors))

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, identities = make_fixture(root)
            package = root / "docs/site/package.json"
            document = json.loads(package.read_text(encoding="utf-8"))
            document["dependencies"]["react"] = "19.2.4"
            package.write_text(json.dumps(document), encoding="utf-8")
            lock = root / "docs/site/package-lock.json"
            lock_document = json.loads(lock.read_text(encoding="utf-8"))
            lock_document["packages"][""]["dependencies"]["react"] = "19.2.4"
            lock_document["packages"]["node_modules/react"] = {
                "version": "19.2.4",
                "integrity": "sha512-fixture",
            }
            lock.write_text(json.dumps(lock_document), encoding="utf-8")
            errors = checker.check_source(root, identities)
            self.assertTrue(any("exact direct set" in error for error in errors))

    @staticmethod
    def _replace(path: Path, old: str, new: str) -> None:
        path.write_text(
            path.read_text(encoding="utf-8").replace(old, new), encoding="utf-8"
        )



if __name__ == "__main__":
    unittest.main()
