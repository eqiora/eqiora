from __future__ import annotations

import contextlib
import io
import sys
import tempfile
import types
import unittest
from contextlib import redirect_stderr
from pathlib import Path
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools/ci"))
sys.path.insert(0, str(REPOSITORY_ROOT / "tools/release"))

import python_distribution_gate  # noqa: E402
from candidate_manifest import ManifestError, REQUIRED_PROFILES  # noqa: E402


class PythonDistributionGateTests(unittest.TestCase):
    REVISION = "1" * 40

    @staticmethod
    def option(argv: list[str], name: str) -> Path:
        return Path(argv[argv.index(name) + 1])

    @staticmethod
    def write_family(directory: Path) -> None:
        directory.mkdir()
        (directory / "eqiora-0.1.0.tar.gz").write_bytes(b"sdist")
        for python in ("311", "312", "313", "314"):
            (directory / f"eqiora-0.1.0-cp{python}-cp{python}-linux.whl").write_bytes(
                f"wheel-{python}".encode()
            )

    def test_required_profiles_are_host_agnostic(self) -> None:
        self.assertEqual(
            REQUIRED_PROFILES,
            ("base", "jax", "matplotlib", "torch", "typing"),
        )

    def test_aggregate_runs_prepare_then_finalize(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            calls: list[list[str]] = []

            def run(argv: list[str], **_kwargs: object) -> str:
                calls.append(argv)
                if argv[2] == "prepare":
                    self.write_family(self.option(argv, "--out"))
                else:
                    output = self.option(argv, "--manifest-out")
                    output.mkdir()
                    (output / "eqiora-0.1.0-python-candidate.json").write_bytes(
                        b"opaque manifest"
                    )
                return ""

            candidate = mock.sentinel.candidate
            with (
                mock.patch.object(python_distribution_gate, "checked_run", side_effect=run),
                mock.patch.object(
                    python_distribution_gate,
                    "source_identity",
                    return_value=types.SimpleNamespace(commit=self.REVISION),
                ),
                mock.patch.object(
                    python_distribution_gate,
                    "load_candidate_family",
                    return_value=candidate,
                ) as load_candidate_family,
                mock.patch.object(python_distribution_gate, "verify_artifacts") as verify_artifacts,
                mock.patch.object(
                    python_distribution_gate, "require_candidate_profile"
                ) as require_profile,
            ):
                observed = python_distribution_gate.build_and_verify_candidate(root)

            self.assertIs(observed, candidate)
            self.assertEqual(len(calls), 2)
            family = self.option(calls[0], "--out")
            metadata = self.option(calls[1], "--manifest-out")
            self.assertEqual(
                calls[1],
                [
                    "python3",
                    "tools/release/python_candidate.py",
                    "finalize",
                    "--expected-commit",
                    self.REVISION,
                    "--artifacts",
                    str(family),
                    "--manifest-out",
                    str(metadata),
                ],
            )
            manifest = metadata / "eqiora-0.1.0-python-candidate.json"
            load_candidate_family.assert_called_once_with(
                manifest,
                family,
                requested_profiles=REQUIRED_PROFILES,
            )
            verify_artifacts.assert_called_once_with(candidate, family)
            self.assertEqual(
                require_profile.call_args_list,
                [mock.call(candidate, profile) for profile in REQUIRED_PROFILES],
            )

    def test_family_mutation_during_finalization_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)

            def run(argv: list[str], **_kwargs: object) -> str:
                if argv[2] == "prepare":
                    self.write_family(self.option(argv, "--out"))
                else:
                    family = self.option(argv, "--artifacts")
                    output = self.option(argv, "--manifest-out")
                    output.mkdir()
                    (output / "eqiora-0.1.0-python-candidate.json").write_bytes(b"manifest")
                    next(family.glob("*.whl")).write_bytes(b"mutated")
                return ""

            with (
                mock.patch.object(python_distribution_gate, "checked_run", side_effect=run),
                mock.patch.object(
                    python_distribution_gate,
                    "source_identity",
                    return_value=types.SimpleNamespace(commit=self.REVISION),
                ),
                self.assertRaisesRegex(RuntimeError, "family inventory changed"),
            ):
                python_distribution_gate.build_and_verify_candidate(root)

    def test_profile_failure_is_a_fail_closed_gate_diagnostic(self) -> None:
        with (
            mock.patch.object(
                python_distribution_gate,
                "build_and_verify_candidate",
                side_effect=ManifestError("candidate jax profile is incomplete"),
            ),
            redirect_stderr(io.StringIO()) as stderr,
        ):
            self.assertEqual(python_distribution_gate.main(), 2)
        self.assertIn("candidate jax profile", stderr.getvalue())

    def test_aggregate_scratch_is_home_backed(self) -> None:
        with tempfile.TemporaryDirectory(dir=Path.home()) as temporary:
            scratch = Path(temporary) / "gate"
            scratch.mkdir()

            def temporary_directory(*_args: object, **kwargs: object) -> object:
                self.assertTrue(
                    Path(kwargs["dir"]).resolve().is_relative_to(Path.home().resolve())
                )
                return contextlib.nullcontext(str(scratch))

            with (
                mock.patch.object(python_distribution_gate, "build_and_verify_candidate") as build,
                mock.patch.object(
                    python_distribution_gate.tempfile,
                    "TemporaryDirectory",
                    side_effect=temporary_directory,
                ),
            ):
                self.assertEqual(python_distribution_gate.main(), 0)
            build.assert_called_once_with(scratch)


if __name__ == "__main__":
    unittest.main()
