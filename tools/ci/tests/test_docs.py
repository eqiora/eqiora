from __future__ import annotations

import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


CI_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = CI_ROOT.parents[1]
sys.path.insert(0, str(CI_ROOT))

from check_docs import BENCHMARKS, benchmark_failures  # noqa: E402


class PythonApiGeneratorTests(unittest.TestCase):
    def test_isolated_startup_preserves_source_tree(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            names = ("generate_python_api.py", "python_api_examples.py")
            for name in names:
                shutil.copy2(REPOSITORY_ROOT / "tools/docs" / name, root / name)
            before = {path.name: path.read_bytes() for path in root.iterdir()}
            result = subprocess.run(
                [sys.executable, "-I", str(root / names[0]), "--help"],
                cwd=root,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(set(path.name for path in root.iterdir()), set(before))
            for name, payload in before.items():
                self.assertEqual((root / name).read_bytes(), payload)


class BenchmarkCitationTests(unittest.TestCase):
    """The benchmark catalogue may not outlive what it cites."""

    def setUp(self) -> None:
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        for relative in (BENCHMARKS, "verify", "crates"):
            (self.root / relative).parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(REPOSITORY_ROOT / "verify", self.root / "verify")
        (self.root / "crates").mkdir(exist_ok=True)
        (self.root / "crates/only.rs").write_text("struct MiniConstantTractionFacet;\n")

    def write(self, body: str) -> list[str]:
        (self.root / BENCHMARKS).write_text(body)
        return benchmark_failures(self.root)

    def test_absent_catalogue_is_not_a_failure(self) -> None:
        self.assertEqual(benchmark_failures(self.root), [])

    def test_unknown_case_is_rejected(self) -> None:
        failures = self.write(
            "## Reproduced today\n\n| a | `case:fluid.does-not-exist` |\n"
        )
        self.assertTrue(
            any("unknown case" in failure for failure in failures), failures
        )

    def test_unverified_case_is_rejected(self) -> None:
        # A case that exists but has not been measured cannot back a claim that
        # the benchmark is reproduced.
        specified = next(
            path
            for path in (self.root / "verify").glob("*/*/case.toml")
            if 'status = "specified"' in path.read_text(encoding="utf-8")
        )
        identifier = (
            specified.read_text(encoding="utf-8").split('id = "', 1)[1].split('"', 1)[0]
        )
        failures = self.write(f"## Reproduced today\n\n| a | `case:{identifier}` |\n")
        self.assertTrue(
            any("whose status is" in failure for failure in failures), failures
        )

    def test_absent_symbol_is_rejected(self) -> None:
        failures = self.write("## Reproduced today\n\n| a | `symbol:NoSuchItem` |\n")
        self.assertTrue(
            any("present in no Rust source" in failure for failure in failures),
            failures,
        )

    def test_undeclared_manifest_key_is_rejected(self) -> None:
        failures = self.write("## Reproduced today\n\n| a | `key:no_such_clause` |\n")
        self.assertTrue(
            any("declared by no case" in failure for failure in failures), failures
        )

    def test_row_without_a_citation_is_rejected(self) -> None:
        # Deleting a citation must not quietly demote a claim to prose.
        failures = self.write("## Reproduced today\n\n| something is reproduced | |\n")
        self.assertTrue(
            any("cites no capability" in failure for failure in failures), failures
        )

    def test_row_declaring_no_capability_is_accepted(self) -> None:
        self.assertEqual(
            self.write("## Reproduced today\n\n| free surface | none declared |\n"), []
        )

    def test_rows_outside_the_cited_sections_are_not_required_to_cite(self) -> None:
        self.assertEqual(self.write("## Citations\n\n| `case:<id>` | a case |\n"), [])

    def test_the_repository_catalogue_passes(self) -> None:
        self.assertEqual(benchmark_failures(REPOSITORY_ROOT), [])


if __name__ == "__main__":
    unittest.main()
