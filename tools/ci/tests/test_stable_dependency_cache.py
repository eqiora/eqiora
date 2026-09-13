from __future__ import annotations

import os
import re
import subprocess
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]


def workflow(name: str) -> str:
    return (ROOT / ".github/workflows" / name).read_text(encoding="utf-8")


def step(source: str, name: str) -> str:
    match = re.search(
        rf"(?ms)^      - name: {re.escape(name)}\n(.*?)(?=^      - |^  \S|\Z)",
        source,
    )
    assert match is not None, name
    return match.group(1)


def script(source: str) -> str:
    body = source.split("        run: |\n", 1)[1]
    return "\n".join(line[10:] for line in body.splitlines())


class StableDependencyCacheTests(unittest.TestCase):
    def test_dispatch_guard_executes_before_selected_source_in_every_workflow(self) -> None:
        sha = "a" * 40
        for filename in ("ci.yml", "python-release-candidate.yml"):
            source = workflow(filename)
            guard = step(source, "Bind manual source to the dispatch revision")
            self.assertIn("SELECTED_COMMIT: ${{ inputs.commit }}", guard)
            self.assertLess(source.index(guard), source.index("uses: actions/checkout@"))
            if filename == "ci.yml":
                self.assertIn("if: github.event_name == 'workflow_dispatch'", guard)
            for ref in ("refs/heads/main", "refs/heads/candidate-review"):
                for selected, accepted in ((sha, True), ("b" * 40, False), ("a" * 7, False),
                                           ("A" * 40, False), ("", False), (sha + "\n", False)):
                    with self.subTest(workflow=filename, ref=ref, selected=selected):
                        result = subprocess.run(
                            ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", script(guard)],
                            env={**os.environ, "SELECTED_COMMIT": selected, "GITHUB_SHA": sha, "GITHUB_REF": ref},
                            cwd=ROOT,
                            capture_output=True,
                            text=True,
                        )
                        self.assertEqual(result.returncode == 0, accepted, result.stderr)

        source = workflow("python-production-publish.yml")
        guard = step(source, "Bind publication authority to protected main")
        self.assertIn("SELECTED_COMMIT: ${{ inputs.commit }}", guard)
        self.assertIn("SELECTED_TAG: ${{ inputs.tag }}", guard)
        self.assertLess(source.index(guard), source.index("uses: actions/checkout@"))
        for ref, tag, accepted in (
            ("refs/heads/main", "v0.1.0a11", True),
            ("refs/heads/candidate-review", "v0.1.0a11", False),
            ("refs/heads/main", "", False),
        ):
            with self.subTest(workflow="python-production-publish.yml", ref=ref, tag=tag):
                result = subprocess.run(
                    ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c", script(guard)],
                    env={
                        **os.environ,
                        "SELECTED_COMMIT": sha,
                        "SELECTED_TAG": tag,
                        "GITHUB_REF": ref,
                    },
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                )
                self.assertEqual(result.returncode == 0, accepted, result.stderr)

    def test_production_checks_tag_object_before_importing_selected_code(self) -> None:
        guard = script(step(workflow("python-production-publish.yml"), "Require exact annotated tag and source commit"))
        imported = guard.index("PYTHONPATH=tools/release")
        for check in ('test "$(git rev-parse HEAD)" = "$RELEASE_COMMIT"',
                      'test "$(git cat-file -t "$tag_object")" = "tag"',
                      'test "$peeled_commit" = "$RELEASE_COMMIT"'):
            self.assertLess(guard.index(check), imported)

        with tempfile.TemporaryDirectory(prefix="eqiora-dispatch-tag-") as directory:
            def git(*arguments: str) -> str:
                return subprocess.check_output(
                    ["git", "-c", "user.name=CI test", "-c", "user.email=ci@example.invalid",
                     "-c", "commit.gpgsign=false", "-c", "tag.gpgsign=false", *arguments],
                    cwd=directory, text=True, stderr=subprocess.DEVNULL,
                ).strip()
            git("init")
            git("commit", "--allow-empty", "-m", "first")
            first = git("rev-parse", "HEAD")
            git("tag", "-a", "v1.0", "-m", "release")
            git("tag", "lightweight")
            git("commit", "--allow-empty", "-m", "second")
            second = git("rev-parse", "HEAD")
            for selected, tag, accepted in ((first, "v1.0", True),
                                             (first, "lightweight", False),
                                             (second, "v1.0", False)):
                git("checkout", "--detach", selected)
                result = subprocess.run(
                    ["bash", "--noprofile", "--norc", "-e", "-o", "pipefail", "-c",
                     "python3() { echo selected-code-ran >&2; printf '1.0'; };\n" + guard],
                    env={**os.environ, "RELEASE_COMMIT": selected, "GITHUB_SHA": selected,
                         "RELEASE_TAG": tag}, cwd=directory, capture_output=True, text=True,
                )
                with self.subTest(tag=tag, selected=selected):
                    self.assertEqual(result.returncode == 0, accepted, result.stderr)
                    self.assertEqual("selected-code-ran" in result.stderr, accepted)

    def test_only_exact_main_manual_run_can_save(self) -> None:
        cache = step(workflow("ci.yml"), "Restore Stable dependencies")
        expression = re.search(r"save-if: \$\{\{ (.*?) \}\}", cache).group(1)
        terms = expression.split(" && ")
        expected = {
            "github.event_name == 'workflow_dispatch'",
            "github.ref == 'refs/heads/main'",
            "needs.changes.outputs.target_sha == github.sha",
        }
        self.assertEqual(set(terms), expected)
        self.assertEqual(len(terms), len(expected))

        def accepts(event: str, ref: str, target: str, head: str) -> bool:
            values = {"github.event_name": event, "github.ref": ref,
                      "needs.changes.outputs.target_sha": target, "github.sha": head}
            def value(token: str) -> str:
                return token[1:-1] if token.startswith("'") else values[token]
            return all(value(left) == value(right) for left, right in
                       (term.split(" == ") for term in terms))

        for event, ref, target, expected_result in (
            ("workflow_dispatch", "refs/heads/main", "a", True),
            ("workflow_dispatch", "refs/heads/main", "b", False),
            ("workflow_dispatch", "refs/heads/candidate-review", "a", False),
            ("pull_request", "refs/pull/1/merge", "a", False),
            ("pull_request", "refs/heads/main", "a", False),
            ("push", "refs/heads/main", "a", False),
        ):
            with self.subTest(event=event, ref=ref, target=target):
                self.assertEqual(accepts(event, ref, target, "a"), expected_result)
        self.assertIn("cache-on-failure: false", cache)

    def test_dependency_cache_is_stable_only_and_isolates_fallback_keys(self) -> None:
        source = workflow("ci.yml")
        cache = step(source, "Restore Stable dependencies")
        self.assertEqual(source.count("Swatinem/rust-cache@"), 1)
        self.assertLess(source.index("  quality:"), source.index(cache))
        self.assertLess(source.index(cache), source.index("\n  msrv:\n"))
        self.assertIn("if: needs.changes.outputs.rust == 'true'", cache)
        self.assertIn("Swatinem/rust-cache@6323deb102c322ba6fcbdcafc7e3dddab59af2b6", cache)
        for setting in ("cache-bin: false", "cache-workspace-crates: false", "cache-all-crates: false",
                        "cmd-format: rustup run stable {0}", "workspaces: . -> target"):
            self.assertIn(setting, cache)
        prefix = next(line for line in cache.splitlines() if "prefix-key:" in line)
        for identity in ("steps.native.outputs.digest", "'.github/workflows/ci.yml'", "'Cargo.toml'",
                         "'tools/ci/rust_quality.py'", "'pyproject.toml'"):
            self.assertIn(identity, prefix)
        self.assertIn("env-vars: ImageOS ImageVersion pythonLocation Python_ROOT_DIR", cache)
        self.assertIn("dpkg-query -W", step(source, "Identify the native build environment"))

    def test_cache_does_not_override_existing_check_profiles_or_commands(self) -> None:
        source = workflow("ci.yml")
        restore = step(source, "Preserve non-test incremental compilation")
        self.assertIn("CARGO_INCREMENTAL=1", restore)
        self.assertLess(source.index("Restore Stable dependencies"), source.index(restore))
        self.assertLess(source.index(restore), source.index("      - name: Formatting"))
        for name in ("Tests", "Full feature tests"):
            body = step(source, name)
            for setting in ('CARGO_INCREMENTAL: "0"', 'CARGO_PROFILE_TEST_INCREMENTAL: "false"',
                            'CARGO_PROFILE_TEST_OPT_LEVEL: "1"', 'CARGO_PROFILE_TEST_DEBUG_ASSERTIONS: "true"',
                            'CARGO_PROFILE_TEST_OVERFLOW_CHECKS: "true"'):
                self.assertIn(setting, body)
        for name, command in {
            "Formatting": "cargo +stable fmt --all -- --check",
            "Clippy": "python tools/ci/rust_quality.py clippy",
            "Tests": "python tools/ci/rust_quality.py test",
            "Full feature tests": "cargo +stable test --workspace --all-targets --all-features --locked --timings",
            "Documentation": "python tools/ci/rust_quality.py doc",
        }.items():
            self.assertIn(f"run: {command}", step(source, name))
