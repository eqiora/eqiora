from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[4]
GENERATOR = REPOSITORY / "tools/docs/generate_interface_reference.py"
OUTPUTS = (
    Path("docs/site/src/content/docs/reference/cli/index.mdx"),
    Path("docs/site/src/content/docs/reference/control-v2/index.mdx"),
    Path("docs/site/src/content/docs/reference/mcp/index.mdx"),
)
FORBIDDEN_GIT_ENVIRONMENT = {
    "GIT_DIR",
    "GIT_WORK_TREE",
    "GIT_INDEX_FILE",
    "GIT_OBJECT_DIRECTORY",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_NAMESPACE",
    "GIT_CEILING_DIRECTORIES",
    "GIT_DISCOVERY_ACROSS_FILESYSTEM",
}
_MISSING = object()


def _run_command(
    arguments: list[str],
    *,
    cwd: Path | None = None,
    environment: dict[str, str] | None = None,
) -> str:
    child_environment = None
    if environment is not None:
        child_environment = os.environ.copy()
        child_environment.update(environment)
    completed = subprocess.run(
        arguments,
        cwd=cwd,
        env=child_environment,
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return completed.stdout.strip()


def _write(path: Path, payload: str, *, executable: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload, encoding="utf-8")
    if executable:
        path.chmod(0o755)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _extract_cli_output(page: str, command: str) -> str:
    match = re.search(
        rf"^\$ {re.escape(command)}\n(?P<output>.*?)\n```$",
        page,
        flags=re.MULTILINE | re.DOTALL,
    )
    if match is None:
        raise AssertionError(f"accepted CLI projection lacks {command}")
    return match.group("output") + "\n"


def _accepted_live_observations() -> tuple[dict[str, str], list[dict]]:
    cli_page = next(path for path in OUTPUTS if "/cli/" in path.as_posix())
    mcp_page = next(path for path in OUTPUTS if "/mcp/" in path.as_posix())
    cli_text = (REPOSITORY / cli_page).read_text(encoding="utf-8")
    cli = {
        "--version": _extract_cli_output(cli_text, "eqiora --version"),
        "--help": _extract_cli_output(cli_text, "eqiora --help"),
        "check --help": _extract_cli_output(cli_text, "eqiora check --help"),
    }
    response_blocks = re.findall(
        r"^Live response:\n\n```json\n(?P<response>.*?)\n```$",
        (REPOSITORY / mcp_page).read_text(encoding="utf-8"),
        flags=re.MULTILINE | re.DOTALL,
    )
    if len(response_blocks) != 2:
        raise AssertionError("accepted MCP projection must contain two live responses")
    return cli, [json.loads(block) for block in response_blocks]


class InterfaceReferenceFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.producer = root / "producer"
        self.observed = root / "observed"
        self.cli_marker = self.observed / "eqiora"
        self.mcp_marker = self.observed / "eqiora-mcp"
        self._populate(self.producer)
        self._commit(self.producer)
        self.source_sha = _run_command(
            ["git", "rev-parse", "HEAD"], cwd=self.producer
        )

    def _populate(self, target: Path) -> None:
        target.mkdir(parents=True)
        self.observed.mkdir(parents=True)
        _write(
            target / "Cargo.toml",
            '[workspace]\nmembers = []\n[workspace.package]\nversion = "0.1.0-alpha.11"\n',
        )
        copied = [
            Path("crates/eqiora-api/schemas/compile-v2.schema.json"),
            Path("verify/interfaces/mcp-stdio-compile-check/case.toml"),
            Path("verify/interfaces/mcp-stdio-compile-check/README.md"),
            *OUTPUTS,
        ]
        for relative in copied:
            destination = target / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(REPOSITORY / relative, destination)

        cli, mcp = _accepted_live_observations()
        cli_script = (
            f"#!{sys.executable}\n"
            "import json\n"
            "import os\n"
            "import pathlib\n"
            "import sys\n"
            "pathlib.Path(os.environ['EQIORA_I01_CAPTURE_ROOT'], 'eqiora').touch()\n"
            f"responses = json.loads({json.dumps(json.dumps(cli))})\n"
            "key = ' '.join(sys.argv[1:])\n"
            "if key not in responses:\n"
            "    raise SystemExit(2)\n"
            "sys.stdout.write(responses[key])\n"
        )
        mcp_script = (
            f"#!{sys.executable}\n"
            "import json\n"
            "import os\n"
            "import pathlib\n"
            "import sys\n"
            "pathlib.Path(os.environ['EQIORA_I01_CAPTURE_ROOT'], 'eqiora-mcp').touch()\n"
            f"responses = json.loads({json.dumps(json.dumps(mcp))})\n"
            "requests = [json.loads(line) for line in sys.stdin if line.strip()]\n"
            "if [request.get('id') for request in requests] != "
            "['docs-discover', 'docs-tools']:\n"
            "    raise SystemExit(2)\n"
            "for response in responses:\n"
            "    print(json.dumps(response, ensure_ascii=False, separators=(',', ':')))\n"
        )
        _write(target / "bin/eqiora", cli_script, executable=True)
        _write(target / "bin/eqiora-mcp", mcp_script, executable=True)

    @staticmethod
    def _commit(repository: Path) -> None:
        _run_command(["git", "init", "-q"], cwd=repository)
        _run_command(["git", "config", "user.name", "I-01 fixture"], cwd=repository)
        _run_command(
            ["git", "config", "user.email", "i-01-fixture@example.invalid"],
            cwd=repository,
        )
        _run_command(["git", "add", "."], cwd=repository)
        _run_command(
            [
                "git",
                "-c",
                "commit.gpgsign=false",
                "commit",
                "-qm",
                "exact interface fixture",
            ],
            cwd=repository,
            environment={
                "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+00:00",
                "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+00:00",
            },
        )

    def archive(self, name: str = "archive") -> Path:
        archive = self.root / name
        archive.mkdir()
        payload = subprocess.run(
            ["git", "archive", "--format=tar", "HEAD"],
            cwd=self.producer,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).stdout
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as members:
            members.extractall(archive, filter="data")
        if (archive / ".git").exists() or (archive / ".git").is_symlink():
            raise AssertionError("direct archive unexpectedly contains .git")
        return archive

    def clear_markers(self) -> None:
        for marker in (self.cli_marker, self.mcp_marker):
            marker.unlink(missing_ok=True)


class InterfaceReferenceArchiveIdentityTests(unittest.TestCase):
    def _snapshot(self, repository: Path) -> dict[Path, tuple[str, int]]:
        return {
            relative: (
                _sha256(repository / relative),
                (repository / relative).stat().st_mtime_ns,
            )
            for relative in OUTPUTS
        }

    def _invoke(
        self,
        fixture: InterfaceReferenceFixture,
        repository: Path,
        source_sha: object,
        *,
        environment_sha: object = _MISSING,
        environment: dict[str, str] | None = None,
    ) -> subprocess.CompletedProcess[str]:
        arguments = [
            sys.executable,
            str(GENERATOR),
            "--repository",
            str(repository),
            "--eqiora-binary",
            str(repository / "bin/eqiora"),
            "--mcp-binary",
            str(repository / "bin/eqiora-mcp"),
            "--check",
        ]
        if source_sha is not _MISSING:
            arguments.extend(("--source-sha", str(source_sha)))
        child_environment = os.environ.copy()
        for name in FORBIDDEN_GIT_ENVIRONMENT | {"EQIORA_SITE_SOURCE_SHA"}:
            child_environment.pop(name, None)
        child_environment["EQIORA_I01_CAPTURE_ROOT"] = str(fixture.observed)
        if environment_sha is not _MISSING:
            child_environment["EQIORA_SITE_SOURCE_SHA"] = str(environment_sha)
        if environment is not None:
            child_environment.update(environment)
        return subprocess.run(
            arguments,
            check=False,
            env=child_environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

    def _assert_positive(
        self,
        fixture: InterfaceReferenceFixture,
        repository: Path,
        source_sha: str,
        *,
        include_environment_sha: bool = True,
        environment: dict[str, str] | None = None,
    ) -> None:
        before = self._snapshot(repository)
        fixture.clear_markers()
        result = self._invoke(
            fixture,
            repository,
            source_sha,
            environment_sha=source_sha if include_environment_sha else _MISSING,
            environment=environment,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(fixture.cli_marker.is_file(), result.stderr)
        self.assertTrue(fixture.mcp_marker.is_file(), result.stderr)
        self.assertEqual(self._snapshot(repository), before)

    def _assert_identity_rejection(
        self,
        fixture: InterfaceReferenceFixture,
        repository: Path,
        source_sha: object,
        *,
        environment_sha: object = _MISSING,
        environment: dict[str, str] | None = None,
    ) -> None:
        before = self._snapshot(repository)
        fixture.clear_markers()
        result = self._invoke(
            fixture,
            repository,
            source_sha,
            environment_sha=environment_sha,
            environment=environment,
        )
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertFalse(fixture.cli_marker.exists(), result.stderr)
        self.assertFalse(fixture.mcp_marker.exists(), result.stderr)
        self.assertEqual(self._snapshot(repository), before)

    def test_00_direct_archive_positive_precedes_identity_mutants(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = InterfaceReferenceFixture(Path(temporary))
            self._assert_positive(fixture, fixture.archive(), fixture.source_sha)

    def test_01_genuine_worktree_positive_precedes_identity_mutants(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = InterfaceReferenceFixture(Path(temporary))
            self._assert_positive(fixture, fixture.producer, fixture.source_sha)

    def test_02_environment_absent_archive_and_worktree_are_admitted(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = InterfaceReferenceFixture(Path(temporary))
            for repository in (fixture.archive(), fixture.producer):
                with self.subTest(repository=repository.name):
                    self._assert_positive(
                        fixture,
                        repository,
                        fixture.source_sha,
                        include_environment_sha=False,
                    )

    def test_03_same_root_same_head_copied_metadata_is_worktree_mode(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = InterfaceReferenceFixture(Path(temporary))
            copied = fixture.archive("copied-metadata")
            shutil.copytree(fixture.producer / ".git", copied / ".git")
            self._assert_positive(fixture, copied, fixture.source_sha)

    def test_04_same_observation_gitfile_and_symlink_are_worktree_mode(self) -> None:
        for kind in ("gitfile", "symlink"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                fixture = InterfaceReferenceFixture(Path(temporary))
                admitted = fixture.archive(f"same-observation-{kind}")
                if kind == "gitfile":
                    _write(admitted / ".git", f"gitdir: {fixture.producer / '.git'}\n")
                else:
                    (admitted / ".git").symlink_to(
                        fixture.producer / ".git", target_is_directory=True
                    )
                self._assert_positive(
                    fixture,
                    admitted,
                    fixture.source_sha,
                    include_environment_sha=False,
                )

    def test_archive_source_sha_shape_and_environment_mismatch_fail_closed(self) -> None:
        invalid = (
            _MISSING,
            "A" * 40,
            "a" * 39,
            "a" * 41,
            "g" * 40,
            "\N{SNOWMAN}" * 40,
            " " + "a" * 40,
            "a" * 40 + " ",
            "a" * 20 + "\n" + "a" * 20,
            "HEAD",
            "main",
            "v0.1.0-alpha.1",
        )
        for source_sha in invalid:
            with (
                self.subTest(source_sha=source_sha),
                tempfile.TemporaryDirectory() as temporary,
            ):
                fixture = InterfaceReferenceFixture(Path(temporary))
                self._assert_identity_rejection(
                    fixture,
                    fixture.archive(),
                    source_sha,
                    environment_sha=fixture.source_sha,
                )
        with tempfile.TemporaryDirectory() as temporary:
            fixture = InterfaceReferenceFixture(Path(temporary))
            self._assert_identity_rejection(
                fixture,
                fixture.archive(),
                fixture.source_sha,
                environment_sha="f" * 40,
            )

    def test_archive_never_falls_back_to_parent_or_ambient_git(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = InterfaceReferenceFixture(Path(temporary))
            ambient = fixture.root / "ambient"
            ambient.mkdir()
            _run_command(["git", "init", "-q"], cwd=ambient)
            _run_command(["git", "config", "user.name", "ambient"], cwd=ambient)
            _run_command(
                ["git", "config", "user.email", "ambient@example.invalid"],
                cwd=ambient,
            )
            _write(ambient / "ambient.txt", "different ambient HEAD\n")
            _run_command(["git", "add", "."], cwd=ambient)
            _run_command(
                ["git", "-c", "commit.gpgsign=false", "commit", "-qm", "ambient"],
                cwd=ambient,
            )
            self.assertNotEqual(
                _run_command(["git", "rev-parse", "HEAD"], cwd=ambient),
                fixture.source_sha,
            )
            archive = fixture.archive("ambient/nested-archive")
            fake_bin = fixture.root / "fake-bin"
            fake_bin.mkdir()
            git_marker = fixture.observed / "git"
            _write(
                fake_bin / "git",
                f"#!{sys.executable}\n"
                "import os\n"
                "import pathlib\n"
                "pathlib.Path(os.environ['EQIORA_I01_CAPTURE_ROOT'], 'git').touch()\n"
                "raise SystemExit(97)\n",
                executable=True,
            )
            path = os.pathsep.join((str(fake_bin), os.environ.get("PATH", "")))
            self._assert_positive(
                fixture,
                archive,
                fixture.source_sha,
                environment={"PATH": path},
            )
            self.assertFalse(git_marker.exists(), "archive mode invoked Git")

    def test_worktree_head_and_environment_mismatches_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = InterfaceReferenceFixture(Path(temporary))
            different = "f" * 40
            self._assert_identity_rejection(
                fixture,
                fixture.producer,
                different,
                environment_sha=different,
            )
            self._assert_identity_rejection(
                fixture,
                fixture.producer,
                fixture.source_sha,
                environment_sha=different,
            )

    def test_inherited_git_settings_are_neutralized_or_rejected(self) -> None:
        values = {
            "GIT_DIR": "missing-git-dir",
            "GIT_WORK_TREE": "missing-work-tree",
            "GIT_INDEX_FILE": "missing-index",
            "GIT_OBJECT_DIRECTORY": "missing-objects",
            "GIT_ALTERNATE_OBJECT_DIRECTORIES": "missing-alternates",
            "GIT_NAMESPACE": "foreign-namespace",
            "GIT_CEILING_DIRECTORIES": "/",
            "GIT_DISCOVERY_ACROSS_FILESYSTEM": "1",
        }
        for name, value in values.items():
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                fixture = InterfaceReferenceFixture(Path(temporary))
                before = self._snapshot(fixture.producer)
                fixture.clear_markers()
                result = self._invoke(
                    fixture,
                    fixture.producer,
                    fixture.source_sha,
                    environment_sha=fixture.source_sha,
                    environment={name: value},
                )
                if result.returncode == 0:
                    self.assertTrue(fixture.cli_marker.is_file(), result.stderr)
                    self.assertTrue(fixture.mcp_marker.is_file(), result.stderr)
                else:
                    self.assertFalse(fixture.cli_marker.exists(), result.stderr)
                    self.assertFalse(fixture.mcp_marker.exists(), result.stderr)
                self.assertEqual(self._snapshot(fixture.producer), before)

    def test_missing_and_malformed_git_fail_before_product_capture(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = InterfaceReferenceFixture(Path(temporary))
            empty_path = fixture.root / "empty-path"
            empty_path.mkdir()
            self._assert_identity_rejection(
                fixture,
                fixture.producer,
                fixture.source_sha,
                environment_sha=fixture.source_sha,
                environment={"PATH": str(empty_path)},
            )

        with tempfile.TemporaryDirectory() as temporary:
            fixture = InterfaceReferenceFixture(Path(temporary))
            fake_bin = fixture.root / "fake-bin"
            fake_bin.mkdir()
            _write(
                fake_bin / "git",
                f"#!{sys.executable}\nprint('not canonical git output')\n",
                executable=True,
            )
            self._assert_identity_rejection(
                fixture,
                fixture.producer,
                fixture.source_sha,
                environment_sha=fixture.source_sha,
                environment={"PATH": str(fake_bin)},
            )

    def test_fake_git_top_level_and_unborn_head_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = InterfaceReferenceFixture(Path(temporary))
            fake_bin = fixture.root / "fake-bin"
            fake_bin.mkdir()
            fake_git = f"""#!{sys.executable}
import sys
if '--show-toplevel' in sys.argv:
    print('/not/the/provided/repository')
elif 'HEAD' in sys.argv:
    print({fixture.source_sha!r})
else:
    raise SystemExit(2)
"""
            _write(fake_bin / "git", fake_git, executable=True)
            self._assert_identity_rejection(
                fixture,
                fixture.producer,
                fixture.source_sha,
                environment_sha=fixture.source_sha,
                environment={"PATH": str(fake_bin)},
            )

        with tempfile.TemporaryDirectory() as temporary:
            fixture = InterfaceReferenceFixture(Path(temporary))
            unborn = fixture.archive("unborn")
            _run_command(["git", "init", "-q"], cwd=unborn)
            self._assert_identity_rejection(
                fixture,
                unborn,
                fixture.source_sha,
                environment_sha=fixture.source_sha,
            )

    def test_redirected_gitfile_and_symlink_mismatches_fail_closed(self) -> None:
        for kind in ("gitfile", "symlink"):
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as temporary:
                fixture = InterfaceReferenceFixture(Path(temporary))
                redirected = fixture.archive(f"redirected-{kind}")
                other = fixture.root / "other"
                other.mkdir()
                _run_command(["git", "init", "-q"], cwd=other)
                _run_command(["git", "config", "user.name", "other"], cwd=other)
                _run_command(
                    ["git", "config", "user.email", "other@example.invalid"],
                    cwd=other,
                )
                _write(other / "other.txt", "other\n")
                _run_command(["git", "add", "."], cwd=other)
                _run_command(["git", "commit", "-qm", "other"], cwd=other)
                if kind == "gitfile":
                    _write(redirected / ".git", f"gitdir: {other / '.git'}\n")
                else:
                    (redirected / ".git").symlink_to(
                        other / ".git", target_is_directory=True
                    )
                self._assert_identity_rejection(
                    fixture,
                    redirected,
                    fixture.source_sha,
                    environment_sha=fixture.source_sha,
                )


if __name__ == "__main__":
    unittest.main()
