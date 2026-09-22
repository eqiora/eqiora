from __future__ import annotations

import contextlib
import hashlib
import importlib
import inspect
import io
import os
import re
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock


CI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CI_ROOT))

from local_verify import (  # noqa: E402
    PlannedCommand,
    ResourceBudget,
    ResourceRequest,
    VerificationFailure,
    VerificationLane,
    VerificationPlan,
    WorkspacePackage,
    build_plan,
    cli_feature_args,
    direct_packages,
    local_changed_paths,
    reverse_dependency_closure,
    run_plan,
)
from classify_changes import impact_plan  # noqa: E402
from check_docs import check as check_docs  # noqa: E402
from verification_scheduler import (  # noqa: E402
    _available_memory_mib,
    cpu_allocations,
)


EVIDENCE_RUN_PREFIX = (
    "cargo",
    "run",
    "--locked",
    "-p",
    "eqiora-verify",
    "--",
    "run",
)


def evidence_runs(plan: VerificationPlan) -> list[PlannedCommand]:
    return [
        item
        for item in plan.commands
        if item.argv[: len(EVIDENCE_RUN_PREFIX)] == EVIDENCE_RUN_PREFIX
    ]


def workspace() -> dict[str, WorkspacePackage]:
    return {
        "eqiora-core": WorkspacePackage(
            "eqiora-core", "crates/eqiora-core", frozenset()
        ),
        "eqiora-compiler": WorkspacePackage(
            "eqiora-compiler", "crates/eqiora-compiler", frozenset({"eqiora-core"})
        ),
        "eqiora": WorkspacePackage(
            "eqiora", "crates/eqiora", frozenset({"eqiora-compiler"})
        ),
        "xtask": WorkspacePackage("xtask", "tools/xtask", frozenset()),
    }


class PackageSelectionTests(unittest.TestCase):
    def test_cli_feature_is_explicit_for_facade_product_checks(self):
        self.assertEqual(
            cli_feature_args(["eqiora"]), ("--features", "eqiora/cli")
        )
        self.assertEqual(cli_feature_args(["eqiora-core"]), ())

    def test_local_change_set_unions_every_git_state(self) -> None:
        with mock.patch(
            "local_verify._git_paths",
            side_effect=[
                {"committed"},
                {"unstaged", "shared"},
                {"staged", "shared"},
                {"untracked"},
            ],
        ):
            self.assertEqual(
                local_changed_paths("base"),
                ("committed", "shared", "staged", "unstaged", "untracked"),
            )

    def test_direct_owner_and_root_manifest_are_fail_closed(self) -> None:
        packages = workspace()
        self.assertEqual(
            direct_packages(["crates/eqiora-core/src/lib.rs"], packages),
            {"eqiora-core"},
        )
        self.assertEqual(direct_packages(["Cargo.lock"], packages), set(packages))

    def test_reverse_dependency_closure_is_transitive(self) -> None:
        self.assertEqual(
            reverse_dependency_closure({"eqiora-core"}, workspace()),
            {"eqiora-core", "eqiora-compiler", "eqiora"},
        )


class PlanTests(unittest.TestCase):
    def test_local_planner_consumes_the_shared_impact_plan_owner(self) -> None:
        with mock.patch("local_verify.impact_plan", wraps=impact_plan) as owner:
            plan = build_plan(
                "fast",
                ["crates/eqiora-core/src/lib.rs"],
                [],
                workspace(),
            )

        owner.assert_called_once_with(
            ["crates/eqiora-core/src/lib.rs"],
            target_authority="local-worktree",
            base_authority="local-change-set",
        )
        self.assertIn("eqiora-core", plan.packages)

    def test_affected_does_not_automatically_run_changed_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "verify" / "interfaces" / "live" / "case.toml"
            manifest.parent.mkdir(parents=True)
            manifest.write_text('id = "interfaces.live"\n', encoding="utf-8")
            plan = build_plan(
                "affected",
                [
                    "verify/interfaces/live/README.md",
                    "verify/interfaces/deleted/README.md",
                ],
                [],
                workspace(),
                root,
            )
        self.assertEqual(plan.cases, ())
        self.assertFalse(
            any(item.label.startswith("Registered evidence") for item in plan.commands)
        )

    def test_periodic_msrv_checks_every_production_feature(self) -> None:
        plan = build_plan("periodic", [], [], workspace())
        msrv = next(item for item in plan.commands if item.label == "MSRV")
        self.assertEqual(msrv.argv[0:2], ("cargo", "+1.89.0"))
        self.assertIn("--all-targets", msrv.argv)
        self.assertIn("--all-features", msrv.argv)
        self.assertIn("--locked", msrv.argv)

    def test_periodic_runs_only_host_cpu_evidence(self) -> None:
        plan = build_plan("periodic", [], [], workspace())
        evidence = next(
            item for item in plan.commands if item.label == "All registered evidence"
        )
        self.assertEqual(
            evidence.argv,
            (*EVIDENCE_RUN_PREFIX, "--environment", "host-cpu"),
        )
        self.assertIn(
            "Physical MPI-CUDA, multi-node MPI, and GPU evidence requires an explicit matching environment run.",
            plan.limitations,
        )

    def test_site_only_changes_select_source_check_without_building(self) -> None:
        expected = (
            sys.executable, "tools/site/check_site.py", "source", "--root", "."
        )
        for path in (
            "docs/site/src/content/docs/index.mdx",
            "docs/site/src/styles/tokens.css",
            "docs/site/public/favicon.svg",
            "tools/site/check_site.py",
        ):
            for tier in ("fast", "affected"):
                with self.subTest(path=path, tier=tier):
                    plan = build_plan(tier, [path], [], workspace())
                    self.assertEqual(plan.packages, ())
                    self.assertEqual(
                        [item.argv for item in plan.commands if item.label == "Site source"],
                        [expected],
                    )
                    if tier == "fast":
                        self.assertEqual([item.argv for item in plan.commands], [expected])
        periodic = build_plan("periodic", [], [], workspace())
        self.assertEqual(
            [item.argv for item in periodic.commands if item.label == "Site source"],
            [expected],
        )
        pr = build_plan("pr", ["docs/site/src/content/docs/index.mdx"], [], workspace())
        self.assertFalse(any(item.label == "Site source" for item in pr.commands))

    def test_fast_plan_keeps_direct_package_and_explicit_case(self) -> None:
        plan = build_plan(
            "fast",
            ["crates/eqiora-core/src/lib.rs"],
            ["language.explicit"],
            workspace(),
        )
        self.assertEqual(plan.packages, ("eqiora-core",))
        self.assertEqual(plan.cases, ("language.explicit",))
        rendered = [item.render() for item in plan.commands]
        self.assertTrue(
            any("cargo test --locked -p eqiora-core" in item for item in rendered)
        )
        evidence = evidence_runs(plan)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].label, "Registered evidence (1 case)")
        self.assertEqual(
            evidence[0].argv,
            (*EVIDENCE_RUN_PREFIX, "--case", "language.explicit"),
        )

    def test_pr_plan_keeps_direct_scope_and_defers_auxiliary_checks(self) -> None:
        plan = build_plan(
            "pr",
            [
                "crates/eqiora-core/src/lib.rs",
                "docs/roadmap.md",
                "tools/ci/local_verify.py",
            ],
            ["language.explicit"],
            workspace(),
        )
        self.assertEqual(plan.packages, ("eqiora-core",))
        self.assertEqual(plan.cases, ("language.explicit",))
        labels = {item.label for item in plan.commands}
        self.assertIn("Formatting", labels)
        self.assertIn("Rust tests", labels)
        self.assertIn("Rust Clippy", labels)
        self.assertNotIn("Documentation contract", labels)
        self.assertNotIn("Public release tree", labels)
        self.assertNotIn("Dependency layers", labels)
        self.assertNotIn("Public facade", labels)
        self.assertNotIn("CI contract tests", labels)
        self.assertNotIn("Rustdoc", labels)
        self.assertEqual(len(evidence_runs(plan)), 1)
        self.assertTrue(any("deferred" in item for item in plan.limitations))

    def test_pr_rust_commands_use_default_targets(self) -> None:
        plan = build_plan(
            "pr",
            ["crates/eqiora-core/src/lib.rs"],
            [],
            workspace(),
        )
        rendered = [item.render() for item in plan.commands]
        self.assertTrue(
            any("cargo test --locked -p eqiora-core" in item for item in rendered)
        )
        self.assertFalse(any("--all-targets" in item for item in rendered))

    def test_fast_rust_commands_keep_all_targets(self) -> None:
        plan = build_plan(
            "fast",
            ["crates/eqiora-core/src/lib.rs"],
            [],
            workspace(),
        )
        rendered = [item.render() for item in plan.commands]
        self.assertTrue(
            any(
                "cargo test --locked -p eqiora-core --all-targets" in item
                for item in rendered
            )
        )

    def test_affected_missing_chrome_defers_only_interaction_tests(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            absent = Path(scratch) / "chrome"
            plan = build_plan(
                "affected",
                ["studio/src/main.ts"],
                [],
                workspace(),
                chrome_executable=absent,
            )
        labels = {item.label for item in plan.commands}
        self.assertNotIn("Studio interaction tests", labels)
        self.assertIn("Studio unit tests", labels)
        self.assertIn("Studio build", labels)
        self.assertFalse(any(label.startswith("Studio native") for label in labels))
        self.assertTrue(any("hosted Studio lane" in item for item in plan.limitations))

    def test_affected_present_chrome_keeps_interaction_tests(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            present = Path(scratch) / "chrome"
            present.touch()
            plan = build_plan(
                "affected",
                ["studio/src/main.ts"],
                [],
                workspace(),
                chrome_executable=present,
            )
        labels = {item.label for item in plan.commands}
        self.assertIn("Studio interaction tests", labels)
        self.assertFalse(any("hosted Studio lane" in item for item in plan.limitations))

    def test_selected_cases_share_one_canonical_runner_invocation(self) -> None:
        first = build_plan(
            "fast",
            [],
            ["language.second", "language.first", "language.second"],
            workspace(),
        )
        second = build_plan(
            "fast",
            [],
            ["language.first", "language.second"],
            workspace(),
        )

        self.assertEqual(first.cases, ("language.first", "language.second"))
        evidence = evidence_runs(first)
        self.assertEqual(len(evidence), 1)
        self.assertEqual(evidence[0].label, "Registered evidence (2 cases)")
        self.assertEqual(
            evidence[0].argv,
            (
                *EVIDENCE_RUN_PREFIX,
                "--case",
                "language.first",
                "--case",
                "language.second",
            ),
        )
        self.assertEqual(first, second)

    def test_candidate_case_serializes_only_its_ci_contracts(self) -> None:
        candidate_plan = build_plan(
            "affected",
            ["tools/ci/local_verify.py"],
            ["interfaces.python-distribution-candidate"],
            workspace(),
        )
        candidate_evidence = evidence_runs(candidate_plan)[0]
        candidate_ci = next(
            item
            for item in candidate_plan.commands
            if item.label == "CI contract tests"
        )
        self.assertEqual(candidate_evidence.lane.name, "root-cargo")
        self.assertIs(candidate_ci.lane, candidate_evidence.lane)

        other_plan = build_plan(
            "affected",
            ["tools/ci/local_verify.py"],
            ["language.explicit"],
            workspace(),
        )
        other_evidence = evidence_runs(other_plan)[0]
        other_ci = next(
            item for item in other_plan.commands if item.label == "CI contract tests"
        )
        self.assertEqual(other_ci.lane.name, "repository")
        self.assertNotEqual(other_ci.lane, other_evidence.lane)

    def test_no_selected_case_schedules_no_runner_invocation(self) -> None:
        plan = build_plan("fast", [], [], workspace())
        self.assertEqual(evidence_runs(plan), [])

    def test_affected_plan_expands_packages_and_adds_rustdoc(self) -> None:
        plan = build_plan(
            "affected",
            ["crates/eqiora-core/src/lib.rs"],
            ["language.explicit"],
            workspace(),
        )
        self.assertEqual(plan.packages, ("eqiora", "eqiora-compiler", "eqiora-core"))
        self.assertEqual(plan.cases, ("language.explicit",))
        self.assertTrue(any(item.label == "Rustdoc" for item in plan.commands))
        self.assertTrue(
            any(item.label == "Evidence manifest inventory" for item in plan.commands)
        )
        rendered = [item.render() for item in plan.commands]
        self.assertFalse(any("--all-features" in item for item in rendered))
        self.assertTrue(any("--case language.explicit" in item for item in rendered))

    def test_affected_checks_isolated_experiment_for_shared_dependency_changes(self) -> None:
        for path in (
            "Cargo.toml",
            "crates/eqiora-compiler/Cargo.toml",
            "crates/eqiora-compiler/src/units/exact_time.rs",
            "crates/eqiora-numerics/src/lib.rs",
        ):
            with self.subTest(path=path):
                plan = build_plan("affected", [path], [], workspace())
                commands = {item.label: item for item in plan.commands}
                for label in ("CubeCL Clippy", "CubeCL tests"):
                    command = commands[label]
                    self.assertIn("--locked", command.argv)
                    self.assertIn(
                        "experiments/cubecl-local-action/Cargo.toml", command.argv
                    )
                    self.assertEqual(command.lane.name, "cubecl")

    def test_affected_plan_does_not_infer_semantic_cases_from_executor_crates(
        self,
    ) -> None:
        plan = build_plan(
            "affected",
            ["crates/eqiora-core/src/lib.rs"],
            [],
            workspace(),
        )
        self.assertEqual(plan.cases, ())
        labels = {item.label for item in plan.commands}
        self.assertIn("Evidence manifest inventory", labels)
        self.assertNotIn("All registered evidence", labels)
        self.assertFalse(
            any(label.startswith("Registered evidence ") for label in labels)
        )

    def assert_documentation_contract(
        self, tier: str, changed_paths: list[str]
    ) -> None:
        checker_paths = {
            "tools/ci/check_docs.py",
            "tools/ci/check_public_release_tree.py",
        }
        plan = build_plan(tier, changed_paths, [], workspace())
        documentation_checks = tuple(
            (index, item)
            for index, item in enumerate(plan.commands)
            if len(item.argv) > 1 and item.argv[1] in checker_paths
        )
        self.assertEqual(
            tuple(item.argv for _index, item in documentation_checks),
            (
                (sys.executable, "tools/ci/check_docs.py", "."),
                (sys.executable, "tools/ci/check_public_release_tree.py", "."),
            ),
        )
        docs_index, docs_check = documentation_checks[0]
        public_tree_index, public_tree_check = documentation_checks[1]
        self.assertEqual(public_tree_index, docs_index + 1)
        self.assertEqual(
            (docs_check.label, public_tree_check.label),
            ("Documentation contract", "Public release tree"),
        )
        self.assertEqual(docs_check.lane.name, "repository")
        self.assertIs(public_tree_check.lane, docs_check.lane)

    def test_fast_documentation_contract_runs_both_hosted_checks_once(self) -> None:
        self.assert_documentation_contract("fast", ["README.md"])

    def test_affected_documentation_contract_runs_both_hosted_checks_once(self) -> None:
        self.assert_documentation_contract("affected", ["README.md"])

    def test_periodic_documentation_contract_runs_both_hosted_checks_once(
        self,
    ) -> None:
        self.assert_documentation_contract("periodic", [])

    def test_ci_infrastructure_change_reuses_fail_closed_surface_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as scratch:
            chrome = Path(scratch) / "chrome"
            chrome.touch()
            plan = build_plan(
                "affected",
                ["tools/ci/local_verify.py"],
                [],
                workspace(),
                chrome_executable=chrome,
            )
        labels = {item.label for item in plan.commands}
        self.assertEqual(plan.packages, tuple(sorted(workspace())))
        self.assertIn("CI contract tests", labels)
        self.assertIn("Root dependency policy", labels)
        self.assertIn("Python isolated wheel and tests", labels)
        self.assertIn("Studio unit tests", labels)
        studio_e2e = next(
            item for item in plan.commands if item.label == "Studio interaction tests"
        )
        self.assertEqual(studio_e2e.argv[-2:], ("--", "--workers=1"))
        self.assertIn("Evidence manifest inventory", labels)
        self.assertNotIn("All registered evidence", labels)
        self.assertFalse(
            any(label.startswith("Registered evidence ") for label in labels)
        )
        python_commands = [
            item for item in plan.commands if item.label.startswith("Python ")
        ]
        self.assertEqual(len(python_commands), 1)
        self.assertEqual(
            {item.argv[0] for item in python_commands},
            {sys.executable},
        )
        self.assertFalse(
            any("maturin develop" in item.render() for item in python_commands)
        )

    def test_jupyter_changes_use_existing_python_lane_for_assets_and_wheel(
        self,
    ) -> None:
        plan = build_plan(
            "affected", ["editor/jupyter/src/language.ts"], [], workspace()
        )
        commands = {item.label: item for item in plan.commands}
        assets = commands["Jupyter prebuilt assets"]
        wheel = commands["Python isolated wheel and tests"]
        self.assertEqual(assets.argv, (sys.executable, "tools/editor/check_jupyter.py"))
        self.assertIs(assets.lane, wheel.lane)
        self.assertNotIn("Studio unit tests", commands)

    def test_unrelated_python_changes_skip_jupyter_asset_rebuild(self) -> None:
        plan = build_plan(
            "affected", ["bindings/python/python/eqiora/__init__.py"], [], workspace()
        )
        labels = {item.label for item in plan.commands}
        self.assertIn("Python isolated wheel and tests", labels)
        self.assertNotIn("Jupyter prebuilt assets", labels)

    def test_studio_npm_lock_change_runs_browser_checks_only(self) -> None:
        plan = build_plan(
            "affected",
            ["studio/package-lock.json"],
            [],
            workspace(),
        )
        labels = {item.label for item in plan.commands}
        self.assertIn("Studio unit tests", labels)
        self.assertFalse(any(label.endswith("dependency policy") for label in labels))

    def test_adapter_dependency_identity_inputs_schedule_ci_contracts(self) -> None:
        inputs = [
            "Cargo.toml",
            "Cargo.lock",
            "pyproject.toml",
            "crates/eqiora-backend-faer/Cargo.toml",
            "crates/eqiora-backend-faer/src/lib.rs",
            "crates/eqiora-backend-rayon/Cargo.toml",
            "crates/eqiora-backend-rayon/src/lib.rs",
        ]
        for path in inputs:
            with self.subTest(path=path):
                plan = build_plan("affected", [path], [], workspace())
                labels = {item.label for item in plan.commands}
                self.assertIn("CI contract tests", labels)

    def test_ci_infrastructure_dependency_policies_share_one_lane(self) -> None:
        plan = build_plan(
            "affected",
            ["tools/ci/local_verify.py"],
            [],
            workspace(),
        )
        lanes = {item.label: item.lane for item in plan.commands}
        self.assertEqual(lanes["Root dependency policy"].name, "dependency-policy")
        self.assertEqual(lanes["Rust tests"].name, "root-cargo")
        self.assertEqual(lanes["Studio unit tests"].name, "studio")
        self.assertNotIn(
            lanes["Root dependency policy"].name,
            {lanes["Rust tests"].name, lanes["Studio unit tests"].name},
        )


class Issue496UnixSocketScratchTests(unittest.TestCase):
    SYNTHETIC_HOME = Path("/home/user1")
    PRIMARY_SCRATCH_ALLOCATIONS = 9
    RECEIPT_ALLOCATIONS = 21
    TOTAL_SHARED_INCREMENTS = PRIMARY_SCRATCH_ALLOCATIONS + RECEIPT_ALLOCATIONS
    MAX_SHARED_SEQUENCE = TOTAL_SHARED_INCREMENTS - 1
    MAX_PROCESS_ID = 4294967295
    SOCKET_RELATIVE = (
        Path(f"eqiora-cli-filesystem-{MAX_PROCESS_ID}-{MAX_SHARED_SEQUENCE}") / "socket"
    )
    SOCKET_SUFFIX = "/" + SOCKET_RELATIVE.as_posix()
    SOCKET_SUFFIX_SHA256 = (
        "bcf9ca8b033c44156fde8731b73e13f2c50ac9dbfa55e1167e8b49898d6cab8e"
    )
    NO_CHILD_DIAGNOSTIC = (
        "scheduler dispatched a child before Unix-socket scratch admission completed"
    )
    UNIX_PATHNAME_MAX = 107
    TMPDIR_ADMITTED_MAX = UNIX_PATHNAME_MAX - len(os.fsencode(SOCKET_SUFFIX))

    @staticmethod
    def plan(*commands: PlannedCommand) -> VerificationPlan:
        return VerificationPlan("affected", (), (), (), commands, ())

    @staticmethod
    def lane(name: str) -> VerificationLane:
        return VerificationLane(name, ResourceRequest(1, 1))

    @classmethod
    def socket_candidate(cls, tmpdir: Path) -> Path:
        return tmpdir / cls.SOCKET_RELATIVE

    @classmethod
    def utf8_boundary_paths(cls) -> tuple[Path, Path, Path]:
        configured_root = cls.SYNTHETIC_HOME / "é"
        padding = (
            cls.TMPDIR_ADMITTED_MAX
            - len(os.fsencode(configured_root))
            - len(os.fsencode("/"))
        )
        tmpdir_107 = configured_root / ("a" * padding)
        tmpdir_108 = configured_root / ("a" * (padding + 1))
        return configured_root, tmpdir_107, tmpdir_108

    @staticmethod
    def filesystem_snapshot(root: Path) -> tuple[tuple[str, str], ...]:
        entries = []
        for path in root.rglob("*"):
            kind = (
                "symlink" if path.is_symlink() else "dir" if path.is_dir() else "file"
            )
            entries.append((path.relative_to(root).as_posix(), kind))
        return tuple(sorted(entries))

    def assert_no_child(self, child: mock.Mock) -> None:
        self.assertEqual(child.call_count, 0, self.NO_CHILD_DIAGNOSTIC)

    @classmethod
    def real_boundary_tmpdirs(cls, authority: Path) -> tuple[Path, Path]:
        authority_characters = len(str(authority))
        authority_bytes = len(os.fsencode(authority))
        if authority_characters != authority_bytes:
            raise AssertionError("boundary authority must have an ASCII spelling")
        padding = cls.TMPDIR_ADMITTED_MAX - authority_bytes - len(os.fsencode("/é"))
        if padding < 0:
            raise AssertionError("home-backed authority is too long for C107")
        tmpdir_107 = authority / ("é" + "a" * padding)
        tmpdir_108 = authority / ("é" + "a" * (padding + 1))
        return tmpdir_107, tmpdir_108

    def test_00_default_path_plan_is_host_neutral_and_bounded(self) -> None:
        observed: list[dict[str, str]] = []

        def execute(
            argv: tuple[str, ...], **kwargs: object
        ) -> subprocess.CompletedProcess[bytes]:
            observed.append(dict(kwargs["env"]))
            return subprocess.CompletedProcess(argv, 0)

        plan = self.plan(
            PlannedCommand(
                "default path witness",
                ("default-path-witness",),
                lane=self.lane("root-cargo"),
            )
        )
        real_home = Path.home().resolve()
        with tempfile.TemporaryDirectory(prefix="h", dir=real_home) as home_directory:
            home = Path(home_directory)
            with mock.patch.dict(os.environ, {"HOME": str(home)}):
                worktree = home / ("w" * 120)
                worktree.mkdir()
                with mock.patch(
                    "verification_scheduler.subprocess.run", side_effect=execute
                ) as child:
                    run_plan(plan, worktree, budget=ResourceBudget(1, 1))

                child.assert_called_once()
                self.assertEqual(len(observed), 1)
                tmpdir = Path(observed[0]["TMPDIR"])
                self.assertTrue(tmpdir.is_absolute())
                self.assertTrue(tmpdir.resolve().is_relative_to(home.resolve()))
                self.assertFalse(tmpdir.exists())
                self.assertNotIn(worktree.name, str(tmpdir))
                lexical_relative = tmpdir.relative_to(home)
                synthetic_tmpdir = self.SYNTHETIC_HOME / lexical_relative
                candidate = self.socket_candidate(synthetic_tmpdir)
                self.assertEqual(len(os.fsencode(self.SYNTHETIC_HOME)), 11)
                self.assertLessEqual(
                    len(os.fsencode(candidate)),
                    self.UNIX_PATHNAME_MAX,
                    f"default TMPDIR shape leaves a "
                    f"{len(os.fsencode(candidate))}-byte Unix-socket candidate",
                )

    def test_01_default_positive_binds_once_and_cleans_tmpdir(self) -> None:
        observed_tmpdirs: list[Path] = []

        def bind_socket(
            argv: tuple[str, ...], **kwargs: object
        ) -> subprocess.CompletedProcess[bytes]:
            tmpdir = Path(dict(kwargs["env"])["TMPDIR"])
            observed_tmpdirs.append(tmpdir)
            self.assertTrue(tmpdir.exists())
            socket_path = self.socket_candidate(tmpdir)
            socket_path.parent.mkdir()
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
                listener.bind(str(socket_path))
            return subprocess.CompletedProcess(argv, 0)

        plan = self.plan(
            PlannedCommand(
                "default socket bind",
                ("bind-default-socket",),
                lane=self.lane("root-cargo"),
            )
        )
        real_home = Path.home().resolve()
        with tempfile.TemporaryDirectory(prefix="h", dir=real_home) as home_directory:
            home = Path(home_directory)
            with mock.patch.dict(os.environ, {"HOME": str(home)}):
                worktree = home / ("w" * 120)
                worktree.mkdir()
                with mock.patch(
                    "verification_scheduler.subprocess.run", side_effect=bind_socket
                ) as child:
                    run_plan(plan, worktree, budget=ResourceBudget(1, 1))

                child.assert_called_once()
                self.assertEqual(len(observed_tmpdirs), 1)
                tmpdir = observed_tmpdirs[0]
                self.assertTrue(tmpdir.resolve().is_relative_to(home.resolve()))
                self.assertLessEqual(
                    len(os.fsencode(self.socket_candidate(tmpdir))),
                    self.UNIX_PATHNAME_MAX,
                )
                self.assertFalse(tmpdir.exists())

    def test_02_two_lanes_get_disjoint_tmp_and_persistent_cargo_roots(self) -> None:
        barrier = threading.Barrier(2)
        environments: dict[str, dict[str, str]] = {}
        environment_lock = threading.Lock()

        def execute(
            argv: tuple[str, ...], **kwargs: object
        ) -> subprocess.CompletedProcess[bytes]:
            environment = dict(kwargs["env"])
            with environment_lock:
                environments[argv[0]] = environment
            self.assertTrue(Path(environment["TMPDIR"]).exists())
            barrier.wait(timeout=1.0)
            return subprocess.CompletedProcess(argv, 0)

        plan = self.plan(
            PlannedCommand("first lane", ("first",), lane=self.lane("first")),
            PlannedCommand("second lane", ("second",), lane=self.lane("second")),
        )
        with (
            tempfile.TemporaryDirectory(prefix="e496-", dir=Path.home()) as directory,
            mock.patch("verification_scheduler.subprocess.run", side_effect=execute),
        ):
            scratch_root = Path(directory)
            run_plan(
                plan,
                scratch_root,
                budget=ResourceBudget(2, 2),
                scratch_root=scratch_root,
            )
            self.assertEqual(set(environments), {"first", "second"})
            tmpdirs = {Path(env["TMPDIR"]) for env in environments.values()}
            lane_roots = {
                Path(env["EQIORA_VERIFY_LANE_ROOT"]) for env in environments.values()
            }
            cargo_roots = {
                Path(env["CARGO_TARGET_DIR"]) for env in environments.values()
            }
            self.assertEqual(len(tmpdirs), 2)
            self.assertEqual(len(lane_roots), 2)
            self.assertEqual(len(cargo_roots), 2)
            self.assertTrue(all(not path.exists() for path in tmpdirs))
            self.assertTrue(all(path.exists() for path in cargo_roots))
            for environment in environments.values():
                lane_root = Path(environment["EQIORA_VERIFY_LANE_ROOT"])
                self.assertTrue(lane_root.is_relative_to(scratch_root))
                self.assertTrue(
                    Path(environment["TMPDIR"]).is_relative_to(scratch_root)
                )
                self.assertTrue(
                    Path(environment["CARGO_TARGET_DIR"]).is_relative_to(lane_root)
                )

    def test_03_overlapping_same_lane_invocations_keep_distinct_tmpdirs(self) -> None:
        observed_tmpdirs: dict[str, Path] = {}
        observed_lock = threading.Lock()
        first_started = threading.Event()
        second_started = threading.Event()
        release_first = threading.Event()
        release_second = threading.Event()

        def execute(
            argv: tuple[str, ...], **kwargs: object
        ) -> subprocess.CompletedProcess[bytes]:
            tmpdir = Path(dict(kwargs["env"])["TMPDIR"])
            with observed_lock:
                observed_tmpdirs[argv[0]] = tmpdir
            self.assertTrue(tmpdir.exists())
            if argv[0] == "first":
                first_started.set()
                self.assertTrue(second_started.wait(timeout=2.0))
                self.assertTrue(release_first.wait(timeout=2.0))
            else:
                second_started.set()
                self.assertTrue(first_started.wait(timeout=2.0))
                self.assertTrue(release_second.wait(timeout=2.0))
            return subprocess.CompletedProcess(argv, 0)

        lane = self.lane("shared")
        first_plan = self.plan(
            PlannedCommand("shared lane first", ("first",), lane=lane)
        )
        second_plan = self.plan(
            PlannedCommand("shared lane second", ("second",), lane=lane)
        )
        with (
            tempfile.TemporaryDirectory(prefix="e496-", dir=Path.home()) as directory,
            mock.patch("verification_scheduler.subprocess.run", side_effect=execute),
        ):
            scratch_root = Path(directory)

            def invoke(plan: VerificationPlan) -> None:
                run_plan(
                    plan,
                    scratch_root,
                    budget=ResourceBudget(1, 1),
                    scratch_root=scratch_root,
                )

            with ThreadPoolExecutor(max_workers=2) as pool:
                second_future = pool.submit(invoke, second_plan)
                first_future = pool.submit(invoke, first_plan)
                try:
                    self.assertTrue(first_started.wait(timeout=2.0))
                    self.assertTrue(second_started.wait(timeout=2.0))
                    with observed_lock:
                        self.assertEqual(set(observed_tmpdirs), {"first", "second"})
                        first_tmpdir = observed_tmpdirs["first"]
                        second_tmpdir = observed_tmpdirs["second"]
                    self.assertNotEqual(first_tmpdir, second_tmpdir)
                    self.assertTrue(first_tmpdir.exists())
                    self.assertTrue(second_tmpdir.exists())

                    release_first.set()
                    first_future.result(timeout=2.0)
                    self.assertFalse(first_tmpdir.exists())
                    self.assertFalse(second_future.done())
                    self.assertTrue(second_tmpdir.exists())
                    live_witness = second_tmpdir / "still-live"
                    live_witness.write_text("second", encoding="ascii")
                    self.assertEqual(live_witness.read_text(encoding="ascii"), "second")
                finally:
                    release_first.set()
                    release_second.set()

                second_future.result(timeout=2.0)
                self.assertFalse(second_tmpdir.exists())

    def test_04_shared_sequence_domain_and_old_default_shape_exceed_the_frozen_socket_budget(
        self,
    ) -> None:
        self.assertEqual(self.PRIMARY_SCRATCH_ALLOCATIONS, 9)
        self.assertEqual(self.RECEIPT_ALLOCATIONS, 21)
        self.assertEqual(
            self.TOTAL_SHARED_INCREMENTS,
            self.PRIMARY_SCRATCH_ALLOCATIONS + self.RECEIPT_ALLOCATIONS,
        )
        self.assertEqual(self.TOTAL_SHARED_INCREMENTS, 30)
        self.assertEqual(self.MAX_SHARED_SEQUENCE, self.TOTAL_SHARED_INCREMENTS - 1)
        self.assertEqual(self.MAX_SHARED_SEQUENCE, 29)
        self.assertEqual(self.MAX_PROCESS_ID, 4294967295)
        self.assertEqual(
            self.SOCKET_RELATIVE.as_posix(),
            "eqiora-cli-filesystem-4294967295-29/socket",
        )
        socket_suffix = os.fsencode(self.SOCKET_SUFFIX)
        self.assertEqual(len(socket_suffix), 43)
        self.assertEqual(
            hashlib.sha256(socket_suffix).hexdigest(), self.SOCKET_SUFFIX_SHA256
        )
        self.assertEqual(
            self.TMPDIR_ADMITTED_MAX + len(socket_suffix), self.UNIX_PATHNAME_MAX
        )

        home = self.SYNTHETIC_HOME
        base = home / ".cache" / "eqiora" / "local-verify" / ("0" * 16)
        old_tmpdir = base / "lanes" / f"root-cargo-{'0' * 8}" / "tmp" / f"run-{'0' * 8}"
        candidate = self.socket_candidate(old_tmpdir)
        self.assertEqual(len(os.fsencode(old_tmpdir)), 98)
        self.assertEqual(len(os.fsencode(candidate)), 141)
        self.assertGreater(len(os.fsencode(candidate)), self.UNIX_PATHNAME_MAX)

        sentinel_key = "EQIORA_ISSUE496_DIAGNOSTIC_SENTINEL"
        sentinel_value = "scrubbed-environment-value"
        sentinel_environment = {sentinel_key: sentinel_value, "SAFE": "ordinary"}
        diagnostic_probe = mock.Mock()
        diagnostic_probe(("sentinel-child",), env=sentinel_environment)
        rendered = io.StringIO()
        outer = self

        class SafeNoChildDiagnosticProbe(unittest.TestCase):
            def runTest(probe_self) -> None:
                outer.assert_no_child(diagnostic_probe)

        probe_result = unittest.TextTestRunner(stream=rendered).run(
            SafeNoChildDiagnosticProbe()
        )
        self.assertEqual(probe_result.testsRun, 1)
        self.assertEqual(len(probe_result.failures), 1)
        self.assertEqual(len(probe_result.errors), 0)
        rendered_failure = rendered.getvalue()
        self.assertIn(self.NO_CHILD_DIAGNOSTIC, rendered_failure)
        self.assertNotIn(sentinel_key, rendered_failure)
        self.assertNotIn(sentinel_value, rendered_failure)
        self.assertNotIn(repr(sentinel_environment), rendered_failure)
        self.assertNotIn("env=", rendered_failure)

    def test_05_controlled_real_64_byte_tmpdir_is_admitted_and_cleaned(self) -> None:
        configured_root, tmpdir_107, tmpdir_108 = self.utf8_boundary_paths()
        self.assertEqual(len(str(self.SYNTHETIC_HOME)), 11)
        self.assertEqual(len(os.fsencode(self.SYNTHETIC_HOME)), 11)
        self.assertEqual(len(str(configured_root)), 13)
        self.assertEqual(len(os.fsencode(configured_root)), 14)
        self.assertEqual(len(str(tmpdir_107)), 63)
        self.assertEqual(len(os.fsencode(tmpdir_107)), 64)
        self.assertEqual(len(str(self.socket_candidate(tmpdir_107))), 106)
        self.assertEqual(len(os.fsencode(self.socket_candidate(tmpdir_107))), 107)
        self.assertEqual(len(str(tmpdir_108)), 64)
        self.assertEqual(len(os.fsencode(tmpdir_108)), 65)
        self.assertEqual(len(str(self.socket_candidate(tmpdir_108))), 107)
        self.assertEqual(len(os.fsencode(self.socket_candidate(tmpdir_108))), 108)

        plan = self.plan(
            PlannedCommand("C107 boundary", ("c107",), lane=self.lane("utf8"))
        )
        home = Path.home().resolve()
        with tempfile.TemporaryDirectory(prefix="x", dir=home) as directory:
            authority = Path(directory)
            controlled_107, _controlled_108 = self.real_boundary_tmpdirs(authority)
            candidate = self.socket_candidate(controlled_107)
            self.assertEqual(len(str(controlled_107)), 63)
            self.assertEqual(len(os.fsencode(controlled_107)), 64)
            self.assertEqual(len(str(candidate)), 106)
            self.assertEqual(len(os.fsencode(candidate)), 107)
            submitted: list[Path] = []
            observed: list[tuple[Path, bool]] = []

            @contextlib.contextmanager
            def controlled_scope(
                received_authority: Path, admit: Callable[[Path], None]
            ) -> Iterator[Path]:
                self.assertEqual(received_authority, authority)
                submitted.append(controlled_107)
                admit(controlled_107)
                controlled_107.mkdir()
                try:
                    yield controlled_107
                finally:
                    controlled_107.rmdir()

            def execute(
                argv: tuple[str, ...], **kwargs: object
            ) -> subprocess.CompletedProcess[bytes]:
                tmpdir = Path(dict(kwargs["env"])["TMPDIR"])
                observed.append((tmpdir, tmpdir.exists()))
                return subprocess.CompletedProcess(argv, 0)

            with (
                mock.patch(
                    "verification_scheduler._lane_tmp_scope",
                    side_effect=controlled_scope,
                    create=True,
                ) as scope,
                mock.patch(
                    "verification_scheduler.subprocess.run", side_effect=execute
                ) as child,
            ):
                run_plan(
                    plan,
                    authority,
                    budget=ResourceBudget(1, 1),
                    scratch_root=authority,
                )

            child.assert_called_once()
            self.assertEqual(
                scope.call_count,
                1,
                "run_plan did not consume the frozen lane TMP scope",
            )
            self.assertEqual(submitted, [controlled_107])
            self.assertEqual(observed, [(controlled_107, True)])
            self.assertFalse(controlled_107.exists())

    def test_06_controlled_real_65_byte_tmpdir_rejects_and_cleans(self) -> None:
        plan = self.plan(
            PlannedCommand("C108 boundary", ("c108",), lane=self.lane("utf8"))
        )
        home = Path.home().resolve()
        with tempfile.TemporaryDirectory(prefix="x", dir=home) as directory:
            authority = Path(directory)
            _controlled_107, controlled_108 = self.real_boundary_tmpdirs(authority)
            candidate = self.socket_candidate(controlled_108)
            self.assertEqual(len(str(controlled_108)), 64)
            self.assertEqual(len(os.fsencode(controlled_108)), 65)
            self.assertEqual(len(str(candidate)), 107)
            self.assertEqual(len(os.fsencode(candidate)), 108)
            submitted: list[Path] = []

            @contextlib.contextmanager
            def controlled_scope(
                received_authority: Path, admit: Callable[[Path], None]
            ) -> Iterator[Path]:
                self.assertEqual(received_authority, authority)
                submitted.append(controlled_108)
                controlled_108.mkdir()
                try:
                    admit(controlled_108)
                    yield controlled_108
                finally:
                    if controlled_108.exists():
                        controlled_108.rmdir()

            rejection: Exception | None = None
            with (
                mock.patch(
                    "verification_scheduler._lane_tmp_scope",
                    side_effect=controlled_scope,
                    create=True,
                ) as scope,
                mock.patch("verification_scheduler.subprocess.run") as child,
            ):
                try:
                    run_plan(
                        plan,
                        authority,
                        budget=ResourceBudget(1, 1),
                        scratch_root=authority,
                    )
                except Exception as caught:  # rejection type is not contractual
                    rejection = caught

            self.assertEqual(
                scope.call_count,
                1,
                "run_plan did not consume the frozen lane TMP scope",
            )
            self.assertEqual(submitted, [controlled_108])
            self.assert_no_child(child)
            self.assertFalse(controlled_108.exists())
            self.assertIsNotNone(rejection)
            assert rejection is not None
            self.assertRegex(str(rejection), r"108")
            self.assertRegex(str(rejection), r"107")

    def test_07_unpatched_64_byte_root_rejects_every_strict_descendant(self) -> None:
        plan = self.plan(
            PlannedCommand(
                "overlong provider falsifier", ("forbidden",), lane=self.lane("utf8")
            )
        )
        home = Path.home().resolve()
        with tempfile.TemporaryDirectory(prefix="x", dir=home) as directory:
            container = Path(directory)
            padding = self.TMPDIR_ADMITTED_MAX - len(os.fsencode(container)) - 1
            self.assertGreaterEqual(padding, 1)
            scratch_root = container / ("a" * padding)
            scratch_root.mkdir()
            self.assertEqual(len(os.fsencode(scratch_root)), 64)
            before_rejection = self.filesystem_snapshot(scratch_root)
            started: list[Path] = []

            def forbidden(
                argv: tuple[str, ...], **kwargs: object
            ) -> subprocess.CompletedProcess[bytes]:
                started.append(Path(dict(kwargs["env"])["TMPDIR"]))
                return subprocess.CompletedProcess(argv, 0)

            rejection: Exception | None = None
            with mock.patch(
                "verification_scheduler.subprocess.run", side_effect=forbidden
            ) as child:
                try:
                    run_plan(
                        plan,
                        container,
                        budget=ResourceBudget(1, 1),
                        scratch_root=scratch_root,
                    )
                except Exception as caught:  # rejection type is not contractual
                    rejection = caught

            after_rejection = self.filesystem_snapshot(scratch_root)
            if started:
                self.assertEqual(len(started), 1)
                self.assertNotEqual(started[0], scratch_root)
                self.assertTrue(started[0].is_relative_to(scratch_root))
                self.assertGreater(
                    len(os.fsencode(self.socket_candidate(started[0]))),
                    self.UNIX_PATHNAME_MAX,
                )
            self.assert_no_child(child)
            self.assertEqual(after_rejection, before_rejection)
            self.assertIsNotNone(rejection)
            assert rejection is not None
            diagnostic = str(rejection)
            self.assertRegex(diagnostic, r"107")
            self.assertTrue(
                any(
                    int(value) > self.UNIX_PATHNAME_MAX
                    for value in re.findall(r"\d+", diagnostic)
                ),
                "overlong-path diagnostic omitted the actual encoded length",
            )

    def test_07a_partial_scope_rejection_prevents_every_lane_child(self) -> None:
        plan = self.plan(
            PlannedCommand("lane A", ("lane-a",), lane=self.lane("lane-a")),
            PlannedCommand("lane B", ("lane-b",), lane=self.lane("lane-b")),
        )
        home = Path.home().resolve()
        with tempfile.TemporaryDirectory(prefix="x", dir=home) as directory:
            authority = Path(directory)
            controlled_107, controlled_108 = self.real_boundary_tmpdirs(authority)
            boundaries = (controlled_107, controlled_108)
            self.assertEqual(
                [len(os.fsencode(self.socket_candidate(path))) for path in boundaries],
                [107, 108],
            )
            allocation_lock = threading.Lock()
            allocated: list[Path] = []
            admission_attempts: list[Path] = []
            entered: list[Path] = []
            released: list[Path] = []
            child_attempted = threading.Event()

            @contextlib.contextmanager
            def controlled_scope(
                received_authority: Path, admit: Callable[[Path], None]
            ) -> Iterator[Path]:
                self.assertEqual(received_authority, authority)
                with allocation_lock:
                    candidate = boundaries[len(allocated)]
                    candidate.mkdir()
                    allocated.append(candidate)
                    admission_attempts.append(candidate)
                try:
                    if candidate == controlled_108:
                        child_attempted.wait(timeout=1.0)
                    admit(candidate)
                    with allocation_lock:
                        entered.append(candidate)
                    yield candidate
                finally:
                    if candidate.exists():
                        candidate.rmdir()
                    with allocation_lock:
                        released.append(candidate)

            def execute(
                argv: tuple[str, ...], **kwargs: object
            ) -> subprocess.CompletedProcess[bytes]:
                child_attempted.set()
                return subprocess.CompletedProcess(argv, 0)

            rejection: Exception | None = None
            with (
                mock.patch(
                    "verification_scheduler._lane_tmp_scope",
                    side_effect=controlled_scope,
                    create=True,
                ) as scope,
                mock.patch(
                    "verification_scheduler.subprocess.run", side_effect=execute
                ) as child,
            ):
                try:
                    run_plan(
                        plan,
                        authority,
                        budget=ResourceBudget(2, 2),
                        scratch_root=authority,
                    )
                except Exception as caught:  # rejection type is not contractual
                    rejection = caught

            self.assertEqual(
                scope.call_count,
                2,
                "run_plan did not request every lane scope before task execution",
            )
            self.assertCountEqual(allocated, boundaries)
            self.assertCountEqual(admission_attempts, boundaries)
            self.assertEqual(entered, [controlled_107])
            self.assert_no_child(child)
            self.assertCountEqual(released, [controlled_107, controlled_108])
            self.assertTrue(all(not path.exists() for path in allocated))
            self.assertIsNotNone(rejection)
            assert rejection is not None
            self.assertRegex(str(rejection), r"108")
            self.assertRegex(str(rejection), r"107")

    def test_08_child_failure_cleans_tmp_but_preserves_cargo_root(self) -> None:
        observed: dict[str, Path] = {}

        def fail(
            argv: tuple[str, ...], **kwargs: object
        ) -> subprocess.CompletedProcess[bytes]:
            environment = dict(kwargs["env"])
            observed.update(
                tmpdir=Path(environment["TMPDIR"]),
                lane_root=Path(environment["EQIORA_VERIFY_LANE_ROOT"]),
                cargo_root=Path(environment["CARGO_TARGET_DIR"]),
            )
            self.assertTrue(observed["tmpdir"].exists())
            raise subprocess.CalledProcessError(23, argv)

        plan = self.plan(
            PlannedCommand(
                "reported child failure", ("fail",), lane=self.lane("failure")
            )
        )
        with tempfile.TemporaryDirectory(prefix="e496-", dir=Path.home()) as directory:
            scratch_root = Path(directory)
            with (
                mock.patch("verification_scheduler.subprocess.run", side_effect=fail),
                self.assertRaises(VerificationFailure) as raised,
            ):
                run_plan(
                    plan,
                    scratch_root,
                    budget=ResourceBudget(1, 1),
                    scratch_root=scratch_root,
                )

            self.assertEqual(
                [failure.returncode for failure in raised.exception.failures], [23]
            )
            self.assertFalse(observed["tmpdir"].exists())
            self.assertTrue(observed["lane_root"].exists())
            self.assertTrue(observed["cargo_root"].exists())

    def test_09_outside_home_configured_root_rejects_before_child(self) -> None:
        plan = self.plan(
            PlannedCommand("must not start", ("forbidden",), lane=self.lane("outside"))
        )
        rejection: Exception | None = None
        with mock.patch("verification_scheduler.subprocess.run") as child:
            try:
                run_plan(
                    plan,
                    Path.home(),
                    budget=ResourceBudget(1, 1),
                    scratch_root=Path("/var/lib/eqiora-issue496-scratch"),
                )
            except Exception as caught:  # rejection type is not contractual
                rejection = caught
        self.assert_no_child(child)
        self.assertIsNotNone(rejection)
        assert rejection is not None
        self.assertRegex(str(rejection), "below the home directory")


class SchedulerTests(unittest.TestCase):
    @staticmethod
    def plan(*commands: PlannedCommand) -> VerificationPlan:
        return VerificationPlan("affected", (), (), (), commands, ())

    @staticmethod
    def lane(
        name: str,
        *,
        cpu_slots: int = 1,
        memory_mib: int = 1,
        gpu_slots: int = 0,
        locks: tuple[str, ...] = (),
    ) -> VerificationLane:
        return VerificationLane(
            name,
            ResourceRequest(cpu_slots, memory_mib, gpu_slots, locks),
        )

    def test_v3_delegates_admission_to_the_callable_scheduler_core(self) -> None:
        core = importlib.import_module("resource_scheduler")
        scheduler = importlib.import_module("verification_scheduler")

        self.assertIs(scheduler.ResourceBudget, core.ResourceBudget)
        self.assertIs(scheduler.ResourceRequest, core.ResourceRequest)
        self.assertNotIn("_fits", vars(scheduler))
        self.assertIn("run_tasks(", inspect.getsource(scheduler.run_plan))

    def test_disjoint_lanes_overlap_and_logs_remain_plan_ordered(self) -> None:
        rendezvous = threading.Barrier(2)
        completion_order: list[str] = []
        completion_lock = threading.Lock()

        def execute(
            argv: tuple[str, ...], **kwargs: object
        ) -> subprocess.CompletedProcess[bytes]:
            rendezvous.wait(timeout=1.0)
            if argv[0] == "slow":
                time.sleep(0.05)
            output = kwargs["stdout"]
            output.write(f"LOG-{argv[0]}\n".encode())
            output.flush()
            with completion_lock:
                completion_order.append(argv[0])
            return subprocess.CompletedProcess(argv, 0)

        plan = self.plan(
            PlannedCommand("slow witness", ("slow",), lane=self.lane("rust")),
            PlannedCommand("fast witness", ("fast",), lane=self.lane("python")),
        )
        stream = io.StringIO()
        with (
            mock.patch("local_verify.subprocess.run", side_effect=execute),
            contextlib.redirect_stdout(stream),
            tempfile.TemporaryDirectory(dir=Path.home()) as directory,
        ):
            run_plan(
                plan,
                Path(directory),
                budget=ResourceBudget(2, 2),
                scratch_root=Path(directory) / "scratch",
            )

        self.assertEqual(completion_order, ["fast", "slow"])
        self.assertEqual(
            [
                line
                for line in stream.getvalue().splitlines()
                if line.startswith("LOG-")
            ],
            ["LOG-slow", "LOG-fast"],
        )

    def test_cpu_memory_gpu_and_named_locks_each_prevent_overlap(self) -> None:
        scenarios = {
            "cpu": (
                self.lane("first", cpu_slots=1),
                self.lane("second", cpu_slots=1),
                ResourceBudget(1, 2, 0),
            ),
            "memory": (
                self.lane("first", memory_mib=2),
                self.lane("second", memory_mib=2),
                ResourceBudget(2, 2, 0),
            ),
            "gpu": (
                self.lane("first", gpu_slots=1),
                self.lane("second", gpu_slots=1),
                ResourceBudget(2, 2, 1),
            ),
            "named lock": (
                self.lane("first", locks=("shared-tool",)),
                self.lane("second", locks=("shared-tool",)),
                ResourceBudget(2, 2, 0),
            ),
        }
        for name, (first, second, budget) in scenarios.items():
            with self.subTest(resource=name):
                second_started = threading.Event()
                overlap_observed = threading.Event()

                def execute(
                    argv: tuple[str, ...], **kwargs: object
                ) -> subprocess.CompletedProcess[bytes]:
                    if argv[0] == "first":
                        if second_started.wait(timeout=0.05):
                            overlap_observed.set()
                    else:
                        second_started.set()
                    return subprocess.CompletedProcess(argv, 0)

                plan = self.plan(
                    PlannedCommand("first", ("first",), lane=first),
                    PlannedCommand("second", ("second",), lane=second),
                )
                with (
                    mock.patch("local_verify.subprocess.run", side_effect=execute),
                    tempfile.TemporaryDirectory(dir=Path.home()) as directory,
                ):
                    run_plan(
                        plan,
                        Path(directory),
                        budget=budget,
                        scratch_root=Path(directory) / "scratch",
                    )
                self.assertFalse(overlap_observed.is_set())

    def test_insufficient_budget_rejects_before_starting_a_child(self) -> None:
        scenarios = {
            "cpu": (ResourceRequest(2, 1), ResourceBudget(1, 1)),
            "memory": (ResourceRequest(1, 2), ResourceBudget(1, 1)),
            "gpu": (ResourceRequest(1, 1, 1), ResourceBudget(1, 1, 0)),
        }
        for name, (request, budget) in scenarios.items():
            with self.subTest(resource=name):
                plan = self.plan(
                    PlannedCommand(
                        "must not start",
                        ("forbidden",),
                        lane=VerificationLane("oversized", request),
                    )
                )
                with (
                    mock.patch("local_verify.subprocess.run") as execute,
                    tempfile.TemporaryDirectory(dir=Path.home()) as directory,
                    self.assertRaisesRegex(ValueError, name),
                ):
                    run_plan(
                        plan,
                        Path(directory),
                        budget=budget,
                        scratch_root=Path(directory) / "scratch",
                    )
                execute.assert_not_called()

    def test_commands_remain_ordered_inside_one_lane(self) -> None:
        observed: list[str] = []

        def execute(
            argv: tuple[str, ...], **kwargs: object
        ) -> subprocess.CompletedProcess[bytes]:
            observed.append(argv[0])
            return subprocess.CompletedProcess(argv, 0)

        lane = self.lane("ordered")
        plan = self.plan(
            PlannedCommand("first", ("first",), lane=lane),
            PlannedCommand("second", ("second",), lane=lane),
        )
        with (
            mock.patch("local_verify.subprocess.run", side_effect=execute),
            tempfile.TemporaryDirectory(dir=Path.home()) as directory,
        ):
            run_plan(
                plan,
                Path(directory),
                budget=ResourceBudget(2, 2),
                scratch_root=Path(directory) / "scratch",
            )
        self.assertEqual(observed, ["first", "second"])

    def test_failure_skips_successor_and_collects_independent_failure(self) -> None:
        rendezvous = threading.Barrier(2)
        observed: list[str] = []
        observed_lock = threading.Lock()

        def execute(
            argv: tuple[str, ...], **kwargs: object
        ) -> subprocess.CompletedProcess[bytes]:
            with observed_lock:
                observed.append(argv[0])
            if argv[0] != "forbidden":
                rendezvous.wait(timeout=1.0)
            returncode = {"first-failure": 7, "independent-failure": 9}.get(argv[0], 0)
            if returncode:
                raise subprocess.CalledProcessError(returncode, argv)
            return subprocess.CompletedProcess(argv, returncode)

        failed_lane = self.lane("failed")
        plan = self.plan(
            PlannedCommand("first failure", ("first-failure",), lane=failed_lane),
            PlannedCommand("must be skipped", ("forbidden",), lane=failed_lane),
            PlannedCommand(
                "independent failure",
                ("independent-failure",),
                lane=self.lane("independent"),
            ),
        )
        with (
            mock.patch("local_verify.subprocess.run", side_effect=execute),
            tempfile.TemporaryDirectory(dir=Path.home()) as directory,
            self.assertRaises(VerificationFailure) as raised,
        ):
            run_plan(
                plan,
                Path(directory),
                budget=ResourceBudget(2, 2),
                scratch_root=Path(directory) / "scratch",
            )

        self.assertCountEqual(observed, ["first-failure", "independent-failure"])
        self.assertEqual(
            [failure.returncode for failure in raised.exception.failures], [7, 9]
        )

    def test_execution_error_preserves_detail_and_skips_lane_successor(self) -> None:
        rendezvous = threading.Barrier(2)
        observed: list[str] = []
        observed_lock = threading.Lock()

        def execute(
            argv: tuple[str, ...], **kwargs: object
        ) -> subprocess.CompletedProcess[bytes]:
            with observed_lock:
                observed.append(argv[0])
            if argv[0] != "forbidden":
                rendezvous.wait(timeout=1.0)
            if argv[0] == "broken":
                raise OSError("missing executable: broken")
            return subprocess.CompletedProcess(argv, 0)

        failed_lane = self.lane("failed")
        plan = self.plan(
            PlannedCommand("broken command", ("broken",), lane=failed_lane),
            PlannedCommand("must be skipped", ("forbidden",), lane=failed_lane),
            PlannedCommand(
                "independent success",
                ("independent",),
                lane=self.lane("independent"),
            ),
        )
        with (
            mock.patch("local_verify.subprocess.run", side_effect=execute),
            tempfile.TemporaryDirectory(dir=Path.home()) as directory,
            self.assertRaises(VerificationFailure) as raised,
        ):
            run_plan(
                plan,
                Path(directory),
                budget=ResourceBudget(2, 2),
                scratch_root=Path(directory) / "scratch",
            )

        self.assertCountEqual(observed, ["broken", "independent"])
        failures = raised.exception.failures
        self.assertEqual(len(failures), 1)
        self.assertIsNone(failures[0].returncode)
        self.assertEqual(failures[0].detail, "missing executable: broken")
        self.assertEqual(failures[0].command.label, "broken command")
        self.assertIn("missing executable: broken", str(raised.exception))

    def test_lane_environment_is_home_backed_contained_and_disjoint(self) -> None:
        environments: dict[str, dict[str, str]] = {}
        rendezvous = threading.Barrier(2)

        def execute(
            argv: tuple[str, ...], **kwargs: object
        ) -> subprocess.CompletedProcess[bytes]:
            environments[argv[0]] = dict(kwargs["env"])
            rendezvous.wait(timeout=1.0)
            return subprocess.CompletedProcess(argv, 0)

        plan = self.plan(
            PlannedCommand("first", ("first",), lane=self.lane("first")),
            PlannedCommand("second", ("second",), lane=self.lane("second")),
        )
        with (
            mock.patch("local_verify.subprocess.run", side_effect=execute),
            tempfile.TemporaryDirectory(dir=Path.home()) as directory,
        ):
            scratch_root = Path(directory) / "scratch"
            run_plan(
                plan,
                Path(directory),
                budget=ResourceBudget(2, 2),
                scratch_root=scratch_root,
            )
            roots = {
                name: Path(environment["EQIORA_VERIFY_LANE_ROOT"])
                for name, environment in environments.items()
            }
            self.assertEqual(len(set(roots.values())), 2)
            for name, root in roots.items():
                self.assertTrue(root.is_relative_to(scratch_root))
                self.assertTrue(
                    Path(environments[name]["TMPDIR"]).is_relative_to(scratch_root)
                )
                self.assertTrue(
                    Path(environments[name]["CARGO_TARGET_DIR"]).is_relative_to(root)
                )
                self.assertEqual(environments[name]["CARGO_BUILD_JOBS"], "1")


class ResourceDetectionTests(unittest.TestCase):
    SYSCONF_VALUES = {"SC_AVPHYS_PAGES": 131072, "SC_PAGE_SIZE": 4096}

    def test_memavailable_is_preferred_over_sysconf_fallback(self) -> None:
        meminfo = (
            "MemTotal:       32000000 kB\n"
            "MemFree:          512000 kB\n"
            "MemAvailable:    2048000 kB\n"
        )
        sysconf = mock.MagicMock(side_effect=self.SYSCONF_VALUES.__getitem__)
        with (
            mock.patch("verification_scheduler.Path.read_text", return_value=meminfo),
            mock.patch("verification_scheduler.os.sysconf", sysconf),
        ):
            self.assertEqual(_available_memory_mib(), 2000)
        sysconf.assert_not_called()

    def test_unreadable_meminfo_falls_back_to_sysconf(self) -> None:
        sysconf = mock.MagicMock(side_effect=self.SYSCONF_VALUES.__getitem__)
        with (
            mock.patch(
                "verification_scheduler.Path.read_text",
                side_effect=OSError("denied"),
            ),
            mock.patch("verification_scheduler.os.sysconf", sysconf),
        ):
            self.assertEqual(_available_memory_mib(), 512)


class CpuAllocationTests(unittest.TestCase):
    # The module lane constants clamp cpu requests to the importing host's
    # cpu_count, so the fixture pins the uncapped plan requests instead.
    PLAN_LANE_REQUESTS = (
        ("repository", 1),
        ("root-cargo", 4),
        ("dependency-policy", 1),
        ("python-candidate", 2),
        ("studio", 2),
        ("cubecl", 2),
    )

    def test_sixty_four_slots_divide_deterministically_across_plan_lanes(self) -> None:
        plan = build_plan("periodic", [], [], workspace())
        self.assertEqual(
            [lane.name for lane in dict.fromkeys(item.lane for item in plan.commands)],
            [name for name, _cpu in self.PLAN_LANE_REQUESTS],
        )
        lanes = tuple(
            VerificationLane(name, ResourceRequest(cpu, 1))
            for name, cpu in self.PLAN_LANE_REQUESTS
        )
        self.assertEqual(
            cpu_allocations(lanes, ResourceBudget(64, 1)),
            {
                "repository": 5,
                "root-cargo": 21,
                "dependency-policy": 5,
                "python-candidate": 11,
                "studio": 11,
                "cubecl": 11,
            },
        )

    def test_exact_budget_keeps_one_job_per_lane(self) -> None:
        lanes = (
            VerificationLane("first", ResourceRequest(1, 1)),
            VerificationLane("second", ResourceRequest(1, 1)),
        )
        self.assertEqual(
            cpu_allocations(lanes, ResourceBudget(2, 1)),
            {"first": 1, "second": 1},
        )


class LocalDocumentationTests(unittest.TestCase):
    def test_dependency_trees_are_outside_the_documentation_contract(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "docs").mkdir()
            (root / "docs/capability-matrix.md").write_text(
                "matrix\n", encoding="utf-8"
            )
            for name in ("README.md", "AGENTS.md", "CONTRIBUTING.md"):
                (root / name).write_text(
                    "[matrix](docs/capability-matrix.md)\n", encoding="utf-8"
                )
            dependency = root / "studio/node_modules/dependency"
            dependency.mkdir(parents=True)
            (dependency / "README.md").write_text(
                "[missing](not-shipped.md)\n", encoding="utf-8"
            )
            self.assertEqual(check_docs(root), [])


if __name__ == "__main__":
    unittest.main()
