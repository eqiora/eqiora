#!/usr/bin/env python3
"""Plan and run repository-owned local verification without hosted CI."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Iterable, Mapping, Sequence

from classify_changes import impact_plan, jupyter_assets_required
from verification_scheduler import (
    CUBECL_LANE,
    DEPENDENCY_POLICY_LANE,
    HOSTED_TEST_PROFILE,
    PYTHON_LANE,
    REPOSITORY_LANE,
    ROOT_CARGO_LANE,
    STUDIO_LANE,
    PlannedCommand,
    ResourceBudget,
    ResourceRequest,
    VerificationFailure,
    VerificationLane,
    VerificationPlan,
    cpu_allocations,
    default_budget,
    run_plan,
)


ROOT = Path(__file__).resolve().parents[2]

# The Playwright configuration pins `channel: "chrome"`, which resolves to this
# executable on Linux; its absence is the one prerequisite that defers the
# Studio interaction tests to the hosted Studio lane.
CHROME_EXECUTABLE = Path("/opt/google/chrome/chrome")

__all__ = [
    "HOSTED_TEST_PROFILE",
    "PlannedCommand",
    "ResourceBudget",
    "ResourceRequest",
    "VerificationFailure",
    "VerificationLane",
    "VerificationPlan",
    "build_plan",
    "run_plan",
]


@dataclass(frozen=True)
class WorkspacePackage:
    name: str
    directory: str
    dependencies: frozenset[str]


def _git_paths(arguments: Sequence[str], root: Path = ROOT) -> set[str]:
    output = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    return {
        entry.decode("utf-8").replace("\\", "/")
        for entry in output.split(b"\0")
        if entry
    }


def local_changed_paths(base: str, root: Path = ROOT) -> tuple[str, ...]:
    """Union committed, staged, unstaged, and untracked change paths."""
    paths = _git_paths(
        ["diff", "--no-renames", "--name-only", "-z", f"{base}...HEAD"], root
    )
    paths.update(_git_paths(["diff", "--no-renames", "--name-only", "-z"], root))
    paths.update(
        _git_paths(["diff", "--cached", "--no-renames", "--name-only", "-z"], root)
    )
    paths.update(_git_paths(["ls-files", "--others", "--exclude-standard", "-z"], root))
    return tuple(sorted(paths))


def load_workspace(root: Path = ROOT) -> dict[str, WorkspacePackage]:
    output = subprocess.run(
        ["cargo", "metadata", "--format-version=1", "--no-deps"],
        cwd=root,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout
    metadata = json.loads(output)
    members = set(metadata["workspace_members"])
    workspace_names = {
        package["name"] for package in metadata["packages"] if package["id"] in members
    }
    packages: dict[str, WorkspacePackage] = {}
    for package in metadata["packages"]:
        if package["id"] not in members:
            continue
        manifest = Path(package["manifest_path"]).resolve()
        directory = manifest.parent.relative_to(root.resolve()).as_posix()
        dependencies = frozenset(
            dependency["name"]
            for dependency in package["dependencies"]
            if dependency["name"] in workspace_names
        )
        packages[package["name"]] = WorkspacePackage(
            package["name"], directory, dependencies
        )
    return packages


def direct_packages(
    paths: Iterable[str], packages: Mapping[str, WorkspacePackage]
) -> set[str]:
    normalized = set(paths)
    if normalized.intersection({"Cargo.toml", "Cargo.lock"}):
        return set(packages)
    selected: set[str] = set()
    for path in normalized:
        candidates = [
            package
            for package in packages.values()
            if path == package.directory or path.startswith(f"{package.directory}/")
        ]
        if candidates:
            selected.add(
                max(candidates, key=lambda package: len(package.directory)).name
            )
    return selected


def reverse_dependency_closure(
    selected: Iterable[str], packages: Mapping[str, WorkspacePackage]
) -> set[str]:
    closure = set(selected)
    changed = True
    while changed:
        changed = False
        for package in packages.values():
            if package.name not in closure and package.dependencies.intersection(
                closure
            ):
                closure.add(package.name)
                changed = True
    return closure


def all_case_ids(root: Path = ROOT) -> set[str]:
    return {
        str(tomllib.loads(path.read_text(encoding="utf-8"))["id"])
        for path in sorted((root / "verify").glob("*/*/case.toml"))
    }


def command(
    label: str,
    *argv: str,
    cwd: str = ".",
    env: Mapping[str, str] | None = None,
    lane: VerificationLane = REPOSITORY_LANE,
) -> PlannedCommand:
    return PlannedCommand(
        label=label,
        argv=tuple(argv),
        cwd=cwd,
        env=tuple(sorted((env or {}).items())),
        lane=lane,
    )


def cli_feature_args(packages: Iterable[str]) -> tuple[str, ...]:
    """Keep the facade's command-line targets in repository product checks."""
    return ("--features", "eqiora/cli") if "eqiora" in packages else ()


def _rust_commands(
    packages: Iterable[str], *, rustdoc: bool, all_targets: bool = True
) -> list[PlannedCommand]:
    selected = sorted(packages)
    if not selected:
        return []
    selectors = tuple(item for package in selected for item in ("-p", package))
    feature_scope = cli_feature_args(selected)
    target_scope = ("--all-targets",) if all_targets else ()
    commands = [
        command(
            "Rust tests",
            "cargo",
            "test",
            "--locked",
            *selectors,
            *feature_scope,
            *target_scope,
            lane=ROOT_CARGO_LANE,
        ),
        command(
            "Rust Clippy",
            "cargo",
            "clippy",
            *selectors,
            *feature_scope,
            *target_scope,
            "--",
            "-D",
            "warnings",
            lane=ROOT_CARGO_LANE,
        ),
    ]
    if rustdoc:
        commands.append(
            command(
                "Rustdoc",
                "cargo",
                "doc",
                *selectors,
                "--no-deps",
                "--locked",
                env={"RUSTDOCFLAGS": "-D warnings"},
                lane=ROOT_CARGO_LANE,
            )
        )
    return commands


def _case_commands(cases: Iterable[str]) -> list[PlannedCommand]:
    selected = sorted(set(cases))
    if not selected:
        return []
    selectors = tuple(item for case in selected for item in ("--case", case))
    noun = "case" if len(selected) == 1 else "cases"
    return [
        command(
            f"Registered evidence ({len(selected)} {noun})",
            "cargo",
            "run",
            "--locked",
            "-p",
            "eqiora-verify",
            "--",
            "run",
            *selectors,
            lane=ROOT_CARGO_LANE,
        )
    ]


def _surface_commands(
    surfaces: Mapping[str, bool], *, jupyter_assets: bool, chrome_available: bool = True
) -> list[PlannedCommand]:
    commands: list[PlannedCommand] = []
    if surfaces["dependency_policy"]:
        commands.append(
            command(
                "Root dependency policy",
                "cargo",
                "deny",
                "--locked",
                "check",
                lane=DEPENDENCY_POLICY_LANE,
            )
        )
    if surfaces["python"] and jupyter_assets:
        commands.append(
            command(
                "Jupyter prebuilt assets",
                sys.executable,
                "tools/editor/check_jupyter.py",
                lane=PYTHON_LANE,
            )
        )
    if surfaces["python"]:
        commands.append(
            command(
                "Python isolated wheel and tests",
                sys.executable,
                "tools/ci/python_package_gate.py",
                lane=PYTHON_LANE,
            )
        )
    if surfaces["studio"]:
        commands.extend(
            [
                command(
                    "Studio quality",
                    "npm",
                    "run",
                    "check",
                    cwd="studio",
                    lane=STUDIO_LANE,
                ),
                command(
                    "Studio unit tests",
                    "npm",
                    "test",
                    cwd="studio",
                    lane=STUDIO_LANE,
                ),
                command(
                    "Studio build",
                    "npm",
                    "run",
                    "build",
                    cwd="studio",
                    lane=STUDIO_LANE,
                ),
                *(
                    (
                        command(
                            "Studio interaction tests",
                            "npm",
                            "run",
                            "test:e2e",
                            "--",
                            "--workers=1",
                            cwd="studio",
                            lane=STUDIO_LANE,
                        ),
                    )
                    if chrome_available
                    else ()
                ),
            ]
        )
    if surfaces["cubecl_experiment"]:
        manifest = "experiments/cubecl-local-action/Cargo.toml"
        commands.extend(
            [
                command(
                    "CubeCL Clippy",
                    "cargo",
                    "clippy",
                    "--manifest-path",
                    manifest,
                    "--locked",
                    "--all-targets",
                    "--",
                    "-D",
                    "warnings",
                    lane=CUBECL_LANE,
                ),
                command(
                    "CubeCL tests",
                    "cargo",
                    "test",
                    "--manifest-path",
                    manifest,
                    "--locked",
                    lane=CUBECL_LANE,
                ),
            ]
        )
    return commands


def build_plan(
    tier: str,
    paths: Sequence[str],
    explicit_cases: Sequence[str],
    packages: Mapping[str, WorkspacePackage],
    root: Path = ROOT,
    *,
    chrome_executable: Path | None = None,
) -> VerificationPlan:
    chrome_available = (
        chrome_executable if chrome_executable is not None else CHROME_EXECUTABLE
    ).exists()
    if tier == "periodic":
        impact = impact_plan(
            [],
            full=True,
            target_authority="local-worktree",
            base_authority="local-periodic-run",
        )
        surfaces = impact.selections()
        selected_packages = set(packages)
        cases = all_case_ids(root)
        ci_contract_lane = (
            ROOT_CARGO_LANE
            if "interfaces.python-distribution-candidate" in cases
            else REPOSITORY_LANE
        )
        commands = [
            command(
                "Documentation contract", sys.executable, "tools/ci/check_docs.py", "."
            ),
            command(
                "Public release tree",
                sys.executable,
                "tools/ci/check_public_release_tree.py",
                ".",
            ),
            command(
                "CI contract tests",
                sys.executable,
                "-m",
                "unittest",
                "discover",
                "-s",
                "tools/ci/tests",
                "-v",
                lane=ci_contract_lane,
            ),
            command(
                "Formatting",
                "cargo",
                "fmt",
                "--all",
                "--",
                "--check",
                lane=ROOT_CARGO_LANE,
            ),
            command(
                "Workspace Clippy",
                "cargo",
                "clippy",
                "--workspace",
                "--all-targets",
                "--all-features",
                "--keep-going",
                "--",
                "-D",
                "warnings",
                lane=ROOT_CARGO_LANE,
            ),
            command(
                "Workspace tests",
                "cargo",
                "test",
                "--workspace",
                "--all-targets",
                "--all-features",
                "--locked",
                lane=ROOT_CARGO_LANE,
            ),
            command(
                "All registered evidence",
                "cargo",
                "run",
                "--locked",
                "-p",
                "eqiora-verify",
                "--",
                "run",
                "--environment",
                "host-cpu",
                lane=ROOT_CARGO_LANE,
            ),
            command(
                "Dependency layers",
                "cargo",
                "xtask",
                "check-layers",
                lane=ROOT_CARGO_LANE,
            ),
            command(
                "Public facade",
                "cargo",
                "xtask",
                "check-facade",
                lane=ROOT_CARGO_LANE,
            ),
            command(
                "Workspace Rustdoc",
                "cargo",
                "doc",
                "--workspace",
                "--no-deps",
                "--locked",
                env={"RUSTDOCFLAGS": "-D warnings"},
                lane=ROOT_CARGO_LANE,
            ),
            command(
                "MSRV",
                "cargo",
                "+1.89.0",
                "check",
                "--workspace",
                "--all-targets",
                "--all-features",
                "--locked",
                lane=ROOT_CARGO_LANE,
            ),
        ]
        commands.extend(
            _surface_commands(
                surfaces,
                jupyter_assets=jupyter_assets_required(impact),
                chrome_available=chrome_available,
            )
        )
        limitations = (
            "Python coverage is the current interpreter, not the complete 3.11-3.14 matrix.",
            "Physical MPI-CUDA, multi-node MPI, and GPU evidence requires an explicit matching environment run.",
            "Studio browser interaction coverage requires its documented Chrome dependency.",
        )
    else:
        impact = impact_plan(
            paths,
            target_authority="local-worktree",
            base_authority="local-change-set",
        )
        surfaces = impact.selections()
        direct = direct_packages(paths, packages)
        selected_packages = (
            reverse_dependency_closure(direct, packages)
            if tier == "affected"
            else direct
        )
        if tier == "affected" and surfaces["rust"] and not direct:
            selected_packages = set(packages)
        cases = set(explicit_cases)
        ci_contract_lane = (
            ROOT_CARGO_LANE
            if "interfaces.python-distribution-candidate" in cases
            else REPOSITORY_LANE
        )
        commands = []
        if surfaces["rust"] or selected_packages:
            commands.append(
                command(
                    "Formatting",
                    "cargo",
                    "fmt",
                    "--all",
                    "--",
                    "--check",
                    lane=ROOT_CARGO_LANE,
                )
            )
        commands.extend(
            _rust_commands(
                selected_packages,
                rustdoc=tier == "affected",
                all_targets=tier != "pr",
            )
        )
        if tier == "affected":
            commands.append(
                command(
                    "Evidence manifest inventory",
                    "cargo",
                    "run",
                    "--locked",
                    "-p",
                    "eqiora-verify",
                    "--",
                    "check",
                    lane=ROOT_CARGO_LANE,
                )
            )
        commands.extend(_case_commands(cases))
        if tier != "pr" and any(
            path.endswith(".md") or path in {"AGENTS.md", "CONTRIBUTING.md"}
            for path in paths
        ):
            commands.append(
                command(
                    "Documentation contract",
                    sys.executable,
                    "tools/ci/check_docs.py",
                    ".",
                )
            )
            commands.append(
                command(
                    "Public release tree",
                    sys.executable,
                    "tools/ci/check_public_release_tree.py",
                    ".",
                )
            )
        if tier != "pr" and any(
            PurePosixPath(path).name == "Cargo.toml"
            or path in {"Cargo.lock", "tools/xtask/src/main.rs"}
            for path in paths
        ):
            commands.append(
                command(
                    "Dependency layers",
                    "cargo",
                    "xtask",
                    "check-layers",
                    lane=ROOT_CARGO_LANE,
                )
            )
        if tier != "pr" and any(
            path
            in {
                "api/eqiora-facade-v1.json",
                "crates/eqiora/src/lib.rs",
                "tools/xtask/Cargo.toml",
                "tools/xtask/src/facade.rs",
                "tools/xtask/src/main.rs",
            }
            for path in paths
        ):
            commands.append(
                command(
                    "Public facade",
                    "cargo",
                    "xtask",
                    "check-facade",
                    lane=ROOT_CARGO_LANE,
                )
            )
        ci_contract_inputs = {
            "Cargo.toml",
            "Cargo.lock",
            "pyproject.toml",
            "crates/eqiora-backend-faer/Cargo.toml",
            "crates/eqiora-backend-faer/src/lib.rs",
            "crates/eqiora-backend-rayon/Cargo.toml",
            "crates/eqiora-backend-rayon/src/lib.rs",
        }
        if tier != "pr" and any(
            path.startswith("tools/ci/") or path in ci_contract_inputs for path in paths
        ):
            commands.append(
                command(
                    "CI contract tests",
                    sys.executable,
                    "-m",
                    "unittest",
                    "discover",
                    "-s",
                    "tools/ci/tests",
                    "-v",
                    lane=ci_contract_lane,
                )
            )
        if tier == "affected":
            commands.extend(
                _surface_commands(
                    surfaces,
                    jupyter_assets=jupyter_assets_required(impact),
                    chrome_available=chrome_available,
                )
            )
        if tier == "pr":
            limitations = (
                "The pr tier is the local iteration loop: formatting, default-target tests and Clippy for directly changed packages, and explicitly named cases only.",
                "Documentation, release-tree, dependency-layer, facade, CI-contract, and surface checks are deferred to hosted pull-request CI or a fast/affected run.",
                "Reverse-dependency closure is not computed; use affected for uncertain dependency closure.",
            )
        else:
            limitations = (
                "Registered evidence runs only when explicitly selected with --case; broad execution belongs to periodic verification.",
                "Default-feature Clippy is the local code gate; optional backend features require their registered case or an explicit environment-specific check.",
                "Environment-dependent hardware and multi-node evidence is not implied unless explicitly run and recorded.",
            )

    if tier != "pr" and surfaces["site"]:
        commands.append(
            command(
                "Site source",
                sys.executable,
                "tools/site/check_site.py",
                "source",
                "--root",
                ".",
            )
        )
        limitations = (*limitations, "Site build and browser checks run in hosted Pages CI.")

    if tier in ("affected", "periodic") and surfaces["studio"] and not chrome_available:
        limitations = (
            *limitations,
            "Studio interaction tests are deferred to the hosted Studio lane: the "
            "local Chrome executable is absent. This deferral is not a local "
            "Studio pass.",
        )
    deduplicated = tuple(dict.fromkeys(commands))
    return VerificationPlan(
        tier=tier,
        paths=tuple(sorted(paths)),
        packages=tuple(sorted(selected_packages)),
        cases=tuple(sorted(cases)),
        commands=deduplicated,
        limitations=limitations,
    )


def render_plan(plan: VerificationPlan, budget: ResourceBudget | None = None) -> str:
    admitted_budget = budget or default_budget()
    lanes = tuple(dict.fromkeys(item.lane for item in plan.commands))
    cargo_jobs = cpu_allocations(lanes, admitted_budget)
    lines = [
        f"tier: {plan.tier}",
        f"changed paths: {len(plan.paths)}",
        f"packages: {', '.join(plan.packages) if plan.packages else 'none'}",
        f"cases: {', '.join(plan.cases) if plan.cases else 'none'}",
        f"budget: cpu={admitted_budget.cpu_slots}, "
        f"memory={admitted_budget.memory_mib} MiB, gpu={admitted_budget.gpu_slots}",
        "lanes:",
    ]
    for lane in lanes:
        resources = lane.resources
        locks = ",".join(resources.locks) if resources.locks else "none"
        lines.append(
            f"  - {lane.name}: cpu-min={resources.cpu_slots}, "
            f"cargo-jobs={cargo_jobs[lane.name]}, "
            f"memory={resources.memory_mib} MiB, gpu={resources.gpu_slots}, "
            f"locks={locks}"
        )
    lines.append("commands:")
    lines.extend(
        f"  {index}. [{item.label}; lane={item.lane.name}] {item.render()}"
        for index, item in enumerate(plan.commands, 1)
    )
    lines.append("limitations:")
    lines.extend(f"  - {limitation}" for limitation in plan.limitations)
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tier", choices=("pr", "fast", "affected", "periodic"))
    parser.add_argument(
        "--base", default="origin/main", help="merge-base comparison ref"
    )
    parser.add_argument(
        "--case", action="append", default=[], help="exact affected case ID"
    )
    parser.add_argument("--plan", action="store_true", help="print without executing")
    parser.add_argument(
        "--cpu-slots",
        type=int,
        help="scheduler CPU admission budget (defaults to detected CPUs)",
    )
    parser.add_argument(
        "--memory-mib",
        type=int,
        help="scheduler memory admission budget (defaults to available memory)",
    )
    parser.add_argument(
        "--gpu-slots",
        type=int,
        help="scheduler GPU admission budget (defaults to zero or the environment)",
    )
    parser.add_argument(
        "--scratch-root",
        type=Path,
        help="home-backed lane root (defaults below ~/.cache/eqiora)",
    )
    arguments = parser.parse_args()
    try:
        paths = (
            () if arguments.tier == "periodic" else local_changed_paths(arguments.base)
        )
        if arguments.tier != "periodic" and not paths and not arguments.case:
            raise ValueError(
                "no local changes or explicit verification cases were selected"
            )
        plan = build_plan(
            arguments.tier,
            paths,
            arguments.case,
            load_workspace(),
        )
        detected = default_budget()
        budget = ResourceBudget(
            arguments.cpu_slots
            if arguments.cpu_slots is not None
            else detected.cpu_slots,
            arguments.memory_mib
            if arguments.memory_mib is not None
            else detected.memory_mib,
            arguments.gpu_slots
            if arguments.gpu_slots is not None
            else detected.gpu_slots,
        )
        print(render_plan(plan, budget))
        if not arguments.plan:
            run_plan(
                plan,
                budget=budget,
                scratch_root=arguments.scratch_root,
            )
    except (
        OSError,
        subprocess.CalledProcessError,
        VerificationFailure,
        RuntimeError,
        UnicodeError,
        ValueError,
        json.JSONDecodeError,
        tomllib.TOMLDecodeError,
    ) as error:
        print(f"local verification failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
