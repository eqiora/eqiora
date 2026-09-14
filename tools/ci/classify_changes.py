#!/usr/bin/env python3
"""Classify a pull-request diff into reviewed CI ownership surfaces.

The classifier is deliberately repository-owned and dependency-free. Unknown
paths fail closed by selecting every conditional job. Non-pull-request runs
are full compatibility runs.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Iterable


SURFACES = (
    "rust",
    "msrv",
    "python",
    "studio",
    "dependency_policy",
    "cubecl_experiment",
)
CLASSIFIED_SURFACES = (*SURFACES, "site")

FULL_SHA = re.compile(r"[0-9a-f]{40}")


@dataclass(frozen=True)
class LaneImpact:
    """One deterministic private lane-selection decision."""

    lane: str
    selected: bool
    reason: str
    owning_changed_inputs: tuple[str, ...]


@dataclass(frozen=True)
class ImpactPlan:
    """The single private owner of hosted and local surface selection."""

    target_authority: str
    base_authority: str
    full: bool
    lanes: tuple[LaneImpact, ...]

    def selections(self) -> dict[str, bool]:
        return {decision.lane: decision.selected for decision in self.lanes}

    def lane(self, name: str) -> LaneImpact:
        for decision in self.lanes:
            if decision.lane == name:
                return decision
        raise KeyError(name)


def documentation_path(path: str) -> bool:
    """Return whether a path is owned entirely by documentation/process."""
    name = PurePosixPath(path).name
    return (
        path.startswith(("docs/", "rfcs/", ".github/ISSUE_TEMPLATE/"))
        or path == ".github/pull_request_template.md"
        or name in {"README.md", "AGENTS.md"}
        or name
        in {
            "CHANGELOG.md",
            "CODE_OF_CONDUCT.md",
            "CONTRIBUTING.md",
            "GOVERNANCE.md",
            "LICENSE",
            "NOTICE",
            "SECURITY.md",
        }
    )


def root_rust_path(path: str) -> bool:
    return path.startswith(("api/", "crates/", "tools/xtask/", "verify/")) or path in {
        "Cargo.toml",
        "Cargo.lock",
        "rustfmt.toml",
    }


def msrv_path(path: str) -> bool:
    return path.startswith(("crates/", "tools/xtask/")) or path in {
        "Cargo.toml",
        "Cargo.lock",
    }


def public_facade_path(path: str) -> bool:
    return path.startswith(("api/", "crates/eqiora/")) or path in {
        "Cargo.toml",
        "Cargo.lock",
    }


def dependency_policy_path(path: str) -> bool:
    name = PurePosixPath(path).name
    return (
        path in {"Cargo.toml", "Cargo.lock"}
        or (name == "Cargo.toml" and path.startswith(("crates/", "tools/xtask/")))
        or path == "deny.toml"
        or path.startswith(".cargo/")
        or path == ".github/dependabot.yml"
    )


SITE_INPUT_FILES = {
    ".cargo/config.toml",
    ".gitattributes",
    ".github/workflows/pages.yml",
    "CHANGELOG.md",
    "Cargo.lock",
    "Cargo.toml",
    "api/eqiora-facade-v1.json",
    "docs/capability-matrix.md",
    "docs/python/api.md",
    "docs/python/differentiation.md",
    "docs/python/execution-and-arrays.md",
    "docs/python/modeling.md",
    "docs/verification/gallery/README.md",
    "examples/decay.eqi",
    "examples/python/exact_cylinder_geometry.py",
    "examples/python/exact_cylinder_mesh.py",
    "examples/python/exact_cylinder_stokes.py",
    "examples/python/textbook_decay.py",
    "examples/steady-flow-past-cylinder.eqi",
    "examples/steady-flow-past-cylinder.geometry.json",
    "mise.lock",
    "mise.toml",
    "mkdocs.yml",
    "crates/eqiora-api/packages/Eqiora.Fluid.Incompressible/src/incompressible.eqi",
    "pyproject.toml",
    "rust-toolchain.toml",
    "crates/eqiora-api/schemas/compile-v2.schema.json",
    "tools/release/python_candidate_common.py",
    "uv.lock",
}


def site_input_path(path: str) -> bool:
    """Return whether the complete documentation artifact can read this path."""
    return (
        path in SITE_INPUT_FILES
        or path.startswith(
            (
                "bindings/python/python/eqiora/",
                "crates/",
                "docs/site/",
                "editor/",
                "tools/docs/",
                "tools/editor/",
                "tools/site/",
                "tools/xtask/",
                "verify/",
            )
        )
        or path == "tools/ci/classify_changes.py"
    )


def recognized_path(path: str) -> bool:
    """Return whether a reviewed ownership rule recognizes this path."""
    return (
        documentation_path(path)
        or root_rust_path(path)
        or msrv_path(path)
        or public_facade_path(path)
        or dependency_policy_path(path)
        or site_input_path(path)
        or path.startswith(
            (
                ".github/workflows/",
                "bindings/python/",
                "crates/eqiora-python/",
                "experiments/cubecl-local-action/",
                "studio/",
                "tools/ci/",
            )
        )
    )


def _normalized_paths(paths: Iterable[str]) -> tuple[str, ...]:
    normalized: set[str] = set()
    for raw_path in paths:
        path = raw_path.replace("\\", "/")
        if path.startswith("./"):
            path = path[2:]
        if not path:
            continue
        normalized.add(path)
    return tuple(sorted(normalized))


def impact_plan(
    paths: Iterable[str],
    *,
    full: bool = False,
    target_authority: str = "unspecified-target",
    base_authority: str = "unspecified-base",
    unsafe_mode: bool = False,
) -> ImpactPlan:
    """Derive one deterministic private plan without changing lane policy."""
    normalized = _normalized_paths(paths)
    owning_inputs = {surface: set() for surface in CLASSIFIED_SURFACES}
    selected = {surface: full for surface in CLASSIFIED_SURFACES}

    if full:
        return ImpactPlan(
            target_authority,
            base_authority,
            True,
            tuple(
                LaneImpact(surface, True, "full compatibility run", normalized)
                for surface in CLASSIFIED_SURFACES
            ),
        )

    for path in normalized:
        if site_input_path(path):
            selected["site"] = True
            owning_inputs["site"].add(path)
        # These displayed programs also execute in the installed Python gate.
        if path.startswith(
            (
                "docs/site/src/content/docs/reference/language/",
                "docs/site/src/content/docs/reference/standard-packages/",
            )
        ) and path.endswith((".mdx", ".eqi")):
            selected["python"] = True
            owning_inputs["python"].add(path)

    if unsafe_mode:
        selected["site"] = True
        owning_inputs["site"].update(normalized)

    if normalized and all(documentation_path(path) for path in normalized):
        return _impact_plan_from_selection(
            selected,
            owning_inputs,
            target_authority=target_authority,
            base_authority=base_authority,
            unsafe_mode=unsafe_mode,
        )

    protected_inputs = tuple(
        path
        for path in normalized
        if path.startswith((".github/workflows/", "tools/ci/"))
    )
    unknown_inputs = tuple(path for path in normalized if not recognized_path(path))
    fail_closed_inputs = protected_inputs or unknown_inputs
    if fail_closed_inputs:
        reason = (
            "protected CI input: full selection"
            if protected_inputs
            else "unrecognized input: full selection"
        )
        return ImpactPlan(
            target_authority,
            base_authority,
            False,
            tuple(
                LaneImpact(surface, True, reason, fail_closed_inputs)
                for surface in CLASSIFIED_SURFACES
            ),
        )

    for path in normalized:
        if root_rust_path(path):
            selected["rust"] = True
            owning_inputs["rust"].add(path)
        if msrv_path(path):
            selected["msrv"] = True
            owning_inputs["msrv"].add(path)
        if path.startswith(
            ("bindings/python/", "crates/eqiora-python/")
        ) or public_facade_path(path):
            selected["python"] = True
            owning_inputs["python"].add(path)
        if path.startswith("studio/") or public_facade_path(path):
            selected["studio"] = True
            owning_inputs["studio"].add(path)
        if dependency_policy_path(path):
            selected["dependency_policy"] = True
            owning_inputs["dependency_policy"].add(path)
        if path.startswith("experiments/cubecl-local-action/"):
            selected["cubecl_experiment"] = True
            owning_inputs["cubecl_experiment"].add(path)

    return _impact_plan_from_selection(
        selected,
        owning_inputs,
        target_authority=target_authority,
        base_authority=base_authority,
        unsafe_mode=unsafe_mode,
    )


def _impact_plan_from_selection(
    selected: dict[str, bool],
    owning_inputs: dict[str, set[str]],
    *,
    target_authority: str,
    base_authority: str,
    unsafe_mode: bool,
) -> ImpactPlan:
    decisions: list[LaneImpact] = []
    for surface in CLASSIFIED_SURFACES:
        inputs = tuple(sorted(owning_inputs[surface]))
        if surface == "site" and unsafe_mode:
            reason = "file mode or type change: full build"
        elif selected[surface]:
            reason = "changed input closure"
        else:
            reason = "unchanged input closure"
        decisions.append(LaneImpact(surface, selected[surface], reason, inputs))
    return ImpactPlan(
        target_authority,
        base_authority,
        False,
        tuple(decisions),
    )


def classify(paths: Iterable[str], *, full: bool = False) -> dict[str, bool]:
    """Compatibility projection of the private impact-plan owner."""
    return impact_plan(paths, full=full).selections()


def _raw_changed_paths(revision_range: str) -> tuple[list[str], bool]:
    """Read paths and file modes from one exact raw Git diff."""
    output = subprocess.run(
        ["git", "diff", "--raw", "--no-renames", "-z", revision_range],
        check=True,
        stdout=subprocess.PIPE,
    ).stdout
    fields = output.split(b"\0")
    if fields[-1:] == [b""]:
        fields.pop()
    if len(fields) % 2:
        raise ValueError("raw diff does not contain header/path pairs")

    paths: list[str] = []
    unsafe_mode = False
    for offset in range(0, len(fields), 2):
        header, raw_path = fields[offset : offset + 2]
        match = re.fullmatch(
            rb":([0-7]{6}) ([0-7]{6}) ([0-9a-f]+) ([0-9a-f]+) ([AMDT])",
            header,
        )
        if match is None or not raw_path:
            raise ValueError("raw diff contains an unknown or ambiguous record")
        old_mode, new_mode = match.group(1), match.group(2)
        regular_or_absent = {b"000000", b"100644"}
        if (
            old_mode not in regular_or_absent
            or new_mode not in regular_or_absent
            or (
                old_mode != b"000000" and new_mode != b"000000" and old_mode != new_mode
            )
        ):
            unsafe_mode = True
        paths.append(raw_path.decode("utf-8"))
    return paths, unsafe_mode


def changed_paths(base: str, head: str) -> tuple[list[str], bool]:
    """Read paths and file modes from the complete merge-base diff."""
    return _raw_changed_paths(f"{base}...{head}")


def snapshot_changed_paths(previous: str, current: str) -> tuple[list[str], bool]:
    """Compare exact trees without assuming ancestry after a rebase."""
    return _raw_changed_paths(f"{previous}..{current}")


def apply_previous_run_reuse(
    plan: ImpactPlan,
    *,
    previous_sha: str,
    current_sha: str,
    workflow: str,
    attestation: dict[str, Any],
) -> ImpactPlan:
    """Reuse authenticated heavy work only for snapshot-identical lane inputs."""
    if not previous_sha or not attestation:
        return plan
    previous = exact_commit(previous_sha, "previous pull-request head")
    if (
        attestation.get("version") != 1
        or attestation.get("workflow") != workflow
        or attestation.get("previous_sha") != previous
        or not isinstance(attestation.get("run_id"), int)
        or not isinstance(attestation.get("run_url"), str)
        or not isinstance(attestation.get("lanes"), dict)
    ):
        raise ValueError("previous-run attestation is malformed or mismatched")
    paths, unsafe_mode = snapshot_changed_paths(previous, current_sha)
    closure_delta = impact_plan(
        paths,
        target_authority=current_sha,
        base_authority=previous,
        unsafe_mode=unsafe_mode,
    )
    run_id = attestation["run_id"]
    run_url = attestation["run_url"]
    canonical_run_url = re.fullmatch(
        r"https://github\.com/[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+/actions/runs/[0-9]+",
        run_url,
    )
    if canonical_run_url is None:
        raise ValueError("previous-run attestation has a non-canonical run URL")
    lanes = attestation["lanes"]
    decisions: list[LaneImpact] = []
    for decision in plan.lanes:
        reusable = (
            decision.selected
            and closure_delta.lane(decision.lane).selected is False
            and lanes.get(decision.lane) is True
        )
        if reusable:
            reason = (
                f"reused successful heavy work from run {run_id} from {previous}; "
                f"exact {decision.lane} input closure unchanged; {run_url}"
            )
            decisions.append(LaneImpact(decision.lane, False, reason, ()))
        else:
            decisions.append(decision)
    return ImpactPlan(
        plan.target_authority,
        plan.base_authority,
        plan.full,
        tuple(decisions),
    )


def exact_commit(value: str, label: str) -> str:
    """Require one full SHA naming an available commit object."""
    if FULL_SHA.fullmatch(value) is None:
        raise ValueError(f"{label} is not one exact lowercase commit SHA")
    expected = value
    resolved = subprocess.run(
        ["git", "rev-parse", "--verify", f"{value}^{{commit}}"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    if FULL_SHA.fullmatch(resolved) is None or resolved != expected:
        raise ValueError(f"{label} did not resolve to one exact commit SHA")
    return resolved


def exact_head(expected: str) -> str:
    """Resolve HEAD and, for dispatches, require the exact requested commit."""
    actual = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        check=True,
        text=True,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    if expected:
        if FULL_SHA.fullmatch(expected) is None:
            raise ValueError(
                "manual full verification requires a full lowercase commit SHA"
            )
        if actual != expected:
            raise ValueError(f"checked out {actual}, expected exact commit {expected}")
    return actual


def render_outputs(
    target_sha: str,
    selected: dict[str, bool],
    *,
    full: bool,
    site_source_sha: str = "",
    site_reason: str = "",
    reasons: dict[str, str] | None = None,
) -> str:
    """Render stable GitHub Actions outputs."""
    lines = [f"target_sha={target_sha}", f"full={'true' if full else 'false'}"]
    lines.extend(
        f"{surface}={'true' if selected[surface] else 'false'}" for surface in SURFACES
    )
    reasons = reasons or {}
    lines.extend(
        f"{surface}_reason={reasons.get(surface, 'changed input closure' if selected[surface] else 'unchanged input closure')}"
        for surface in SURFACES
    )
    lines.append(f"site={'true' if selected['site'] else 'false'}")
    lines.append(f"site_source_sha={site_source_sha or target_sha}")
    lines.append(
        "site_reason="
        + (
            site_reason
            or (
                "changed input closure"
                if selected["site"]
                else "unchanged input closure"
            )
        )
    )
    versions = ["3.11", "3.12", "3.13", "3.14"] if full else ["3.11", "3.14"]
    lines.append(f"python_versions={json.dumps(versions, separators=(',', ':'))}")
    return "\n".join(lines)


def append_github_outputs(path: Path, rendered: str) -> None:
    """Validate one complete decision before appending it to GitHub outputs."""
    expected_keys = (
        "target_sha",
        "full",
        *SURFACES,
        *(f"{surface}_reason" for surface in SURFACES),
        "site",
        "site_source_sha",
        "site_reason",
        "python_versions",
    )
    records = [line.partition("=") for line in rendered.splitlines()]
    if any(not separator for _, separator, _ in records):
        raise ValueError("classification output contains a malformed record")
    keys = tuple(key for key, _, _ in records)
    if keys != expected_keys:
        raise ValueError(
            "classification output keys are missing, duplicated, or reordered"
        )
    values = {key: value for key, _, value in records}
    if (
        FULL_SHA.fullmatch(values["target_sha"]) is None
        or FULL_SHA.fullmatch(values["site_source_sha"]) is None
    ):
        raise ValueError("classification output contains an invalid source identity")
    for key in ("full", *SURFACES, "site"):
        if values[key] not in {"true", "false"}:
            raise ValueError(
                f"classification output contains an invalid {key} decision"
            )
    for surface in SURFACES:
        if not values[f"{surface}_reason"]:
            raise ValueError(f"classification output omits the {surface} reason")
    try:
        versions = json.loads(values["python_versions"])
    except json.JSONDecodeError as error:
        raise ValueError(
            "classification output contains invalid Python versions"
        ) from error
    expected_versions = (
        ["3.11", "3.12", "3.13", "3.14"]
        if values["full"] == "true"
        else ["3.11", "3.14"]
    )
    if versions != expected_versions:
        raise ValueError("classification output contains inconsistent Python versions")
    if values["site"] == "false" and (
        values["full"] != "false"
        or (
            values["site_reason"] != "unchanged input closure"
            and not values["site_reason"].startswith("reused successful heavy work from run")
        )
    ):
        raise ValueError("a quick site decision is incomplete or inconsistent")
    if values["full"] == "true" and values["site"] != "true":
        raise ValueError("a full decision must select the documentation site")
    if not values["site_reason"]:
        raise ValueError("classification output omits its decision reason")
    if not path.is_absolute() or (os.path.lexists(path) and path.is_symlink()):
        raise ValueError("GitHub output must be an absolute non-symlink path")
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(rendered)
        handle.write("\n")


def emit_outputs(arguments: argparse.Namespace, rendered: str) -> None:
    if arguments.github_output:
        append_github_outputs(Path(arguments.github_output), rendered)
    else:
        print(rendered)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event", required=True)
    parser.add_argument("--base", default="")
    parser.add_argument("--head", default="")
    parser.add_argument("--requested-commit", default="")
    parser.add_argument("--github-output", default="")
    parser.add_argument("--previous", default="")
    parser.add_argument("--reuse-attestation", default="")
    parser.add_argument("--workflow", choices=("ci.yml", "pages.yml"), default="ci.yml")
    arguments = parser.parse_args()
    if not arguments.previous:
        arguments.previous = os.environ.get("EQIORA_PREVIOUS_SHA", "")
    if not arguments.reuse_attestation:
        arguments.reuse_attestation = os.environ.get("EQIORA_REUSE_ATTESTATION", "")
    environment_workflow = os.environ.get("EQIORA_REUSE_WORKFLOW", "")
    if environment_workflow:
        if environment_workflow not in {"ci.yml", "pages.yml"}:
            print("CI classification failed: invalid reuse workflow", file=sys.stderr)
            return 2
        arguments.workflow = environment_workflow

    try:
        target_sha = exact_head(arguments.requested_commit)
        main_pages = (
            arguments.event == "push"
            and arguments.workflow == "pages.yml"
            and os.environ.get("GITHUB_REF") == "refs/heads/main"
        )
        full = arguments.event != "pull_request"
        if full:
            paths: list[str] = []
            site_source_sha = target_sha
            site_reason = "non-pull-request full build"
        else:
            if not arguments.base or not arguments.head:
                raise ValueError(
                    "pull-request classification requires base and head SHAs"
                )
            site_source_sha = exact_commit(arguments.base, "pull-request base")
            exact_commit(arguments.head, "pull-request head")
            paths, unsafe_mode = changed_paths(arguments.base, arguments.head)
            if not paths:
                raise ValueError("pull-request diff is unexpectedly empty")
        plan = impact_plan(
            paths,
            full=full,
            target_authority=target_sha,
            base_authority=site_source_sha,
            unsafe_mode=unsafe_mode if not full else False,
        )
        if (not full or main_pages) and arguments.previous and arguments.reuse_attestation:
            attestation_path = Path(arguments.reuse_attestation)
            if not attestation_path.is_absolute() or attestation_path.is_symlink():
                raise ValueError(
                    "reuse attestation must be an absolute non-symlink path"
                )
            attestation = json.loads(attestation_path.read_text(encoding="utf-8"))
            if not isinstance(attestation, dict):
                raise ValueError("reuse attestation is not an object")
            if main_pages and (
                attestation.get("event") != "push" or attestation.get("pull_request") != 0
            ):
                raise ValueError("main Pages requires a successful deployment attestation")
            plan = apply_previous_run_reuse(
                plan,
                previous_sha=arguments.previous,
                current_sha=target_sha,
                workflow=arguments.workflow,
                attestation=attestation,
            )
            if main_pages and not plan.lane("site").selected:
                full = False
                unsafe_mode = False
                plan = replace(plan, full=False)
        selected = plan.selections()
        if not full:
            if selected["site"]:
                site_source_sha = target_sha
                if unsafe_mode:
                    site_reason = "file mode or type change: full build"
                elif any(not recognized_path(path) for path in paths):
                    site_reason = "unrecognized input: full build"
                else:
                    site_reason = plan.lane("site").reason
            else:
                site_reason = plan.lane("site").reason
                if site_reason.startswith("reused successful heavy work from run"):
                    site_source_sha = arguments.previous
    except (
        OSError,
        subprocess.CalledProcessError,
        UnicodeDecodeError,
        ValueError,
    ) as error:
        if arguments.event == "pull_request":
            try:
                target_sha = exact_head("")
            except (OSError, subprocess.CalledProcessError, ValueError):
                pass
            else:
                selected = impact_plan(
                    [],
                    full=True,
                    target_authority=target_sha,
                    base_authority=target_sha,
                ).selections()
                print(f"CI classification failed closed: {error}", file=sys.stderr)
                rendered = render_outputs(
                    target_sha,
                    selected,
                    full=True,
                    site_source_sha=target_sha,
                    site_reason="classification failure: full build",
                )
                try:
                    emit_outputs(arguments, rendered)
                except (OSError, ValueError) as output_error:
                    print(
                        f"CI classification output failed: {output_error}",
                        file=sys.stderr,
                    )
                    return 2
                return 0
        print(f"CI classification failed: {error}", file=sys.stderr)
        return 2

    rendered = render_outputs(
        target_sha,
        selected,
        full=full,
        site_source_sha=site_source_sha,
        site_reason=site_reason,
        reasons={decision.lane: decision.reason for decision in plan.lanes},
    )
    try:
        emit_outputs(arguments, rendered)
    except (OSError, ValueError) as error:
        print(f"CI classification output failed: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
