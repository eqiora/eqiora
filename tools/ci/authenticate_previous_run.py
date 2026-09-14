#!/usr/bin/env python3
"""Authenticate previous PR work or the latest successful main Pages deployment.

This helper deliberately makes no lane-selection decision.  It only emits an
ephemeral attestation for exact successful jobs and witness steps from the
same pull request, or the main run that actually deployed Pages. The classifier proves that a
lane's complete input closure is unchanged before consuming the attestation.
Any unavailable or ambiguous GitHub state produces an empty attestation.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Callable


FULL_SHA = re.compile(r"[0-9a-f]{40}")

CI_WITNESSES = {
    "rust": (("Stable quality gate", "Tests"),),
    "msrv": (("MSRV 1.89", "Check every production feature and target"),),
    "python": (
        ("Python 3.11 installed wheel", "Test installed wheel"),
        ("Python 3.14 installed wheel", "Test installed wheel"),
    ),
    "studio": (("Studio browser projection", "Accessible interaction tests"),),
    "dependency_policy": (
        ("Dependency policy", "Check root dependency policy"),
    ),
    "cubecl_experiment": (
        ("Isolated CubeCL contract experiment", "Device-independent contract tests"),
    ),
}

PAGES_WITNESSES = {
    "site": (
        (
            "Build and verify static documentation",
            "Build and verify with only loopback networking",
        ),
    ),
}


def _request_json(url: str, token: str) -> Any:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "eqiora-ci-reuse",
        },
    )
    with urllib.request.urlopen(request, timeout=20) as response:
        if response.status != 200:
            raise ValueError(f"GitHub API returned HTTP {response.status}")
        payload = json.load(response)
    return payload


def authenticate(
    *,
    repository: str,
    pull_request: int,
    previous_sha: str,
    workflow: str,
    fetch: Callable[[str], Any],
) -> dict[str, Any]:
    """Return an exact prior-run attestation or raise on uncertain state."""
    if FULL_SHA.fullmatch(previous_sha) is None:
        raise ValueError("previous head is not one exact commit SHA")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("repository identity is malformed")
    if pull_request <= 0:
        raise ValueError("pull-request number is invalid")
    if workflow not in {"ci.yml", "pages.yml"}:
        raise ValueError("workflow is not eligible for reuse")

    encoded_workflow = urllib.parse.quote(workflow, safe="")
    query = urllib.parse.urlencode(
        {
            "event": "pull_request",
            "head_sha": previous_sha,
            "status": "completed",
            "per_page": "100",
        }
    )
    runs_url = (
        f"https://api.github.com/repos/{repository}/actions/workflows/"
        f"{encoded_workflow}/runs?{query}"
    )
    runs_payload = fetch(runs_url)
    runs = runs_payload.get("workflow_runs")
    if not isinstance(runs, list):
        raise ValueError("workflow-run response omits its run list")
    if runs_payload.get("total_count") != len(runs):
        raise ValueError("workflow-run response is incomplete")
    candidates = []
    for run in runs:
        if not isinstance(run, dict):
            continue
        if (
            run.get("head_sha") == previous_sha
            and run.get("event") == "pull_request"
            and run.get("status") == "completed"
            and (
                run.get("conclusion") == "success"
                or (workflow == "ci.yml" and run.get("conclusion") == "failure")
            )
            and isinstance(run.get("id"), int)
            # A rebase removes the old head from the live commit-to-PR index.
            # The completed run retains its PR association; its head_sha above
            # identifies the tested commit, not the PR object's current head.
            and isinstance(run.get("pull_requests"), list)
            and any(
                isinstance(pull, dict) and pull.get("number") == pull_request
                for pull in run["pull_requests"]
            )
        ):
            candidates.append(run)
    if not candidates:
        raise ValueError("did not find an eligible completed prior workflow run")
    if len({run["id"] for run in candidates}) != len(candidates):
        raise ValueError("duplicate prior workflow run identity")

    newest = None
    for run in sorted(candidates, key=lambda candidate: candidate["id"], reverse=True):
        attestation = _authenticate_pr_run(
            run, repository=repository, pull_request=pull_request,
            previous_sha=previous_sha, workflow=workflow, fetch=fetch,
        )
        if newest is None:
            newest = attestation
        if any(attestation["lanes"].values()):
            return attestation
    assert newest is not None
    return newest


def _authenticate_pr_run(
    run: dict[str, Any], *, repository: str, pull_request: int,
    previous_sha: str, workflow: str, fetch: Callable[[str], Any],
) -> dict[str, Any]:
    """Bind one run without combining witnesses from different executions."""
    run_id = run["id"]
    expected_run_url = f"https://github.com/{repository}/actions/runs/{run_id}"
    if run.get("html_url") != expected_run_url:
        raise ValueError("prior workflow run URL is not its canonical repository URL")
    jobs_payload = fetch(
        f"https://api.github.com/repos/{repository}/actions/runs/{run_id}/jobs"
        "?filter=latest&per_page=100"
    )
    jobs = jobs_payload.get("jobs")
    if not isinstance(jobs, list):
        raise ValueError("workflow-job response omits its job list")
    if jobs_payload.get("total_count") != len(jobs):
        raise ValueError("workflow-job response is incomplete")

    successful_steps: set[tuple[str, str]] = set()
    for job in jobs:
        if (
            not isinstance(job, dict)
            or job.get("status") != "completed"
            or job.get("conclusion") != "success"
        ):
            continue
        job_name = job.get("name")
        steps = job.get("steps")
        if not isinstance(job_name, str) or not isinstance(steps, list):
            continue
        for step in steps:
            if (
                isinstance(step, dict)
                and isinstance(step.get("name"), str)
                and step.get("status") == "completed"
                and step.get("conclusion") == "success"
            ):
                successful_steps.add((job_name, step["name"]))

    witnesses = CI_WITNESSES if workflow == "ci.yml" else PAGES_WITNESSES
    lanes = {
        lane: all(witness in successful_steps for witness in required)
        for lane, required in witnesses.items()
    }
    return {
        "version": 1,
        "repository": repository,
        "pull_request": pull_request,
        "workflow": workflow,
        "previous_sha": previous_sha,
        "run_id": run_id,
        "run_url": expected_run_url,
        "lanes": lanes,
    }


def authenticate_deployed_pages(
    *, repository: str, fetch: Callable[[str], Any]
) -> dict[str, Any]:
    """Bind the latest Pages deployment only after its publishing run succeeded."""
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", repository):
        raise ValueError("repository identity is malformed")
    api = f"https://api.github.com/repos/{repository}"
    deployments = fetch(f"{api}/deployments?environment=github-pages&per_page=1")
    if not isinstance(deployments, list) or len(deployments) != 1:
        raise ValueError("latest Pages deployment is unavailable")
    deployment = deployments[0]
    if (
        not isinstance(deployment, dict)
        or type(deployment.get("id")) is not int
        or deployment.get("environment") != "github-pages"
        or deployment.get("ref") != "main"
        or not isinstance(deployment.get("sha"), str)
        or FULL_SHA.fullmatch(deployment["sha"]) is None
    ):
        raise ValueError("latest deployment is not an exact main Pages deployment")
    statuses = fetch(f"{api}/deployments/{deployment['id']}/statuses?per_page=1")
    if (
        not isinstance(statuses, list)
        or len(statuses) != 1
        or not isinstance(statuses[0], dict)
        or statuses[0].get("state") != "success"
        or statuses[0].get("environment") != "github-pages"
    ):
        raise ValueError("latest Pages deployment did not succeed")
    match = re.fullmatch(
        rf"https://github\.com/{re.escape(repository)}/actions/runs/([0-9]+)/job/([0-9]+)",
        str(statuses[0].get("log_url", "")),
    )
    if match is None:
        raise ValueError("deployment has no canonical publishing job")
    run_id, job_id = map(int, match.groups())
    run = fetch(f"{api}/actions/runs/{run_id}")
    if (
        not isinstance(run, dict)
        or run.get("id") != run_id
        or run.get("event") != "push"
        or run.get("head_branch") != "main"
        or run.get("head_sha") != deployment["sha"]
        or run.get("path") != ".github/workflows/pages.yml"
        or run.get("status") != "completed"
        or run.get("conclusion") != "success"
    ):
        raise ValueError("publishing run is not a successful main Pages push")
    payload = fetch(f"{api}/actions/runs/{run_id}/jobs?filter=latest&per_page=100")
    if not isinstance(payload, dict):
        raise ValueError("publishing job response is malformed")
    jobs = payload.get("jobs")
    if not isinstance(jobs, list) or payload.get("total_count") != len(jobs):
        raise ValueError("publishing job list is incomplete")
    required = {
        (
            "Build and verify static documentation",
            "Build and verify with only loopback networking",
        ),
        ("Deploy static documentation", "Deploy GitHub Pages artifact"),
    }
    completed = set()
    for job in jobs:
        if not isinstance(job, dict) or job.get("conclusion") != "success":
            continue
        if not isinstance(job.get("name"), str):
            continue
        if job.get("name") == "Deploy static documentation" and job.get("id") != job_id:
            continue
        steps = job.get("steps")
        if not isinstance(steps, list):
            continue
        for step in steps:
            if (
                isinstance(step, dict)
                and isinstance(step.get("name"), str)
                and step.get("status") == "completed"
                and step.get("conclusion") == "success"
            ):
                completed.add((job.get("name"), step.get("name")))
    if not required <= completed:
        raise ValueError("publishing run lacks successful build and deployment steps")
    return {
        "version": 1,
        "repository": repository,
        "pull_request": 0,
        "workflow": "pages.yml",
        "event": "push",
        "previous_sha": deployment["sha"],
        "run_id": run_id,
        "run_url": f"https://github.com/{repository}/actions/runs/{run_id}",
        "lanes": {"site": True},
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True)
    parser.add_argument("--pull-request", required=True, type=int)
    parser.add_argument("--previous-sha", required=True)
    parser.add_argument("--workflow", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--deployed-pages", action="store_true")
    arguments = parser.parse_args()
    output = Path(arguments.output)
    if not output.is_absolute() or (os.path.lexists(output) and output.is_symlink()):
        print(
            "reuse attestation output must be an absolute non-symlink path",
            file=sys.stderr,
        )
        return 2
    token = os.environ.get("GITHUB_TOKEN", "")
    attestation: dict[str, Any] = {}
    try:
        if not token:
            raise ValueError("GitHub token is unavailable")
        if arguments.deployed_pages:
            if arguments.workflow != "pages.yml" or arguments.pull_request != 0:
                raise ValueError("deployment reuse is only available for main Pages")
            attestation = authenticate_deployed_pages(
                repository=arguments.repository,
                fetch=lambda url: _request_json(url, token),
            )
        else:
            attestation = authenticate(
                repository=arguments.repository,
                pull_request=arguments.pull_request,
                previous_sha=arguments.previous_sha,
                workflow=arguments.workflow,
                fetch=lambda url: _request_json(url, token),
            )
    except (OSError, ValueError, urllib.error.URLError, json.JSONDecodeError) as error:
        print(
            f"Previous-run reuse unavailable; running current lanes: {error}",
            file=sys.stderr,
        )
    output.write_text(json.dumps(attestation, sort_keys=True) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
