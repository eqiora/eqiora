from __future__ import annotations

import io
import json
import sys
import unittest
from pathlib import Path
from unittest import mock


REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPOSITORY_ROOT / "tools/release"))

from validate_candidate_run import (  # noqa: E402
    CandidateRunError,
    fetch_candidate_run,
    validate_candidate_run,
)


class Response(io.BytesIO):
    def __enter__(self) -> "Response":
        return self

    def __exit__(self, *arguments: object) -> None:
        self.close()


class CandidateRunTests(unittest.TestCase):
    repository = "nkiyohara/eqiora"
    run_id = 12345
    commit = "a" * 40
    tag = "v0.1.0"

    def accepted_payload(self) -> dict[str, object]:
        return {
            "id": self.run_id,
            "event": "workflow_dispatch",
            "path": ".github/workflows/python-release-candidate.yml",
            "head_branch": "main",
            "head_sha": self.commit,
            "display_title": f"Python candidate / {self.commit}",
            "status": "completed",
            "conclusion": "success",
            "repository": {"full_name": self.repository},
            "head_repository": {"full_name": self.repository},
        }

    def test_accepts_only_the_complete_successful_release_commit_run(self) -> None:
        for path in (
            ".github/workflows/python-release-candidate.yml",
            ".github/workflows/python-release-candidate.yml@main",
        ):
            with self.subTest(path=path):
                payload = self.accepted_payload()
                payload["path"] = path
                validate_candidate_run(
                    payload,
                    repository=self.repository,
                    run_id=self.run_id,
                    expected_commit=self.commit,
                    expected_tag=self.tag,
                )

        payload = self.accepted_payload()
        payload["head_branch"] = self.tag
        validate_candidate_run(
            payload,
            repository=self.repository,
            run_id=self.run_id,
            expected_commit=self.commit,
            expected_tag=self.tag,
        )

    def test_rejects_incomplete_failed_or_unrelated_runs(self) -> None:
        mutations = {
            "wrong ID": ("id", self.run_id + 1),
            "wrong event": ("event", "pull_request"),
            "wrong workflow": ("path", ".github/workflows/ci.yml"),
            "workflow from unprotected ref": (
                "path",
                ".github/workflows/python-release-candidate.yml@release-candidate",
            ),
            "unprotected ref": ("head_branch", "release-candidate"),
            "wrong definition commit": ("head_sha", "b" * 40),
            "wrong dispatch input": (
                "display_title",
                f"Python candidate / {'b' * 40}",
            ),
            "still running": ("status", "in_progress"),
            "failed replay": ("conclusion", "failure"),
        }
        for label, (key, value) in mutations.items():
            with self.subTest(label=label):
                payload = self.accepted_payload()
                payload[key] = value
                with self.assertRaises(CandidateRunError):
                    validate_candidate_run(
                        payload,
                        repository=self.repository,
                        run_id=self.run_id,
                        expected_commit=self.commit,
                        expected_tag=self.tag,
                    )

    def test_rejects_a_run_from_another_repository(self) -> None:
        for key in ("repository", "head_repository"):
            with self.subTest(key=key):
                payload = self.accepted_payload()
                payload[key] = {"full_name": "attacker/eqiora"}
                with self.assertRaisesRegex(
                    CandidateRunError, "differs from the release repository"
                ):
                    validate_candidate_run(
                        payload,
                        repository=self.repository,
                        run_id=self.run_id,
                        expected_commit=self.commit,
                        expected_tag=self.tag,
                    )

    def test_fetch_uses_the_read_only_run_endpoint_and_bearer_token(self) -> None:
        payload = self.accepted_payload()
        opener = mock.Mock(return_value=Response(json.dumps(payload).encode()))

        observed = fetch_candidate_run(
            api_url="https://api.github.test",
            repository=self.repository,
            run_id=self.run_id,
            token="not-a-real-token",
            opener=opener,
        )

        self.assertEqual(observed, payload)
        request = opener.call_args.args[0]
        self.assertEqual(
            request.full_url,
            f"https://api.github.test/repos/nkiyohara/eqiora/actions/runs/{self.run_id}",
        )
        self.assertEqual(request.headers["Authorization"], "Bearer not-a-real-token")


class ProductionWorkflowTests(unittest.TestCase):
    def test_authenticates_the_candidate_before_cross_run_artifact_download(self) -> None:
        workflow = (
            REPOSITORY_ROOT / ".github/workflows/python-production-publish.yml"
        ).read_text(encoding="utf-8")
        candidate = (
            REPOSITORY_ROOT / ".github/workflows/python-release-candidate.yml"
        ).read_text(encoding="utf-8")
        verify_job = workflow.split("  verify:\n", maxsplit=1)[1].split(
            "\n  publish:", maxsplit=1
        )[0]

        authentication = verify_job.index(
            'python3 "$RUNNER_TEMP/validate_candidate_run.py"'
        )
        download = verify_job.index("actions/download-artifact@")
        self.assertLess(authentication, download)
        self.assertIn("actions: read", verify_job)
        self.assertIn('--repository "$GITHUB_REPOSITORY"', verify_job)
        self.assertIn('--expected-commit "$RELEASE_COMMIT"', verify_job)
        self.assertIn('--expected-tag "${{ inputs.tag }}"', verify_job)
        self.assertIn('git cat-file -t "$tag_object"', verify_job)
        self.assertIn('git cat-file -t "$peeled_commit"', verify_job)
        self.assertIn("run-name: Python candidate / ${{ inputs.commit }}", candidate)


if __name__ == "__main__":
    unittest.main()
