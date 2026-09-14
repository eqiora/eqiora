from __future__ import annotations

import io
import json
import os
import copy
import subprocess
import sys
import unittest
import urllib.error
import urllib.parse
from contextlib import redirect_stderr, redirect_stdout
from email.message import Message
from pathlib import Path
from unittest import mock


CI_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(CI_ROOT))

import check_trust_boundary as trust_boundary  # noqa: E402
from check_trust_boundary import (  # noqa: E402
    changed_file_names,
    fetch_changed_files,
    protected_changes,
    protected_path,
)


class Response(io.BytesIO):
    def __init__(
        self,
        payload: bytes,
        *,
        content_type: str = "application/json",
        content_length: int | None = None,
        include_content_length: bool = True,
    ) -> None:
        super().__init__(payload)
        self.status = 200
        self.headers = Message()
        self.headers["Content-Type"] = content_type
        if include_content_length:
            self.headers["Content-Length"] = str(
                len(payload) if content_length is None else content_length
            )

    def getcode(self) -> int:
        return self.status

    def __enter__(self) -> "Response":
        return self

    def __exit__(self, *arguments: object) -> None:
        self.close()


BASE_REPOSITORY = "base-owner/eqiora"
HEAD_REPOSITORY = "base-owner/eqiora"
BASE_SHA = "1" * 40
HEAD_SHA = "2" * 40
PULL_NUMBER = 461
ARCHITECTURE_DEBT = "tools/ci/architecture-debt.toml"
FORMATTER = "crates/eqiora-lang/src/formatter.rs"
PARSER = "crates/eqiora-lang/src/parser.rs"
FROZEN_MAX_RAW_BLOB_BYTES = 1_048_576

FORMATTER_DEBT_ENTRY = b"""[[file_lines]]\n\
path = "crates/eqiora-lang/src/formatter.rs"\n\
ceiling = 1050\n\
reason = "existing formatter debt"\n\
removal = "split formatter"\n\
\n"""

BASE_LEDGER = b"""# Architecture debt ledger\n\
[limits]\n\
production_file_lines = 1000\n\
test_file_lines = 2000\n\
public_items_per_crate = 128\n\
\n\
[[file_lines]]\n\
path = "crates/eqiora-lang/src/formatter.rs"\n\
ceiling = 1050\n\
reason = "existing formatter debt"\n\
removal = "split formatter"\n\
\n\
[[file_lines]]\n\
path = "crates/eqiora-lang/src/parser.rs"\n\
ceiling = 2701\n\
reason = "existing parser debt"\n\
removal = "split parser"\n\
\n\
[[file_lines]]\n\
path = "crates/eqiora-schema/src/model.rs"\n\
ceiling = 1200\n\
reason = "unrelated existing debt"\n\
removal = "split model"\n\
\n\
[[public_surface]]\n\
crate = "eqiora-lang"\n\
ceiling = 128\n\
reason = "existing public surface"\n\
removal = "name a smaller facade"\n\
\n\
[[glob_reexports]]\n\
path = "crates/eqiora-lang/src/lib.rs"\n\
identities = ["always | syntax::* | module file crates/eqiora-lang/src/syntax.rs"]\n\
reason = "existing glob"\n\
removal = "name exports"\n"""


def physical_lines(count: int, *, final_newline: bool = True) -> bytes:
    if count == 0:
        return b""
    if final_newline:
        return b"// inert source line\n" * count
    return b"// inert source line\n" * (count - 1) + b"// inert source line"


def sized_physical_source(size: int, lines: int = 1010) -> bytes:
    prefix = b"x\n" * (lines - 1)
    return prefix + b"x" * (size - len(prefix))


def replace_once(payload: bytes, old: bytes, new: bytes) -> bytes:
    if payload.count(old) != 1:
        raise AssertionError(f"fixture token is not unique: {old!r}")
    return payload.replace(old, new, 1)


def exact_ratchet_ledger(*, include_parser: bool = True) -> bytes:
    candidate = replace_once(BASE_LEDGER, b"ceiling = 1050", b"ceiling = 1010")
    if include_parser:
        candidate = replace_once(candidate, b"ceiling = 2701", b"ceiling = 2608")
    return candidate


def ordinary_files(count: int) -> list[dict[str, str]]:
    return [
        {
            "filename": f"docs/ordinary-{index}.md",
            "status": "modified",
            "sha": f"{index + 1000:040x}",
        }
        for index in range(count)
    ]


class FakeGitHub:
    """Small authenticated GitHub API double; candidate URLs are never routes."""

    def __init__(
        self,
        *,
        include_parser: bool = True,
        head_repository: str = HEAD_REPOSITORY,
    ) -> None:
        product_files = [
            {
                "filename": FORMATTER,
                "status": "modified",
                "sha": "3" * 40,
                "raw_url": "https://candidate.invalid/execute-head.py",
                "contents_url": "https://candidate.invalid/execute-head.py",
            },
            {
                "filename": ARCHITECTURE_DEBT,
                "status": "modified",
                "sha": "4" * 40,
                "raw_url": "https://candidate.invalid/untrusted-ledger.toml",
                "contents_url": "https://candidate.invalid/untrusted-ledger.toml",
            },
        ]
        if include_parser:
            product_files.insert(
                1,
                {
                    "filename": PARSER,
                    "status": "modified",
                    "sha": "5" * 40,
                    "raw_url": "https://candidate.invalid/execute-parser.py",
                    "contents_url": "https://candidate.invalid/execute-parser.py",
                },
            )
            product_files.extend(
                {
                    "filename": f"docs/natural-equation-{index}.md",
                    "status": "modified",
                    "sha": f"{index + 10:040x}",
                }
                for index in range(12)
            )

        self.files = product_files
        self.pull = {
            "number": PULL_NUMBER,
            "changed_files": len(self.files),
            "base": {
                "sha": BASE_SHA,
                "repo": {"full_name": BASE_REPOSITORY},
            },
            "head": {
                "sha": HEAD_SHA,
                "repo": {"full_name": head_repository},
            },
        }
        self.event_file_count = len(self.files)
        self.blobs: dict[tuple[str, str, str], bytes] = {
            (BASE_REPOSITORY, BASE_SHA, ARCHITECTURE_DEBT): BASE_LEDGER,
            (
                head_repository,
                HEAD_SHA,
                ARCHITECTURE_DEBT,
            ): exact_ratchet_ledger(include_parser=include_parser),
            (BASE_REPOSITORY, BASE_SHA, FORMATTER): physical_lines(1050),
            (
                head_repository,
                HEAD_SHA,
                FORMATTER,
            ): physical_lines(1010, final_newline=False),
            (BASE_REPOSITORY, BASE_SHA, PARSER): physical_lines(2701),
            (head_repository, HEAD_SHA, PARSER): physical_lines(2608),
        }
        self.content_types: dict[tuple[str, str, str], str] = {}
        self.declared_lengths: dict[tuple[str, str, str], int] = {}
        self.omitted_lengths: set[tuple[str, str, str]] = set()
        self.requests: list[object] = []
        self.http_failure_path: str | None = None
        self.pull_content_type = "application/json"
        self.files_content_type = "application/json"
        self.identity_reads = 0
        self.move_boundary: str | None = None
        self.compare_files = copy.deepcopy(self.files)
        self.compare_overrides: dict[str, object] = {}
        self.compare_reads = 0
        self.compare_http_error = False
        self.compare_link: str | None = None
        self.compare_content_type = "application/json"
        self.aba_restore_pending = False
        self.pull_files_reads = 0

    def _move_head(self) -> None:
        self.pull["head"]["sha"] = "c" * 40

    def _response(self, payload: bytes, *, content_type: str) -> Response:
        return Response(payload, content_type=content_type)

    def open(self, request: object, timeout: int = 0) -> Response:
        del timeout
        self.requests.append(request)
        full_url = getattr(request, "full_url")
        parsed = urllib.parse.urlparse(full_url)
        if parsed.scheme != "https" or parsed.netloc != "api.github.test":
            raise AssertionError(f"untrusted or unexpected URL followed: {full_url}")

        headers = getattr(request, "headers")
        if headers.get("Authorization") != "Bearer not-a-real-token":
            raise AssertionError("every API request must remain authenticated")
        if headers.get("X-github-api-version") != "2022-11-28":
            raise AssertionError("every API request must pin the GitHub API version")

        decoded_path = urllib.parse.unquote(parsed.path)
        if self.aba_restore_pending:
            self.pull["head"]["sha"] = HEAD_SHA
            self.aba_restore_pending = False
        if self.http_failure_path and self.http_failure_path in decoded_path:
            raise urllib.error.HTTPError(full_url, 503, "unavailable", {}, None)

        prefix = f"/repos/{BASE_REPOSITORY}/pulls/{PULL_NUMBER}"
        if decoded_path == prefix and not parsed.query:
            self.identity_reads += 1
            return self._response(
                json.dumps(self.pull).encode(), content_type=self.pull_content_type
            )
        if decoded_path == f"{prefix}/files":
            self.pull_files_reads += 1
            if self.move_boundary == "files":
                self._move_head()
            if self.move_boundary == "aba":
                self._move_head()
                self.aba_restore_pending = True
            page = urllib.parse.parse_qs(parsed.query).get("page", ["1"])[0]
            payload = self.files if page == "1" else []
            return self._response(
                json.dumps(payload).encode(), content_type=self.files_content_type
            )

        compare_path = f"/repos/{BASE_REPOSITORY}/compare/{BASE_SHA}...{HEAD_SHA}"
        if decoded_path == compare_path and not parsed.query:
            self.compare_reads += 1
            if self.compare_http_error:
                raise urllib.error.HTTPError(full_url, 503, "unavailable", {}, None)
            payload: dict[str, object] = {
                "status": "ahead",
                "ahead_by": 1,
                "behind_by": 0,
                "total_commits": 1,
                "base_commit": {"sha": BASE_SHA},
                "merge_base_commit": {"sha": BASE_SHA},
                "commits": [{"sha": HEAD_SHA}],
                "files": self.compare_files,
            }
            payload.update(self.compare_overrides)
            response = self._response(
                json.dumps(payload).encode(), content_type=self.compare_content_type
            )
            if self.compare_link is not None:
                response.headers["Link"] = self.compare_link
            return response

        marker = "/contents/"
        if marker in decoded_path:
            if self.move_boundary == "content":
                self._move_head()
            repository_path, blob_path = decoded_path.split(marker, maxsplit=1)
            repository = repository_path.removeprefix("/repos/")
            ref = urllib.parse.parse_qs(parsed.query).get("ref", [""])[0]
            key = (repository, ref, blob_path)
            if key not in self.blobs:
                raise urllib.error.HTTPError(full_url, 404, "missing", {}, None)
            payload = self.blobs[key]
            return Response(
                payload,
                content_type=self.content_types.get(key, "application/octet-stream"),
                content_length=self.declared_lengths.get(key),
                include_content_length=key not in self.omitted_lengths,
            )

        raise urllib.error.HTTPError(full_url, 404, "unexpected route", {}, None)


class CoupledRatchetEvidenceTests(unittest.TestCase):
    def run_classifier(
        self,
        api: FakeGitHub,
        *,
        base_repository: str = BASE_REPOSITORY,
        head_repository: str = HEAD_REPOSITORY,
        base_sha: str = BASE_SHA,
        head_sha: str = HEAD_SHA,
        pull_number: int = PULL_NUMBER,
        expected_file_count: int | None = None,
    ) -> tuple[int, str, str]:
        count = (
            api.event_file_count if expected_file_count is None else expected_file_count
        )
        arguments = [
            "check_trust_boundary.py",
            "--api-url",
            "https://api.github.test",
            "--repository",
            base_repository,
            "--head-repository",
            head_repository,
            "--pull-number",
            str(pull_number),
            "--expected-file-count",
            str(count),
            "--base-sha",
            base_sha,
            "--head-sha",
            head_sha,
        ]
        stdout = io.StringIO()
        stderr = io.StringIO()
        with (
            mock.patch.object(sys, "argv", arguments),
            mock.patch.dict(os.environ, {"GITHUB_TOKEN": "not-a-real-token"}),
            mock.patch.object(
                trust_boundary.urllib.request, "urlopen", side_effect=api.open
            ),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            try:
                result = trust_boundary.main()
            except SystemExit as error:
                result = int(error.code)
        self.assertEqual(
            api.pull_files_reads,
            0,
            "trust decisions must not consult mutable /pulls/{number}/files",
        )
        return result, stdout.getvalue(), stderr.getvalue()

    def assert_certified(self, api: FakeGitHub) -> None:
        result, stdout, stderr = self.run_classifier(api)
        self.assertEqual(result, 0, stderr)
        self.assertIn("coupled exact architecture ratchet", stdout.lower())
        self.assertEqual(api.compare_reads, 1)
        self.assertEqual(
            api.pull_files_reads,
            0,
            "mutable pull-file metadata cannot authorize any success path",
        )

    def assert_rejected(self, api: FakeGitHub, **arguments: object) -> None:
        result, stdout, stderr = self.run_classifier(api, **arguments)
        self.assertNotEqual(result, 0, stdout)
        self.assertTrue(stderr.strip(), "a rejection must explain itself on stderr")
        self.assertNotIn("unrecognized arguments", stderr.lower())

    def assert_ordinary_certified(self, api: FakeGitHub) -> None:
        result, stdout, stderr = self.run_classifier(api)
        self.assertEqual(result, 0, stderr)
        self.assertIn("does not change protected", stdout.lower())
        self.assertEqual(api.compare_reads, 1)
        self.assertEqual(api.pull_files_reads, 0)

    def retired_formatter_api(self, *, head_lines: int = 1000) -> FakeGitHub:
        api = FakeGitHub(include_parser=False)
        api.blobs[(HEAD_REPOSITORY, HEAD_SHA, ARCHITECTURE_DEBT)] = replace_once(
            BASE_LEDGER, FORMATTER_DEBT_ENTRY, b""
        )
        api.blobs[(HEAD_REPOSITORY, HEAD_SHA, FORMATTER)] = physical_lines(head_lines)
        return api

    def test_00_single_existing_file_line_ceiling_can_repay_exactly(self) -> None:
        api = FakeGitHub(include_parser=False)
        self.assert_certified(api)

    def test_file_line_exception_can_retire_at_the_ordinary_role_limit(self) -> None:
        self.assert_certified(self.retired_formatter_api())

    def test_file_line_exception_cannot_retire_while_the_source_exceeds_the_limit(
        self,
    ) -> None:
        self.assert_rejected(self.retired_formatter_api(head_lines=1001))

    def test_file_line_retirement_cannot_substitute_another_path(self) -> None:
        api = self.retired_formatter_api()
        replacement = FORMATTER_DEBT_ENTRY.replace(
            FORMATTER.encode(), b"crates/eqiora-lang/src/replacement.rs"
        )
        api.blobs[(HEAD_REPOSITORY, HEAD_SHA, ARCHITECTURE_DEBT)] = replace_once(
            BASE_LEDGER, FORMATTER_DEBT_ENTRY, replacement
        )
        self.assert_rejected(api)

    def test_file_line_retirement_does_not_allow_deleting_another_section(self) -> None:
        api = self.retired_formatter_api()
        public_surface = b"""[[public_surface]]\n\
crate = "eqiora-lang"\n\
ceiling = 128\n\
reason = "existing public surface"\n\
removal = "name a smaller facade"\n\
\n"""
        api.blobs[(HEAD_REPOSITORY, HEAD_SHA, ARCHITECTURE_DEBT)] = replace_once(
            api.blobs[(HEAD_REPOSITORY, HEAD_SHA, ARCHITECTURE_DEBT)],
            public_surface,
            b"",
        )
        self.assert_rejected(api)

    def test_01_pr461_shaped_pair_is_certified_at_exact_physical_counts(self) -> None:
        api = FakeGitHub()
        self.assertEqual(api.pull["changed_files"], 15)
        self.assertEqual(
            api.blobs[(HEAD_REPOSITORY, HEAD_SHA, ARCHITECTURE_DEBT)],
            exact_ratchet_ledger(),
        )
        formatter = api.blobs[(HEAD_REPOSITORY, HEAD_SHA, FORMATTER)]
        parser = api.blobs[(HEAD_REPOSITORY, HEAD_SHA, PARSER)]
        self.assertFalse(formatter.endswith(b"\n"))
        self.assertEqual(formatter.count(b"\n") + 1, 1010)
        self.assertEqual(parser.count(b"\n"), 2608)
        self.assert_certified(api)

    def test_02_bound_fork_head_repository_is_supported(self) -> None:
        fork = "fork-owner/eqiora"
        api = FakeGitHub(include_parser=False, head_repository=fork)
        result, stdout, stderr = self.run_classifier(api, head_repository=fork)
        self.assertEqual(result, 0, stderr)
        self.assertIn("coupled exact architecture ratchet", stdout.lower())

    def test_03_head_blobs_are_inert_and_candidate_urls_are_never_followed(
        self,
    ) -> None:
        api = FakeGitHub(include_parser=False)
        executable_line = b'raise RuntimeError("head source was executed")\n'
        api.blobs[(HEAD_REPOSITORY, HEAD_SHA, FORMATTER)] = executable_line * 1010
        with (
            mock.patch("builtins.exec", side_effect=AssertionError("head exec")),
            mock.patch.object(
                subprocess, "Popen", side_effect=AssertionError("head subprocess")
            ),
        ):
            self.assert_certified(api)
        self.assertTrue(api.requests)
        self.assertEqual(
            {
                urllib.parse.urlparse(request.full_url).netloc
                for request in api.requests
            },
            {"api.github.test"},
        )

    def test_05_changed_file_inventory_stays_bound_during_acquisition(self) -> None:
        stable = FakeGitHub()
        self.assert_certified(stable)
        self.assertEqual(stable.compare_reads, 1)
        self.assertTrue(
            any(
                f"/compare/{BASE_SHA}...{HEAD_SHA}" in request.full_url
                for request in stable.requests
            )
        )

        aba = FakeGitHub()
        aba.move_boundary = "aba"
        aba.compare_files[-1] = {
            "filename": ".github/workflows/ci.yml",
            "status": "modified",
            "sha": "f" * 40,
        }
        self.assert_rejected(aba)
        self.assertEqual(aba.pull["head"]["sha"], HEAD_SHA)
        self.assertEqual(aba.compare_reads, 1)

    def test_06_immutable_compare_identity_and_completeness_fail_closed(self) -> None:
        mutants: list[tuple[str, FakeGitHub]] = []

        omitted = FakeGitHub()
        omitted.compare_files[0] = {
            "filename": "docs/replacement-with-same-count.md",
            "status": "modified",
            "sha": "e" * 40,
        }
        mutants.append(("same-count omission", omitted))

        truncated = FakeGitHub()
        truncated.compare_files.pop()
        mutants.append(("truncated files", truncated))

        malformed = FakeGitHub()
        del malformed.compare_files[0]["status"]
        mutants.append(("malformed file entry", malformed))

        renamed = FakeGitHub()
        renamed.compare_files[0]["status"] = "renamed"
        renamed.compare_files[0]["previous_filename"] = FORMATTER
        mutants.append(("rename ambiguity", renamed))

        for field, value in (
            ("status", "diverged"),
            ("base_commit", {"sha": HEAD_SHA}),
            ("merge_base_commit", {"sha": HEAD_SHA}),
            ("commits", [{"sha": "c" * 40}]),
        ):
            api = FakeGitHub()
            api.compare_overrides[field] = value
            mutants.append((f"wrong compare {field}", api))

        swapped = FakeGitHub()
        swapped.compare_overrides.update(
            {
                "base_commit": {"sha": HEAD_SHA},
                "merge_base_commit": {"sha": HEAD_SHA},
                "commits": [{"sha": BASE_SHA}],
            }
        )
        mutants.append(("swapped compare refs", swapped))

        paginated = FakeGitHub()
        paginated.compare_link = '<https://api.github.test/next>; rel="next"'
        mutants.append(("pagination ambiguity", paginated))

        failed = FakeGitHub()
        failed.compare_http_error = True
        mutants.append(("compare HTTP error", failed))

        for name, api in mutants:
            with self.subTest(name=name):
                self.assert_rejected(api)

    def test_07_immutable_compare_inventory_has_a_299_file_boundary(self) -> None:
        at_limit = FakeGitHub(include_parser=False)
        at_limit.files.extend(
            {
                "filename": f"docs/limit-{index}.md",
                "status": "modified",
                "sha": f"{index + 100:040x}",
            }
            for index in range(297)
        )
        at_limit.event_file_count = 299
        at_limit.pull["changed_files"] = at_limit.event_file_count
        at_limit.compare_files = copy.deepcopy(at_limit.files)
        self.assert_certified(at_limit)

        for declared_count in (300, 301):
            with self.subTest(declared_count=declared_count):
                over_limit = FakeGitHub(include_parser=False)
                over_limit.files.extend(
                    {
                        "filename": f"docs/over-{index}.md",
                        "status": "modified",
                        "sha": f"{index + 100:040x}",
                    }
                    for index in range(declared_count - 2)
                )
                over_limit.event_file_count = declared_count
                over_limit.pull["changed_files"] = over_limit.event_file_count
                over_limit.compare_files = copy.deepcopy(over_limit.files[:300])
                self.assertEqual(
                    len(over_limit.compare_files), min(declared_count, 300)
                )
                self.assert_rejected(over_limit, expected_file_count=declared_count)

    def test_08_ordinary_decisions_use_only_the_immutable_inventory(self) -> None:
        docs = FakeGitHub(include_parser=False)
        docs.files = ordinary_files(2)
        docs.compare_files = copy.deepcopy(docs.files)
        docs.event_file_count = 2
        docs.pull["changed_files"] = docs.event_file_count
        self.assert_ordinary_certified(docs)

        mutable_safe = FakeGitHub(include_parser=False)
        mutable_safe.files = ordinary_files(2)
        mutable_safe.compare_files = [
            ordinary_files(1)[0],
            {
                "filename": ".github/workflows/ci.yml",
                "status": "modified",
                "sha": "f" * 40,
            },
        ]
        mutable_safe.event_file_count = 2
        mutable_safe.pull["changed_files"] = mutable_safe.event_file_count
        self.assert_rejected(mutable_safe)
        self.assertEqual(mutable_safe.compare_reads, 1)
        self.assertEqual(mutable_safe.pull_files_reads, 0)

        immutable_safe = FakeGitHub(include_parser=False)
        immutable_safe.files = [
            {
                "filename": ".github/workflows/ci.yml",
                "status": "modified",
                "sha": "f" * 40,
            },
            {
                "filename": ARCHITECTURE_DEBT,
                "status": "modified",
                "sha": "e" * 40,
            },
        ]
        immutable_safe.compare_files = ordinary_files(2)
        immutable_safe.event_file_count = 2
        immutable_safe.pull["changed_files"] = immutable_safe.event_file_count
        self.assert_ordinary_certified(immutable_safe)

    def test_09_ordinary_inventory_has_the_same_299_file_boundary(self) -> None:
        at_limit = FakeGitHub(include_parser=False)
        at_limit.files = ordinary_files(299)
        at_limit.compare_files = copy.deepcopy(at_limit.files)
        at_limit.event_file_count = 299
        at_limit.pull["changed_files"] = at_limit.event_file_count
        self.assert_ordinary_certified(at_limit)

        capped = FakeGitHub(include_parser=False)
        capped.files = ordinary_files(300)
        capped.compare_files = copy.deepcopy(capped.files)
        capped.event_file_count = 300
        capped.pull["changed_files"] = capped.event_file_count
        self.assert_rejected(capped, expected_file_count=300)
        self.assertEqual(capped.compare_reads, 0)
        self.assertEqual(capped.pull_files_reads, 0)

    def public_surface_api(self, *, combined: bool = False) -> FakeGitHub:
        api = FakeGitHub(include_parser=False)
        base = exact_ratchet_ledger(include_parser=False) if combined else BASE_LEDGER
        api.blobs[(HEAD_REPOSITORY, HEAD_SHA, ARCHITECTURE_DEBT)] = replace_once(
            base, b"ceiling = 128", b"ceiling = 126"
        )
        return api

    def test_public_surface_and_combined_decreases_are_certified(self) -> None:
        for combined in (False, True):
            with self.subTest(combined=combined):
                self.assert_certified(self.public_surface_api(combined=combined))

    def test_public_surface_requires_source_for_every_reduced_crate(self) -> None:
        api = self.public_surface_api()
        base_key = (BASE_REPOSITORY, BASE_SHA, ARCHITECTURE_DEBT)
        head_key = (HEAD_REPOSITORY, HEAD_SHA, ARCHITECTURE_DEBT)
        api.blobs[base_key] += b'\n[[public_surface]]\ncrate = "eqiora-other"\nceiling = 8\n'
        api.blobs[head_key] += b'\n[[public_surface]]\ncrate = "eqiora-other"\nceiling = 7\n'
        self.assert_rejected(api)

    def test_public_surface_accepts_added_removed_modified_sources_but_not_rename(self) -> None:
        for status in ("added", "removed", "modified"):
            with self.subTest(status=status):
                api = self.public_surface_api()
                api.compare_files[0]["status"] = status
                self.assert_certified(api)
        api = self.public_surface_api()
        api.compare_files[0].update(status="renamed", previous_filename="crates/eqiora-lang/src/old.rs")
        self.assert_rejected(api)

    def test_public_surface_zero_is_a_valid_strict_integer_decrease(self) -> None:
        api = self.public_surface_api()
        api.blobs[(HEAD_REPOSITORY, HEAD_SHA, ARCHITECTURE_DEBT)] = replace_once(
            BASE_LEDGER, b"ceiling = 128", b"ceiling = 0"
        )
        self.assert_certified(api)

    def test_public_surface_count_is_left_to_the_required_architecture_check(self) -> None:
        api = self.public_surface_api()
        # Trust consumes only authenticated change identities and inert ledgers.
        # This source is not parsed or executed, and does not prove the item count.
        del api.blobs[(BASE_REPOSITORY, BASE_SHA, FORMATTER)]
        del api.blobs[(HEAD_REPOSITORY, HEAD_SHA, FORMATTER)]
        self.assert_certified(api)
        self.assertFalse(any(FORMATTER in request.full_url for request in api.requests))

    def test_public_surface_invalid_or_non_decreasing_ceiling_fails_closed(self) -> None:
        for value in (b"128", b"129", b"-1", b"true", b"126.0", b'"126"', b"1_26"):
            with self.subTest(value=value):
                api = self.public_surface_api()
                api.blobs[(HEAD_REPOSITORY, HEAD_SHA, ARCHITECTURE_DEBT)] = replace_once(
                    BASE_LEDGER, b"ceiling = 128", b"ceiling = " + value
                )
                self.assert_rejected(api)

    def test_public_surface_inventory_order_and_metadata_are_immutable(self) -> None:
        other = b'\n[[public_surface]]\ncrate = "eqiora-other"\nceiling = 42\n'
        base = BASE_LEDGER + other
        exact = replace_once(base, b"ceiling = 128", b"ceiling = 126")
        mutations = {
            "add": exact + b'\n[[public_surface]]\ncrate = "eqiora-new"\nceiling = 2\n',
            "delete": exact.removesuffix(other),
            "order": other + exact.removesuffix(other),
            "identity": exact.replace(b'crate = "eqiora-lang"', b'crate = "eqiora-renamed"'),
            "metadata": exact.replace(b"existing public surface", b"different public surface"),
            "extra metadata": exact + b'reason = "new metadata"\n',
            "whitespace": exact.replace(b"ceiling = 126", b"ceiling  = 126"),
            "mixed increase": exact.replace(b"ceiling = 42", b"ceiling = 43"),
            "duplicate": exact + other,
        }
        for name, ledger in mutations.items():
            with self.subTest(name=name):
                api = self.public_surface_api()
                api.blobs[(BASE_REPOSITORY, BASE_SHA, ARCHITECTURE_DEBT)] = base
                api.blobs[(HEAD_REPOSITORY, HEAD_SHA, ARCHITECTURE_DEBT)] = ledger
                self.assert_rejected(api)

    def test_public_surface_requires_rust_change_inside_each_exact_crate(self) -> None:
        for path, status in (
            ("docs/notes.md", "modified"),
            ("crates/eqiora-lang-other/src/lib.rs", "modified"),
            ("crates/eqiora-lang/Cargo.toml", "modified"),
            (FORMATTER, "unchanged"),
            ("crates/eqiora-lang/../eqiora-other/src/lib.rs", "modified"),
        ):
            with self.subTest(path=path, status=status):
                api = self.public_surface_api()
                api.compare_files[0].update(filename=path, status=status)
                self.assert_rejected(api)
        api = self.public_surface_api()
        for key in ((BASE_REPOSITORY, BASE_SHA), (HEAD_REPOSITORY, HEAD_SHA)):
            api.blobs[(*key, ARCHITECTURE_DEBT)] = api.blobs[(*key, ARCHITECTURE_DEBT)].replace(
                b'crate = "eqiora-lang"', b'crate = "../eqiora-lang"'
            )
        self.assert_rejected(api)

    def test_public_surface_mixed_protected_edits_never_use_the_exception(self) -> None:
        for path in ("tools/ci/check_trust_boundary.py", ".github/workflows/ci.yml"):
            with self.subTest(path=path):
                api = self.public_surface_api(combined=True)
                api.compare_files.append({"filename": path, "status": "modified", "sha": "f" * 40})
                api.event_file_count += 1
                api.pull["changed_files"] = api.event_file_count
                self.assert_rejected(api)

    def test_10_non_decreasing_and_non_exact_numeric_changes_are_rejected(self) -> None:
        mutations = {
            "equal": b"ceiling = 1050",
            "raise": b"ceiling = 1051",
            "non-exact lower": b"ceiling = 1009",
            "zero": b"ceiling = 0",
            "cross ordinary limit": b"ceiling = 1000",
            "below ordinary limit": b"ceiling = 999",
        }
        for name, replacement in mutations.items():
            with self.subTest(name=name):
                api = FakeGitHub(include_parser=False)
                api.blobs[(HEAD_REPOSITORY, HEAD_SHA, ARCHITECTURE_DEBT)] = (
                    replace_once(BASE_LEDGER, b"ceiling = 1050", replacement)
                )
                if name in {"cross ordinary limit", "below ordinary limit"}:
                    api.blobs[(HEAD_REPOSITORY, HEAD_SHA, FORMATTER)] = physical_lines(
                        int(replacement.removeprefix(b"ceiling = "))
                    )
                self.assert_rejected(api)

        mixed = FakeGitHub()
        ledger = replace_once(BASE_LEDGER, b"ceiling = 1050", b"ceiling = 1010")
        ledger = replace_once(ledger, b"ceiling = 2701", b"ceiling = 2702")
        mixed.blobs[(HEAD_REPOSITORY, HEAD_SHA, ARCHITECTURE_DEBT)] = ledger
        self.assert_rejected(mixed)

    def test_11_only_existing_ceiling_digit_tokens_may_change(self) -> None:
        exact = exact_ratchet_ledger(include_parser=False)
        formatter_entry = b"""[[file_lines]]\n\
path = "crates/eqiora-lang/src/formatter.rs"\n\
ceiling = 1010\n\
reason = "existing formatter debt"\n\
removal = "split formatter"\n"""
        parser_entry = b"""[[file_lines]]\n\
path = "crates/eqiora-lang/src/parser.rs"\n\
ceiling = 2701\n\
reason = "existing parser debt"\n\
removal = "split parser"\n"""
        mutations = {
            "add entry": exact
            + b'\n[[file_lines]]\npath = "crates/new.rs"\nceiling = 1001\n',
            "delete entry": exact.replace(parser_entry, b"", 1),
            "reorder entry": exact.replace(formatter_entry, b"", 1)
            + b"\n"
            + formatter_entry,
            "repoint path": replace_once(
                exact,
                b'path = "crates/eqiora-lang/src/formatter.rs"',
                b'path = "crates/eqiora-lang/src/not-formatter.rs"',
            ),
            "reason": replace_once(
                exact,
                b'reason = "existing formatter debt"',
                b'reason = "rewritten reason"',
            ),
            "removal": replace_once(
                exact,
                b'removal = "split formatter"',
                b'removal = "later"',
            ),
            "comment": exact.replace(
                b"# Architecture debt ledger", b"# Architecture debt ledger edited", 1
            ),
            "whitespace": exact.replace(b"ceiling = 1010", b"ceiling  = 1010", 1),
            "line endings": exact.replace(b"\n", b"\r\n"),
            "TOML integer spelling": exact.replace(
                b"ceiling = 1010", b"ceiling = 1_010", 1
            ),
            "global limit": exact.replace(
                b"production_file_lines = 1000",
                b"production_file_lines = 999",
                1,
            ),
            "glob": exact.replace(b"always | syntax::*", b"always | ast::*", 1),
        }
        for name, ledger in mutations.items():
            with self.subTest(name=name):
                api = FakeGitHub(include_parser=False)
                api.blobs[(HEAD_REPOSITORY, HEAD_SHA, ARCHITECTURE_DEBT)] = ledger
                self.assert_rejected(api)

    def test_12_entry_uniqueness_utf8_and_toml_are_fail_closed(self) -> None:
        exact = exact_ratchet_ledger(include_parser=False)
        duplicate_path = (
            exact
            + b"""\n[[file_lines]]\n\
path = "crates/eqiora-lang/src/formatter.rs"\n\
ceiling = 1010\n\
reason = "duplicate"\n\
removal = "duplicate"\n"""
        )
        mutations = {
            "duplicate file_lines path": duplicate_path,
            "duplicate TOML key": exact.replace(
                b"ceiling = 1010", b"ceiling = 1010\nceiling = 1009", 1
            ),
            "malformed TOML": exact + b"\n[[file_lines]\n",
            "invalid UTF-8": exact + b"\xff",
        }
        for name, ledger in mutations.items():
            with self.subTest(name=name):
                api = FakeGitHub(include_parser=False)
                api.blobs[(HEAD_REPOSITORY, HEAD_SHA, ARCHITECTURE_DEBT)] = ledger
                self.assert_rejected(api)

    def test_13_base_and_head_measurements_are_bound_to_their_exact_blobs(self) -> None:
        base_mismatch = FakeGitHub(include_parser=False)
        base_mismatch.blobs[(BASE_REPOSITORY, BASE_SHA, FORMATTER)] = physical_lines(
            1049
        )
        self.assert_rejected(base_mismatch)

        head_mismatch = FakeGitHub(include_parser=False)
        head_mismatch.blobs[(HEAD_REPOSITORY, HEAD_SHA, FORMATTER)] = physical_lines(
            1009
        )
        self.assert_rejected(head_mismatch)

        empty = FakeGitHub(include_parser=False)
        empty.blobs[(HEAD_REPOSITORY, HEAD_SHA, FORMATTER)] = b""
        self.assert_rejected(empty)

    def test_14_required_source_and_protected_path_metadata_are_exact(self) -> None:
        mutations: list[tuple[str, FakeGitHub]] = []

        missing_source = FakeGitHub(include_parser=False)
        missing_source.compare_files = [
            entry
            for entry in missing_source.compare_files
            if entry["filename"] != FORMATTER
        ]
        missing_source.event_file_count = len(missing_source.compare_files)
        missing_source.pull["changed_files"] = missing_source.event_file_count
        mutations.append(("ratcheted source absent", missing_source))

        same_basename = FakeGitHub(include_parser=False)
        same_basename.compare_files[0]["filename"] = "crates/other/src/formatter.rs"
        mutations.append(("different path with same basename", same_basename))

        ledger_rename = FakeGitHub(include_parser=False)
        ledger_rename.compare_files[1]["previous_filename"] = ARCHITECTURE_DEBT
        ledger_rename.compare_files[1]["filename"] = "docs/architecture-debt.toml"
        ledger_rename.compare_files[1]["status"] = "renamed"
        mutations.append(("ledger rename", ledger_rename))

        source_rename = FakeGitHub(include_parser=False)
        source_rename.compare_files[0]["previous_filename"] = FORMATTER
        source_rename.compare_files[0]["filename"] = "crates/eqiora-lang/src/format.rs"
        source_rename.compare_files[0]["status"] = "renamed"
        mutations.append(("source rename", source_rename))

        missing_status = FakeGitHub(include_parser=False)
        del missing_status.compare_files[0]["status"]
        mutations.append(("incomplete source metadata", missing_status))

        duplicate = FakeGitHub(include_parser=False)
        duplicate.compare_files.append(dict(duplicate.compare_files[0]))
        duplicate.event_file_count = len(duplicate.compare_files)
        duplicate.pull["changed_files"] = duplicate.event_file_count
        mutations.append(("duplicate current filename", duplicate))

        mixed_protected = FakeGitHub(include_parser=False)
        mixed_protected.compare_files.append(
            {"filename": ".github/workflows/ci.yml", "status": "modified"}
        )
        mixed_protected.event_file_count = len(mixed_protected.compare_files)
        mixed_protected.pull["changed_files"] = mixed_protected.event_file_count
        mutations.append(("mixed protected path", mixed_protected))

        other_protected = FakeGitHub(include_parser=False)
        other_protected.compare_files = [
            {"filename": "tools/xtask/src/architecture.rs", "status": "modified"}
        ]
        other_protected.event_file_count = 1
        other_protected.pull["changed_files"] = other_protected.event_file_count
        mutations.append(("other protected path", other_protected))

        for name, api in mutations:
            with self.subTest(name=name):
                self.assert_rejected(api)

    def test_15_event_identity_and_provider_counts_are_exactly_bound(self) -> None:
        identity_mutations: list[tuple[str, dict[str, object]]] = [
            ("abbreviated base SHA", {"base_sha": BASE_SHA[:12]}),
            ("abbreviated head SHA", {"head_sha": HEAD_SHA[:12]}),
            ("wrong base SHA", {"base_sha": "a" * 40}),
            ("wrong head SHA", {"head_sha": "b" * 40}),
            ("wrong base repository", {"base_repository": "other/eqiora"}),
            ("wrong head repository", {"head_repository": "fork/eqiora"}),
            ("base repository has extra component", {"base_repository": "a/b/c"}),
            (
                "head repository injects query",
                {"head_repository": "fork/eqiora?ref=main"},
            ),
            ("wrong pull number", {"pull_number": PULL_NUMBER + 1}),
            ("wrong event count", {"expected_file_count": 14}),
        ]
        for name, arguments in identity_mutations:
            with self.subTest(name=name):
                self.assert_rejected(FakeGitHub(), **arguments)

        moved_head = FakeGitHub()
        moved_head.pull["head"]["sha"] = "c" * 40
        self.assert_rejected(moved_head)

        moved_base = FakeGitHub()
        moved_base.pull["base"]["sha"] = "d" * 40
        self.assert_rejected(moved_base)

        fork_mismatch = FakeGitHub()
        fork_mismatch.pull["head"]["repo"]["full_name"] = "fork/eqiora"
        self.assert_rejected(fork_mismatch)

        provider_count = FakeGitHub()
        provider_count.pull["changed_files"] = len(provider_count.files) + 1
        self.assert_rejected(provider_count)

        truncated = FakeGitHub()
        truncated.compare_files.pop()
        self.assert_rejected(truncated, expected_file_count=15)

    def test_16_missing_http_wrong_content_type_and_oversize_blobs_fail_closed(
        self,
    ) -> None:
        missing_keys = (
            ("base ledger", (BASE_REPOSITORY, BASE_SHA, ARCHITECTURE_DEBT)),
            ("head ledger", (HEAD_REPOSITORY, HEAD_SHA, ARCHITECTURE_DEBT)),
            ("base source", (BASE_REPOSITORY, BASE_SHA, FORMATTER)),
            ("head source", (HEAD_REPOSITORY, HEAD_SHA, FORMATTER)),
        )
        for name, key in missing_keys:
            with self.subTest(name=f"missing {name}"):
                missing = FakeGitHub(include_parser=False)
                del missing.blobs[key]
                self.assert_rejected(missing)

        http_error = FakeGitHub(include_parser=False)
        http_error.http_failure_path = ARCHITECTURE_DEBT
        self.assert_rejected(http_error)

        wrong_pull_type = FakeGitHub(include_parser=False)
        wrong_pull_type.pull_content_type = "text/html"
        self.assert_rejected(wrong_pull_type)

        wrong_compare_type = FakeGitHub(include_parser=False)
        wrong_compare_type.compare_content_type = "text/html"
        self.assert_rejected(wrong_compare_type)

        wrong_type = FakeGitHub(include_parser=False)
        source_key = (HEAD_REPOSITORY, HEAD_SHA, FORMATTER)
        wrong_type.content_types[source_key] = "text/html"
        self.assert_rejected(wrong_type)

        self.assertEqual(
            getattr(trust_boundary, "MAX_BLOB_BYTES", None),
            FROZEN_MAX_RAW_BLOB_BYTES,
        )
        for omitted in (False, True):
            with self.subTest(boundary="max body", omitted=omitted):
                exact = FakeGitHub(include_parser=False)
                exact.blobs[source_key] = sized_physical_source(
                    FROZEN_MAX_RAW_BLOB_BYTES
                )
                if omitted:
                    exact.omitted_lengths.add(source_key)
                self.assert_certified(exact)

        for declared in (None, 1):
            with self.subTest(boundary="max+1 body", declared=declared):
                oversize = FakeGitHub(include_parser=False)
                oversize.blobs[source_key] = sized_physical_source(
                    FROZEN_MAX_RAW_BLOB_BYTES + 1
                )
                if declared is None:
                    oversize.omitted_lengths.add(source_key)
                else:
                    oversize.declared_lengths[source_key] = declared
                self.assert_rejected(oversize)

        declared_oversize = FakeGitHub(include_parser=False)
        declared_oversize.declared_lengths[source_key] = FROZEN_MAX_RAW_BLOB_BYTES + 1
        self.assert_rejected(declared_oversize)

    def test_04_workflow_binds_exact_event_identity_without_head_checkout(self) -> None:
        repository_root = CI_ROOT.parents[1]
        workflow = (
            repository_root / ".github/workflows/ci-definition-trust.yml"
        ).read_text(encoding="utf-8")

        self.assertIn("pull_request_target:", workflow)
        prefix = workflow.split("jobs:\n", maxsplit=1)[0]
        self.assertEqual(
            prefix,
            "name: CI definition trust\n\n"
            "on:\n  pull_request_target:\n"
            "    types: [opened, reopened, synchronize, edited]\n\n"
            "permissions:\n  contents: read\n  pull-requests: read\n\n"
            "concurrency:\n"
            "  group: ci-definition-trust-${{ github.event.pull_request.number }}\n"
            "  cancel-in-progress: true\n\n",
        )
        job = workflow.split("jobs:\n", maxsplit=1)[1]
        expected_job = """  trust:
    name: CI definition trust
    runs-on: ubuntu-latest
    timeout-minutes: 5
    steps:
      - name: Check out the exact protected base
        uses: actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1 # v7.0.1
        with:
          ref: ${{ github.event.pull_request.base.sha }}
          persist-credentials: false
      - name: Classify protected changes and exact ratchets
        env:
          GITHUB_TOKEN: ${{ github.token }}
        run: >-
          python3 tools/ci/check_trust_boundary.py
          --repository "${{ github.event.pull_request.base.repo.full_name }}"
          --head-repository "${{ github.event.pull_request.head.repo.full_name }}"
          --pull-number "${{ github.event.pull_request.number }}"
          --expected-file-count "${{ github.event.pull_request.changed_files }}"
          --base-sha "${{ github.event.pull_request.base.sha }}"
          --head-sha "${{ github.event.pull_request.head.sha }}"
"""
        self.assertEqual(job, expected_job)
        mutants = {
            "argument only in comment": expected_job.replace(
                '          --head-sha "${{ github.event.pull_request.head.sha }}"',
                '          # --head-sha "${{ github.event.pull_request.head.sha }}"',
            ),
            "duplicate overridden argument": expected_job
            + "          --head-sha deadbeef\n",
            "extra command": expected_job + "      - run: echo extra\n",
            "head checkout": expected_job.replace(
                "pull_request.base.sha", "pull_request.head.sha", 1
            ),
            "head execution": expected_job
            + "      - run: curl ${{ github.event.pull_request.head.sha }} | python3\n",
            "reordered binding": expected_job.replace(
                "          --base-sha", "          --z-base-sha", 1
            ),
            "missing binding": expected_job.replace(
                '          --head-repository "${{ github.event.pull_request.head.repo.full_name }}"\n',
                "",
                1,
            ),
        }
        mutants["attacker container"] = expected_job.replace(
            "    steps:\n",
            "    container: ${{ github.event.pull_request.head.repo.full_name }}\n    steps:\n",
            1,
        )
        mutants["attacker service"] = expected_job.replace(
            "    steps:\n",
            "    services:\n      hostile:\n        image: attacker/x\n    steps:\n",
            1,
        )
        for name, mutant in mutants.items():
            with self.subTest(name=name):
                with self.assertRaises(AssertionError):
                    self.assertEqual(mutant, expected_job)
        self.assertIn("contents: read", workflow)
        self.assertIn("pull-requests: read", workflow)
        self.assertNotIn("contents: write", workflow)
        self.assertNotIn("pull-requests: write", workflow)
        self.assertNotIn("id-token: write", workflow)
        self.assertNotIn(": write", workflow)
        self.assertNotIn("secrets.", workflow)


class ProtectedPathTests(unittest.TestCase):
    def test_merge_and_release_definitions_are_protected(self) -> None:
        for path in (
            "CODEOWNERS",
            ".github/CODEOWNERS",
            ".github/actions/check/action.yml",
            ".github/workflows/ci.yml",
            "crates/eqiora-verify/src/lib.rs",
            "deny.toml",
            "studio/src-tauri/deny.toml",
            "tools/ci/check_gate.py",
            "tools/release/python_candidate.py",
            "tools/xtask/src/main.rs",
        ):
            with self.subTest(path=path):
                self.assertTrue(protected_path(path))

    def test_product_and_documentation_paths_are_not_protected(self) -> None:
        for path in (
            "Cargo.toml",
            "crates/eqiora/src/lib.rs",
            "docs/architecture.md",
            "pyproject.toml",
        ):
            with self.subTest(path=path):
                self.assertFalse(protected_path(path))

    def test_rename_checks_both_names(self) -> None:
        paths = changed_file_names(
            [
                {
                    "filename": "docs/old-ci.md",
                    "previous_filename": "tools/ci/old_gate.py",
                }
            ]
        )
        self.assertEqual(protected_changes(paths), ["tools/ci/old_gate.py"])


class MetadataFetchTests(unittest.TestCase):
    def test_fetches_all_pages_with_read_only_headers(self) -> None:
        first = [{"filename": f"docs/page-{index}.md"} for index in range(100)]
        second = [{"filename": ".github/workflows/ci.yml"}]
        opener = mock.Mock(
            side_effect=[
                Response(json.dumps(first).encode()),
                Response(json.dumps(second).encode()),
            ]
        )

        paths = fetch_changed_files(
            api_url="https://api.github.test",
            repository="owner/project",
            pull_number=7,
            expected_file_count=101,
            token="not-a-real-token",
            opener=opener,
        )

        self.assertEqual(len(paths), 101)
        self.assertEqual(paths[-1], ".github/workflows/ci.yml")
        requests = [call.args[0] for call in opener.call_args_list]
        self.assertTrue(requests[0].full_url.endswith("per_page=100&page=1"))
        self.assertTrue(requests[1].full_url.endswith("per_page=100&page=2"))
        self.assertEqual(
            requests[0].headers["Authorization"], "Bearer not-a-real-token"
        )

    def test_rejects_api_truncation_and_oversized_pull_requests(self) -> None:
        truncated = mock.Mock(
            return_value=Response(
                json.dumps(
                    [{"filename": f"docs/page-{index}.md"} for index in range(99)]
                ).encode()
            )
        )
        with self.assertRaisesRegex(ValueError, "count differs"):
            fetch_changed_files(
                api_url="https://api.github.test",
                repository="owner/project",
                pull_number=7,
                expected_file_count=100,
                token="not-a-real-token",
                opener=truncated,
            )

        unopened = mock.Mock()
        with self.assertRaisesRegex(ValueError, "3000-file API trust boundary"):
            fetch_changed_files(
                api_url="https://api.github.test",
                repository="owner/project",
                pull_number=7,
                expected_file_count=3001,
                token="not-a-real-token",
                opener=unopened,
            )
        unopened.assert_not_called()

    def test_malformed_metadata_fails_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "no filename"):
            changed_file_names([{"status": "modified"}])


if __name__ == "__main__":
    unittest.main()
