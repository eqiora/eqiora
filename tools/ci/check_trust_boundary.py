#!/usr/bin/env python3
"""Reject pull requests that alter the definitions of public trust gates.

This program runs from the protected base revision under ``pull_request_target``.
It treats pull-request metadata and explicitly fetched head blobs as bounded inert
data; it never checks out, imports, or executes code from the pull-request head.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tomllib
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable, Iterable, Mapping
from typing import Any


PROTECTED_PREFIXES = (
    ".github/actions/",
    ".github/workflows/",
    "crates/eqiora-verify/",
    "tools/ci/",
    "tools/release/",
    "tools/xtask/",
)
PROTECTED_PATHS = frozenset(
    {
        "CODEOWNERS",
        ".github/CODEOWNERS",
        "deny.toml",
        "studio/src-tauri/deny.toml",
    }
)
ARCHITECTURE_DEBT = "tools/ci/architecture-debt.toml"
MAX_VISIBLE_FILES = 3_000
PAGE_SIZE = 100
MAX_PAGES = MAX_VISIBLE_FILES // PAGE_SIZE + 1
MAX_BLOB_BYTES = 1_048_576
MAX_COMPARE_FILES = 299
FULL_SHA = re.compile(r"[0-9a-f]{40}\Z")
REPOSITORY_COMPONENT = re.compile(r"[A-Za-z0-9_.-]+\Z")
RUST_SOURCE_PATH = re.compile(
    r"crates/[A-Za-z0-9_.-]+/(?:[A-Za-z0-9_.-]+/)*[A-Za-z0-9_.-]+\.rs\Z"
)
RATCHET_SECTIONS = {
    name: re.compile(
        rb"(?m)^\[\[" + name.encode("ascii") + rb"\]\]\r?$.*?(?=^\[\[|\Z)",
        re.DOTALL,
    )
    for name in ("file_lines", "public_surface")
}
CRATE_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
CEILING_LINE = re.compile(rb"(?m)^ceiling = (0|[1-9][0-9]*)\r?$")
COMPARE_FILE_STATUSES = frozenset(
    {"added", "removed", "modified", "renamed", "copied", "changed", "unchanged"}
)


def protected_path(path: str) -> bool:
    """Return whether *path* defines merge or release trust."""
    normalized = path.replace("\\", "/").removeprefix("./")
    return normalized in PROTECTED_PATHS or normalized.startswith(PROTECTED_PREFIXES)


def changed_file_names(payload: Any) -> list[str]:
    """Validate one GitHub pull-files response and return both path identities."""
    if not isinstance(payload, list):
        raise ValueError("GitHub pull-files response must be a JSON array")

    names: list[str] = []
    for index, entry in enumerate(payload):
        if not isinstance(entry, dict):
            raise ValueError(f"changed-file entry {index} must be an object")
        filename = entry.get("filename")
        if not isinstance(filename, str) or not filename:
            raise ValueError(f"changed-file entry {index} has no filename")
        names.append(filename)
        previous = entry.get("previous_filename")
        if previous is not None:
            if not isinstance(previous, str) or not previous:
                raise ValueError(
                    f"changed-file entry {index} has invalid previous_filename"
                )
            names.append(previous)
    return names


def _validated_repository(repository: str) -> str:
    components = repository.split("/")
    if (
        len(components) != 2
        or any(component in {"", ".", ".."} for component in components)
        or any(
            REPOSITORY_COMPONENT.fullmatch(component) is None
            for component in components
        )
    ):
        raise ValueError("repository must be an owner/name pair")
    return "/".join(urllib.parse.quote(component, safe="") for component in components)


def _validated_api_root(api_url: str) -> str:
    parsed = urllib.parse.urlsplit(api_url)
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
        or "/../" in f"{parsed.path}/"
    ):
        raise ValueError("GitHub API URL must be an absolute trusted HTTPS root")
    return api_url.rstrip("/")


def _request(url: str, *, token: str, accept: str) -> urllib.request.Request:
    if not token:
        raise ValueError("GITHUB_TOKEN is required")
    return urllib.request.Request(
        url,
        headers={
            "Accept": accept,
            "Authorization": f"Bearer {token}",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )


def _read_bounded_response(
    response: Any,
    *,
    request_url: str,
    allowed_content_types: frozenset[str],
) -> bytes:
    status = response.getcode()
    if status != 200:
        raise ValueError(f"GitHub API returned HTTP {status}")

    final_url = response.geturl() if hasattr(response, "geturl") else request_url
    if final_url != request_url:
        raise ValueError("GitHub API response redirected outside the fixed request")

    content_type = response.headers.get_content_type().lower()
    if content_type not in allowed_content_types:
        raise ValueError(
            f"GitHub API returned unexpected content type {content_type!r}"
        )

    declared_text = response.headers.get("Content-Length")
    if declared_text is not None:
        if not declared_text.isascii() or not declared_text.isdecimal():
            raise ValueError("GitHub API returned an invalid Content-Length")
        if int(declared_text) > MAX_BLOB_BYTES:
            raise ValueError(f"GitHub API body exceeds {MAX_BLOB_BYTES} bytes")

    payload = response.read(MAX_BLOB_BYTES + 1)
    if len(payload) > MAX_BLOB_BYTES:
        raise ValueError(f"GitHub API body exceeds {MAX_BLOB_BYTES} bytes")
    return payload


def _fetch_json(
    url: str,
    *,
    token: str,
    opener: Callable[..., Any],
) -> Any:
    request = _request(url, token=token, accept="application/vnd.github+json")
    with opener(request, timeout=30) as response:
        payload = _read_bounded_response(
            response,
            request_url=url,
            allowed_content_types=frozenset(
                {"application/json", "application/vnd.github+json"}
            ),
        )
    return json.loads(payload)


def _fetch_changed_file_entries(
    *,
    api_url: str,
    repository: str,
    pull_number: int,
    expected_file_count: int,
    token: str,
    opener: Callable[..., Any],
) -> list[dict[str, Any]]:
    if pull_number <= 0:
        raise ValueError("pull number must be positive")
    if expected_file_count <= 0:
        raise ValueError("expected changed-file count must be positive")
    if expected_file_count > MAX_VISIBLE_FILES:
        raise ValueError(
            f"pull request exceeds the {MAX_VISIBLE_FILES}-file API trust boundary"
        )

    root = _validated_api_root(api_url)
    encoded_repository = _validated_repository(repository)
    entries: list[dict[str, Any]] = []
    for page in range(1, MAX_PAGES + 1):
        url = (
            f"{root}/repos/{encoded_repository}/pulls/{pull_number}/files"
            f"?per_page={PAGE_SIZE}&page={page}"
        )
        payload = _fetch_json(url, token=token, opener=opener)
        changed_file_names(payload)
        entries.extend(payload)
        if len(payload) < PAGE_SIZE:
            if len(entries) != expected_file_count:
                raise ValueError(
                    "GitHub changed-file metadata count differs: "
                    f"expected {expected_file_count}, observed {len(entries)}"
                )
            return entries
    raise ValueError("pull-file metadata did not terminate within the API boundary")


def fetch_changed_files(
    *,
    api_url: str,
    repository: str,
    pull_number: int,
    expected_file_count: int,
    token: str,
    opener: Callable[..., Any] = urllib.request.urlopen,
) -> list[str]:
    """Read every changed filename through the read-only GitHub REST API."""
    entries = _fetch_changed_file_entries(
        api_url=api_url,
        repository=repository,
        pull_number=pull_number,
        expected_file_count=expected_file_count,
        token=token,
        opener=opener,
    )
    return changed_file_names(entries)


def _require_full_sha(value: str, label: str) -> None:
    if FULL_SHA.fullmatch(value) is None:
        raise ValueError(f"{label} must be a full lowercase commit SHA")


def _require_exact_pull_identity(
    payload: Any,
    *,
    repository: str,
    head_repository: str,
    pull_number: int,
    expected_file_count: int,
    base_sha: str,
    head_sha: str,
) -> None:
    if not isinstance(payload, dict):
        raise ValueError("GitHub pull response must be a JSON object")
    try:
        observed = (
            payload["number"],
            payload["changed_files"],
            payload["base"]["repo"]["full_name"],
            payload["base"]["sha"],
            payload["head"]["repo"]["full_name"],
            payload["head"]["sha"],
        )
    except (KeyError, TypeError) as error:
        raise ValueError("GitHub pull response has incomplete identity") from error
    if (
        type(observed[0]) is not int
        or type(observed[1]) is not int
        or any(not isinstance(value, str) for value in observed[2:])
    ):
        raise ValueError("GitHub pull response has malformed identity fields")
    expected = (
        pull_number,
        expected_file_count,
        repository,
        base_sha,
        head_repository,
        head_sha,
    )
    if observed != expected:
        raise ValueError("GitHub pull identity differs from the bound event identity")


def _validate_exact_file_entries(
    payload: Any, expected_file_count: int
) -> list[dict[str, Any]]:
    if expected_file_count > MAX_COMPARE_FILES:
        raise ValueError(
            "coupled exact ratchets require fewer than the 300-file compare cap"
        )
    changed_file_names(payload)
    if len(payload) != expected_file_count:
        raise ValueError(
            "GitHub compare file count differs: "
            f"expected {expected_file_count}, observed {len(payload)}"
        )

    entries: list[dict[str, Any]] = payload
    for index, entry in enumerate(entries):
        status = entry.get("status")
        sha = entry.get("sha")
        if status not in COMPARE_FILE_STATUSES:
            raise ValueError(f"compare file entry {index} has an invalid status")
        if not isinstance(sha, str) or FULL_SHA.fullmatch(sha) is None:
            raise ValueError(f"compare file entry {index} has an invalid blob SHA")
        previous = entry.get("previous_filename")
        if status == "renamed":
            if not isinstance(previous, str) or not previous:
                raise ValueError(
                    f"renamed compare file entry {index} has no previous filename"
                )
        elif previous is not None:
            raise ValueError(
                f"non-renamed compare file entry {index} has a previous filename"
            )
    names = changed_file_names(entries)
    if len(names) != len(set(names)):
        raise ValueError("changed-file metadata has duplicate path identities")
    return entries


def _fetch_exact_compare_inventory(
    *,
    api_url: str,
    repository: str,
    base_sha: str,
    head_sha: str,
    expected_file_count: int,
    token: str,
    opener: Callable[..., Any],
) -> list[dict[str, Any]]:
    if expected_file_count > MAX_COMPARE_FILES:
        raise ValueError(
            "coupled exact ratchets require fewer than the 300-file compare cap"
        )
    root = _validated_api_root(api_url)
    encoded_repository = _validated_repository(repository)
    _require_full_sha(base_sha, "compare base SHA")
    _require_full_sha(head_sha, "compare head SHA")
    url = f"{root}/repos/{encoded_repository}/compare/{base_sha}...{head_sha}"
    request = _request(url, token=token, accept="application/vnd.github+json")
    with opener(request, timeout=30) as response:
        if response.headers.get("Link") is not None:
            raise ValueError("GitHub compare response is paginated")
        body = _read_bounded_response(
            response,
            request_url=url,
            allowed_content_types=frozenset(
                {"application/json", "application/vnd.github+json"}
            ),
        )
    payload = json.loads(body)
    if not isinstance(payload, dict):
        raise ValueError("GitHub compare response must be a JSON object")
    try:
        status = payload["status"]
        observed_base = payload["base_commit"]["sha"]
        observed_merge_base = payload["merge_base_commit"]["sha"]
        commits = payload["commits"]
        files = payload["files"]
    except (KeyError, TypeError) as error:
        raise ValueError("GitHub compare response has incomplete identity") from error
    if status != "ahead":
        raise ValueError("GitHub compare status is not exact base-to-head ancestry")
    if observed_base != base_sha or observed_merge_base != base_sha:
        raise ValueError("GitHub compare base or merge-base identity differs")
    if (
        not isinstance(commits, list)
        or not commits
        or not isinstance(commits[-1], dict)
        or commits[-1].get("sha") != head_sha
    ):
        raise ValueError("GitHub compare does not terminate at the bound head")
    return _validate_exact_file_entries(files, expected_file_count)


def _fetch_blob(
    *,
    api_url: str,
    repository: str,
    revision: str,
    path: str,
    token: str,
    opener: Callable[..., Any],
) -> bytes:
    root = _validated_api_root(api_url)
    encoded_repository = _validated_repository(repository)
    _require_full_sha(revision, "blob revision")
    if not path or path.startswith("/") or "\\" in path:
        raise ValueError("blob path must be a repository-relative POSIX path")
    components = path.split("/")
    if any(component in {"", ".", ".."} for component in components):
        raise ValueError("blob path contains an unsafe component")
    encoded_path = "/".join(
        urllib.parse.quote(component, safe="") for component in components
    )
    url = (
        f"{root}/repos/{encoded_repository}/contents/{encoded_path}"
        f"?ref={urllib.parse.quote(revision, safe='')}"
    )
    request = _request(
        url,
        token=token,
        accept="application/vnd.github.raw+json",
    )
    with opener(request, timeout=30) as response:
        return _read_bounded_response(
            response,
            request_url=url,
            allowed_content_types=frozenset(
                {"application/octet-stream", "application/vnd.github.raw+json"}
            ),
        )


def protected_changes(paths: Iterable[str]) -> list[str]:
    """Return unique protected paths in stable order."""
    return sorted({path for path in paths if protected_path(path)})


def _ratchet_entries(payload: bytes, section: str) -> tuple[Mapping[str, Any], ...]:
    try:
        parsed = tomllib.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, tomllib.TOMLDecodeError) as error:
        raise ValueError("architecture debt is not valid UTF-8 TOML") from error
    entries = parsed.get(section, [])
    if not isinstance(entries, list):
        raise ValueError(f"architecture debt {section} must be an array")
    key = "path" if section == "file_lines" else "crate"
    identity_pattern = RUST_SOURCE_PATH if section == "file_lines" else CRATE_NAME
    minimum = 1 if section == "file_lines" else 0
    seen: set[str] = set()
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"{section} entry {index} is not a table")
        identity = entry.get(key)
        ceiling = entry.get("ceiling")
        if not isinstance(identity, str) or identity_pattern.fullmatch(identity) is None:
            raise ValueError(f"{section} entry {index} has an invalid {key}")
        if identity in seen:
            raise ValueError(f"{section} {key} is duplicated: {identity}")
        if type(ceiling) is not int or ceiling < minimum:
            raise ValueError(f"{section} entry {identity} has an invalid ceiling")
        seen.add(identity)
    return tuple(entries)


def _reconstruct_exact_ratchet(
    base_blob: bytes,
    head_blob: bytes,
) -> dict[str, dict[str, tuple[int, int]]]:
    replacements: list[tuple[int, int, bytes]] = []
    changed: dict[str, dict[str, tuple[int, int]]] = {}
    for name, section_pattern in RATCHET_SECTIONS.items():
        base_entries = _ratchet_entries(base_blob, name)
        head_entries = _ratchet_entries(head_blob, name)
        if len(base_entries) != len(head_entries):
            raise ValueError(f"{name} entry inventory changed")
        sections = tuple(section_pattern.finditer(base_blob))
        if len(sections) != len(base_entries):
            raise ValueError(f"base {name} text does not match its TOML inventory")
        key = "path" if name == "file_lines" else "crate"
        changed[name] = {}
        for base_entry, head_entry, section in zip(
            base_entries, head_entries, sections, strict=True
        ):
            identity = base_entry[key]
            if head_entry.get(key) != identity:
                raise ValueError(f"{name} {key} or order changed")
            base_metadata = dict(base_entry)
            head_metadata = dict(head_entry)
            base_ceiling = base_metadata.pop("ceiling")
            head_ceiling = head_metadata.pop("ceiling")
            if base_metadata != head_metadata:
                raise ValueError(f"{name} entry metadata changed for {identity}")
            matches = tuple(CEILING_LINE.finditer(section.group(0)))
            if len(matches) != 1:
                raise ValueError(f"{name} entry {identity} has no unique ceiling token")
            token = matches[0].group(1)
            if token != str(base_ceiling).encode("ascii"):
                raise ValueError(f"{name} entry {identity} has non-canonical base text")
            if head_ceiling != base_ceiling:
                if head_ceiling >= base_ceiling:
                    raise ValueError(f"{name} ceiling does not strictly lower for {identity}")
                replacements.append((
                    section.start() + matches[0].start(1),
                    section.start() + matches[0].end(1),
                    str(head_ceiling).encode("ascii"),
                ))
                changed[name][identity] = (base_ceiling, head_ceiling)
    if not replacements:
        raise ValueError("architecture debt has no strict ceiling ratchet")
    reconstructed = bytearray()
    cursor = 0
    for start, end, replacement in sorted(replacements):
        reconstructed.extend(base_blob[cursor:start])
        reconstructed.extend(replacement)
        cursor = end
    reconstructed.extend(base_blob[cursor:])
    if bytes(reconstructed) != head_blob:
        raise ValueError("architecture debt changed outside exact ceiling digits")
    return changed


def _physical_lines(payload: bytes) -> int:
    if not payload:
        return 0
    return payload.count(b"\n") + (not payload.endswith(b"\n"))


def _role_limit(path: str, limits: Mapping[str, Any]) -> int:
    is_test = (
        "/tests/" in path
        or "/benches/" in path
        or "/examples/" in path
        or path.rsplit("/", maxsplit=1)[-1] == "tests.rs"
    )
    key = "test_file_lines" if is_test else "production_file_lines"
    limit = limits.get(key)
    if type(limit) is not int or limit <= 0:
        raise ValueError(f"base {key} limit is invalid")
    return limit


def _certify_coupled_exact_ratchet(
    *,
    api_url: str,
    repository: str,
    head_repository: str,
    base_sha: str,
    head_sha: str,
    entries: list[dict[str, Any]],
    token: str,
    opener: Callable[..., Any],
) -> None:
    current_names = [entry.get("filename") for entry in entries]
    if any(not isinstance(path, str) or not path for path in current_names):
        raise ValueError("changed-file metadata has an invalid current filename")
    if len(set(current_names)) != len(current_names):
        raise ValueError("changed-file metadata has duplicate current filenames")
    all_names = changed_file_names(entries)
    if len(set(all_names)) != len(all_names):
        raise ValueError("changed-file metadata has duplicate path identities")

    debt_entries = [
        entry for entry in entries if entry.get("filename") == ARCHITECTURE_DEBT
    ]
    if len(debt_entries) != 1:
        raise ValueError("architecture debt must be one exact current filename")
    if (
        debt_entries[0].get("status") != "modified"
        or "previous_filename" in debt_entries[0]
    ):
        raise ValueError("architecture debt rename or non-modification is ineligible")

    base_ledger = _fetch_blob(
        api_url=api_url,
        repository=repository,
        revision=base_sha,
        path=ARCHITECTURE_DEBT,
        token=token,
        opener=opener,
    )
    head_ledger = _fetch_blob(
        api_url=api_url,
        repository=head_repository,
        revision=head_sha,
        path=ARCHITECTURE_DEBT,
        token=token,
        opener=opener,
    )
    ratchets = _reconstruct_exact_ratchet(base_ledger, head_ledger)
    base_parsed = tomllib.loads(base_ledger.decode("utf-8"))
    limits = base_parsed.get("limits")
    if not isinstance(limits, dict):
        raise ValueError("base architecture debt has no limits table")

    by_name = {entry["filename"]: entry for entry in entries}
    for crate in ratchets["public_surface"]:
        if not any(
            path.startswith(f"crates/{crate}/")
            and RUST_SOURCE_PATH.fullmatch(path) is not None
            and all(part not in {".", ".."} for part in path.split("/"))
            and metadata.get("status") in {"added", "modified", "removed"}
            and "previous_filename" not in metadata
            for path, metadata in by_name.items()
        ):
            raise ValueError(f"public surface ratchet has no changed Rust source for {crate}")

    for path, (base_ceiling, head_ceiling) in ratchets["file_lines"].items():
        metadata = by_name.get(path)
        if (
            metadata is None
            or metadata.get("status") != "modified"
            or "previous_filename" in metadata
        ):
            raise ValueError(f"ratcheted source is not an exact modified path: {path}")
        if head_ceiling <= _role_limit(path, limits):
            raise ValueError(f"ratchet crosses the ordinary role limit for {path}")

        base_source = _fetch_blob(
            api_url=api_url,
            repository=repository,
            revision=base_sha,
            path=path,
            token=token,
            opener=opener,
        )
        head_source = _fetch_blob(
            api_url=api_url,
            repository=head_repository,
            revision=head_sha,
            path=path,
            token=token,
            opener=opener,
        )
        if _physical_lines(base_source) != base_ceiling:
            raise ValueError(f"base source does not match its frozen ceiling: {path}")
        if _physical_lines(head_source) != head_ceiling:
            raise ValueError(f"head source does not match its exact ratchet: {path}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository", required=True)
    parser.add_argument("--head-repository", required=True)
    parser.add_argument("--pull-number", required=True, type=int)
    parser.add_argument("--expected-file-count", required=True, type=int)
    parser.add_argument("--base-sha", required=True)
    parser.add_argument("--head-sha", required=True)
    parser.add_argument(
        "--api-url", default=os.environ.get("GITHUB_API_URL", "https://api.github.com")
    )
    arguments = parser.parse_args()
    token = os.environ.get("GITHUB_TOKEN", "")

    try:
        root = _validated_api_root(arguments.api_url)
        encoded_repository = _validated_repository(arguments.repository)
        _validated_repository(arguments.head_repository)
        _require_full_sha(arguments.base_sha, "base SHA")
        _require_full_sha(arguments.head_sha, "head SHA")
        if arguments.pull_number <= 0:
            raise ValueError("pull number must be positive")
        if arguments.expected_file_count <= 0:
            raise ValueError("expected changed-file count must be positive")

        pull_url = f"{root}/repos/{encoded_repository}/pulls/{arguments.pull_number}"
        pull = _fetch_json(
            pull_url,
            token=token,
            opener=urllib.request.urlopen,
        )
        _require_exact_pull_identity(
            pull,
            repository=arguments.repository,
            head_repository=arguments.head_repository,
            pull_number=arguments.pull_number,
            expected_file_count=arguments.expected_file_count,
            base_sha=arguments.base_sha,
            head_sha=arguments.head_sha,
        )
        entries = _fetch_exact_compare_inventory(
            api_url=arguments.api_url,
            repository=arguments.repository,
            base_sha=arguments.base_sha,
            head_sha=arguments.head_sha,
            expected_file_count=arguments.expected_file_count,
            token=token,
            opener=urllib.request.urlopen,
        )
        paths = changed_file_names(entries)
        rejected = protected_changes(paths)
        if rejected == [ARCHITECTURE_DEBT]:
            _certify_coupled_exact_ratchet(
                api_url=arguments.api_url,
                repository=arguments.repository,
                head_repository=arguments.head_repository,
                base_sha=arguments.base_sha,
                head_sha=arguments.head_sha,
                entries=entries,
                token=token,
                opener=urllib.request.urlopen,
            )
            print("Coupled exact architecture ratchet certified by the protected base")
            return 0
    except (
        OSError,
        ValueError,
        json.JSONDecodeError,
        tomllib.TOMLDecodeError,
        urllib.error.HTTPError,
        urllib.error.URLError,
    ) as error:
        print(f"CI definition trust check failed closed: {error}", file=sys.stderr)
        return 2

    if rejected:
        print(
            "This pull request changes protected merge or release definitions. "
            "It requires an explicit maintainer ruleset bypass after independent "
            "local verification; this check cannot approve its own replacement.",
            file=sys.stderr,
        )
        for path in rejected:
            print(f"- {path}", file=sys.stderr)
        return 1

    print("Pull request does not change protected merge or release definitions")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
