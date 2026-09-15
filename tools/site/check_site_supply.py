"""Private source/archive/browser supply policy for the site checker."""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
from pathlib import Path

try:
    import tools.site.check_site_artifact as _artifact
except ModuleNotFoundError as error:
    if error.name not in {"tools", "tools.site", "tools.site.check_site_artifact"}:
        raise
    import check_site_artifact as _artifact

_EXPECTED_ARTIFACT = Path(__file__).with_name("check_site_artifact.py").resolve()
if Path(_artifact.__file__ or "").resolve() != _EXPECTED_ARTIFACT:
    raise ImportError(
        "site checker artifact owner did not resolve to its exact sibling"
    )
_ARTIFACT_EXPORTS = (
    "MAX_FILES",
    "MAX_FILE_BYTES",
    "MAX_TOTAL_BYTES",
    "MAX_HTML_BYTES",
    "WAKE_SHA256",
    "PRESSURE_SHA256",
    "SOCIAL_SHA256",
    "FAVICON_SHA256",
    "APPLE_TOUCH_SHA256",
    "OLD_SOCIAL_SHA256",
    "OLD_SOCIAL_LINE",
    "SiteIdentities",
    "PRODUCTION_IDENTITIES",
    "ROUTES",
    "SITEMAP_ROUTES",
    "WAKE_ALT",
    "PRESSURE_ALT",
    "PRESSURE_CAPTION",
    "CASE_SOURCE_PATHS",
    "CASE_EVIDENCE_PATHS",
    "sha256",
    "check_exact_source",
    "check_artifact",
)
if _artifact.__all__ != _ARTIFACT_EXPORTS:
    raise ImportError("site checker artifact owner exposes an unexpected interface")

__all__ = (
    "FULL_CHROMIUM_VERSION_STDOUT_HEX",
    "FULL_CHROMIUM_VERSION_STDOUT",
    "DIRECT_SOURCE_ARCHIVE_COMMAND",
    "EXACT_LINK_PAYLOAD_SHA256",
    "EXACT_TREE_LINK_COMMAND",
    "EXACT_EXTRACTED_LINK_COMMAND",
    "OFFLINE_WORKFLOW_TOKENS",
    "FORBIDDEN_WORKFLOW_TOKENS",
    "check_source_topology",
    "check_runner_source_topology_text",
    "check_browser_supply",
    "check_runner_browser_supply_text",
)

FULL_CHROMIUM_VERSION_STDOUT_HEX = (
    "476f6f676c65204368726f6d6520666f722054657374696e67203135312e302e373932322e3334200a"
)
FULL_CHROMIUM_VERSION_STDOUT = bytes.fromhex(FULL_CHROMIUM_VERSION_STDOUT_HEX)
DIRECT_SOURCE_ARCHIVE_COMMAND = (
    'git archive --format=tar "$GITHUB_SHA" | tar -xf - -C "$scratch/source"'
)
EXACT_LINK_PAYLOAD_SHA256 = (
    "a54ff182c7e8acf56acfd6e4b9c3ff41e2c41a31c9b211b2deb9df75d9a478f9"
)
EXACT_TREE_LINK_COMMAND = (
    "git cat-file blob \"$GITHUB_SHA:CLAUDE.md\" | sha256sum | cut -d ' ' -f 1"
)
EXACT_EXTRACTED_LINK_COMMAND = (
    "readlink -n \"$scratch/source/CLAUDE.md\" | sha256sum | cut -d ' ' -f 1"
)
OFFLINE_WORKFLOW_TOKENS = (
    "ubuntu-24.04",
    "eqiora-pw-1.62.1-r1234",
    "playwright install --with-deps chromium",
    "chromium.executablePath()",
    "chromium-1234/chrome-linux64/chrome",
    'sha256sum "$browser_path"',
    'stat -c %s "$browser_path"',
    "EQIORA_SITE_BROWSER_SHA256=",
    "EQIORA_SITE_BROWSER_BYTES=",
    '--expected-executable-sha256 "$EQIORA_SITE_BROWSER_SHA256"',
    '--expected-executable-bytes "$EQIORA_SITE_BROWSER_BYTES"',
    FULL_CHROMIUM_VERSION_STDOUT_HEX,
    "check_site.py browser-supply",
    "unshare --net",
    "ip link set lo up",
    "setpriv",
    "npm_config_offline=true",
    "CARGO_NET_OFFLINE=true",
    DIRECT_SOURCE_ARCHIVE_COMMAND,
    "EQIORA_SITE_SOURCE_ROOT=$scratch/source",
    'test "$(git rev-parse HEAD)" = "$GITHUB_SHA"',
    'git ls-tree -r "$GITHUB_SHA"',
    'git ls-tree "$GITHUB_SHA" -- AGENTS.md',
    EXACT_TREE_LINK_COMMAND,
    'git cat-file blob "$GITHUB_SHA:AGENTS.md"',
    EXACT_EXTRACTED_LINK_COMMAND,
    EXACT_LINK_PAYLOAD_SHA256,
    'cmp "$scratch/source/AGENTS.md" "$scratch/expected-AGENTS.md"',
    "120000",
    "100644",
    "AGENTS.md",
    "source_links=",
    'case "$source_links" in',
    'if test -n "$source_links"; then',
    'if test -L "$scratch/source/CLAUDE.md"; then',
)
FORBIDDEN_WORKFLOW_TOKENS = (
    "playwright install --with-deps --only-shell chromium",
    "HeadlessChrome 151.0.7922.34",
    "tar --exclude",
    "--dereference",
    "cp -L",
    "rsync -L",
    "tar -h",
    'rm "$scratch/source/CLAUDE.md"',
)


def check_source_topology(
    root: Path, expected_agents_sha256: str | None = None
) -> list[str]:
    errors: list[str] = []
    if not root.is_dir() or root.is_symlink():
        return ["site source root must be a real directory"]
    resolved_root = root.resolve()
    pruned = root / "docs/site/node_modules"
    links: list[Path] = []
    for current, directories, filenames in os.walk(root, topdown=True):
        current_path = Path(current)
        kept_directories: list[str] = []
        for name in directories:
            candidate = current_path / name
            if candidate == pruned:
                continue
            if candidate.is_symlink():
                links.append(candidate)
            else:
                kept_directories.append(name)
        directories[:] = kept_directories
        links.extend(
            candidate
            for name in filenames
            if (candidate := current_path / name).is_symlink()
        )
    claude = root / "CLAUDE.md"
    if not links:
        return errors
    if links != [claude]:
        observed = sorted(str(path.relative_to(root)) for path in links)
        return [f"site source has unadmitted symlinks: {observed}"]
    try:
        payload = os.readlink(claude)
    except OSError as error:
        return [f"site source CLAUDE.md link is unreadable: {error}"]
    if payload != "AGENTS.md":
        errors.append("site source CLAUDE.md must contain exact link payload AGENTS.md")
    agents = root / "AGENTS.md"
    try:
        agents_mode = agents.lstat().st_mode
    except OSError as error:
        return [*errors, f"site source AGENTS.md target is unavailable: {error}"]
    if agents.is_symlink() or not stat.S_ISREG(agents_mode):
        return [
            *errors,
            "site source AGENTS.md target must be a regular non-symlink file",
        ]
    try:
        if agents.resolve(strict=True) != resolved_root / "AGENTS.md":
            errors.append("site source AGENTS.md target escapes the authenticated root")
    except OSError as error:
        errors.append(f"site source AGENTS.md target cannot be resolved: {error}")
    if expected_agents_sha256 is not None:
        if not re.fullmatch(r"[0-9a-f]{64}", expected_agents_sha256):
            errors.append("expected AGENTS.md SHA-256 is malformed")
        elif _artifact.sha256(agents) != expected_agents_sha256:
            errors.append("site source AGENTS.md differs from the same-commit Git blob")
    return errors


def check_runner_source_topology_text(text: str) -> list[str]:
    errors: list[str] = []
    if text.count("check_site.py source-topology") != 2:
        errors.append(
            "offline runner must check source topology exactly before and after"
        )
    if re.search(
        r'find\s+"\$EQIORA_SITE_SOURCE_ROOT"[\s\S]{0,256}-type l -print -quit', text
    ):
        errors.append("offline runner retains blanket source-symlink rejection")
    return errors


def check_browser_supply(
    site_root: Path,
    browser_cache: Path,
    expected_executable_sha256: str,
    expected_executable_bytes: int,
    environment: dict[str, str] | None = None,
) -> list[str]:
    errors: list[str] = []
    if not re.fullmatch(r"[0-9a-f]{64}", expected_executable_sha256):
        errors.append("online full Chromium SHA-256 identity is malformed")
    if expected_executable_bytes <= 0:
        errors.append("online full Chromium byte identity is malformed")
    if not site_root.is_dir() or site_root.is_symlink():
        return ["Playwright site root must be a real directory"]
    if browser_cache.name != "eqiora-pw-1.62.1-r1234":
        errors.append("browser cache must use exact Playwright/revision identity")
    if not browser_cache.is_dir() or browser_cache.is_symlink():
        errors.append("browser cache must be a real directory")
    elif not browser_cache.is_absolute() or browser_cache.resolve() != browser_cache:
        errors.append("browser cache must be an absolute canonical path")
    try:
        lock = json.loads((site_root / "package-lock.json").read_text(encoding="utf-8"))
        packages = lock["packages"]
        browsers = json.loads(
            (site_root / "node_modules/playwright-core/browsers.json").read_text(
                encoding="utf-8"
            )
        )["browsers"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as error:
        return [*errors, f"locked Playwright metadata is unavailable: {error}"]
    if not isinstance(packages, dict) or not isinstance(browsers, list):
        return [*errors, "locked Playwright metadata has the wrong shape"]
    for package in ("@playwright/test", "playwright", "playwright-core"):
        entry = packages.get(f"node_modules/{package}")
        if not isinstance(entry, dict) or entry.get("version") != "1.62.1":
            errors.append(f"browser supply must retain locked {package} 1.62.1")
        try:
            installed = json.loads(
                (site_root / "node_modules" / package / "package.json").read_text(
                    encoding="utf-8"
                )
            )
        except (OSError, json.JSONDecodeError) as error:
            errors.append(f"installed {package} metadata is unavailable: {error}")
        else:
            if not isinstance(installed, dict) or installed.get("version") != "1.62.1":
                errors.append(f"browser supply must retain installed {package} 1.62.1")
    chromium = [
        entry
        for entry in browsers
        if isinstance(entry, dict) and entry.get("name") == "chromium"
    ]
    if len(chromium) != 1 or (
        chromium[0].get("revision"),
        chromium[0].get("browserVersion"),
    ) != ("1234", "151.0.7922.34"):
        errors.append("browser supply metadata must select Chromium 1234/151.0.7922.34")
    supplied_environment = dict(os.environ if environment is None else environment)
    supplied_environment["PLAYWRIGHT_BROWSERS_PATH"] = str(browser_cache)
    try:
        resolved = subprocess.run(
            [
                "node",
                "-e",
                "const {chromium}=require('playwright');process.stdout.write(chromium.executablePath())",
            ],
            cwd=site_root,
            check=False,
            env=supplied_environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return [*errors, f"locked Playwright executable resolution failed: {error}"]
    expected = browser_cache / "chromium-1234/chrome-linux64/chrome"
    if (
        resolved.returncode != 0
        or resolved.stderr
        or resolved.stdout != os.fsencode(expected)
    ):
        errors.append("locked Playwright did not resolve the exact full Chromium path")
    executable = Path(os.fsdecode(resolved.stdout)) if resolved.stdout else expected
    try:
        mode = executable.lstat().st_mode
    except OSError as error:
        return [*errors, f"full Chromium executable is unavailable: {error}"]
    if executable.is_symlink() or not stat.S_ISREG(mode):
        return [*errors, "full Chromium executable must be a regular non-symlink file"]
    if not os.access(executable, os.X_OK):
        errors.append("full Chromium executable is not executable")
    if executable.stat().st_size != expected_executable_bytes:
        errors.append("full Chromium byte length changed after online verification")
    if _artifact.sha256(executable) != expected_executable_sha256:
        errors.append("full Chromium SHA-256 changed after online verification")
    try:
        with executable.open("rb") as browser:
            magic = browser.read(4)
        if magic != b"\x7fELF":
            errors.append(
                "full Chromium executable must be the acquired binary, not a shim"
            )
    except OSError as error:
        return [*errors, f"full Chromium executable cannot be read: {error}"]
    if errors:
        return errors
    try:
        version = subprocess.run(
            [str(executable), "--version"],
            check=False,
            env={**supplied_environment, "LC_ALL": "C", "TZ": "UTC"},
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return [f"full Chromium version observation failed: {error}"]
    if (
        version.returncode != 0
        or version.stderr
        or version.stdout != FULL_CHROMIUM_VERSION_STDOUT
    ):
        errors.append("full Chromium must emit the exact 41-byte locked version stdout")
    return errors


def check_runner_browser_supply_text(text: str) -> list[str]:
    errors: list[str] = []
    if text.count("check_site.py browser-supply") != 1:
        errors.append("offline runner must verify the locked full browser exactly once")
    for token in (
        '--expected-executable-sha256 "$EQIORA_SITE_BROWSER_SHA256"',
        '--expected-executable-bytes "$EQIORA_SITE_BROWSER_BYTES"',
    ):
        if token not in text:
            errors.append(f"offline runner omits online browser identity {token!r}")
    if "HeadlessChrome 151.0.7922.34" in text:
        errors.append("offline runner retains obsolete headless-shell identity")
    return errors
