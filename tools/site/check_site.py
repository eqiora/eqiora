#!/usr/bin/env python3
"""Verify and serve the bounded Eqiora static-site artifact without network access."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import mimetypes
import re
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

try:
    import tools.site.check_site_artifact as _site_artifact
    import tools.site.check_site_html as _site_html
    import tools.site.check_site_supply as _site_supply
except ModuleNotFoundError as error:
    if error.name not in {
        "tools",
        "tools.site",
        "tools.site.check_site_artifact",
        "tools.site.check_site_html",
        "tools.site.check_site_supply",
    }:
        raise
    import check_site_artifact as _site_artifact
    import check_site_html as _site_html
    import check_site_supply as _site_supply

_ARTIFACT_EXPORTS = (
    "MAX_FILES",
    "MAX_FILE_BYTES",
    "MAX_TOTAL_BYTES",
    "MAX_HTML_BYTES",
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
    "PRESSURE_ALT",
    "PRESSURE_CAPTION",
    "CASE_SOURCE_PATHS",
    "CASE_EVIDENCE_PATHS",
    "sha256",
    "check_exact_source",
    "check_artifact",
)
_SUPPLY_EXPORTS = (
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
for module, filename, exports in (
    (_site_artifact, "check_site_artifact.py", _ARTIFACT_EXPORTS),
    (_site_html, "check_site_html.py", ("HtmlInspection", "normalize", "read_html")),
    (_site_supply, "check_site_supply.py", _SUPPLY_EXPORTS),
):
    registered = sys.modules.get(f"tools.site.{filename.removesuffix('.py')}")
    if registered is not None and registered is not module:
        raise ImportError(
            f"site checker {filename} did not resolve to its exact sibling"
        )
    if (
        Path(module.__file__ or "").resolve()
        != Path(__file__).with_name(filename).resolve()
    ):
        raise ImportError(
            f"site checker {filename} did not resolve to its exact sibling"
        )
    if module.__all__ != exports:
        raise ImportError(f"site checker {filename} exposes an unexpected interface")
if _site_supply.subprocess is not subprocess:
    raise ImportError(
        "site checker subprocess facade does not share its supply binding"
    )

HtmlInspection = _site_html.HtmlInspection
normalize = _site_html.normalize
read_html = _site_html.read_html
MAX_FILES = _site_artifact.MAX_FILES
MAX_FILE_BYTES = _site_artifact.MAX_FILE_BYTES
MAX_TOTAL_BYTES = _site_artifact.MAX_TOTAL_BYTES
MAX_HTML_BYTES = _site_artifact.MAX_HTML_BYTES
PRESSURE_SHA256 = _site_artifact.PRESSURE_SHA256
SOCIAL_SHA256 = _site_artifact.SOCIAL_SHA256
FAVICON_SHA256 = _site_artifact.FAVICON_SHA256
APPLE_TOUCH_SHA256 = _site_artifact.APPLE_TOUCH_SHA256
OLD_SOCIAL_SHA256 = _site_artifact.OLD_SOCIAL_SHA256
OLD_SOCIAL_LINE = _site_artifact.OLD_SOCIAL_LINE
PRESSURE_ALT = _site_artifact.PRESSURE_ALT
PRESSURE_CAPTION = _site_artifact.PRESSURE_CAPTION
SiteIdentities = _site_artifact.SiteIdentities
PRODUCTION_IDENTITIES = _site_artifact.PRODUCTION_IDENTITIES
ROUTES = _site_artifact.ROUTES
SITEMAP_ROUTES = _site_artifact.SITEMAP_ROUTES
CASE_SOURCE_PATHS = _site_artifact.CASE_SOURCE_PATHS
CASE_EVIDENCE_PATHS = _site_artifact.CASE_EVIDENCE_PATHS
sha256 = _site_artifact.sha256
FULL_CHROMIUM_VERSION_STDOUT_HEX = _site_supply.FULL_CHROMIUM_VERSION_STDOUT_HEX
FULL_CHROMIUM_VERSION_STDOUT = _site_supply.FULL_CHROMIUM_VERSION_STDOUT
DIRECT_SOURCE_ARCHIVE_COMMAND = _site_supply.DIRECT_SOURCE_ARCHIVE_COMMAND
EXACT_LINK_PAYLOAD_SHA256 = _site_supply.EXACT_LINK_PAYLOAD_SHA256
EXACT_TREE_LINK_COMMAND = _site_supply.EXACT_TREE_LINK_COMMAND
EXACT_EXTRACTED_LINK_COMMAND = _site_supply.EXACT_EXTRACTED_LINK_COMMAND
OFFLINE_WORKFLOW_TOKENS = _site_supply.OFFLINE_WORKFLOW_TOKENS
FORBIDDEN_WORKFLOW_TOKENS = _site_supply.FORBIDDEN_WORKFLOW_TOKENS
check_source_topology = _site_supply.check_source_topology
check_runner_source_topology_text = _site_supply.check_runner_source_topology_text
check_browser_supply = _site_supply.check_browser_supply
check_runner_browser_supply_text = _site_supply.check_runner_browser_supply_text

ACTION_USE = re.compile(r"^\s*uses:\s*([^@\s]+)@([^\s#]+)", re.MULTILINE)
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*]\(([^)\n]+)\)")
HTML_REFERENCE = re.compile(r"""(?:href|src)=["']([^"']+)["']""", re.IGNORECASE)


@dataclass(frozen=True)
class ReleaseIdentity:
    cargo: str
    python: str


DIRECT_PINS = {
    "astro": "7.2.8",
    "@astrojs/starlight": "0.41.7",
    "@astrojs/mdx": "7.0.6",
    "@astrojs/markdown-satteri": "0.3.6",
    "satteri": "0.9.5",
    "katex": "0.18.4",
    "@playwright/test": "1.62.1",
    "@axe-core/playwright": "4.13.0",
    "axe-core": "4.13.0",
}
DEVELOPMENT_PACKAGES = {"@playwright/test", "@axe-core/playwright", "axe-core"}
RUNTIME_PINS = {
    name: version
    for name, version in DIRECT_PINS.items()
    if name not in DEVELOPMENT_PACKAGES
}
DEVELOPMENT_PINS = {
    name: version for name, version in DIRECT_PINS.items() if name in DEVELOPMENT_PACKAGES
}
ROOT_DEPENDENCY_SECTIONS = {
    "dependencies": RUNTIME_PINS,
    "devDependencies": DEVELOPMENT_PINS,
    "optionalDependencies": {},
    "peerDependencies": {},
}
FORBIDDEN_CLIENT_PACKAGES = {"react", "react-dom"}
PROVIDER_PATHS = (
    "docs/site/src/components/site/ExactSourceLink.astro",
    "docs/site/src/components/site/ReleaseIdentity.astro",
)


def _destination(raw: str) -> str:
    destination = raw.strip()
    if destination.startswith("<") and ">" in destination:
        return destination[1 : destination.index(">")]
    return destination.split(maxsplit=1)[0]


def check_markdown_links(site_root: Path) -> list[str]:
    """Retained source-link helper used by the pre-successor unit suite."""

    errors: list[str] = []
    resolved_site = site_root.resolve()
    for document in sorted(site_root.rglob("*.md")):
        text = document.read_text(encoding="utf-8")
        destinations = [
            _destination(match.group(1)) for match in MARKDOWN_LINK.finditer(text)
        ]
        destinations.extend(match.group(1) for match in HTML_REFERENCE.finditer(text))
        for destination in destinations:
            if not destination or destination.startswith("#"):
                continue
            parsed = urlsplit(destination)
            if parsed.scheme:
                if parsed.scheme not in {"https", "mailto"}:
                    errors.append(
                        f"{document}: unsupported link scheme {parsed.scheme!r}"
                    )
                continue
            if destination.startswith("//"):
                errors.append(f"{document}: scheme-relative URL is not allowed")
                continue
            path_text = unquote(parsed.path)
            if not path_text:
                continue
            candidate = (
                site_root / path_text.lstrip("/")
                if path_text.startswith("/")
                else document.parent / path_text
            )
            candidate = candidate.resolve()
            try:
                candidate.relative_to(resolved_site)
            except ValueError:
                errors.append(
                    f"{document}: local link escapes docs/site: {destination}"
                )
                continue
            if candidate.is_dir():
                candidate = candidate / "index.md"
            elif candidate.suffix == "":
                candidate = candidate.with_suffix(".md")
            if not candidate.exists():
                errors.append(f"{document}: missing local link target: {destination}")
    return errors


def derive_release_identity(
    root: Path,
) -> tuple[ReleaseIdentity | None, list[str]]:
    cargo_path = root / "Cargo.toml"
    mapper_path = root / "tools/release/python_candidate_common.py"
    if (
        not cargo_path.is_file()
        or cargo_path.is_symlink()
        or not mapper_path.is_file()
        or mapper_path.is_symlink()
    ):
        return None, ["Cargo version or accepted Python release mapper is absent"]
    try:
        cargo_document = tomllib.loads(cargo_path.read_text(encoding="utf-8"))
        cargo_version = cargo_document["workspace"]["package"]["version"]
        module_name = (
            "_eqiora_site_release_mapper_"
            + hashlib.sha256(str(mapper_path).encode("utf-8")).hexdigest()
        )
        specification = importlib.util.spec_from_file_location(module_name, mapper_path)
        if specification is None or specification.loader is None:
            raise ValueError("release mapper cannot be loaded")
        module = importlib.util.module_from_spec(specification)
        sys.modules[module_name] = module
        previous_bytecode_policy = sys.dont_write_bytecode
        sys.dont_write_bytecode = True
        try:
            specification.loader.exec_module(module)
        finally:
            sys.dont_write_bytecode = previous_bytecode_policy
        python_version = module.python_distribution_version(cargo_version)
    except Exception as error:  # noqa: BLE001 - a source mapper failure is a denial
        return None, [f"release identity derivation failed: {error}"]
    finally:
        if "module_name" in locals():
            sys.modules.pop(module_name, None)
    if not isinstance(cargo_version, str) or not re.fullmatch(
        r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)",
        cargo_version,
    ):
        return None, ["Cargo version is not the required release SemVer identity"]
    if python_version != cargo_version:
        return None, ["Python mapper did not preserve the release identity"]
    return ReleaseIdentity(cargo_version, python_version), []


def check_workflow_text(text: str) -> list[str]:
    errors: list[str] = []
    trigger = text.split("permissions:", maxsplit=1)[0]
    trigger_contract = (
        "  pull_request:\n    types: [opened, reopened, synchronize, edited]\n",
        "  push:\n    branches:\n      - main\n",
        "  workflow_dispatch:\n",
    )
    if any(token not in trigger for token in trigger_contract) or "paths:" in trigger:
        errors.append(
            "Pages must run one unfiltered required job for every pull request"
        )

    build_match = re.search(
        r"(?ms)^  build:\n(?P<body>.*?)(?=^  [a-zA-Z0-9_-]+:\n|\Z)", text
    )
    build = build_match.group("body") if build_match is not None else ""
    classifier_match = re.search(
        r"(?ms)^      - name: Classify the complete documentation input closure\n"
        r"(?P<body>.*?)(?=^      - name: |\Z)",
        build,
    )
    classifier = classifier_match.group("body") if classifier_match is not None else ""
    expected_classifier = (
        "        id: site_closure\n"
        "        env:\n"
        "          SITE_EVENT_NAME: ${{ github.event_name }}\n"
        "          SITE_BASE_SHA: ${{ github.event.pull_request.base.sha }}\n"
        "          SITE_HEAD_SHA: ${{ github.event.pull_request.head.sha }}\n"
        "        run: |\n"
        "          exec python3 tools/ci/classify_changes.py \\\n"
        "            --event \"$SITE_EVENT_NAME\" \\\n"
        "            --base \"$SITE_BASE_SHA\" \\\n"
        "            --head \"$SITE_HEAD_SHA\" \\\n"
        "            --github-output \"$GITHUB_OUTPUT\"\n"
    )
    if classifier != expected_classifier:
        errors.append("Pages omits the repository-owned input-closure classifier")

    full_condition = "if: steps.site_closure.outputs.site == 'true'"
    full_steps = (
        "Configure GitHub Pages",
        "Bind exact temporary supply roots",
        "Restore the exact Playwright browser cache",
        "Install the locked mise toolchain",
        "Prepare the exact clean source and unique home-backed scratch",
        "Supply locked native, Rust, Python, Node, and browser inputs",
        "Build and verify with only loopback networking",
        "Recheck exact inputs and checked artifact",
        "Upload GitHub Pages artifact",
    )
    for name in full_steps:
        marker = f"      - name: {name}\n        {full_condition}\n"
        if marker not in text:
            errors.append(f"Pages full build step is not fail-closed: {name}")
    cleanup = (
        "      - name: Retain an authenticated empty site scratch for runner teardown\n"
        "        if: always() && steps.site_closure.outputs.site == 'true'\n"
    )
    if cleanup not in text:
        errors.append("Pages cleanup is not bound to a selected full build")
    deploy_match = re.search(
        r"(?ms)^  deploy:\n(?P<body>.*?)(?=^  [a-zA-Z0-9_-]+:\n|\Z)", text
    )
    deploy = deploy_match.group("body") if deploy_match is not None else ""
    deploy_header = deploy.split("    steps:\n", maxsplit=1)[0]
    deploy_header = "\n".join(
        line
        for line in deploy_header.splitlines()
        if not line.lstrip().startswith("#")
    )
    deploy_if = (
        "    if: >-\n"
        "      github.event_name == 'push' &&\n"
        "      github.ref == 'refs/heads/main' &&\n"
        "      needs.build.outputs.full_build == 'true'\n"
    )
    if (
        deploy_if not in deploy_header + "\n"
        or len(re.findall(r"(?m)^    if:", deploy_header)) != 1
        or not re.search(r"(?m)^    needs: build$", deploy_header)
    ):
        errors.append(
            "Pages deployment is not bound to an authenticated main full build"
        )
    for action, revision in ACTION_USE.findall(text):
        if not re.fullmatch(r"[0-9a-f]{40}", revision):
            errors.append(f"Pages action {action} is not pinned to a full SHA")
    for token in OFFLINE_WORKFLOW_TOKENS:
        if token not in text:
            errors.append(f"Pages workflow omits offline/supply boundary {token!r}")
    for token in FORBIDDEN_WORKFLOW_TOKENS:
        if token in text:
            errors.append(
                f"Pages workflow uses forbidden supply substitution {token!r}"
            )
    archive = text.find(DIRECT_SOURCE_ARCHIVE_COMMAND)
    source_export = text.find('echo "EQIORA_SITE_SOURCE_ROOT=', archive + 1)
    before_archive = (
        'git ls-tree -r "$GITHUB_SHA"',
        "source_links=",
        'case "$source_links" in',
        'if test -n "$source_links"; then',
        EXACT_TREE_LINK_COMMAND,
        'git ls-tree "$GITHUB_SHA" -- AGENTS.md',
        'git cat-file blob "$GITHUB_SHA:AGENTS.md"',
    )
    after_archive = (
        'if test -n "$source_links"; then',
        '\n            test -L "$scratch/source/CLAUDE.md"\n',
        EXACT_EXTRACTED_LINK_COMMAND,
        'cmp "$scratch/source/AGENTS.md" "$scratch/expected-AGENTS.md"',
        'elif test -e "$scratch/source/CLAUDE.md" || test -L "$scratch/source/CLAUDE.md"; then',
    )
    before_positions = [text.find(token) for token in before_archive]
    after_positions = [text.find(token, archive + 1) for token in after_archive]
    archive_window = text[archive:source_export] if 0 <= archive < source_export else ""
    archive_bound = (
        archive >= 0
        and source_export > archive
        and all(0 <= position < archive for position in before_positions)
        and before_positions == sorted(before_positions)
        and all(archive < position < source_export for position in after_positions)
        and after_positions == sorted(after_positions)
        and 'unlink "$scratch/source/CLAUDE.md"' not in archive_window
        and 'rm -f "$scratch/source/CLAUDE.md"' not in archive_window
    )
    if not archive_bound:
        errors.append("Pages archive must bind the tracked link after extraction")
    browser_admission = (
        'expected_browser_sha256="0b20b130e7edd9dd51873be867761295fe0cfad490c2b9a64f95bd3cfc08fa71"',
        'expected_browser_bytes="290614600"',
        'browser_sha256="$(sha256sum "$browser_path" | cut -d \' \' -f 1)"',
        'browser_bytes="$(stat -c %s "$browser_path")"',
        'test "$browser_sha256" = "$expected_browser_sha256"',
        'test "$browser_bytes" = "$expected_browser_bytes"',
        'EQIORA_SITE_BROWSER_SHA256="$expected_browser_sha256"',
        'EQIORA_SITE_BROWSER_BYTES="$expected_browser_bytes"',
        "export EQIORA_SITE_BROWSER_SHA256 EQIORA_SITE_BROWSER_BYTES",
        "check_site.py browser-supply",
        'version_hex="$("$browser_path" --version | od -An -tx1 | tr -d \'[:space:]\')"',
        'test "$version_hex" = "$expected_browser_version_hex"',
    )
    browser_positions = [text.find(token) for token in browser_admission]
    if any(
        position < 0 for position in browser_positions
    ) or browser_positions != sorted(browser_positions):
        errors.append("Pages browser identity must precede execution and propagation")
    ordered = (
        'git ls-tree -r "$GITHUB_SHA"',
        DIRECT_SOURCE_ARCHIVE_COMMAND,
        EXACT_EXTRACTED_LINK_COMMAND,
        "playwright install --with-deps chromium",
        "chromium.executablePath()",
        FULL_CHROMIUM_VERSION_STDOUT_HEX,
        "check_site.py browser-supply",
        "unshare --net",
    )
    positions = [text.find(token) for token in ordered]
    if all(position >= 0 for position in positions) and positions != sorted(positions):
        errors.append("Pages archive/browser supply checks are out of causal order")
    return errors


def check_source(
    root: Path,
    identities: SiteIdentities = PRODUCTION_IDENTITIES,
    release_identity: ReleaseIdentity | None = None,
) -> list[str]:
    errors: list[str] = []
    if release_identity is None:
        release_identity, release_errors = derive_release_identity(root)
        errors.extend(release_errors)
    site = root / "docs/site"
    for obsolete in (root / "mkdocs.yml", site / "assets/social-card.svg"):
        if obsolete.exists() or obsolete.is_symlink():
            errors.append(
                f"obsolete successor source remains: {obsolete.relative_to(root)}"
            )
    errors.extend(
        _site_artifact.check_exact_source(
            site / "src/assets/gallery/exact-cylinder-pressure-presentation.png",
            identities.pressure,
            "admitted pressure media",
        )
    )
    errors.extend(
        _site_artifact.check_exact_source(
            site / "public/social-card.svg", identities.social, "timeless social card"
        )
    )
    errors.extend(
        _site_artifact.check_exact_source(
            site / "public/favicon.svg", identities.favicon, "favicon"
        )
    )
    errors.extend(
        _site_artifact.check_exact_source(
            site / "public/apple-touch-icon.png",
            identities.apple_touch,
            "apple touch icon",
        )
    )
    provider_requirements = {
        root / "tools/ci/classify_changes.py": (
            "SITE_INPUT_FILES",
            "site_input_path",
            "recognized_path",
            "exact_commit",
        ),
        site / "src/components/site/ExactSourceLink.astro": (
            "EQIORA_SITE_SOURCE_SHA",
            "blob",
            "tree",
        ),
        site / "src/components/site/ReleaseIdentity.astro": (
            "EQIORA_SITE_CARGO_VERSION",
            "EQIORA_SITE_PYTHON_VERSION",
        ),
    }
    for provider, tokens in provider_requirements.items():
        try:
            if provider.is_symlink() or not provider.is_file():
                raise OSError("not a regular file")
            provider_text = provider.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            errors.append(
                f"accepted provider is missing or invalid: {provider.relative_to(root)}: {error}"
            )
            continue
        for token in tokens:
            if token not in provider_text:
                errors.append(
                    f"accepted provider {provider.relative_to(root)} omits {token!r}"
                )

    provider_consumers = {
        site / "src/content/docs/evidence/index.mdx": (
            "@components/site/ExactSourceLink.astro",
            "<ExactSourceLink",
        ),
        site / "astro.config.mjs": (
            "src/components/site/ExactSourceLink.astro",
            "src/components/site/ReleaseIdentity.astro",
        ),
    }
    for consumer, tokens in provider_consumers.items():
        try:
            if consumer.is_symlink() or not consumer.is_file():
                raise OSError("not a regular file")
            consumer_text = consumer.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as error:
            errors.append(
                f"provider consumer is missing or invalid: {consumer.relative_to(root)}: {error}"
            )
            continue
        for token in tokens:
            if token not in consumer_text:
                errors.append(
                    f"provider consumer {consumer.relative_to(root)} omits {token!r}"
                )

    package = site / "package.json"
    lock = site / "package-lock.json"
    try:
        package_data = json.loads(package.read_text(encoding="utf-8"))
        lock_data = json.loads(lock.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"site package/lock is missing or invalid: {error}")
    else:
        dependencies = package_data.get("dependencies")
        development = package_data.get("devDependencies")
        lock_packages = lock_data.get("packages", {})
        lock_root = lock_packages.get("", {})
        for label, document in (("package", package_data), ("lock root", lock_root)):
            for section, expected in ROOT_DEPENDENCY_SECTIONS.items():
                if document.get(section, {}) != expected:
                    errors.append(
                        f"site {label} {section} differs from the exact direct set"
                    )
        declared = {
            **(dependencies if isinstance(dependencies, dict) else {}),
            **(development if isinstance(development, dict) else {}),
        }
        for name, expected in DIRECT_PINS.items():
            if declared.get(name) != expected:
                errors.append(f"site package must pin {name} exactly to {expected}")
            entry = lock_packages.get(f"node_modules/{name}", {})
            if entry.get("version") != expected:
                errors.append(f"site lock must resolve {name} exactly to {expected}")
            if not isinstance(entry.get("integrity"), str) or not entry[
                "integrity"
            ].startswith("sha512-"):
                errors.append(
                    f"site lock lacks registry integrity for {name}@{expected}"
                )
        if package_data.get("engines") != {"node": "24.18.1", "npm": "11.16.0"}:
            errors.append("site package must pin Node 24.18.1 and npm 11.16.0")
        if lock_root.get("engines") != package_data.get("engines"):
            errors.append("site lock root engines differ from package.json")
        for package_path, entry in lock_packages.items():
            if re.search(r"(?:^|/)node_modules/(?:react|react-dom)$", package_path):
                errors.append(f"site lock realizes client framework {package_path!r}")
            for section in ROOT_DEPENDENCY_SECTIONS:
                edges = entry.get(section, {})
                if isinstance(edges, dict) and any(
                    name in FORBIDDEN_CLIENT_PACKAGES
                    or str(specification).startswith(("npm:react@", "npm:react-dom@"))
                    for name, specification in edges.items()
                ):
                    errors.append(
                        f"site lock {package_path!r} has forbidden {section} edge"
                    )

    workflow = root / ".github/workflows/pages.yml"
    try:
        workflow_text = workflow.read_text(encoding="utf-8")
    except OSError as error:
        errors.append(f"missing Pages workflow: {error}")
    else:
        errors.extend(check_workflow_text(workflow_text))

    runner = root / "tools/site/run_offline_site_checks.sh"
    try:
        runner_text = runner.read_text(encoding="utf-8")
    except OSError as error:
        errors.append(f"missing offline site runner: {error}")
    else:
        errors.extend(check_runner_source_topology_text(runner_text))
        errors.extend(check_runner_browser_supply_text(runner_text))
        runner_tokens = (
            "build_products.py",
            "generate_interface_reference.py",
            "--repository",
            "--eqiora-binary",
            "--mcp-binary",
            "--check",
            "build_rust_reference.py",
            "--rustdoc-root",
            "--output",
            "check_site.py check",
            "check_site.py serve",
            "npm_config_offline",
            "CARGO_NET_OFFLINE",
            "EQIORA_SITE_CARGO_VERSION",
            "EQIORA_SITE_PYTHON_VERSION",
            "tools.site.tests.test_site_tools",
        )
        for token in runner_tokens:
            if token not in runner_text:
                errors.append(
                    f"offline runner omits frozen orchestration token {token!r}"
                )

    source_files = [
        source
        for suffix in (
            "*.astro",
            "*.css",
            "*.js",
            "*.json",
            "*.md",
            "*.mdx",
            "*.mjs",
            "*.svg",
            "*.ts",
        )
        for source in (site / "src").rglob(suffix)
    ]
    for source in sorted(source_files):
        text = source.read_text(encoding="utf-8")
        if OLD_SOCIAL_LINE in text:
            errors.append(
                f"site source retains deprecated social-card copy: {source.relative_to(root)}"
            )
        if re.search(r"\bfetch\s*\(", text):
            errors.append(
                f"build/runtime content fetch is forbidden: {source.relative_to(root)}"
            )
        if re.search(r"(?:src|poster)\s*=\s*[\"']https?://", text):
            errors.append(
                f"external executable asset reference: {source.relative_to(root)}"
            )
    return errors


def check_artifact(
    artifact: Path,
    source_sha: str,
    expected_python_version: str,
    identities: SiteIdentities = PRODUCTION_IDENTITIES,
) -> list[str]:
    """Retain the checker-level call shape while delegating sole artifact policy."""

    return _site_artifact.check_artifact(
        artifact,
        source_sha,
        expected_python_version,
        identities,
        maximum_files=MAX_FILES,
        maximum_total_bytes=MAX_TOTAL_BYTES,
    )


def check_site(
    root: Path,
    artifact: Path,
    source_sha: str,
    identities: SiteIdentities = PRODUCTION_IDENTITIES,
) -> list[str]:
    release_identity, _ = derive_release_identity(root)
    return [
        *check_source(root, identities),
        *check_artifact(
            artifact,
            source_sha,
            release_identity.python if release_identity is not None else "",
            identities,
        ),
    ]


class SiteRequestHandler(BaseHTTPRequestHandler):
    artifact: Path

    def do_HEAD(self) -> None:  # noqa: N802
        self._send(send_body=False)

    def do_GET(self) -> None:  # noqa: N802
        self._send(send_body=True)

    def _send(self, *, send_body: bool) -> None:
        parsed = urlsplit(self.path)
        relative = PurePosixPath(unquote(parsed.path).lstrip("/"))
        unsafe = relative.is_absolute() or ".." in relative.parts
        if unsafe:
            self.send_error(400)
            return
        candidate = self.artifact.joinpath(*relative.parts)
        if parsed.path.endswith("/") or not relative.parts:
            candidate /= "index.html"
        elif candidate.is_dir():
            self.send_response(308)
            self.send_header("Location", parsed.path + "/")
            self.end_headers()
            return
        status_code = 200
        if not candidate.is_file() or candidate.is_symlink():
            candidate = self.artifact / "404.html"
            status_code = 404
        try:
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(self.artifact)
            payload = resolved.read_bytes()
        except (OSError, ValueError):
            self.send_error(404)
            return
        content_type = (
            mimetypes.guess_type(resolved.name)[0] or "application/octet-stream"
        )
        if content_type.startswith("text/") or content_type in {
            "application/javascript",
            "application/json",
            "image/svg+xml",
        }:
            content_type += "; charset=utf-8"
        self.send_response(status_code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        if send_body:
            self.wfile.write(payload)

    def log_message(self, format: str, *args: object) -> None:
        print(f"site server: {format % args}", file=sys.stderr)


def serve(artifact: Path, host: str, port: int) -> int:
    resolved = artifact.resolve(strict=True)
    if not resolved.is_dir() or artifact.is_symlink():
        raise ValueError("artifact must be a real directory")
    handler = type(
        "BoundSiteRequestHandler", (SiteRequestHandler,), {"artifact": resolved}
    )
    server = ThreadingHTTPServer((host, port), handler)
    print(f"site server: http://{host}:{server.server_port}", flush=True)
    server.serve_forever()
    return 0


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    source = subparsers.add_parser("source")
    source.add_argument("--root", type=Path, required=True)
    check = subparsers.add_parser("check")
    check.add_argument("--root", type=Path, required=True)
    check.add_argument("--artifact", type=Path, required=True)
    check.add_argument("--source-sha", required=True)
    server = subparsers.add_parser("serve")
    server.add_argument("--artifact", type=Path, required=True)
    server.add_argument("--host", default="127.0.0.1")
    server.add_argument("--port", type=int, default=4173)
    topology = subparsers.add_parser("source-topology")
    topology.add_argument("--root", type=Path, required=True)
    topology.add_argument("--expected-agents-sha256")
    browser = subparsers.add_parser("browser-supply")
    browser.add_argument("--site-root", type=Path, required=True)
    browser.add_argument("--browser-cache", type=Path, required=True)
    browser.add_argument("--expected-executable-sha256", required=True)
    browser.add_argument("--expected-executable-bytes", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "source":
        errors = check_source(args.root.resolve())
        if errors:
            print("\n".join(f"site source: {error}" for error in errors), file=sys.stderr)
            return 1
        print("site source: checks passed (build and browser checks run separately)")
        return 0
    if args.command == "serve":
        try:
            return serve(args.artifact, args.host, args.port)
        except (OSError, ValueError) as error:
            print(f"site server: {error}", file=sys.stderr)
            return 1
    if args.command == "source-topology":
        errors = check_source_topology(args.root, args.expected_agents_sha256)
        if errors:
            print(
                "\n".join(f"site source: {error}" for error in errors), file=sys.stderr
            )
            return 1
        print("site source: exact optional CLAUDE.md topology admitted")
        return 0
    if args.command == "browser-supply":
        errors = check_browser_supply(
            args.site_root,
            args.browser_cache,
            args.expected_executable_sha256,
            args.expected_executable_bytes,
        )
        if errors:
            print(
                "\n".join(f"site browser: {error}" for error in errors), file=sys.stderr
            )
            return 1
        print("site browser: exact locked full Chromium supply admitted")
        return 0
    errors = check_site(args.root.resolve(), args.artifact.resolve(), args.source_sha)
    if errors:
        print("\n".join(f"site check: {error}" for error in errors), file=sys.stderr)
        return 1
    print("site check: exact static artifact satisfies the bounded public contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
