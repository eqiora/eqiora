"""Private admitted-inventory and artifact-policy composition."""

from __future__ import annotations

import hashlib
import os
import re
import stat
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    import tools.site.check_site_html as _html
    import tools.site.check_site_references as _references
    import tools.site.check_site_rustdoc as _rustdoc
    import tools.site.check_site_starlight as _starlight
except ModuleNotFoundError as error:
    if error.name not in {
        "tools",
        "tools.site",
        "tools.site.check_site_html",
        "tools.site.check_site_references",
        "tools.site.check_site_rustdoc",
        "tools.site.check_site_starlight",
    }:
        raise
    import check_site_html as _html
    import check_site_references as _references
    import check_site_rustdoc as _rustdoc
    import check_site_starlight as _starlight


def _require(module: object, filename: str, exports: tuple[str, ...]) -> None:
    expected = Path(__file__).with_name(filename).resolve()
    registered = sys.modules.get(f"tools.site.{filename.removesuffix('.py')}")
    if registered is not None and registered is not module:
        raise ImportError(
            f"site checker {filename} did not resolve to its exact sibling"
        )
    if Path(module.__file__ or "").resolve() != expected:
        raise ImportError(
            f"site checker {filename} did not resolve to its exact sibling"
        )
    if module.__all__ != exports:
        raise ImportError(f"site checker {filename} exposes an unexpected interface")


_require(_html, "check_site_html.py", ("HtmlInspection", "normalize", "read_html"))
_require(_references, "check_site_references.py", ("check_references",))
_require(_rustdoc, "check_site_rustdoc.py", ("RUSTDOC_ROOT", "check_rustdoc"))
_require(
    _starlight,
    "check_site_starlight.py",
    (
        "STARLIGHT_ROUTES",
        "SITEMAP_ROUTES",
        "WAKE_ALT",
        "PRESSURE_ALT",
        "PRESSURE_CAPTION",
        "CASE_SOURCE_PATHS",
        "CASE_EVIDENCE_PATHS",
        "check_starlight",
    ),
)

__all__ = (
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

MAX_FILES = 20_000
MAX_FILE_BYTES = 32 * 1024 * 1024
MAX_TOTAL_BYTES = 512 * 1024 * 1024
MAX_HTML_BYTES = 4 * 1024 * 1024
_PUBLIC_ASSETS = Path(__file__).resolve().parents[2] / "docs/site/public"


def _source_asset_digest(name: str) -> str:
    path = _PUBLIC_ASSETS / name
    # Missing source input cannot match any artifact SHA-256. Let the ordinary
    # source/artifact checks report it instead of failing while importing them.
    return hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else ""


# Publication preserves the current source assets, not an obsolete generation.
_pressure = (
    _PUBLIC_ASSETS.parent / "src/assets/gallery/exact-cylinder-pressure-presentation.png"
)
_wake = _PUBLIC_ASSETS.parent / "src/assets/gallery/karman-vortex-street-poster.png"
WAKE_SHA256 = hashlib.sha256(_wake.read_bytes()).hexdigest() if _wake.is_file() else ""
PRESSURE_SHA256 = (
    hashlib.sha256(_pressure.read_bytes()).hexdigest() if _pressure.is_file() else ""
)
SOCIAL_SHA256 = _source_asset_digest("social-card.svg")
FAVICON_SHA256 = _source_asset_digest("favicon.svg")
APPLE_TOUCH_SHA256 = _source_asset_digest("apple-touch-icon.png")
OLD_SOCIAL_SHA256 = "3b9be694357a6db29674e82eabfdb63738d0e40bf70b3f00163737b490b9128b"
OLD_SOCIAL_LINE = "Open-source computational engineering · Alpha 0.1.0a1"
PRESSURE_ALT = _starlight.PRESSURE_ALT
PRESSURE_CAPTION = _starlight.PRESSURE_CAPTION
CASE_SOURCE_PATHS = _starlight.CASE_SOURCE_PATHS
CASE_EVIDENCE_PATHS = _starlight.CASE_EVIDENCE_PATHS
SITEMAP_ROUTES = _starlight.SITEMAP_ROUTES
WAKE_ALT = _starlight.WAKE_ALT
ROUTES = {
    **_starlight.STARLIGHT_ROUTES,
    "/reference/rust/api/eqiora/struct.Diagnostic.html": "reference/rust/api/eqiora/struct.Diagnostic.html",
}
SOURCE_SHA = re.compile(r"^[0-9a-f]{40}$")


@dataclass(frozen=True)
class SiteIdentities:
    """Current source-asset identities."""

    wake: str = WAKE_SHA256
    pressure: str = PRESSURE_SHA256
    social: str = SOCIAL_SHA256
    favicon: str = FAVICON_SHA256
    apple_touch: str = APPLE_TOUCH_SHA256


PRODUCTION_IDENTITIES = SiteIdentities()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_exact_source(path: Path, expected: str, label: str) -> list[str]:
    if not path.is_file() or path.is_symlink():
        return [f"missing exact {label}: {path}"]
    observed = sha256(path)
    return (
        []
        if observed == expected
        else [f"{label} digest mismatch: expected {expected}, got {observed}"]
    )


def _inventory(
    artifact: Path,
    maximum_files: int,
    maximum_total_bytes: int,
) -> tuple[list[Path], list[str]]:
    files: list[Path] = []
    errors: list[str] = []
    total = 0
    if not artifact.is_dir() or artifact.is_symlink():
        return [], [f"artifact must be a real directory: {artifact}"]
    for current, directories, names in os.walk(artifact, followlinks=False):
        directories.sort()
        names.sort()
        for name in [*directories, *names]:
            path = Path(current) / name
            try:
                details = path.lstat()
            except OSError as error:
                errors.append(
                    f"artifact entry is unavailable: {path.relative_to(artifact)}: {error}"
                )
                continue
            if stat.S_ISLNK(details.st_mode):
                errors.append(
                    f"artifact contains symlink: {path.relative_to(artifact)}"
                )
        for name in names:
            path = Path(current) / name
            details = path.lstat()
            if not stat.S_ISREG(details.st_mode):
                errors.append(
                    f"artifact contains non-regular file: {path.relative_to(artifact)}"
                )
                continue
            if details.st_size > MAX_FILE_BYTES:
                errors.append(
                    f"artifact file exceeds read cap: {path.relative_to(artifact)}"
                )
            total += details.st_size
            files.append(path)
            if len(files) > maximum_files:
                return files, [*errors, f"artifact exceeds {maximum_files} files"]
            if total > maximum_total_bytes:
                return files, [*errors, f"artifact exceeds {maximum_total_bytes} bytes"]
    return files, errors


def _check_identities(
    artifact: Path,
    files: list[Path],
    file_digests: dict[Path, str],
    identities: SiteIdentities,
) -> list[str]:
    errors: list[str] = []
    digest_paths: dict[str, list[Path]] = {}
    for path, digest in file_digests.items():
        digest_paths.setdefault(digest, []).append(path)
    for relative, expected in {
        "social-card.svg": identities.social,
        "favicon.svg": identities.favicon,
        "apple-touch-icon.png": identities.apple_touch,
    }.items():
        path = artifact / relative
        if not path.is_file() or file_digests.get(path) != expected:
            errors.append(
                f"public asset {relative} does not have admitted digest {expected}"
            )
    if len(digest_paths.get(identities.social, [])) != 1:
        errors.append(
            "assembled site must expose exactly one timeless social-card byte identity"
        )
    if digest_paths.get(OLD_SOCIAL_SHA256):
        errors.append("assembled site exposes the deprecated social-card bytes")
    if len(digest_paths.get(identities.pressure, [])) != 1:
        errors.append("assembled site must expose exactly one admitted pressure image")
    if len(digest_paths.get(identities.wake, [])) != 1:
        errors.append("assembled site must expose exactly one admitted wake poster")
    for path in files:
        if path.stat().st_size > MAX_HTML_BYTES or path.suffix.casefold() not in {
            ".html",
            ".svg",
            ".xml",
            ".json",
            ".txt",
            ".css",
            ".js",
        }:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if OLD_SOCIAL_LINE in text:
            errors.append(
                f"assembled site exposes deprecated social-card copy: {path.relative_to(artifact)}"
            )
        if "assets/social-card.svg" in text:
            errors.append(
                f"assembled site references the legacy social-card route: {path.relative_to(artifact)}"
            )
        if (
            "/blob/main/" in text
            or "/tree/main/" in text
            or "EQIORA_SITE_SOURCE_SHA" in text
        ):
            errors.append(
                f"{path.relative_to(artifact)}: unresolved or branch-relative source identity"
            )
    return errors


def _inspect_html(
    artifact: Path,
    files: list[Path],
) -> tuple[dict[Path, tuple[str, object]], list[str]]:
    inspections: dict[Path, tuple[str, object]] = {}
    errors: list[str] = []
    for path in files:
        if path.suffix.casefold() != ".html":
            continue
        try:
            inspections[path] = _html.read_html(path, MAX_HTML_BYTES)
        except (OSError, UnicodeDecodeError, ValueError) as error:
            errors.append(f"invalid HTML {path.relative_to(artifact)}: {error}")
    return inspections, errors


def check_artifact(
    artifact: Path,
    source_sha: str,
    expected_python_version: str,
    identities: SiteIdentities = PRODUCTION_IDENTITIES,
    *,
    maximum_files: int = MAX_FILES,
    maximum_total_bytes: int = MAX_TOTAL_BYTES,
) -> list[str]:
    if not SOURCE_SHA.fullmatch(source_sha):
        return ["source SHA must be exactly 40 lowercase hexadecimal characters"]
    artifact = artifact.resolve()
    files, errors = _inventory(artifact, maximum_files, maximum_total_bytes)
    if errors:
        return errors
    file_digests = {path: sha256(path) for path in files}
    errors.extend(_check_identities(artifact, files, file_digests, identities))
    inspections, inspection_errors = _inspect_html(artifact, files)
    errors.extend(inspection_errors)
    rustdoc = {
        path: value
        for path, value in inspections.items()
        if path.relative_to(artifact).is_relative_to(_rustdoc.RUSTDOC_ROOT)
    }
    starlight = {
        path: value for path, value in inspections.items() if path not in rustdoc
    }
    errors.extend(
        _starlight.check_starlight(
            artifact,
            starlight,
            file_digests,
            identities.wake,
            identities.pressure,
            identities.favicon,
            source_sha,
            expected_python_version,
            maximum_files,
        )
    )
    errors.extend(_rustdoc.check_rustdoc(artifact, rustdoc))
    errors.extend(
        _references.check_references(
            artifact, files, inspections, starlight, source_sha
        )
    )
    return errors
