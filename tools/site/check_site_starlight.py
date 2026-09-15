"""Private Starlight shell, route, and child-policy composition."""

from __future__ import annotations

import sys
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

try:
    import tools.site.check_site_html as _html
    import tools.site.check_site_sitemap as _sitemap
    import tools.site.check_site_starlight_content as _content
except ModuleNotFoundError as error:
    if error.name not in {
        "tools",
        "tools.site",
        "tools.site.check_site_html",
        "tools.site.check_site_sitemap",
        "tools.site.check_site_starlight_content",
    }:
        raise
    import check_site_html as _html
    import check_site_sitemap as _sitemap
    import check_site_starlight_content as _content


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
_require(_sitemap, "check_site_sitemap.py", ("SITEMAP_ROUTES", "check_sitemap"))
_require(
    _content,
    "check_site_starlight_content.py",
    (
        "WAKE_ALT",
        "PRESSURE_ALT",
        "PRESSURE_CAPTION",
        "CASE_SOURCE_PATHS",
        "CASE_EVIDENCE_PATHS",
        "check_starlight_content",
    ),
)

__all__ = (
    "STARLIGHT_ROUTES",
    "SITEMAP_ROUTES",
    "WAKE_ALT",
    "PRESSURE_ALT",
    "PRESSURE_CAPTION",
    "CASE_SOURCE_PATHS",
    "CASE_EVIDENCE_PATHS",
    "check_starlight",
)

SITEMAP_ROUTES = _sitemap.SITEMAP_ROUTES
WAKE_ALT = _content.WAKE_ALT
PRESSURE_ALT = _content.PRESSURE_ALT
PRESSURE_CAPTION = _content.PRESSURE_CAPTION
CASE_SOURCE_PATHS = _content.CASE_SOURCE_PATHS
CASE_EVIDENCE_PATHS = _content.CASE_EVIDENCE_PATHS

SITE_ORIGIN = "https://eqiora.org"
STARLIGHT_ROUTES = {
    "/": "index.html",
    "/contributing/architecture/": "contributing/architecture/index.html",
    "/capabilities/": "capabilities/index.html",
    "/guides/how-eqiora-fits-together/": "guides/how-eqiora-fits-together/index.html",
    "/contributing/": "contributing/index.html",
    "/evidence/": "evidence/index.html",
    "/gallery/": "gallery/index.html",
    "/gallery/exact-cylinder-steady-stokes/": "gallery/exact-cylinder-steady-stokes/index.html",
    "/gallery/mixed-boundary-elasticity/": "gallery/mixed-boundary-elasticity/index.html",
    "/gallery/karman-vortex-street/": "gallery/karman-vortex-street/index.html",
    "/get-started/": "get-started/index.html",
    "/guides/": "guides/index.html",
    "/guides/run-and-inspect/": "guides/run-and-inspect/index.html",
    "/guides/differentiation/": "guides/differentiation/index.html",
    "/guides/execution-and-arrays/": "guides/execution-and-arrays/index.html",
    "/guides/modeling/": "guides/modeling/index.html",
    "/reference/": "reference/index.html",
    "/reference/language/": "reference/language/index.html",
    "/reference/language/composition/": "reference/language/composition/index.html",
    "/reference/language/declarations/": "reference/language/declarations/index.html",
    "/reference/language/equations/": "reference/language/equations/index.html",
    "/reference/language/units/": "reference/language/units/index.html",
    "/reference/standard-packages/": "reference/standard-packages/index.html",
    "/reference/standard-packages/continuum/": "reference/standard-packages/continuum/index.html",
    "/reference/standard-packages/electrical/": "reference/standard-packages/electrical/index.html",
    "/reference/cli/": "reference/cli/index.html",
    "/reference/control-v2/": "reference/control-v2/index.html",
    "/reference/mcp/": "reference/mcp/index.html",
    "/reference/python/": "reference/python/index.html",
    "/reference/python/diff/": "reference/python/diff/index.html",
    "/reference/python/eqiora/": "reference/python/eqiora/index.html",
    "/reference/python/fem/": "reference/python/fem/index.html",
    "/reference/python/formulation/": "reference/python/formulation/index.html",
    "/reference/python/fvm/": "reference/python/fvm/index.html",
    "/reference/python/lang/": "reference/python/lang/index.html",
    "/reference/python/solve/": "reference/python/solve/index.html",
    "/reference/python/time/": "reference/python/time/index.html",
    "/reference/python/viewer/": "reference/python/viewer/index.html",
    "/reference/python/fluid/": "reference/python/fluid/index.html",
    "/reference/python/fsi/": "reference/python/fsi/index.html",
    "/reference/python/geometry/": "reference/python/geometry/index.html",
    "/reference/python/jax/": "reference/python/jax/index.html",
    "/reference/python/matplotlib/": "reference/python/matplotlib/index.html",
    "/reference/python/meshing/": "reference/python/meshing/index.html",
    "/reference/python/solid/": "reference/python/solid/index.html",
    "/reference/python/torch/": "reference/python/torch/index.html",
    "/reference/python/trajectory/": "reference/python/trajectory/index.html",
    "/reference/rust/": "reference/rust/index.html",
    "/release-notes/": "release-notes/index.html",
    "/learn/": "learn/index.html",
    "/learn/mathematical-modeling/": "learn/mathematical-modeling/index.html",
    "/learn/mathematical-modeling/algebraic-relations-networks/": "learn/mathematical-modeling/algebraic-relations-networks/index.html",
    "/learn/mathematical-modeling/boundary-interface-conditions/": "learn/mathematical-modeling/boundary-interface-conditions/index.html",
    "/learn/mathematical-modeling/conservation-laws/": "learn/mathematical-modeling/conservation-laws/index.html",
    "/learn/mathematical-modeling/constitutive-laws/": "learn/mathematical-modeling/constitutive-laws/index.html",
    "/learn/mathematical-modeling/fields-spatial-domains/": "learn/mathematical-modeling/fields-spatial-domains/index.html",
    "/learn/mathematical-modeling/models-not-simulations/": "learn/mathematical-modeling/models-not-simulations/index.html",
    "/learn/mathematical-modeling/ordinary-differential-equations/": "learn/mathematical-modeling/ordinary-differential-equations/index.html",
    "/learn/mathematical-modeling/quantities-dimensions-units/": "learn/mathematical-modeling/quantities-dimensions-units/index.html",
    "/reference/standard-packages/controls/": "reference/standard-packages/controls/index.html",
    "/reference/python/units/": "reference/python/units/index.html",
    "/reference/python/colab/": "reference/python/colab/index.html",
    "/learn/circuits-dynamics/component-laws/": "learn/circuits-dynamics/component-laws/index.html",
    "/learn/circuits-dynamics/conserving-networks/": "learn/circuits-dynamics/conserving-networks/index.html",
    "/learn/circuits-dynamics/decay-and-storage/": "learn/circuits-dynamics/decay-and-storage/index.html",
    "/learn/circuits-dynamics/": "learn/circuits-dynamics/index.html",
    "/learn/circuits-dynamics/resistor-divider/": "learn/circuits-dynamics/resistor-divider/index.html",
    "/learn/circuits-dynamics/voltage-current-power/": "learn/circuits-dynamics/voltage-current-power/index.html",
    "/learn/fluid-mechanics/computing-incompressible-flow/": "learn/fluid-mechanics/computing-incompressible-flow/index.html",
    "/learn/fluid-mechanics/continuum-and-statics/": "learn/fluid-mechanics/continuum-and-statics/index.html",
    "/learn/fluid-mechanics/cylinder-wakes/": "learn/fluid-mechanics/cylinder-wakes/index.html",
    "/learn/fluid-mechanics/exact-parallel-flows/": "learn/fluid-mechanics/exact-parallel-flows/index.html",
    "/learn/fluid-mechanics/": "learn/fluid-mechanics/index.html",
    "/learn/fluid-mechanics/mass-and-momentum/": "learn/fluid-mechanics/mass-and-momentum/index.html",
    "/learn/fluid-mechanics/references/": "learn/fluid-mechanics/references/index.html",
    "/learn/fluid-mechanics/stokes-and-boundaries/": "learn/fluid-mechanics/stokes-and-boundaries/index.html",
    "/learn/heat-transfer/assessing-a-thermal-calculation/": "learn/heat-transfer/assessing-a-thermal-calculation/index.html",
    "/learn/heat-transfer/boundaries-and-interfaces/": "learn/heat-transfer/boundaries-and-interfaces/index.html",
    "/learn/heat-transfer/conservation-and-fourier-law/": "learn/heat-transfer/conservation-and-fourier-law/index.html",
    "/learn/heat-transfer/heated-body-in-eqiora/": "learn/heat-transfer/heated-body-in-eqiora/index.html",
    "/learn/heat-transfer/": "learn/heat-transfer/index.html",
    "/learn/heat-transfer/steady-conduction/": "learn/heat-transfer/steady-conduction/index.html",
    "/learn/heat-transfer/transient-storage/": "learn/heat-transfer/transient-storage/index.html",
    "/learn/inverse-problems/differentiating-a-solved-model/": "learn/inverse-problems/differentiating-a-solved-model/index.html",
    "/learn/inverse-problems/heat-source-recovery/": "learn/inverse-problems/heat-source-recovery/index.html",
    "/learn/inverse-problems/": "learn/inverse-problems/index.html",
    "/learn/inverse-problems/measurements-to-parameters/": "learn/inverse-problems/measurements-to-parameters/index.html",
    "/learn/inverse-problems/noise-scaling-and-design/": "learn/inverse-problems/noise-scaling-and-design/index.html",
    "/learn/inverse-problems/sensitivity-and-identifiability/": "learn/inverse-problems/sensitivity-and-identifiability/index.html",
    "/learn/numerical-simulation/errors-and-residuals/": "learn/numerical-simulation/errors-and-residuals/index.html",
    "/learn/numerical-simulation/finite-volume-balance/": "learn/numerical-simulation/finite-volume-balance/index.html",
    "/learn/numerical-simulation/": "learn/numerical-simulation/index.html",
    "/learn/numerical-simulation/refinement-and-reproducibility/": "learn/numerical-simulation/refinement-and-reproducibility/index.html",
    "/learn/numerical-simulation/time-integration/": "learn/numerical-simulation/time-integration/index.html",
    "/learn/numerical-simulation/weak-forms-and-finite-elements/": "learn/numerical-simulation/weak-forms-and-finite-elements/index.html",
    "/learn/structural-mechanics/": "learn/structural-mechanics/index.html",
    "/learn/structural-mechanics/interpreting-results/": "learn/structural-mechanics/interpreting-results/index.html",
    "/learn/structural-mechanics/loads-energy/": "learn/structural-mechanics/loads-energy/index.html",
    "/learn/structural-mechanics/mixed-boundary-square/": "learn/structural-mechanics/mixed-boundary-square/index.html",
    "/learn/structural-mechanics/strain-stress/": "learn/structural-mechanics/strain-stress/index.html",
    "/learn/structural-mechanics/virtual-work/": "learn/structural-mechanics/virtual-work/index.html",
    "/404.html": "404.html",
}


class _ShellInspection(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.header_depth = 0
        self.hidden_depth = 0
        self.active: list[dict[str, object]] = []
        self.titles: list[dict[str, object]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {name.casefold(): value or "" for name, value in attrs}
        if tag == "header":
            self.header_depth += 1
        if (
            tag in {"script", "style", "template", "svg"}
            or "hidden" in values
            or values.get("aria-hidden") == "true"
        ):
            self.hidden_depth += 1
        if tag == "a" and "site-title" in values.get("class", "").split():
            record: dict[str, object] = {
                "attrs": values,
                "header": self.header_depth > 0,
                "images": [],
                "text": [],
            }
            self.active.append(record)
            self.titles.append(record)
        elif tag == "img" and self.active:
            images = self.active[-1]["images"]
            assert isinstance(images, list)
            images.append(values)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self.active:
            self.active.pop()
        if tag in {"script", "style", "template", "svg"} and self.hidden_depth:
            self.hidden_depth -= 1
        if tag == "header" and self.header_depth:
            self.header_depth -= 1

    def handle_data(self, data: str) -> None:
        if self.active and not self.hidden_depth:
            text = self.active[-1]["text"]
            assert isinstance(text, list)
            text.append(data)


def _same_origin_file(artifact: Path, page: Path, value: str) -> Path | None:
    parsed = urlsplit(value)
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        return None
    if parsed.scheme or value.startswith("//"):
        if (
            parsed.scheme not in {"http", "https"}
            or f"{parsed.scheme}://{parsed.netloc}" != SITE_ORIGIN
        ):
            return None
    raw_path = unquote(parsed.path)
    target = (
        artifact / raw_path.lstrip("/")
        if raw_path.startswith("/")
        else page.parent / raw_path
    )
    if target.is_symlink():
        return None
    target = target.resolve()
    try:
        target.relative_to(artifact)
    except ValueError:
        return None
    return target if target.is_file() else None


def _check_shell(
    artifact: Path,
    route: str,
    page: Path,
    raw: str,
    file_digests: dict[Path, str],
    favicon_digest: str,
) -> list[str]:
    errors: list[str] = []
    inspection = _ShellInspection()
    inspection.feed(raw)
    inspection.close()
    if len(inspection.titles) != 1 or not inspection.titles[0]["header"]:
        errors.append(
            f"{route}: site-title home link must appear exactly once in the page banner"
        )
        return errors
    title = inspection.titles[0]
    attrs = title["attrs"]
    text = title["text"]
    images = title["images"]
    assert (
        isinstance(attrs, dict) and isinstance(text, list) and isinstance(images, list)
    )
    if attrs.get("href") != "/":
        errors.append(f"{route}: header home link href must be exactly '/'")
    forbidden_names = {"aria-label", "aria-labelledby", "title"}
    visible_name = " ".join("".join(text).split())
    image_names = any(
        image.get("alt") != "" or forbidden_names.intersection(image)
        for image in images
    )
    if visible_name != "Eqiora" or forbidden_names.intersection(attrs) or image_names:
        errors.append(
            f"{route}: header home link must derive its name only from visible 'Eqiora'"
        )
    if len(images) != 1:
        errors.append(f"{route}: header brand asset must appear exactly once")
        return errors
    target = _same_origin_file(artifact, page, images[0].get("src", ""))
    if target is None:
        errors.append(f"{route}: header brand asset must be same-origin")
    elif file_digests.get(target) != favicon_digest:
        errors.append(f"{route}: header brand asset has the wrong digest")
    return errors


def _check_head(route: str, parser: object) -> list[str]:
    errors: list[str] = []
    canonicals = [
        link.get("href", "")
        for link in parser.links
        if "canonical" in link.get("rel", "").split()
    ]
    canonical_route = "/404/" if route == "/404.html" else route
    expected = f"{SITE_ORIGIN}{canonical_route}"
    if canonicals != [expected]:
        errors.append(
            f"{route}: canonical must be exactly {expected!r}, got {canonicals!r}"
        )
    properties = {
        (meta.get("property") or meta.get("name"), meta.get("content"))
        for meta in parser.metas
    }
    if ("og:image", f"{SITE_ORIGIN}/social-card.svg") not in properties:
        errors.append(f"{route}: missing exact same-origin Open Graph image")
    rels = {(link.get("rel", ""), link.get("href", "")) for link in parser.links}
    if not any("icon" in rel.split() and href == "/favicon.svg" for rel, href in rels):
        errors.append(f"{route}: missing exact favicon link")
    if not any(
        "apple-touch-icon" in rel.split() and href == "/apple-touch-icon.png"
        for rel, href in rels
    ):
        errors.append(f"{route}: missing exact apple-touch-icon link")
    return errors


def check_starlight(
    artifact: Path,
    inspections: dict[Path, tuple[str, object]],
    file_digests: dict[Path, str],
    wake_digest: str,
    pressure_digest: str,
    favicon_digest: str,
    source_sha: str,
    expected_python_version: str,
    maximum_sitemap_urls: int,
) -> list[str]:
    errors: list[str] = []
    for path, (raw, parser) in sorted(inspections.items()):
        relative = path.relative_to(artifact)
        route = (
            "/"
            if relative == Path("index.html")
            else f"/{relative.parent.as_posix()}/"
            if relative.name == "index.html"
            else f"/{relative.as_posix()}"
        )
        errors.extend(_check_head(route, parser))
        errors.extend(
            _check_shell(
                artifact,
                route,
                path,
                raw,
                file_digests,
                favicon_digest,
            )
        )
        if parser.inline_handlers:
            errors.append(f"{relative}: inline event handlers are forbidden")
    for route, relative in STARLIGHT_ROUTES.items():
        path = artifact / relative
        if path not in inspections:
            errors.append(f"missing required Starlight route {route}: {relative}")
    for displaced in ("api", "examples", "textbooks", "python", "concepts", "architecture"):
        if any(path.is_relative_to(artifact / displaced) for path in inspections):
            errors.append(f"displaced reader route remains published: /{displaced}/")
    case = inspections.get(artifact / "gallery/exact-cylinder-steady-stokes/index.html")
    if case and "<form" in case[0].casefold():
        errors.append(
            "gallery/exact-cylinder-steady-stokes/index.html: forms imply an uncontracted case interaction"
        )
    errors.extend(
        _content.check_starlight_content(
            artifact,
            inspections,
            file_digests,
            wake_digest,
            pressure_digest,
            favicon_digest,
            source_sha,
            expected_python_version,
        )
    )
    if not (artifact / "pagefind/pagefind.js").is_file():
        errors.append("Pagefind JavaScript entry is missing")
    try:
        robots = (artifact / "robots.txt").read_text(encoding="utf-8")
    except OSError as error:
        errors.append(f"robots.txt is missing: {error}")
    else:
        if (
            robots
            != "User-agent: *\nAllow: /\nSitemap: https://eqiora.org/sitemap-index.xml\n"
        ):
            errors.append("robots.txt differs from the exact public crawl boundary")
    errors.extend(_sitemap.check_sitemap(artifact, maximum_sitemap_urls))
    return errors
