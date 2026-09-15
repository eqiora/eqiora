"""Private bounded sitemap-graph policy for the static-site checker."""

from __future__ import annotations

from collections import Counter
from pathlib import Path, PurePosixPath
from urllib.parse import urlsplit
from xml.etree import ElementTree

__all__ = ("SITEMAP_ROUTES", "check_sitemap")

SITE_ORIGIN = "https://eqiora.org"
MAX_CHILDREN = 16
SITEMAP_ROUTES = (
    "/",
    "/contributing/architecture/",
    "/capabilities/",
    "/guides/how-eqiora-fits-together/",
    "/contributing/",
    "/evidence/",
    "/gallery/",
    "/gallery/exact-cylinder-steady-stokes/",
    "/gallery/mixed-boundary-elasticity/",
    "/gallery/karman-vortex-street/",
    "/get-started/",
    "/guides/",
    "/guides/run-and-inspect/",
    "/guides/differentiation/",
    "/guides/execution-and-arrays/",
    "/guides/modeling/",
    "/reference/",
    "/reference/language/",
    "/reference/language/composition/",
    "/reference/language/declarations/",
    "/reference/language/equations/",
    "/reference/language/units/",
    "/reference/standard-packages/",
    "/reference/standard-packages/continuum/",
    "/reference/standard-packages/electrical/",
    "/reference/cli/",
    "/reference/python/",
    "/reference/python/diff/",
    "/reference/python/eqiora/",
    "/reference/python/fem/",
    "/reference/python/formulation/",
    "/reference/python/fvm/",
    "/reference/python/lang/",
    "/reference/python/solve/",
    "/reference/python/time/",
    "/reference/python/viewer/",
    "/reference/python/fluid/",
    "/reference/python/fsi/",
    "/reference/python/geometry/",
    "/reference/python/jax/",
    "/reference/python/matplotlib/",
    "/reference/python/meshing/",
    "/reference/python/solid/",
    "/reference/python/torch/",
    "/reference/python/trajectory/",
    "/reference/rust/",
    "/release-notes/",
    "/learn/",
    "/learn/mathematical-modeling/",
    "/learn/mathematical-modeling/algebraic-relations-networks/",
    "/learn/mathematical-modeling/boundary-interface-conditions/",
    "/learn/mathematical-modeling/conservation-laws/",
    "/learn/mathematical-modeling/constitutive-laws/",
    "/learn/mathematical-modeling/fields-spatial-domains/",
    "/learn/mathematical-modeling/models-not-simulations/",
    "/learn/mathematical-modeling/ordinary-differential-equations/",
    "/learn/mathematical-modeling/quantities-dimensions-units/",
    "/reference/standard-packages/controls/",
    "/reference/python/units/",
    "/reference/python/colab/",
    "/learn/circuits-dynamics/component-laws/",
    "/learn/circuits-dynamics/conserving-networks/",
    "/learn/circuits-dynamics/decay-and-storage/",
    "/learn/circuits-dynamics/",
    "/learn/circuits-dynamics/resistor-divider/",
    "/learn/circuits-dynamics/voltage-current-power/",
    "/learn/fluid-mechanics/computing-incompressible-flow/",
    "/learn/fluid-mechanics/continuum-and-statics/",
    "/learn/fluid-mechanics/cylinder-wakes/",
    "/learn/fluid-mechanics/exact-parallel-flows/",
    "/learn/fluid-mechanics/",
    "/learn/fluid-mechanics/mass-and-momentum/",
    "/learn/fluid-mechanics/references/",
    "/learn/fluid-mechanics/stokes-and-boundaries/",
    "/learn/heat-transfer/assessing-a-thermal-calculation/",
    "/learn/heat-transfer/boundaries-and-interfaces/",
    "/learn/heat-transfer/conservation-and-fourier-law/",
    "/learn/heat-transfer/heated-body-in-eqiora/",
    "/learn/heat-transfer/",
    "/learn/heat-transfer/steady-conduction/",
    "/learn/heat-transfer/transient-storage/",
    "/learn/inverse-problems/differentiating-a-solved-model/",
    "/learn/inverse-problems/heat-source-recovery/",
    "/learn/inverse-problems/",
    "/learn/inverse-problems/measurements-to-parameters/",
    "/learn/inverse-problems/noise-scaling-and-design/",
    "/learn/inverse-problems/sensitivity-and-identifiability/",
    "/learn/numerical-simulation/errors-and-residuals/",
    "/learn/numerical-simulation/finite-volume-balance/",
    "/learn/numerical-simulation/",
    "/learn/numerical-simulation/refinement-and-reproducibility/",
    "/learn/numerical-simulation/time-integration/",
    "/learn/numerical-simulation/weak-forms-and-finite-elements/",
    "/learn/structural-mechanics/",
    "/learn/structural-mechanics/interpreting-results/",
    "/learn/structural-mechanics/loads-energy/",
    "/learn/structural-mechanics/mixed-boundary-square/",
    "/learn/structural-mechanics/strain-stress/",
    "/learn/structural-mechanics/virtual-work/",
)


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse(path: Path) -> tuple[ElementTree.Element | None, list[str]]:
    try:
        text = path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as error:
        return None, [f"sitemap XML is missing or invalid: {error}"]
    if "<!DOCTYPE" in text.upper() or "<!ENTITY" in text.upper():
        return None, ["sitemap XML forbids DOCTYPE and entities"]
    try:
        return ElementTree.fromstring(text), []
    except ElementTree.ParseError as error:
        return None, [f"sitemap XML is missing or invalid: {error}"]


def _locations(root: ElementTree.Element) -> list[str]:
    return [
        (node.text or "").strip()
        for node in root.iter()
        if _local_name(node.tag) == "loc"
    ]


def _admit_url(value: str, *, child: bool) -> tuple[str | None, str | None]:
    parsed = urlsplit(value)
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        label = "sitemap child" if child else "sitemap URL"
        return None, f"{label} must not contain query, fragment, or userinfo"
    if f"{parsed.scheme}://{parsed.netloc}" != SITE_ORIGIN:
        label = "sitemap child" if child else "sitemap URL"
        return None, f"{label} must be exact same-origin"
    path = parsed.path
    if not path.startswith("/") or ".." in PurePosixPath(path).parts:
        label = "sitemap child path" if child else "sitemap URL path"
        return None, f"{label} escapes the artifact"
    return path, None


def _url_set(
    root: ElementTree.Element,
    *,
    maximum_urls: int,
) -> tuple[list[str], list[str]]:
    if _local_name(root.tag) != "urlset":
        return [], ["sitemap child must be a URL set"]
    urls: list[str] = []
    errors: list[str] = []
    for value in _locations(root):
        path, error = _admit_url(value, child=False)
        if error:
            errors.append(error)
        elif path is not None:
            urls.append(path)
        if len(urls) > maximum_urls:
            errors.append(f"sitemap exceeds {maximum_urls} URLs")
            break
    if len(urls) != len(set(urls)):
        errors.append("sitemap contains duplicate URLs")
    return urls, errors


def check_sitemap(artifact: Path, maximum_urls: int) -> list[str]:
    index = artifact / "sitemap-index.xml"
    if index.is_symlink() or not index.is_file():
        return ["sitemap-index.xml is missing or invalid"]
    root, errors = _parse(index)
    if root is None:
        return errors
    if _local_name(root.tag) == "urlset":
        urls, child_errors = _url_set(root, maximum_urls=maximum_urls)
        errors.extend(child_errors)
    elif _local_name(root.tag) == "sitemapindex":
        children = _locations(root)
        if len(children) > MAX_CHILDREN:
            errors.append(f"sitemap exceeds {MAX_CHILDREN} children")
        if len(children) != len(set(children)):
            errors.append("duplicate sitemap child")
        urls = []
        seen_children: set[Path] = set()
        for value in children[: MAX_CHILDREN + 1]:
            path_text, error = _admit_url(value, child=True)
            if error:
                errors.append(error)
                continue
            assert path_text is not None
            relative = PurePosixPath(path_text.lstrip("/"))
            child = artifact.joinpath(*relative.parts)
            linked = any(
                artifact.joinpath(*relative.parts[:index]).is_symlink()
                for index in range(1, len(relative.parts) + 1)
            )
            if linked or not child.is_file():
                errors.append(f"sitemap child is missing: {path_text}")
                continue
            child = child.resolve()
            try:
                child.relative_to(artifact)
            except ValueError:
                errors.append("sitemap child path escapes the artifact")
                continue
            if child in seen_children:
                continue
            seen_children.add(child)
            child_root, child_errors = _parse(child)
            errors.extend(child_errors)
            if child_root is None:
                continue
            child_urls, child_errors = _url_set(
                child_root,
                maximum_urls=maximum_urls,
            )
            errors.extend(child_errors)
            urls.extend(child_urls)
            if len(urls) > maximum_urls:
                errors.append(f"sitemap exceeds {maximum_urls} URLs")
                break
        if len(urls) != len(set(urls)):
            errors.append("sitemap contains duplicate URLs")
    else:
        return [*errors, "sitemap-index.xml must be a sitemap index or URL set"]
    counts = Counter(urls)
    for route, count in counts.items():
        if count > 1:
            errors.append(f"duplicate sitemap route {route}")
    for route in SITEMAP_ROUTES:
        if counts[route] == 0:
            errors.append(f"sitemap omits required route {route}")
    if counts["/404.html"]:
        errors.append("sitemap must not publish the 404 artifact route")
    return errors
