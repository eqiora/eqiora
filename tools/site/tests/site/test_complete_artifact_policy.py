from __future__ import annotations

import base64
import importlib.util
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from fixture import REPOSITORY, SOURCE_SHA, checker, make_fixture


SITE_ORIGIN = "https://eqiora.org"
MAX_FILES = 20_000
MAX_DATA_URL_BYTES = 1024 * 1024
RUSTDOC_ROOT = Path("reference/rust/api")
FRAGMENT_ID = "impl-LifetimeProbe%3C'a%3E-for-Thing"
QUOTED_FRAGMENT_ID = 'impl-Quoted%3C"Marker%3E-for-Thing'
NUMERIC_FRAGMENT_ID = "123"
LINE_RANGE_FRAGMENT = "1-325"
BRAND_PATH = "/assets/eqiora-mark.BN8rmEAl.svg"
WAKE_PATH = "/assets/karman-vortex.Bada55aa.png"
WAKE_ALT = "Cell-average vorticity in a Kármán vortex street behind a circular cylinder."
PRESSURE_PATH = "/assets/exact-cylinder-pressure.C0ffee42.png"
PRESSURE_ALT = "Steady Stokes pressure around a cylinder, with the current mesh and pressure scale in pascals."
PRESSURE_CAPTION = "Steady Stokes pressure on a 0.025 m target mesh."
RENDERED_SOURCE_SENTENCE = (
    "This website is a curated projection, not a parallel specification. "
    "Detailed contracts remain in the repository’s architecture, RFCs, "
    "capability matrix, and validated verify manifests."
)
REFERENCE_GUIDANCE = "Look up a declaration, find a physical building block, or inspect an API signature."

SOURCE_PATHS = (
    "examples/python/exact_cylinder_stokes.py",
    "examples/python/exact_cylinder_geometry.py",
    "examples/python/exact_cylinder_mesh.py",
    "examples/steady-flow-past-cylinder.eqi",
    "crates/eqiora-api/packages/Eqiora.Fluid.Incompressible/src/incompressible.eqi",
)

STAGES = (
    ("problem-setup", "1", "Problem setup"),
    ("model-definition", "2", "Eqiora model definition"),
    ("mesh-and-boundaries", "3", "Mesh and boundaries"),
    ("submit-and-result", "4", "Submit and result"),
    ("pressure-visualization", "5", "Pressure visualization"),
    ("reading-pressure", "6", "Reading the pressure plot"),
)
ST_STARLIGHT_ROUTES = (
    "/",
    "/contributing/architecture/",
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
ABSENT_REFERENCES = (
    (
        "eqiora/api/trait.ReferenceRunObserver.html",
        "../../trait.impl/eqiora_api/reference_run/trait.ReferenceRunObserver.js",
    ),
    (
        "eqiora/api/trait.ScalarEllipticRunObserver.html",
        "../../trait.impl/eqiora_api/spatial/plan/trait.ScalarEllipticRunObserver.js",
    ),
    (
        "eqiora/backends/mpi/trait.MpiRankLocalCsrAction.html",
        "../../../trait.impl/eqiora_backend_mpi/runtime/trait.MpiRankLocalCsrAction.js",
    ),
    (
        "eqiora/device/trait.CommandQueue.html",
        "../../trait.impl/eqiora_device/queue/trait.CommandQueue.js",
    ),
    (
        "eqiora/device/trait.DeviceBuffer.html",
        "../../trait.impl/eqiora_device/buffer/trait.DeviceBuffer.js",
    ),
    (
        "eqiora/device/trait.Fence.html",
        "../../trait.impl/eqiora_device/queue/trait.Fence.js",
    ),
)

NAVIGATION = (
    '<nav class="sidebar"><a href="/learn/">Learn</a><a href="/guides/">Guides</a>'
    '<a href="/gallery/">Gallery</a><a href="/reference/">Reference</a>'
    '<a href="/get-started/">Get started</a>'
    '<a href="/release-notes/">Releases</a>'
    '<a href="/contributing/">Contributing</a>'
    '<a href="https://github.com/nkiyohara/eqiora">GitHub</a></nav>'
)
OLD_SHELL = (
    '<header><a class="site-title" href="/"><img src="/assets/brand.svg" alt="">'
    f'<span>Eqiora</span></a>{NAVIGATION}</header>'
)
SITE_TITLE = (
    f'<a class="site-title" href="/"><img src="{BRAND_PATH}" '
    'alt=""><span>Eqiora</span></a>'
)
SHELL = f"""<header>
{SITE_TITLE}
{NAVIGATION}
</header>"""
BACK_CONTROL = (
    '<a id="back" href="javascript:void(0)" onclick="history.back();">Back</a>'
)
SAFE_FONT_DATA = "data:font/woff2;base64," + base64.b64encode(b"wOF2oracle").decode(
    "ascii"
)
SAFE_SVG = b'<svg xmlns="http://www.w3.org/2000/svg"><path d="M0 0"/></svg>'
SAFE_SVG_DATA = "data:image/svg+xml;base64," + base64.b64encode(SAFE_SVG).decode(
    "ascii"
)
SAFE_SVG_LF = SAFE_SVG_DATA.replace("base64,", "base64,\\\n")
SAFE_SVG_CRLF = SAFE_SVG_DATA.replace("base64,", "base64,\\\r\n")
SAFE_SVG_CR = SAFE_SVG_DATA.replace("base64,", "base64,\\\r")


def _write(path: Path, value: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(value, encoding="utf-8")


def _replace(path: Path, old: str, new: str, count: int = 1) -> None:
    text = path.read_text(encoding="utf-8")
    observed = text.count(old)
    if observed != count:
        raise AssertionError(
            f"fixture replacement count for {path}: expected {count}, got {observed}"
        )
    path.write_text(text.replace(old, new), encoding="utf-8")


def _set_main(path: Path, body: str) -> None:
    text = path.read_text(encoding="utf-8")
    start = text.index("<main>")
    end = text.index("</main>", start)
    path.write_text(text[: start + 6] + body + text[end:], encoding="utf-8")


def _exact_link(relative: str, label: str, fragment: str = "") -> str:
    return (
        f'<a href="https://github.com/nkiyohara/eqiora/blob/{SOURCE_SHA}/'
        f'{relative}{fragment}">{label}</a>'
    )


def _math(tex: str, display: bool = False) -> str:
    display_open = '<span class="katex-display">' if display else ""
    display_close = "</span>" if display else ""
    display_attribute = ' display="block"' if display else ""
    return (
        f'{display_open}<span class="katex"><span class="katex-mathml">'
        f"<math{display_attribute}><semantics><mrow><mi>x</mi></mrow>"
        f'<annotation encoding="application/x-tex">{tex}</annotation>'
        f'</semantics></math></span><span class="katex-html" aria-hidden="true">'
        f"x</span></span>{display_close}"
    )


def _stage(identifier: str, step: str, title: str, body: str) -> str:
    heading = f"{identifier}-title"
    return (
        f'<section class="eq-stage" aria-labelledby="{heading}" '
        f'data-step="{step}" id="{identifier}"><header><h2 id="{heading}">'
        f'<a class="eq-stage-marker" href="#{identifier}">Stage {step} {title}</a>'
        f"</h2></header><div>{body}</div></section>"
    )


def _case_body() -> str:
    links: list[str] = []
    for relative in SOURCE_PATHS:
        label = Path(relative).name
        links.append(_exact_link(relative, label))

    sentinel = _exact_link(
        "examples/python/exact_cylinder_stokes.py",
        "Eqiora source form: Python resolve/run path",
        "#L51-L67",
    )

    source_form = """<p><strong>Eqiora source form</strong></p><pre>relation momentum on fluid {
  -div(
    2 * dynamic_viscosity * symmetric_part(grad(velocity))
    - isotropic_lift(pressure)
  ) - grad(force_potential) = 0;
}
relation incompressibility on fluid {
  div(velocity) = 0;
}</pre>"""
    stage_bodies = (
        "<p>The fluid region is a 2.2m by 0.41m channel with an exact authored circle.</p>"
        + _math("H"),
        "<p>The accepted stress construction is rendered at build time.</p>"
        + _math(
            r"\boldsymbol{\sigma}(\boldsymbol{u},p)=2\mu\,\operatorname{sym}(\nabla\boldsymbol{u})-p\boldsymbol{I}",
            display=True,
        )
        + source_form,
        _math(r"\nabla\cdot\boldsymbol{u}=0"),
        "<p>The immutable intent resolves to one Plan, Run, and Result.</p>"
        + sentinel,
        f'<figure><img src="{PRESSURE_PATH}" alt="{PRESSURE_ALT}"><figcaption>'
        f"{PRESSURE_CAPTION}</figcaption></figure>",
        '<a href="/learn/fluid-mechanics/">Continue with fluid mechanics</a>'
        + " ".join(links),
    )
    sections = "".join(
        _stage(identifier, step, title, body)
        for (identifier, step, title), body in zip(STAGES, stage_bodies, strict=True)
    )
    return "<h1>Exact-cylinder steady Stokes</h1>" + sections


def _rustdoc_page(body: str) -> str:
    return (
        "<!doctype html><html><head>"
        '<link rel="stylesheet" href="/reference/rust/api/static.files/rustdoc.css">'
        '<script src="/reference/rust/api/static.files/main.js"></script>'
        '<link rel="icon" href="/favicon.svg" type="image/svg+xml">'
        '<a class="eqiora-return" href="/reference/rust/" '
        'aria-label="Back to the Eqiora Rust reference">Back to Eqiora docs</a>'
        f"</head><body><main>{body}</main></body></html>"
    )


def _sitemap_index(children: tuple[str, ...] = ("/sitemap-0.xml",)) -> str:
    locations = "".join(
        f"<sitemap><loc>{SITE_ORIGIN}{child}</loc></sitemap>" for child in children
    )
    return (
        '<?xml version="1.0"?><sitemapindex '
        f'xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{locations}'
        "</sitemapindex>"
    )


def _url_set(routes: tuple[str, ...] = ST_STARLIGHT_ROUTES) -> str:
    locations = "".join(
        f"<url><loc>{SITE_ORIGIN}{route}</loc></url>" for route in routes
    )
    return (
        '<?xml version="1.0"?><urlset '
        f'xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">{locations}</urlset>'
    )


def _ordinary(root: Path):
    artifact, identities = make_fixture(root)

    brand = artifact / "assets/brand.svg"
    wake = artifact / "assets/wake.png"
    pressure = artifact / "assets/pressure.png"
    _write(artifact / BRAND_PATH.removeprefix("/"), brand.read_bytes())
    _write(artifact / WAKE_PATH.removeprefix("/"), wake.read_bytes())
    _write(artifact / PRESSURE_PATH.removeprefix("/"), pressure.read_bytes())
    brand.unlink()
    wake.unlink()
    pressure.unlink()

    for page in sorted(artifact.rglob("*.html")):
        relative = page.relative_to(artifact)
        if relative.is_relative_to(RUSTDOC_ROOT):
            continue
        _replace(page, OLD_SHELL, SHELL)

    home = artifact / "index.html"
    _replace(home, 'src="/assets/wake.png"', f'src="{WAKE_PATH}"')
    _replace(
        home,
        "This website is a curated projection, not a parallel specification. "
        "Detailed contracts remain in the repository's architecture, RFCs, "
        "capability matrix, and validated verify manifests.",
        RENDERED_SOURCE_SENTENCE,
    )
    _set_main(
        artifact / "gallery/exact-cylinder-steady-stokes/index.html", _case_body()
    )
    _set_main(
        artifact / "reference/index.html",
        "<h1>Reference</h1><p>Python Rust CLI</p>"
        f"<p>{REFERENCE_GUIDANCE}</p>",
    )

    _write(artifact / "assets/KaTeX_Main-Regular.woff2", b"wOF2local-katex")
    _write(
        artifact / "assets/site.css",
        "@font-face{font-family:KaTeX;src:url('./KaTeX_Main-Regular.woff2')}\n"
        f"@font-face{{font-family:Oracle;src:url('{SAFE_FONT_DATA}')}}\n"
        f'.safe-svg{{background-image:url("{SAFE_SVG_DATA}")}}\n'
        f'.safe-svg-lf{{background-image:url("{SAFE_SVG_LF}")}}\n'
        f'.safe-svg-crlf{{background-image:url("{SAFE_SVG_CRLF}")}}\n'
        f'.safe-svg-cr{{background-image:url("{SAFE_SVG_CR}")}}\n',
    )

    _write(artifact / "sitemap-index.xml", _sitemap_index())
    _write(artifact / "sitemap-0.xml", _url_set())

    rustdoc = artifact / RUSTDOC_ROOT
    _write(rustdoc / "static.files/rustdoc.css", "body{color:CanvasText}\n")
    _write(rustdoc / "static.files/main.js", "window.searchState = {};\n")
    _write(
        rustdoc / "eqiora/index.html",
        _rustdoc_page(
            '<h1>Crate eqiora</h1><a href="module/index.html">Module item</a>'
            f'<a href="struct.Diagnostic.html#{FRAGMENT_ID}">Diagnostic impl</a>'
            f"<a href='struct.Diagnostic.html#{QUOTED_FRAGMENT_ID}'>Quoted ID</a>"
            f'<a href="struct.Diagnostic.html#{NUMERIC_FRAGMENT_ID}">Numeric ID</a>'
            f'<a href="../src/eqiora/lib.rs.html#{LINE_RANGE_FRAGMENT}">Source</a>'
        ),
    )
    _write(
        rustdoc / "src/eqiora/lib.rs.html",
        _rustdoc_page("<h1>Source</h1><span id=1>1</span><span id=325>325</span>"),
    )
    _write(
        rustdoc / "eqiora/struct.Diagnostic.html",
        _rustdoc_page(
            f'<h1>Struct Diagnostic</h1><section id="{FRAGMENT_ID}">Impl</section>'
            f"<section id='{QUOTED_FRAGMENT_ID}'>Quoted</section>"
            f"<section id={NUMERIC_FRAGMENT_ID}>Numeric</section>"
            '<a href="index.html">Crate eqiora</a>'
        ),
    )
    _write(
        rustdoc / "eqiora/module/index.html",
        _rustdoc_page(
            '<h1>Module item</h1><a href="../struct.Diagnostic.html">Diagnostic</a>'
            '<a href="https://doc.rust-lang.org/std/">Rust standard library</a>'
        ),
    )
    for relative, reference in ABSENT_REFERENCES:
        _write(
            rustdoc / relative,
            _rustdoc_page(
                f'<h1>Generated trait</h1><a href="{reference}">Implementors</a>'
            ),
        )
    for name in ("help.html", "settings.html"):
        _write(
            rustdoc / name,
            _rustdoc_page(
                '<h1>Rustdoc utility</h1><a href="./index.html">All crates</a>'
                '<a href="./index.html">Crate list</a>' + BACK_CONTROL
            ),
        )
    return artifact, identities


def _artifact_errors(artifact: Path, identities) -> list[str]:
    return checker.check_artifact(artifact, SOURCE_SHA, "0.1.0", identities)


def _append_main(path: Path, value: str) -> None:
    _replace(path, "</main>", value + "</main>")


class CompleteArtifactPolicyTests(unittest.TestCase):
    def test_00_mixed_starlight_rustdoc_positive_then_mutants(self) -> None:
        # This complete positive is deliberately first. Against accepted
        # predecessor 7555fcbdeb676b24781bffe5a5cd2f52e70011e5 it is the
        # causal RED and stops the method
        # before any mutant can execute or receive credit.
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact, identities = _ordinary(root)
            self.assertEqual(_artifact_errors(artifact, identities), [])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact, identities = _ordinary(root)
            rust_index = artifact / RUSTDOC_ROOT / "eqiora/index.html"
            _append_main(
                rust_index,
                '<section id="impl-Instrument-for-T">'
                '<a href="crate::Span">Span</a>'
                '<a href="super::Span::current()">current span</a>'
                "</section>"
                '<section id="impl-WithSubscriber-for-T">'
                '<a href="super::Subscriber">subscriber</a>'
                '<a href="dispatcher#setting-the-default-subscriber">default</a>'
                "</section>",
            )
            self.assertEqual(_artifact_errors(artifact, identities), [])

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact, identities = _ordinary(root)
            added_route = "/reference/python/new-module/"
            source = (artifact / "guides/index.html").read_text(encoding="utf-8")
            _write(
                artifact / "reference/python/new-module/index.html",
                source.replace(
                    f'{SITE_ORIGIN}/guides/',
                    f'{SITE_ORIGIN}{added_route}',
                ),
            )
            _write(
                artifact / "sitemap-0.xml",
                _url_set((*ST_STARLIGHT_ROUTES, added_route)),
            )
            self.assertEqual(_artifact_errors(artifact, identities), [])

        def reject(label: str, mutate, expected: str) -> None:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                artifact, identities = _ordinary(root)
                mutate(artifact)
                errors = _artifact_errors(artifact, identities)
                self.assertTrue(
                    any(expected in error for error in errors),
                    f"{label}: expected {expected!r} in {errors!r}",
                )

        home = Path("index.html")
        case = Path("gallery/exact-cylinder-steady-stokes/index.html")
        reference = Path("reference/index.html")

        reject(
            "required artifact route missing",
            lambda artifact: (artifact / "guides/index.html").unlink(),
            "missing required Starlight route /guides/",
        )
        reject(
            "additional route without the common shell",
            lambda artifact: _write(
                artifact / "unadmitted/index.html",
                "<!doctype html><html><body><main><h1>Extra</h1></main></body></html>",
            ),
            "/unadmitted/: canonical must be exactly",
        )
        reject(
            "site title outside banner",
            lambda artifact: _replace(
                artifact / home, SHELL, f"<header></header>\n{SITE_TITLE}"
            ),
            "site-title home link must appear exactly once in the page banner",
        )
        reject(
            "anchor aria-label substitutes the visible brand name",
            lambda artifact: _replace(
                artifact / home,
                'class="site-title" href="/"',
                'class="site-title" href="/" aria-label="Eqiora"',
            ),
            "header home link must derive its name only from visible 'Eqiora'",
        )
        reject(
            "site title wrong destination",
            lambda artifact: _replace(
                artifact / home,
                'class="site-title" href="/"',
                'class="site-title" href="/get-started/"',
            ),
            "header home link href must be exactly '/'",
        )
        reject(
            "brand asset is not same-origin",
            lambda artifact: _replace(
                artifact / home,
                f'src="{BRAND_PATH}"',
                'src="https://example.com/brand.svg"',
            ),
            "header brand asset must be same-origin",
        )
        reject(
            "wrong brand bytes",
            lambda artifact: _write(
                artifact / BRAND_PATH.removeprefix("/"), b"not the Eqiora mark"
            ),
            "header brand asset has the wrong digest",
        )
        reject(
            "home wake alt changed",
            lambda artifact: _replace(
                artifact / home, WAKE_ALT, "Decorative wake image"
            ),
            "featured walkthrough must expose the admitted wake image",
        )
        reject(
            "gallery pressure alt changed",
            lambda artifact: _replace(
                artifact / case, PRESSURE_ALT, "Decorative pressure image"
            ),
            "gallery walkthrough must expose the admitted pressure image",
        )
        reject(
            "pressure bytes changed",
            lambda artifact: _write(
                artifact / PRESSURE_PATH.removeprefix("/"), b"wrong pressure bytes"
            ),
            "admitted pressure image has the wrong digest",
        )
        reject(
            "home wake duplicated",
            lambda artifact: _replace(
                artifact / home,
                f'<img src="{WAKE_PATH}" alt="{WAKE_ALT}">',
                f'<img src="{WAKE_PATH}" alt="{WAKE_ALT}">'
                f'<img src="{WAKE_PATH}" alt="{WAKE_ALT}">',
            ),
            "featured walkthrough must expose exactly one admitted wake image",
        )
        reject(
            "gallery pressure duplicated",
            lambda artifact: _replace(
                artifact / case,
                f'<img src="{PRESSURE_PATH}" alt="{PRESSURE_ALT}">',
                f'<img src="{PRESSURE_PATH}" alt="{PRESSURE_ALT}">'
                f'<img src="{PRESSURE_PATH}" alt="{PRESSURE_ALT}">',
            ),
            "gallery walkthrough must expose exactly one admitted pressure image",
        )
        reject(
            "404 canonical uses the Starlight route",
            lambda artifact: _replace(
                artifact / "404.html",
                'href="https://eqiora.org/404/"',
                'href="https://eqiora.org/404.html"',
            ),
            "canonical must be exactly 'https://eqiora.org/404/'",
        )
        reject(
            "404 exception does not admit unrelated missing links",
            lambda artifact: _append_main(
                artifact / home,
                '<a href="/unrelated-missing/">Unrelated missing page</a>',
            ),
            "broken or escaping link",
        )
        reject(
            "title-only stage heading",
            lambda artifact: _replace(
                artifact / case, "Stage 1 Problem setup", "Problem setup"
            ),
            "Stage 1 Problem setup",
        )
        reject(
            "misnumbered stage",
            lambda artifact: _replace(
                artifact / case, 'data-step="3"', 'data-step="4"'
            ),
            "six ordered semantic stages",
        )
        reject(
            "stage ID changed",
            lambda artifact: _replace(
                artifact / case, 'id="mesh-and-boundaries"', 'id="mesh-stage"'
            ),
            "six ordered semantic stages",
        )
        reject(
            "stage labelled by the wrong heading",
            lambda artifact: _replace(
                artifact / case,
                'aria-labelledby="submit-and-result-title"',
                'aria-labelledby="model-definition-title"',
            ),
            "six ordered semantic stages",
        )
        reject(
            "stage heading loses separator space",
            lambda artifact: _replace(
                artifact / case,
                "Stage 5 Pressure visualization",
                "Stage 5Pressure visualization",
            ),
            "Stage 5 Pressure visualization",
        )

        def reorder_stages(artifact: Path) -> None:
            page = artifact / case
            text = page.read_text(encoding="utf-8")
            first = text.index('<section class="eq-stage"')
            second = text.index('<section class="eq-stage"', first + 1)
            third = text.index('<section class="eq-stage"', second + 1)
            stage_one = text[first:second]
            stage_two = text[second:third]
            page.write_text(
                text[:first] + stage_two + stage_one + text[third:], encoding="utf-8"
            )

        reject("reordered stages", reorder_stages, "six ordered semantic stages")
        inline_math = _math("H")
        reject(
            "missing KaTeX HTML half",
            lambda artifact: _replace(
                artifact / case,
                inline_math,
                inline_math.replace('class="katex-html"', 'class="rendered-html"', 1),
            ),
            "KaTeX HTML and MathML",
        )
        reject(
            "missing TeX annotation",
            lambda artifact: _replace(
                artifact / case,
                inline_math,
                inline_math.replace(
                    ' encoding="application/x-tex"', ' encoding="text/plain"', 1
                ),
            ),
            "nonempty TeX annotation",
        )
        reject(
            "empty TeX annotation",
            lambda artifact: _replace(
                artifact / case,
                ">H</annotation>",
                "></annotation>",
            ),
            "nonempty TeX annotation",
        )
        reject(
            "duplicate TeX annotation",
            lambda artifact: _replace(
                artifact / case,
                inline_math,
                inline_math.replace(
                    "</annotation>",
                    '</annotation><annotation encoding="application/x-tex">duplicate</annotation>',
                    1,
                ),
            ),
            "exactly one TeX annotation",
        )
        reject(
            "raw math delimiter",
            lambda artifact: _append_main(artifact / case, "<p>$$ leaked target</p>"),
            "raw target math delimiters",
        )
        reject(
            "accepted source form absent",
            lambda artifact: _replace(
                artifact / case, "div(velocity) = 0;", "divergence omitted;"
            ),
            "accepted Eqiora source form",
        )

        accepted_link = _exact_link(
            "examples/python/exact_cylinder_stokes.py",
            "Eqiora source form: Python resolve/run path",
            "#L51-L67",
        )
        accepted_href = accepted_link.split('href="', 1)[1].split('"', 1)[0]
        accepted_label = "Eqiora source form: Python resolve/run path"
        link_mutants = (
            ("empty", f'<a href="">{accepted_label}</a>'),
            ("fragment-only", f'<a href="#model-definition">{accepted_label}</a>'),
            ("javascript", f'<a href="javascript:void(0)">{accepted_label}</a>'),
            (
                "form",
                f'<form action="/get-started/"><button>{accepted_label}</button></form>',
            ),
            (
                "role-button",
                f'<a href="{accepted_href}" role="button">{accepted_label}</a>',
            ),
            (
                "handler-backed",
                f'<a href="{accepted_href}" onclick="submitCase()">{accepted_label}</a>',
            ),
        )
        for label, replacement in link_mutants:
            reject(
                f"accepted navigation changed to {label} control",
                lambda artifact, replacement=replacement: _replace(
                    artifact / case, accepted_link, replacement
                ),
                "navigation link became an action control",
            )
        for label, replacement in (
            ("missing lines", accepted_href.removesuffix("#L51-L67")),
            ("wrong lines", accepted_href.replace("#L51-L67", "#L44-L57")),
            (
                "wrong exact head",
                accepted_href.replace(SOURCE_SHA, "b" * 40),
            ),
        ):
            reject(
                f"accepted sentinel {label}",
                lambda artifact, replacement=replacement: _replace(
                    artifact / case, accepted_href, replacement
                ),
                "accepted source-form sentinel must be the exact-head L51-L67 anchor",
            )

        reject(
            "reference landing guidance omitted",
            lambda artifact: _replace(
                artifact / reference,
                REFERENCE_GUIDANCE,
                "Choose the interface you need.",
            ),
            REFERENCE_GUIDANCE,
        )

        index = Path("sitemap-index.xml")
        child = Path("sitemap-0.xml")
        reject(
            "external sitemap child",
            lambda artifact: _write(
                artifact / index,
                _sitemap_index().replace(
                    f"{SITE_ORIGIN}/sitemap-0.xml", "https://example.com/sitemap.xml"
                ),
            ),
            "sitemap child must be exact same-origin",
        )
        reject(
            "traversing sitemap child",
            lambda artifact: _write(
                artifact / index, _sitemap_index(("/../sitemap-0.xml",))
            ),
            "sitemap child path escapes the artifact",
        )
        reject(
            "missing sitemap child",
            lambda artifact: (artifact / child).unlink(),
            "sitemap child is missing",
        )
        reject(
            "duplicate sitemap child",
            lambda artifact: _write(
                artifact / index, _sitemap_index(("/sitemap-0.xml", "/sitemap-0.xml"))
            ),
            "duplicate sitemap child",
        )
        reject(
            "nested sitemap index",
            lambda artifact: _write(
                artifact / child, _sitemap_index(("/sitemap-1.xml",))
            ),
            "sitemap child must be a URL set",
        )
        reject(
            "cyclic sitemap index",
            lambda artifact: _write(
                artifact / child, _sitemap_index(("/sitemap-index.xml",))
            ),
            "sitemap child must be a URL set",
        )
        reject(
            "sitemap DTD",
            lambda artifact: _write(
                artifact / index,
                '<!DOCTYPE sitemapindex [<!ENTITY x "x">]>' + _sitemap_index(),
            ),
            "sitemap XML forbids DOCTYPE and entities",
        )
        reject(
            "route omitted from child URL set",
            lambda artifact: _write(
                artifact / child,
                _url_set(
                    tuple(
                        route for route in ST_STARLIGHT_ROUTES if route != "/reference/"
                    )
                ),
            ),
            "sitemap omits required route /reference/",
        )
        reject(
            "duplicate route in child URL set",
            lambda artifact: _write(
                artifact / child,
                _url_set(ST_STARLIGHT_ROUTES + ("/reference/",)),
            ),
            "duplicate sitemap route /reference/",
        )
        reject(
            "404 artifact route added to sitemap",
            lambda artifact: _write(
                artifact / child,
                _url_set(ST_STARLIGHT_ROUTES + ("/404.html",)),
            ),
            "sitemap must not publish the 404 artifact route",
        )
        reject(
            "sitemap child query",
            lambda artifact: _write(
                artifact / index, _sitemap_index(("/sitemap-0.xml?copy=1",))
            ),
            "sitemap child must not contain query, fragment, or userinfo",
        )
        reject(
            "sitemap child fragment",
            lambda artifact: _write(
                artifact / index, _sitemap_index(("/sitemap-0.xml#copy",))
            ),
            "sitemap child must not contain query, fragment, or userinfo",
        )
        reject(
            "sitemap child cap",
            lambda artifact: _write(
                artifact / index,
                _sitemap_index(tuple(f"/sitemap-{item}.xml" for item in range(17))),
            ),
            "sitemap exceeds 16 children",
        )

        stylesheet = Path("assets/site.css")
        data_mutants = (
            (
                "malformed base64",
                SAFE_FONT_DATA,
                "data:font/woff2;base64,%%%",
                "malformed CSS data URL",
            ),
            (
                "malformed percent encoding",
                SAFE_SVG_DATA,
                "data:image/svg+xml,%ZZ",
                "malformed CSS data URL",
            ),
            (
                "wrong media type",
                SAFE_FONT_DATA,
                "data:text/plain;base64,d09GMg==",
                "CSS data URL media type is not admitted",
            ),
            (
                "wrong WOFF2 signature",
                SAFE_FONT_DATA,
                "data:font/woff2;base64,bm90LWEtZm9udA==",
                "WOFF2 data URL lacks the wOF2 signature",
            ),
        )
        for label, old, new, expected in data_mutants:
            reject(
                label,
                lambda artifact, old=old, new=new: _replace(
                    artifact / stylesheet, old, new
                ),
                expected,
            )
        reject(
            "HTML data URL",
            lambda artifact: _append_main(
                artifact / home,
                '<img src="data:image/svg+xml;base64,PHN2Zy8+" alt="inline">',
            ),
            "data URL is forbidden in HTML",
        )

        active_svgs = (
            ("script", b"<svg><script/></svg>"),
            ("handler", b'<svg onload="run()"/>'),
            ("foreign object", b"<svg><foreignObject/></svg>"),
            ("external reference", b'<svg><image href="https://example.com/x"/></svg>'),
            ("animation", b'<svg><animate attributeName="x"/></svg>'),
            ("style", b"<svg><style>*{display:block}</style></svg>"),
            ("DTD and entity", b'<!DOCTYPE svg [<!ENTITY x "x">]><svg>&x;</svg>'),
        )
        for label, payload in active_svgs:
            replacement = "data:image/svg+xml;base64," + base64.b64encode(
                payload
            ).decode("ascii")
            reject(
                f"active SVG {label}",
                lambda artifact, replacement=replacement: _replace(
                    artifact / stylesheet, SAFE_SVG_DATA, replacement
                ),
                "SVG data URL contains active content",
            )
        raw_oversize = (
            "data:image/svg+xml,"
            + "%20" * ((MAX_DATA_URL_BYTES // 3) + 1)
            + "%3Csvg%20xmlns%3D%22http%3A%2F%2Fwww.w3.org%2F2000%2Fsvg%22%2F%3E"
        )
        reject(
            "raw data URL cap",
            lambda artifact: _replace(
                artifact / stylesheet, SAFE_SVG_DATA, raw_oversize
            ),
            "raw data URL exceeds 1048576 bytes",
        )
        decoded_oversize = "data:font/woff2;base64," + base64.b64encode(
            b"wOF2" + b"x" * MAX_DATA_URL_BYTES
        ).decode("ascii")
        reject(
            "decoded data URL cap",
            lambda artifact: _replace(
                artifact / stylesheet, SAFE_FONT_DATA, decoded_oversize
            ),
            "decoded data URL exceeds 1048576 bytes",
        )
        reject(
            "unterminated CSS URL",
            lambda artifact: _replace(
                artifact / stylesheet,
                f'url("{SAFE_SVG_DATA}")',
                f'url("{SAFE_SVG_DATA}',
            ),
            "unterminated CSS url()",
        )
        reject(
            "escaped closing quote rejects an escape-blind CSS URL scanner",
            lambda artifact: _replace(
                artifact / stylesheet,
                f'url("{SAFE_SVG_DATA}")',
                f'url("{SAFE_SVG_DATA}\\")',
            ),
            "unterminated CSS url()",
        )

        def css_url_cap(artifact: Path) -> None:
            value = "".join(
                "a{background:url('./KaTeX_Main-Regular.woff2')}\n" for _ in range(4097)
            )
            _write(artifact / stylesheet, value)

        reject("CSS URL cap", css_url_cap, "CSS file exceeds 4096 URL references")

        diagnostic = RUSTDOC_ROOT / "eqiora/struct.Diagnostic.html"
        rust_index = RUSTDOC_ROOT / "eqiora/index.html"
        rust_source = RUSTDOC_ROOT / "src/eqiora/lib.rs.html"
        reject(
            "unadmitted site-absolute Rustdoc link adjacent to owned landing",
            lambda artifact: _append_main(
                artifact / rust_index,
                '<a href="/reference/rust/extra/">Unowned Rust reference</a>',
            ),
            "Rustdoc reference escapes exact root '/reference/rust/extra/'",
        )
        reject(
            "percent-decoded fragment",
            lambda artifact: _replace(
                artifact / rust_index,
                f"#{FRAGMENT_ID}",
                "#impl-LifetimeProbe<'a>-for-Thing",
            ),
            "missing raw Rustdoc fragment target",
        )
        reject(
            "changed Rustdoc fragment",
            lambda artifact: _replace(
                artifact / rust_index, f"#{FRAGMENT_ID}", f"#{FRAGMENT_ID}-changed"
            ),
            "missing raw Rustdoc fragment target",
        )
        reject(
            "missing Rustdoc fragment ID",
            lambda artifact: _replace(
                artifact / diagnostic, f'id="{FRAGMENT_ID}"', 'id="different"'
            ),
            "missing raw Rustdoc fragment target",
        )
        reject(
            "missing single-quoted Rustdoc fragment ID containing a double quote",
            lambda artifact: _replace(
                artifact / diagnostic,
                f"id='{QUOTED_FRAGMENT_ID}'",
                "id='different-quoted'",
            ),
            "missing raw Rustdoc fragment target",
        )
        reject(
            "missing unquoted numeric Rustdoc fragment ID",
            lambda artifact: _replace(
                artifact / diagnostic,
                f"id={NUMERIC_FRAGMENT_ID}",
                "id=124",
            ),
            "missing raw Rustdoc fragment target",
        )
        reject(
            "duplicate Rustdoc fragment ID",
            lambda artifact: _append_main(
                artifact / diagnostic, f'<span id="{FRAGMENT_ID}">duplicate</span>'
            ),
            "duplicate Rustdoc target ID",
        )
        reject(
            "duplicate unquoted numeric Rustdoc fragment ID",
            lambda artifact: _append_main(
                artifact / diagnostic,
                f"<span id={NUMERIC_FRAGMENT_ID}>duplicate</span>",
            ),
            "duplicate Rustdoc target ID",
        )
        reject(
            "wrong-document Rustdoc fragment",
            lambda artifact: _replace(
                artifact / rust_index,
                f"struct.Diagnostic.html#{FRAGMENT_ID}",
                f"module/index.html#{FRAGMENT_ID}",
            ),
            "missing raw Rustdoc fragment target",
        )
        reject(
            "missing Rustdoc line-range start endpoint",
            lambda artifact: _replace(
                artifact / rust_source, "<span id=1>1</span>", ""
            ),
            "missing raw Rustdoc fragment target",
        )
        reject(
            "duplicate Rustdoc line-range end endpoint",
            lambda artifact: _append_main(
                artifact / rust_source, "<span id=325>duplicate</span>"
            ),
            "duplicate Rustdoc target ID",
        )
        reject(
            "reversed Rustdoc line range",
            lambda artifact: _replace(
                artifact / rust_index,
                f"#{LINE_RANGE_FRAGMENT}",
                "#325-1",
            ),
            "missing raw Rustdoc fragment target",
        )

        def textual_line_range(artifact: Path) -> None:
            _replace(
                artifact / rust_index,
                f"#{LINE_RANGE_FRAGMENT}",
                "#begin-end",
            )
            _append_main(
                artifact / rust_source,
                "<span id=begin>begin</span><span id=end>end</span>",
            )

        reject(
            "arbitrary textual Rustdoc range",
            textual_line_range,
            "missing raw Rustdoc fragment target",
        )

        help_page = RUSTDOC_ROOT / "help.html"
        back_mutants = (
            ("tag", '<button id="back" onclick="history.back();">Back</button>'),
            ("ID", BACK_CONTROL.replace('id="back"', 'id="return"')),
            ("href", BACK_CONTROL.replace("javascript:void(0)", "javascript:back()")),
            ("handler", BACK_CONTROL.replace("history.back();", "history.go(-1);")),
            ("label", BACK_CONTROL.replace(">Back<", ">Return<")),
            (
                "additional handler",
                BACK_CONTROL.replace('onclick="', 'onfocus="focus()" onclick="'),
            ),
            (
                "javascript half only",
                BACK_CONTROL.replace(' onclick="history.back();"', ""),
            ),
            ("handler half only", BACK_CONTROL.replace("javascript:void(0)", "#top")),
        )
        for label, replacement in back_mutants:
            reject(
                f"Back control {label}",
                lambda artifact, replacement=replacement: _replace(
                    artifact / help_page, BACK_CONTROL, replacement
                ),
                "invalid generated Rustdoc Back control",
            )
        reject(
            "Back control duplicate",
            lambda artifact: _replace(
                artifact / help_page, BACK_CONTROL, BACK_CONTROL + BACK_CONTROL
            ),
            "invalid generated Rustdoc Back control",
        )

        def back_wrong_file(artifact: Path) -> None:
            _replace(artifact / help_page, BACK_CONTROL, "")
            _append_main(artifact / rust_index, BACK_CONTROL)

        reject(
            "Back control wrong file",
            back_wrong_file,
            "invalid generated Rustdoc Back control",
        )

        first_source, first_reference = ABSENT_REFERENCES[0]
        absent_source = RUSTDOC_ROOT / first_source
        reject(
            "tracing span link without blanket implementation",
            lambda artifact: _append_main(
                artifact / RUSTDOC_ROOT / "eqiora/index.html",
                '<a href="crate::Span">Span</a>',
            ),
            "unsafe Rustdoc reference",
        )
        reject(
            "tracing dispatcher link without blanket implementation",
            lambda artifact: _append_main(
                artifact / RUSTDOC_ROOT / "eqiora/index.html",
                '<a href="dispatcher#setting-the-default-subscriber">default</a>',
            ),
            "unadmitted missing Rustdoc reference",
        )
        reject(
            "extra absent Rustdoc target",
            lambda artifact: _append_main(
                artifact / rust_index, '<a href="missing-generated.js">Missing</a>'
            ),
            "unadmitted missing Rustdoc reference",
        )

        def move_absent_reference(artifact: Path) -> None:
            _replace(
                artifact / absent_source,
                f'<a href="{first_reference}">Implementors</a>',
                "",
            )
            _append_main(
                artifact / rust_index,
                f'<a href="{first_reference}">Implementors</a>',
            )

        reject(
            "moved absent Rustdoc reference",
            move_absent_reference,
            "Rustdoc reference escapes exact root",
        )
        reject(
            "glob-like absent Rustdoc imitation",
            lambda artifact: _append_main(
                artifact / absent_source,
                '<a href="../../trait.impl/eqiora_api/reference_run/other.js">Other</a>',
            ),
            "unadmitted missing Rustdoc reference",
        )
        def make_absent_target_nonregular(artifact: Path) -> None:
            target = (artifact / absent_source).parent / first_reference
            target.resolve().mkdir(parents=True)

        reject(
            "expected absent Rustdoc target becomes nonregular",
            make_absent_target_nonregular,
            "Rustdoc target has wrong type",
        )

        reject(
            "extra public crate root",
            lambda artifact: _write(
                artifact / RUSTDOC_ROOT / "eqiora_mcp/index.html",
                _rustdoc_page("<h1>Crate eqiora_mcp</h1>"),
            ),
            "unexpected Rustdoc crate root 'eqiora_mcp'",
        )
        reject(
            "missing public crate root",
            lambda artifact: (artifact / rust_index).unlink(),
            "missing exact Rustdoc crate root eqiora/index.html",
        )
        reject(
            "private crate root",
            lambda artifact: _write(
                artifact / RUSTDOC_ROOT / "eqiora_core/index.html",
                _rustdoc_page("<h1>Crate eqiora_core</h1>"),
            ),
            "unexpected Rustdoc crate root 'eqiora_core'",
        )
        reject(
            "doubled Rustdoc public prefix",
            lambda artifact: _write(
                artifact / RUSTDOC_ROOT / "reference/rust/api/eqiora/index.html",
                _rustdoc_page("<h1>Crate eqiora</h1>"),
            ),
            "doubled Rustdoc public prefix",
        )
        reject(
            "unprefixed Rustdoc crate",
            lambda artifact: _write(
                artifact / "eqiora/index.html",
                _rustdoc_page("<h1>Crate eqiora</h1>"),
            ),
            "/eqiora/: canonical must be exactly",
        )
        reject(
            "Rustdoc-like owner collision outside exact root",
            lambda artifact: _write(
                artifact / "reference/rust/api-copy/eqiora/index.html",
                _rustdoc_page("<h1>Crate eqiora</h1>"),
            ),
            "/reference/rust/api-copy/eqiora/: canonical must be exactly",
        )

        def add_symlink(artifact: Path) -> None:
            (artifact / RUSTDOC_ROOT / "linked.css").symlink_to(
                artifact / RUSTDOC_ROOT / "static.files/rustdoc.css"
            )

        reject("artifact symlink", add_symlink, "artifact contains symlink")

        def overflow_file_cap(artifact: Path) -> None:
            current = sum(1 for path in artifact.rglob("*") if path.is_file())
            overflow = artifact / "overflow"
            overflow.mkdir()
            for item in range(MAX_FILES - current + 1):
                (overflow / f"{item:05d}").touch()

        reject("raw file-count cap", overflow_file_cap, "artifact exceeds 20000 files")

        # The split helpers are exact siblings, not ambient extension points.
        counterfeit = types.ModuleType("tools.site.check_site_artifact")
        counterfeit.__file__ = "/counterfeit/check_site_artifact.py"
        counterfeit.__all__ = ()
        checker_path = REPOSITORY / "tools/site/check_site.py"
        specification = importlib.util.spec_from_file_location(
            "alpha2_counterfeit_site_check", checker_path
        )
        self.assertIsNotNone(specification)
        assert specification is not None and specification.loader is not None
        candidate = importlib.util.module_from_spec(specification)
        with mock.patch.dict(
            sys.modules,
            {"tools.site.check_site_artifact": counterfeit},
        ):
            with self.assertRaisesRegex(ImportError, "exact sibling"):
                specification.loader.exec_module(candidate)


if __name__ == "__main__":
    unittest.main()
