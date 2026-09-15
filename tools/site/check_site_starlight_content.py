import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

__all__ = (
    "WAKE_ALT",
    "PRESSURE_ALT",
    "PRESSURE_CAPTION",
    "CASE_SOURCE_PATHS",
    "check_starlight_content",
)

WAKE_ALT = "Cell-average vorticity in a Kármán vortex street behind a circular cylinder."
PRESSURE_ALT = "Steady Stokes pressure around a cylinder, with the current mesh and pressure scale in pascals."
PRESSURE_CAPTION = "Steady Stokes pressure on a 0.025 m target mesh."
REFERENCE_GUIDANCE = "Look up a declaration, find a physical building block, or inspect an API signature."
MODELING_FOUNDATION_CHAPTERS = (
    ("algebraic-relations-networks", "Algebraic relations and networks"),
    ("boundary-interface-conditions", "Boundary and interface conditions"),
    ("conservation-laws", "Conservation laws"),
    ("constitutive-laws", "Constitutive laws"),
    ("fields-spatial-domains", "Fields and spatial domains"),
    ("models-not-simulations", "Models are not simulations"),
    ("ordinary-differential-equations", "Ordinary differential equations"),
    ("quantities-dimensions-units", "Quantities, dimensions, and units"),
)
STAGES = (
    ("problem-setup", "1", "Problem setup"),
    ("model-definition", "2", "Eqiora model definition"),
    ("mesh-and-boundaries", "3", "Mesh and boundaries"),
    ("submit-and-result", "4", "Submit and result"),
    ("pressure-visualization", "5", "Pressure visualization"),
    ("reading-pressure", "6", "Reading the pressure plot"),
)
ADMITTED_SOURCE_PATH = "examples/python/exact_cylinder_stokes.py"
ADMITTED_SOURCE_FRAGMENT = "#L51-L67"
ADMITTED_SOURCE_LABEL = "Eqiora source form: Python resolve/run path"
CASE_SOURCE_PATHS = (
    ADMITTED_SOURCE_PATH,
    "examples/python/exact_cylinder_geometry.py",
    "examples/python/exact_cylinder_mesh.py",
    "examples/steady-flow-past-cylinder.eqi",
    "crates/eqiora-api/packages/Eqiora.Fluid.Incompressible/src/incompressible.eqi",
)

EXECUTION_CONTROL = re.compile(
    r"\b(?:run|submit|reset|start|begin|try|solv\w*|execut\w*|simulat\w*|comput\w*|calculat\w*|launch\w*|evaluat\w*|process\w*|generat\w*|analy[sz]\w*|predict\w*)\b",
    re.IGNORECASE,
)


class _ContentInspection(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stages: list[tuple[str, str, str]] = []
        self.math: list[dict[str, list[str]]] = []
        self.math_stack: list[dict[str, list[str]]] = []
        self.annotations: list[tuple[dict[str, list[str]], list[str]] | None] = []
        self.katex_mathml = 0
        self.katex_html = 0
        self.elements: list[tuple[str, bool, str]] = []
        self.active_ids: list[str] = []
        self.id_accessible_text: dict[str, list[str]] = {}
        self.id_attrs: dict[str, dict[str, str]] = {}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {name.casefold(): value or "" for name, value in attrs}
        classes = values.get("class", "").split()
        parent_hidden = self.elements[-1][1] if self.elements else False
        hidden = (
            parent_hidden
            or tag in {"script", "style", "template", "svg"}
            or "hidden" in values
            or values.get("aria-hidden") == "true"
        )
        if not hidden:
            for identifier in self.active_ids:
                self.id_accessible_text[identifier].append(" ")
        identifier = values.get("id", "")
        if identifier:
            self.id_accessible_text.setdefault(identifier, [])
            self.id_attrs[identifier] = values
            self.active_ids.append(identifier)
        self.elements.append((tag, hidden, identifier))
        if "katex-mathml" in classes:
            self.katex_mathml += 1
        if "katex-html" in classes:
            self.katex_html += 1
        if tag == "section" and "eq-stage" in classes:
            self.stages.append(
                (
                    values.get("id", ""),
                    values.get("data-step", ""),
                    values.get("aria-labelledby", ""),
                )
            )
        if tag == "math":
            record: dict[str, list[str]] = {"annotations": []}
            self.math.append(record)
            self.math_stack.append(record)
        elif tag == "annotation":
            if self.math_stack and values.get("encoding") == "application/x-tex":
                self.annotations.append((self.math_stack[-1], []))
            else:
                self.annotations.append(None)

    def handle_endtag(self, tag: str) -> None:
        if tag == "annotation" and self.annotations:
            current = self.annotations.pop()
            if current is not None:
                record, chunks = current
                record["annotations"].append(" ".join("".join(chunks).split()))
        elif tag == "math" and self.math_stack:
            self.math_stack.pop()
        for index in range(len(self.elements) - 1, -1, -1):
            if self.elements[index][0] != tag:
                continue
            removed = self.elements[index:]
            if not self.elements[index][1]:
                for identifier in self.active_ids:
                    self.id_accessible_text[identifier].append(" ")
            del self.elements[index:]
            for _, _, identifier in reversed(removed):
                if identifier:
                    self.active_ids.remove(identifier)
            break

    def handle_data(self, data: str) -> None:
        if self.annotations and self.annotations[-1] is not None:
            current = self.annotations[-1]
            assert current is not None
            current[1].append(data)
        if not (self.elements and self.elements[-1][1]):
            for identifier in self.active_ids:
                self.id_accessible_text[identifier].append(data)

    def accessible_name(self, identifier: str, seen: tuple[str, ...] = ()) -> str:
        if identifier in seen:
            return ""
        attrs = self.id_attrs.get(identifier, {})
        labelled = attrs.get("aria-labelledby", "").split()
        if labelled:
            value = " ".join(
                self.accessible_name(item, (*seen, identifier)) for item in labelled
            )
        elif "aria-label" in attrs:
            value = attrs["aria-label"]
        else:
            value = "".join(self.id_accessible_text.get(identifier, []))
        return " ".join(value.split())


def _same_origin_file(artifact: Path, page: Path, value: str) -> Path | None:
    parsed = urlsplit(value)
    if parsed.query or parsed.fragment or parsed.username or parsed.password:
        return None
    if parsed.scheme or value.startswith("//"):
        if (
            parsed.scheme not in {"http", "https"}
            or f"{parsed.scheme}://{parsed.netloc}" != "https://eqiora.org"
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


def _check_admitted_image(
    artifact: Path,
    page_path: Path,
    page: object,
    file_digests: dict[Path, str],
    admitted_digest: str,
    admitted_alt: str,
    subject: str,
    label: str,
) -> list[str]:
    resolved = [
        (image, _same_origin_file(artifact, page_path, image.get("src", "")))
        for image in page.images
    ]
    candidates = [
        (image, target)
        for image, target in resolved
        if image.get("alt") == admitted_alt
        or (target is not None and file_digests.get(target) == admitted_digest)
    ]
    if len(candidates) != 1:
        qualifier = "exactly one " if len(candidates) > 1 else "the "
        return [
            f"{label} must expose {qualifier}admitted {subject} image with exact alt text"
        ]
    image, target = candidates[0]
    if image.get("alt") != admitted_alt or target is None:
        return [f"{label} must expose the admitted {subject} image with exact alt text"]
    if file_digests.get(target) != admitted_digest:
        return [f"{label}: admitted {subject} image has the wrong digest"]
    return []


def _check_home(
    artifact: Path,
    home: object,
    expected_python_version: str,
    file_digests: dict[Path, str],
    wake_digest: str,
    favicon_digest: str,
    enhanced: bool,
) -> list[str]:
    errors: list[str] = []
    report = errors.append
    if f"Release {expected_python_version}" not in home.visible_text:
        report("/: current release must be visible")
    start = home.visible_text.find("Featured walkthrough")
    end = home.visible_text.find("Get started", start + 1)
    featured = home.visible_text[start:end].casefold()
    for widening in (
        "flagship",
        "validated flow",
        "production ready",
        "general solver",
        "all backends",
        "benchmark",
        "interactive",
        "run now",
    ):
        if widening in featured:
            report(f"/: featured walkthrough widens its claim with {widening!r}")
    errors.extend(
        _check_admitted_image(
            artifact,
            artifact / "index.html",
            home,
            file_digests,
            wake_digest,
            WAKE_ALT,
            "wake",
            "/: featured walkthrough",
        )
    )
    return errors


def _check_case(
    artifact: Path,
    raw: str,
    page: object,
    file_digests: dict[Path, str],
    pressure_digest: str,
    source_sha: str,
    enhanced: bool,
) -> list[str]:
    errors: list[str] = []
    report = errors.append
    markup = _ContentInspection()
    markup.feed(raw)
    markup.close()
    headings = [heading for _, heading in page.headings]
    if enhanced:
        expected_headings = [f"Stage {step} {title}" for _, step, title in STAGES]
        expected_sections = [
            (identifier, step, f"{identifier}-title") for identifier, step, _ in STAGES
        ]
        accessible = [
            markup.accessible_name(label) for _, _, label in expected_sections
        ]
        if markup.stages != expected_sections or accessible != expected_headings:
            report(
                f"Cylinder route must expose six ordered semantic stages: {expected_headings!r}"
            )
    else:
        titles = [title for _, _, title in STAGES]
        if [heading for heading in headings if heading in titles] != titles:
            report("Cylinder route must expose six ordered semantic stages")
    if not any(
        display == "block" and wrapper for display, wrapper in page.math
    ) or not any(display != "block" and not wrapper for display, wrapper in page.math):
        report(
            "Cylinder route must contain distinct block and inline MathML/KaTeX output"
        )
    if markup.katex_mathml == 0 or markup.katex_mathml != markup.katex_html:
        report("Cylinder route must retain both KaTeX HTML and MathML output")
    if enhanced:
        for math in markup.math:
            annotations = math["annotations"]
            if not annotations or (len(annotations) == 1 and not annotations[0]):
                report(
                    "Cylinder route must retain a nonempty TeX annotation for each rendered formula"
                )
                break
            if len(annotations) != 1:
                report(
                    "Cylinder route must retain exactly one TeX annotation for each rendered formula"
                )
                break
    if any(item in page.visible_text for item in ("$$", "\\[", "\\]", "\\(", "\\)")):
        report("Cylinder route exposes raw target math delimiters")
    source_tokens = (
        (
            "relation momentum on fluid",
            "2 * dynamic_viscosity * symmetric_part(grad(velocity))",
            "- isotropic_lift(pressure)",
            "relation incompressibility on fluid",
            "div(velocity) = 0;",
        )
        if enhanced
        else (
            "Eqiora source form",
            "sigma(u,p) = 2 mu sym(grad(u)) - p I",
            "-div(sigma(u,p)) - grad(phi) = 0",
            "div(u) = 0",
        )
    )
    if any(token not in page.visible_text for token in source_tokens):
        report("Cylinder route omits the accepted Eqiora source form")
    errors.extend(
        _check_admitted_image(
            artifact,
            artifact / "gallery/exact-cylinder-steady-stokes/index.html",
            page,
            file_digests,
            pressure_digest,
            PRESSURE_ALT,
            "pressure",
            "gallery walkthrough",
        )
    )
    admitted_label = ADMITTED_SOURCE_LABEL
    source_base = f"https://github.com/nkiyohara/eqiora/blob/{source_sha}/"
    admitted_href = source_base + ADMITTED_SOURCE_PATH + ADMITTED_SOURCE_FRAGMENT
    sentinels: list[tuple[str, dict[str, str], str, str]] = []
    for tag, attrs, label in page.interactives:
        names = page.id_text
        labelled_ids = attrs.get("aria-labelledby", "").split()
        labelled = " ".join(" ".join(names.get(item, [])) for item in labelled_ids)
        accessible = " ".join(
            (
                labelled or attrs.get("aria-label") or label or attrs.get("title", "")
            ).split()
        )
        handlers = any(name.startswith("on") for name in attrs)
        stage_navigation = tag == "a" and any(
            "eq-stage-marker" in attrs.get("class", "").split()
            and attrs.get("href") == f"#{identifier}"
            and accessible == f"Stage {step} {title}"
            for identifier, step, title in STAGES
        )
        if label == admitted_label or accessible == admitted_label:
            sentinels.append((tag, attrs, label, accessible))
            href = attrs.get("href", "")
            action = (
                tag != "a"
                or attrs.get("role") == "button"
                or handlers
                or not href
                or href.startswith(("#", "javascript:"))
            )
            if action:
                report("Cylinder route navigation link became an action control")
            elif href != admitted_href:
                report(
                    "Cylinder route accepted source-form sentinel must be the exact-head L51-L67 anchor"
                )
            continue
        href = attrs.get("href", "")
        section_navigation = (
            tag == "a" and href.startswith("#") and unquote(href[1:]) in page.id_text
        )
        navigation = stage_navigation or section_navigation or (
            tag == "a" and href.startswith(source_base)
        )
        action = not navigation or attrs.get("role") == "button" or handlers
        if action and EXECUTION_CONTROL.search(f"{label} {accessible}"):
            report(
                f"Cylinder route contains an uncontracted execution control {(label, accessible)!r}"
            )
    if len(sentinels) != 1:
        report("Cylinder route must expose one uniquely labelled source-form sentinel")
    hrefs = {href for href, _ in page.anchors}
    for relative in CASE_SOURCE_PATHS:
        expected = source_base + relative
        if expected not in hrefs:
            report(f"Cylinder route omits exact-head source link {relative}")
    if PRESSURE_CAPTION not in page.visible_text:
        report("Cylinder route omits the exact admitted caption")
    return errors


def check_starlight_content(
    artifact: Path,
    inspections: dict[Path, tuple[str, object]],
    file_digests: dict[Path, str],
    wake_digest: str,
    pressure_digest: str,
    favicon_digest: str,
    source_sha: str,
    expected_python_version: str,
) -> list[str]:
    errors: list[str] = []
    case_value = inspections.get(
        artifact / "gallery/exact-cylinder-steady-stokes/index.html"
    )
    enhanced = bool(case_value and 'class="eq-stage"' in case_value[0])
    home_value = inspections.get(artifact / "index.html")
    if home_value:
        errors += _check_home(
            artifact,
            home_value[1],
            expected_python_version,
            file_digests,
            wake_digest,
            favicon_digest,
            enhanced,
        )
    if case_value:
        errors += _check_case(
            artifact,
            case_value[0],
            case_value[1],
            file_digests,
            pressure_digest,
            source_sha,
            enhanced,
        )
    reference = inspections.get(artifact / "reference/index.html")
    if reference:
        guidance = (
            REFERENCE_GUIDANCE
            if enhanced
            else "API presence is not verification or maturity."
        )
        required = ("Python", "Rust", "CLI", guidance)
        for phrase in required:
            if phrase not in reference[1].visible_text:
                errors.append(f"reference landing omits {phrase!r}")
    home = home_value[1] if home_value else None
    if home:
        anchors = home.anchors
        destinations = (
            ("/learn/", "Learn"),
            ("/guides/", "Guides"),
            ("/gallery/", "Gallery"),
            ("/reference/", "Reference"),
            ("/get-started/", "Get started"),
            ("/capabilities/", "Capabilities"),
            ("/release-notes/", "Releases"),
            ("/contributing/", "Contributing"),
            ("https://github.com/nkiyohara/eqiora", "GitHub"),
        )
        positions = []
        for destination in destinations:
            try:
                positions.append(anchors.index(destination))
            except ValueError:
                errors.append(f"public navigation omits {destination!r}")
        if len(positions) == len(destinations) and positions != sorted(positions):
            errors.append("public navigation does not preserve primary then secondary order")
        if ("/evidence/", "Evidence") in anchors:
            errors.append("technical Evidence catalog remains in primary navigation")

    capabilities_value = inspections.get(artifact / "capabilities/index.html")
    learn_value = inspections.get(artifact / "learn/index.html")
    evidence_value = inspections.get(artifact / "evidence/index.html")
    if capabilities_value:
        capabilities = capabilities_value[1]
        required = ("Capabilities", "Thermal")
        for phrase in required:
            if phrase not in capabilities.visible_text:
                errors.append(f"capabilities landing omits {phrase!r}")
    if learn_value:
        for phrase in ("Start a learning path", "Browse by topic"):
            if phrase not in learn_value[1].visible_text:
                errors.append(f"Learn landing omits {phrase!r}")
        if not any(href == "/learn/mathematical-modeling/" for href, _ in learn_value[1].anchors):
            errors.append("Learn landing omits the mathematical-modeling path")
    path_value = inspections.get(artifact / "learn/mathematical-modeling/index.html")
    if path_value:
        page = path_value[1]
        for phrase in ("Build a model", "Extend it through space", "Put it to use"):
            if phrase not in page.visible_text:
                errors.append(f"learning path omits {phrase!r}")
        for chapter_slug, chapter_title in MODELING_FOUNDATION_CHAPTERS:
            destination = (f"/learn/mathematical-modeling/{chapter_slug}/", chapter_title)
            if destination not in page.anchors:
                errors.append(f"learning path omits published lesson {chapter_title!r}")
    for slug, heading in (
        ("modeling", "Native declarations"),
        ("execution-and-arrays", "Structured failures"),
        ("differentiation", "Evaluate a point and its derivatives"),
    ):
        guide_value = inspections.get(artifact / f"guides/{slug}/index.html")
        if guide_value:
            for phrase in ("View guide source", heading):
                if phrase not in guide_value[1].visible_text:
                    errors.append(f"guide {slug!r} omits maintained content {phrase!r}")
    for slug, title in MODELING_FOUNDATION_CHAPTERS:
        value = inspections.get(
            artifact / f"learn/mathematical-modeling/{slug}/index.html"
        )
        if not value:
            continue
        page = value[1]
        for phrase in (
            title,
            "Learning outcomes",
            "Deliberate failure",
            "Exercises",
        ):
            if phrase not in page.visible_text:
                errors.append(f"Learn lesson {title!r} omits {phrase!r}")
        if (
            "/learn/mathematical-modeling/",
            "Back to the series map",
        ) not in page.anchors:
            errors.append(f"Learn lesson {title!r} omits its learning-path return route")
    if evidence_value:
        evidence = evidence_value[1]
        for phrase in ("Checking a claim", "Find the source", "Run a selected check"):
            if phrase not in evidence.visible_text:
                errors.append(f"verification guide omits {phrase!r}")
    if capabilities_value and learn_value and case_value and evidence_value:
        route_chain = (
            (
                learn_value[1],
                "/gallery/exact-cylinder-steady-stokes/",
                "Learn landing",
            ),
            (
                case_value[1],
                "/capabilities/#exact-cylinder-steady-stokes",
                "Gallery walkthrough",
            ),
            (
                capabilities_value[1],
                "/evidence/",
                "capability summary",
            ),
        )
        for page, href, label in route_chain:
            if href not in {anchor for anchor, _ in page.anchors}:
                errors.append(f"{label} omits the static learning-to-evidence route {href}")
    return errors
