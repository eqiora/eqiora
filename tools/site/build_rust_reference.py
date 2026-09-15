#!/usr/bin/env python3
"""Validate and stage the all-features Eqiora facade rustdoc tree."""

from __future__ import annotations

import argparse
import difflib
import html
import json
import re
import shutil
import sys
from dataclasses import dataclass
from html.parser import HTMLParser
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parents[2]
FACADE = ROOT / "api/eqiora-facade-v1.json"
LANDING = ROOT / "docs/site/src/content/docs/reference/rust/index.mdx"
FACADE_SCHEMA = "eqiora.facade-inventory/v1"
PUBLIC_RUSTDOC_PREFIX = "/reference/rust/api/eqiora/"
ALLOWED_SITE_LINKS = {"/favicon.svg", "/reference/rust/"}
VOID_TAGS = frozenset(
    {
        "area",
        "base",
        "br",
        "col",
        "embed",
        "hr",
        "img",
        "input",
        "link",
        "meta",
        "param",
        "source",
        "track",
        "wbr",
    }
)
RAW_LINK_ATTRIBUTES = frozenset({"class", "data-notable-ty", "href", "title"})
ORDERED_IMPLEMENTATION_LIST_IDS = frozenset(
    {
        "trait-implementations-list",
        "synthetic-implementations-list",
        "blanket-implementations-list",
    }
)
SPECIAL_HIDEME_LABELS = frozenset(
    {
        "This enum is marked as non-exhaustive",
        "This struct is marked as non-exhaustive",
    }
)


class RustReferenceError(ValueError):
    """The Rust reference cannot be generated or staged safely."""


@dataclass(frozen=True, order=True)
class FacadePath:
    path: str
    classification: str
    provider: str | None = None


class _References(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.urls: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        del tag
        for name, value in attrs:
            if name in {"href", "src"} and value is not None:
                self.urls.append(value)


@dataclass
class _Element:
    tag: str
    attrs: list[tuple[str, str | None]]
    opening: str
    closing: str = ""
    children: list[_Element | str] | None = None

    def __post_init__(self) -> None:
        if self.children is None:
            self.children = []

    def attr(self, name: str) -> str | None:
        matches = [value for key, value in self.attrs if key == name]
        if len(matches) > 1:
            raise RustReferenceError(f"duplicate {name!r} attribute on <{self.tag}>")
        return matches[0] if matches else None

    def classes(self) -> set[str]:
        return set((self.attr("class") or "").split())

    def elements(self) -> list[_Element]:
        assert self.children is not None
        return [child for child in self.children if isinstance(child, _Element)]

    def descendants(self) -> list[_Element]:
        result: list[_Element] = []
        for child in self.elements():
            result.append(child)
            result.extend(child.descendants())
        return result


class _DocumentParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=False)
        self.root = _Element("#document", [], "")
        self.stack = [self.root]

    def _append(self, item: _Element | str) -> None:
        assert self.stack[-1].children is not None
        self.stack[-1].children.append(item)

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        element = _Element(tag, attrs, self.get_starttag_text())
        self._append(element)
        if tag not in VOID_TAGS:
            self.stack.append(element)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self._append(_Element(tag, attrs, self.get_starttag_text()))

    def handle_endtag(self, tag: str) -> None:
        if len(self.stack) == 1 or self.stack[-1].tag != tag:
            current = self.stack[-1].tag
            raise RustReferenceError(
                f"malformed HTML: closing </{tag}> while <{current}> is open"
            )
        self.stack[-1].closing = f"</{tag}>"
        self.stack.pop()

    def handle_data(self, data: str) -> None:
        self._append(data)

    def handle_entityref(self, name: str) -> None:
        self._append(f"&{name};")

    def handle_charref(self, name: str) -> None:
        self._append(f"&#{name};")

    def handle_comment(self, data: str) -> None:
        self._append(f"<!--{data}-->")

    def handle_decl(self, decl: str) -> None:
        self._append(f"<!{decl}>")

    def unknown_decl(self, data: str) -> None:
        self._append(f"<![{data}]>")

    def handle_pi(self, data: str) -> None:
        self._append(f"<?{data}>")

    def close(self) -> None:
        super().close()
        if len(self.stack) != 1:
            raise RustReferenceError(f"malformed HTML: unclosed <{self.stack[-1].tag}>")


@dataclass
class _ProjectionStats:
    files: int = 0
    html_files: int = 0
    projected_pages: int = 0
    toggle_summaries: int = 0
    direct_sections: int = 0
    signature_links: int = 0
    hideme_labels: int = 0
    description_labels: int = 0
    special_hideme_labels: int = 0

    def add(self, other: _ProjectionStats) -> None:
        for field in self.__dataclass_fields__:
            setattr(self, field, getattr(self, field) + getattr(other, field))


def _parse_document(source: str, context: str) -> _Element:
    parser = _DocumentParser()
    try:
        parser.feed(source)
        parser.close()
    except (RustReferenceError, ValueError) as error:
        raise RustReferenceError(f"cannot parse {context}: {error}") from error
    return parser.root


def _render(item: _Element | str) -> str:
    if isinstance(item, str):
        return item
    assert item.children is not None
    return (
        item.opening + "".join(_render(child) for child in item.children) + item.closing
    )


def _direct(node: _Element, tag: str) -> list[_Element]:
    return [child for child in node.elements() if child.tag == tag]


def _node_text(node: _Element) -> str:
    assert node.children is not None
    return "".join(
        child if isinstance(child, str) else _node_text(child)
        for child in node.children
    )


def _active_hrefs(root: _Element) -> list[str]:
    return [
        node.attr("href") or ""
        for node in root.descendants()
        if node.tag == "a" and node.attr("href") is not None
    ]


def _script_markup(root: _Element) -> list[str]:
    return [_render(node) for node in root.descendants() if node.tag == "script"]


def _implementation_key(implementation: _Element, context: str) -> str:
    if implementation.tag == "section" and "impl" in implementation.classes():
        identifier = implementation.attr("id")
        if not identifier:
            raise RustReferenceError(
                f"{context}: implementation section lacks one stable id"
            )
        return identifier
    summaries = _direct(implementation, "summary")
    if len(summaries) != 1:
        raise RustReferenceError(
            f"{context}: implementation toggle must have exactly one direct summary"
        )
    sections = _direct(summaries[0], "section")
    if len(sections) != 1 or not sections[0].attr("id"):
        raise RustReferenceError(
            f"{context}: implementation toggle lacks one stable direct section id"
        )
    return sections[0].attr("id") or ""


def _canonicalize_implementation_order(root: _Element, context: str) -> None:
    containers = [
        node
        for node in root.descendants()
        if node.attr("id") in ORDERED_IMPLEMENTATION_LIST_IDS
    ]
    identities = [container.attr("id") for container in containers]
    if len(identities) != len(set(identities)):
        raise RustReferenceError(f"{context}: duplicate implementation-list identity")
    for container in containers:
        assert container.children is not None
        if any(
            isinstance(child, str) and child.strip() for child in container.children
        ):
            raise RustReferenceError(
                f"{context}: implementation list contains non-whitespace text"
            )
        elements = [
            child for child in container.children if isinstance(child, _Element)
        ]
        if any(
            not (
                child.tag == "details"
                and "toggle" in child.classes()
                or child.tag == "section"
                and "impl" in child.classes()
            )
            for child in elements
        ):
            raise RustReferenceError(
                f"{context}: implementation list contains an unknown element"
            )
        all_keys = [_implementation_key(element, context) for element in elements]
        if len(all_keys) != len(set(all_keys)):
            raise RustReferenceError(
                f"{context}: implementation list has duplicate stable section ids"
            )
        # Rustdoc deliberately renders expandable and non-expandable impls in
        # separate runs. Preserve those no-JavaScript presentation runs while
        # removing compiler traversal order from each run.
        for tag in ("details", "section"):
            positions = [
                offset
                for offset, child in enumerate(container.children)
                if isinstance(child, _Element) and child.tag == tag
            ]
            implementations = [container.children[offset] for offset in positions]
            ordered = sorted(
                implementations,
                key=lambda child: (
                    _implementation_key(child, context)
                    if isinstance(child, _Element)
                    else ""
                ),
            )
            for offset, implementation in zip(positions, ordered, strict=True):
                container.children[offset] = implementation


def _canonical_start(tag: str, attrs: list[tuple[str, str | None]]) -> str:
    rendered = [f"<{tag}"]
    for name, value in attrs:
        if value is None:
            rendered.append(f" {name}")
        else:
            rendered.append(f' {name}="{html.escape(value, quote=True)}"')
    rendered.append(">")
    return "".join(rendered)


def _project_anchor(anchor: _Element, context: str) -> str:
    names = [name for name, _ in anchor.attrs]
    if len(names) != len(set(names)) or set(names) - RAW_LINK_ATTRIBUTES:
        raise RustReferenceError(f"{context}: unexpected nested anchor attributes")
    href = anchor.attr("href")
    if href is None or anchor.attr("id") is not None:
        raise RustReferenceError(f"{context}: invalid nested active anchor")
    if any(node.tag == "a" for node in anchor.descendants()):
        raise RustReferenceError(f"{context}: nested anchor shape")
    clone = _render(anchor)
    anchor.tag = "span"
    anchor.attrs = [
        ("data-eqiora-href" if name == "href" else name, value)
        for name, value in anchor.attrs
    ]
    anchor.opening = _canonical_start("span", anchor.attrs)
    anchor.closing = "</span>"
    return clone


def _project_document(source: str, context: str) -> tuple[str, _ProjectionStats]:
    root = _parse_document(source, context)
    _canonicalize_implementation_order(root, context)
    original_hrefs = _active_hrefs(root)
    original_scripts = _script_markup(root)
    nodes = root.descendants()
    if any(any(name == "data-eqiora-href" for name, _ in node.attrs) for node in nodes):
        raise RustReferenceError(f"{context}: existing projected-link marker")
    if any(
        {"eqiora-signature-links", "eqiora-signature-links__label"} & node.classes()
        for node in nodes
    ):
        raise RustReferenceError(f"{context}: existing projection group marker")

    stats = _ProjectionStats()
    for details in [
        node for node in nodes if node.tag == "details" and "toggle" in node.classes()
    ]:
        summaries = _direct(details, "summary")
        if len(summaries) != 1:
            raise RustReferenceError(
                f"{context}: toggle details must have exactly one direct summary"
            )
        summary = summaries[0]
        stats.toggle_summaries += 1
        sections = _direct(summary, "section")
        if len(sections) > 1:
            raise RustReferenceError(f"{context}: multiple direct summary sections")
        raw_links = [
            node
            for node in summary.descendants()
            if node.tag == "a" and node.attr("href") is not None
        ]
        if sections:
            if not sections[0].attr("id") or not raw_links:
                raise RustReferenceError(
                    f"{context}: link-bearing direct section shape changed"
                )
            stats.direct_sections += 1
            clones = [_project_anchor(link, context) for link in raw_links]
            stats.signature_links += len(clones)
            group = (
                '<div class="eqiora-signature-links">'
                '<span class="eqiora-signature-links__label">Signature links:</span>'
                + "".join(clones)
                + "</div>"
            )
            assert details.children is not None
            details.children.insert(details.children.index(summary) + 1, group)
        elif "hideme" in summary.classes():
            if raw_links:
                raise RustReferenceError(
                    f"{context}: hideme summary has an active link"
                )
            labels = _direct(summary, "span")
            if len(labels) != 1:
                raise RustReferenceError(
                    f"{context}: hideme summary must have exactly one direct label"
                )
            label = _node_text(labels[0]).strip()
            stats.hideme_labels += 1
            if label == "Expand description":
                if labels[0].children != ["Expand description"]:
                    raise RustReferenceError(
                        f"{context}: generic description label shape changed"
                    )
                labels[0].children = ["Description"]
                stats.description_labels += 1
            elif label in SPECIAL_HIDEME_LABELS or re.fullmatch(
                r"Show [1-9][0-9]* (?:fields|variants)", label
            ):
                stats.special_hideme_labels += 1
            else:
                raise RustReferenceError(
                    f"{context}: unexpected hideme label {label!r}"
                )
        else:
            raise RustReferenceError(f"{context}: unexpected toggle-summary shape")

    projected = _render(root)
    output = _parse_document(projected, f"projected {context}")
    if _active_hrefs(output) != original_hrefs:
        raise RustReferenceError(f"{context}: active href order changed")
    if _script_markup(output) != original_scripts:
        raise RustReferenceError(f"{context}: upstream script markup changed")
    if any(
        node.tag == "a" and node.attr("href") is not None
        for summary in [node for node in output.descendants() if node.tag == "summary"]
        for node in summary.descendants()
    ):
        raise RustReferenceError(f"{context}: nested active link survived projection")
    projected_sources = [
        node
        for node in output.descendants()
        if node.tag == "span" and node.attr("data-eqiora-href") is not None
    ]
    groups = [
        node
        for node in output.descendants()
        if "eqiora-signature-links" in node.classes()
    ]
    if (
        len(projected_sources) != stats.signature_links
        or len(groups) != stats.direct_sections
    ):
        raise RustReferenceError(f"{context}: projected marker count drift")
    return projected, stats


def _project_tree(root: Path, *, write: bool) -> _ProjectionStats:
    regular = sorted(path for path in root.rglob("*") if path.is_file())
    documents = [path for path in regular if path.suffix == ".html"]
    stats = _ProjectionStats(files=len(regular), html_files=len(documents))
    for document in documents:
        relative = document.relative_to(root).as_posix()
        projected, page = _project_document(
            document.read_text(encoding="utf-8"), relative
        )
        if page.signature_links:
            page.projected_pages = 1
        stats.add(page)
        if write:
            document.write_text(projected, encoding="utf-8", newline="")
    return stats


def _object(value: Any, context: str, keys: set[str]) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != keys:
        raise RustReferenceError(f"{context} does not match its frozen shape")
    return value


def _text(value: Any, context: str) -> str:
    if not isinstance(value, str) or not value or any(ord(c) < 0x20 for c in value):
        raise RustReferenceError(f"{context} must be non-empty printable text")
    return value


def _exports(value: Any, context: str) -> list[tuple[str, str]]:
    if not isinstance(value, list):
        raise RustReferenceError(f"{context} must be an array")
    result: list[tuple[str, str]] = []
    for offset, raw in enumerate(value):
        entry = _object(raw, f"{context}[{offset}]", {"name", "from"})
        name = _text(entry["name"], f"{context}[{offset}].name")
        provider = _text(entry["from"], f"{context}[{offset}].from")
        if "::" in name or any(character.isspace() for character in name):
            raise RustReferenceError(f"{context}[{offset}].name is not one item name")
        result.append((name, provider))
    return result


def _load_facade() -> tuple[list[FacadePath], list[FacadePath]]:
    try:
        payload = FACADE.read_bytes()
        document = json.loads(payload)
    except (OSError, json.JSONDecodeError) as error:
        raise RustReferenceError(f"cannot read facade inventory: {error}") from error
    root = _object(
        document,
        "facade",
        {"schema", "crate", "source", "root", "stable_namespaces"},
    )
    if root["schema"] != FACADE_SCHEMA or root["crate"] != "eqiora":
        raise RustReferenceError("facade schema or crate identity changed")
    if root["source"] != "crates/eqiora/src/lib.rs":
        raise RustReferenceError("facade source owner changed")

    root_inventory = _object(
        root["root"],
        "facade.root",
        {"stable_exports", "stable_modules", "transitional_modules"},
    )
    modules: list[FacadePath] = []
    for classification, key in (
        ("stable", "stable_modules"),
        ("transitional", "transitional_modules"),
    ):
        raw_modules = root_inventory[key]
        if not isinstance(raw_modules, list):
            raise RustReferenceError(f"facade.root.{key} must be an array")
        for offset, value in enumerate(raw_modules):
            name = _text(value, f"facade.root.{key}[{offset}]")
            modules.append(FacadePath(f"eqiora::{name}", classification))

    items = [
        FacadePath(f"eqiora::{name}", "stable", provider)
        for name, provider in _exports(
            root_inventory["stable_exports"], "facade.root.stable_exports"
        )
    ]
    namespaces = root["stable_namespaces"]
    if not isinstance(namespaces, list):
        raise RustReferenceError("facade.stable_namespaces must be an array")
    for offset, raw_namespace in enumerate(namespaces):
        context = f"facade.stable_namespaces[{offset}]"
        namespace = _object(
            raw_namespace,
            context,
            {"path", "source_module", "stable_exports", "transitional_exports"},
        )
        path = _text(namespace["path"], f"{context}.path")
        source_module = _text(namespace["source_module"], f"{context}.source_module")
        if path != f"eqiora::{source_module}":
            raise RustReferenceError(f"{context} path/source module disagreement")
        for classification, key in (
            ("stable", "stable_exports"),
            ("transitional", "transitional_exports"),
        ):
            items.extend(
                FacadePath(f"{path}::{name}", classification, provider)
                for name, provider in _exports(namespace[key], f"{context}.{key}")
            )

    modules.sort()
    items.sort()
    if len({entry.path for entry in modules}) != len(modules):
        raise RustReferenceError("facade module paths are not unique")
    if len({entry.path for entry in items}) != len(items):
        raise RustReferenceError("facade item paths are not unique")
    return modules, items


def _facade_counts(
    modules: list[FacadePath], items: list[FacadePath]
) -> dict[str, int]:
    return {
        "modules": len(modules),
        "stable_modules": sum(entry.classification == "stable" for entry in modules),
        "transitional_modules": sum(
            entry.classification == "transitional" for entry in modules
        ),
        "items": len(items),
        "stable_items": sum(entry.classification == "stable" for entry in items),
        "transitional_items": sum(
            entry.classification == "transitional" for entry in items
        ),
    }


def _resolved_directory(path: Path, context: str) -> Path:
    if path.is_symlink() or not path.is_dir():
        raise RustReferenceError(f"{context} must be a real directory: {path}")
    return path.resolve(strict=True)


def _reject_symlinks(root: Path) -> None:
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise RustReferenceError(f"rustdoc tree contains a symlink: {path}")


def _validate_rustdoc_root(root: Path) -> Path:
    rustdoc = _resolved_directory(root, "--rustdoc-root")
    _reject_symlinks(rustdoc)
    crate = rustdoc / "eqiora"
    if not (crate / "index.html").is_file():
        raise RustReferenceError("rustdoc root is missing eqiora/index.html")
    if (rustdoc / "eqiora_mcp").exists():
        raise RustReferenceError("rustdoc contains the private eqiora_mcp root")
    crate_roots = sorted(
        child.name
        for child in rustdoc.iterdir()
        if child.is_dir() and (child / "index.html").is_file()
    )
    if crate_roots != ["eqiora"]:
        raise RustReferenceError(
            f"rustdoc contains non-facade crate roots: {crate_roots}"
        )
    _validate_html_references(rustdoc)
    return rustdoc


def _validate_html_references(root: Path) -> None:
    for document in sorted(root.rglob("*.html")):
        parser = _References()
        try:
            body = document.read_text(encoding="utf-8")
            parser.feed(body)
        except (OSError, UnicodeError) as error:
            raise RustReferenceError(f"cannot parse {document}: {error}") from error
        for raw_url in parser.urls:
            parsed = urlsplit(html.unescape(raw_url))
            if parsed.scheme or parsed.netloc or not parsed.path:
                continue
            path_text = unquote(parsed.path)
            if path_text.startswith("/"):
                if path_text not in ALLOWED_SITE_LINKS:
                    raise RustReferenceError(
                        f"{document} contains an unowned site-absolute link: {path_text}"
                    )
                continue
            target = (document.parent / path_text).resolve(strict=False)
            try:
                target.relative_to(root)
            except ValueError as error:
                raise RustReferenceError(
                    f"{document} contains a link escaping rustdoc: {raw_url}"
                ) from error
            if target.is_dir():
                target = target / "index.html"
            if not target.is_file():
                relative = target.relative_to(root)
                # Rustdoc emits async implementor-shard references even when a
                # public re-exported trait has no implementors and therefore no
                # shard file. Those sparse shards are optional rustdoc data,
                # not a missing copied asset. Every other local reference is
                # required to resolve.
                if (
                    len(relative.parts) >= 2
                    and relative.parts[0] in {"trait.impl", "type.impl"}
                    and relative.suffix == ".js"
                ):
                    continue
                # `tracing::instrument::WithSubscriber` is a blanket impl on
                # every documented type. Its upstream method prose contains
                # this crate-relative link, even when rustdoc renders the impl
                # into another crate under `--no-deps`. The tracing dispatcher
                # page is intentionally outside this facade-only artifact.
                if (
                    raw_url == "dispatcher#setting-the-default-subscriber"
                    and 'id="impl-WithSubscriber-for-T"' in body
                ):
                    continue
                # Standalone rustdoc emits help/settings navigation to a
                # workspace index that Cargo intentionally does not create for
                # a single selected package. The published entry point is the
                # required eqiora/index.html, never this absent aggregate.
                if document.parent == root and relative == Path("index.html"):
                    continue
                raise RustReferenceError(
                    f"{document} references a missing rustdoc asset: {raw_url}"
                )


def _item_target(crate: Path, item: FacadePath) -> str:
    parts = item.path.split("::")
    if parts[0] != "eqiora" or len(parts) < 2:
        raise RustReferenceError(f"invalid facade item path: {item.path}")
    parent_parts = parts[1:-1]
    name = parts[-1]
    parent = crate.joinpath(*parent_parts)
    module_target = parent / name / "index.html"
    if module_target.is_file():
        return PurePosixPath(*parent_parts, name, "index.html").as_posix()
    candidates = sorted(parent.glob(f"*.{name}.html"))
    if len(candidates) != 1:
        relative = PurePosixPath(*parent_parts).as_posix()
        raise RustReferenceError(
            f"{item.path} has {len(candidates)} rustdoc targets below {relative or '.'}"
        )
    return PurePosixPath(*parent_parts, candidates[0].name).as_posix()


def _module_target(crate: Path, module: FacadePath) -> str:
    parts = module.path.split("::")
    target = crate.joinpath(*parts[1:], "index.html")
    if not target.is_file():
        raise RustReferenceError(f"{module.path} has no rustdoc module target")
    return PurePosixPath(*parts[1:], "index.html").as_posix()


def _link(path: str, target: str) -> str:
    return f"[`{path}`]({PUBLIC_RUSTDOC_PREFIX}{target})"


def _render_landing(
    rustdoc: Path, modules: list[FacadePath], items: list[FacadePath]
) -> str:
    crate = rustdoc / "eqiora"
    lines = [
        "---",
        'title: "Rust API"',
        'description: "Browse Rust modules, types, functions, and signatures."',
        "---",
        "",
        "import ExactSourceLink from '@components/site/ExactSourceLink.astro';",
        "",
        "{/* Generated by tools/site/build_rust_reference.py; do not edit. */}",
        "",
        "[Open the Rust API documentation](/reference/rust/api/eqiora/) to browse",
        "types, functions, examples, and signatures with rustdoc navigation and search.",
        "",
        "The reference includes all optional features. Enable the features and install",
        "the native dependencies needed by your application; see the",
        '<ExactSourceLink kind="blob" path="docs/rust-api.md">Rust setup guide</ExactSourceLink>.',
        "",
        '<ExactSourceLink kind="blob" path="crates/eqiora/src/lib.rs">View source</ExactSourceLink>',
        "",
        "## Compile and inspect a model",
        "",
        "Add `eqiora` to a Cargo project as described in the setup guide above,",
        "then put this in `src/main.rs` and run `cargo run`:",
        "",
        "```rust",
        "use eqiora::api::ModelDocument;",
        "use eqiora::kernel::KernelNode;",
        "use eqiora::language::NotationProfile;",
        "",
        "fn main() {",
        '    let model = ModelDocument::compile("decay.eqi", r#"',
        "        model decay(parameter rate: 1 / s = 1) {",
        "            state x: 1;",
        "            initial { x = 1; }",
        "            relation flow { derivative(x) + rate * x = 0; }",
        "        }",
        '    "#).expect("model must compile");',
        "",
        "    for node in model.program().nodes() {",
        "        if let KernelNode::Relation(relation) = node {",
        "            let equations = model.render_equations(",
        "                relation.id().into(), NotationProfile::Plain,",
        '            ).expect("equations must render");',
        "            for equation in equations {",
        '                println!("{}", equation.text());',
        "            }",
        "        }",
        "    }",
        "}",
        "```",
        "",
        "The program checks dimensions and names, then prints the compiled",
        "equations as readable mathematics. Compilation creates an immutable Model;",
        "it does not advance time. The [Python workflow](/reference/python/)",
        "shows the corresponding compile, resolve, and run operations.",
        "",
        "## Work with physical quantities",
        "",
        "```rust",
        "use eqiora::aliases::Velocity;",
        "",
        "let inlet = Velocity::new(3.0);",
        "let correction = Velocity::new(0.5);",
        'println!("{} m/s", (inlet + correction).value());',
        "```",
        "",
        "The result is `3.5 m/s`. Quantities carry their dimension in the Rust",
        "type; adding a velocity to a length is a type error. Values use SI units.",
        "",
        "## Save and reload a Model",
        "",
        "Continue with the compiled `model` from the first example:",
        "",
        "```rust",
        'let bytes = model.canonical_json().expect("model must serialize");',
        'std::fs::write("decay.model.json", &bytes).expect("file must be writable");',
        'let restored = ModelDocument::replay(&bytes).expect("model must reload");',
        "assert!(model.structurally_equivalent(&restored).unwrap());",
        "```",
        "",
        "Replay restores the model's mathematical content. Keep the original `.eqi`",
        "source for editing and source-level presentation.",
        "",
        "## Modules",
        "",
        "Choose a module below, then use rustdoc's search to find a type or method.",
        "",
        "| Module |",
        "| --- |",
    ]
    for module in modules:
        lines.append(
            f"| {_link(module.path, _module_target(crate, module))} |"
        )
    lines.extend(
        [
            "",
            "## Complete item index",
            "",
            "| Item |",
            "| --- |",
        ]
    )
    for item in items:
        assert item.provider is not None
        lines.append(
            f"| {_link(item.path, _item_target(crate, item))} |"
        )
    lines.append("")
    return "\n".join(lines)


def _update_or_check_landing(expected: str, update: bool) -> None:
    if update:
        LANDING.parent.mkdir(parents=True, exist_ok=True)
        LANDING.write_text(expected, encoding="utf-8", newline="\n")
        print(f"updated {LANDING.relative_to(ROOT)}")
        return
    if not LANDING.is_file():
        raise RustReferenceError(f"generated landing is missing: {LANDING}")
    actual = LANDING.read_text(encoding="utf-8")
    if actual != expected:
        diff = "\n".join(
            difflib.unified_diff(
                actual.splitlines(),
                expected.splitlines(),
                fromfile=LANDING.relative_to(ROOT).as_posix(),
                tofile="generated",
                lineterm="",
            )
        )
        raise RustReferenceError(f"generated Rust landing is stale:\n{diff}")
    print(f"checked {LANDING.relative_to(ROOT)}")


def _empty_output(path: Path) -> Path:
    output = _resolved_directory(path, "--output")
    if any(output.iterdir()):
        raise RustReferenceError(f"--output must be empty: {output}")
    return output


def _stage(rustdoc: Path, output: Path) -> Path:
    expected_projection = _project_tree(rustdoc, write=False)
    destination = output / "reference" / "rust" / "api"
    destination.parent.mkdir(parents=True, exist_ok=False)
    shutil.copytree(rustdoc, destination, symlinks=False)
    _validate_rustdoc_root(destination)
    if _project_tree(destination, write=True) != expected_projection:
        raise RustReferenceError("staged Rustdoc accessibility projection changed")
    _validate_rustdoc_root(destination)
    return destination


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rustdoc-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--update-index",
        action="store_true",
        help="update the committed classified facade landing before staging",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        rustdoc = _validate_rustdoc_root(args.rustdoc_root)
        output = _empty_output(args.output)
        modules, items = _load_facade()
        expected = _render_landing(rustdoc, modules, items)
        _update_or_check_landing(expected, args.update_index)
        destination = _stage(rustdoc, output)
    except (OSError, RustReferenceError) as error:
        print(f"Rust reference: {error}", file=sys.stderr)
        return 1
    print(
        "Rust reference: staged complete facade rustdoc at "
        f"{destination} ({len(modules)} modules + {len(items)} items = "
        f"{len(modules) + len(items)} paths)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
