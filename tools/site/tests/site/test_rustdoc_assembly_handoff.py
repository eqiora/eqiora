from __future__ import annotations

import hashlib
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path


REPOSITORY = Path(__file__).resolve().parents[4]
BUILDER = REPOSITORY / "tools/site/build_rust_reference.py"
RUNNER = REPOSITORY / "tools/site/run_offline_site_checks.sh"
LANDING = REPOSITORY / "docs/site/src/content/docs/reference/rust/index.mdx"
PUBLIC_PREFIX = Path("reference/rust/api")
HANDOFF_SUFFIX = PUBLIC_PREFIX
RUSTDOC_LINK = re.compile(
    r"\[`eqiora(?:::[^`]+)?`\]"
    r"\(/reference/rust/api/eqiora/(?P<target>[^)]+)\)"
)
SYNTHETIC_HTML_FILES = 256
SYNTHETIC_PROJECTED_PAGES = 8
SYNTHETIC_DIRECT_SECTIONS = 16
SYNTHETIC_SIGNATURE_LINKS = 36
SYNTHETIC_DESCRIPTION_LABELS = 7
SYNTHETIC_SPECIAL_HIDEME_LABELS = 1
SYNTHETIC_NON_HTML_FILES = 5


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8", newline="\n")


def _projection_page(
    *,
    first_section: int,
    section_count: int,
    three_link_sections: int,
    description_labels: int,
    special_labels: int,
) -> str:
    markup = ["<!doctype html><title>Synthetic Rustdoc projection</title>"]
    for section in range(first_section, first_section + section_count):
        link_count = 3 if section < three_link_sections else 2
        links = "".join(
            f'<a href="#s{section}">signature {link}</a>'
            for link in range(link_count)
        )
        markup.append(
            '<details class="toggle"><summary>'
            f'<section id="s{section}">{links}</section>'
            "</summary></details>"
        )
    markup.extend(
        '<details class="toggle"><summary class="hideme">'
        "<span>Expand description</span></summary></details>"
        for _ in range(description_labels)
    )
    markup.extend(
        '<details class="toggle"><summary class="hideme">'
        "<span>Show 13 fields</span></summary></details>"
        for _ in range(special_labels)
    )
    return "".join(markup) + "\n"


def _make_compiler_projection(root: Path) -> None:
    """Create the finite production-shape facade consumed by the real builder CLI."""
    crate = root / "eqiora"
    _write(crate / "index.html", "<!doctype html><title>Crate eqiora</title>\n")
    landing = LANDING.read_text(encoding="utf-8")
    targets = [match.group("target") for match in RUSTDOC_LINK.finditer(landing)]
    if not targets or len(set(targets)) != len(targets):
        raise AssertionError("accepted Rust landing must name unique facade targets")
    for target in targets:
        _write(crate / target, "<!doctype html><title>Eqiora facade target</title>\n")

    html_files = sorted(path for path in root.rglob("*.html"))
    for offset in range(SYNTHETIC_HTML_FILES - len(html_files)):
        path = crate / "__synthetic_projection" / f"page-{offset:04d}.html"
        _write(path, "<!doctype html><title>Synthetic Rustdoc page</title>\n")
        html_files.append(path)
    html_files.sort()
    if len(html_files) != SYNTHETIC_HTML_FILES:
        raise AssertionError("synthetic Rustdoc HTML count is not production-shaped")

    first_section = 0
    descriptions_written = 0
    special_written = 0
    three_link_sections = (
        SYNTHETIC_SIGNATURE_LINKS - 2 * SYNTHETIC_DIRECT_SECTIONS
    )
    for page, path in enumerate(html_files[:SYNTHETIC_PROJECTED_PAGES]):
        section_count = SYNTHETIC_DIRECT_SECTIONS // SYNTHETIC_PROJECTED_PAGES
        if page < SYNTHETIC_DIRECT_SECTIONS % SYNTHETIC_PROJECTED_PAGES:
            section_count += 1

        hideme_count = (
            SYNTHETIC_DESCRIPTION_LABELS + SYNTHETIC_SPECIAL_HIDEME_LABELS
        ) // SYNTHETIC_PROJECTED_PAGES
        if page < (
            SYNTHETIC_DESCRIPTION_LABELS + SYNTHETIC_SPECIAL_HIDEME_LABELS
        ) % SYNTHETIC_PROJECTED_PAGES:
            hideme_count += 1
        description_count = min(
            hideme_count, SYNTHETIC_DESCRIPTION_LABELS - descriptions_written
        )
        special_count = hideme_count - description_count
        _write(
            path,
            _projection_page(
                first_section=first_section,
                section_count=section_count,
                three_link_sections=three_link_sections,
                description_labels=description_count,
                special_labels=special_count,
            ),
        )
        first_section += section_count
        descriptions_written += description_count
        special_written += special_count

    if (
        first_section != SYNTHETIC_DIRECT_SECTIONS
        or descriptions_written != SYNTHETIC_DESCRIPTION_LABELS
        or special_written != SYNTHETIC_SPECIAL_HIDEME_LABELS
    ):
        raise AssertionError("synthetic Rustdoc projection totals drifted")
    for offset in range(SYNTHETIC_NON_HTML_FILES):
        _write(
            crate / "__synthetic_projection" / f"asset-{offset:04d}.bin",
            f"bounded synthetic Rustdoc asset {offset}\n",
        )


def _make_astro(root: Path) -> None:
    for relative in (
        "index.html",
        "404.html",
        "pagefind/pagefind.js",
        "robots.txt",
        "sitemap-index.xml",
    ):
        _write(root / relative, f"ordinary Astro artifact: {relative}\n")


def _runner_post_builder_slice(source: str) -> str:
    """Return the existing runner seam after builder success through assembly."""
    lines = source.splitlines()
    try:
        builder = lines.index("python3 tools/site/build_rust_reference.py \\")
    except ValueError as error:
        raise AssertionError("runner lost the accepted Rust-reference builder call") from error
    builder_end = next(
        index
        for index in range(builder + 1, len(lines))
        if '--output "$EQIORA_SITE_RUSTDOC_STAGE"' in lines[index]
    )
    try:
        assembler = lines.index("python3 tools/site/assemble_site.py \\", builder_end)
    except ValueError as error:
        raise AssertionError("runner lost the accepted site assembler call") from error
    assembler_end = next(
        index
        for index in range(assembler + 1, len(lines))
        if '--scratch-root "$assembly_scratch"' in lines[index]
    )
    return "\n".join(lines[builder_end + 1 : assembler_end + 1]) + "\n"


def _replace_assembler_root(segment: str, expression: str) -> str:
    lines = segment.splitlines()
    options = [index for index, line in enumerate(lines) if "--rustdoc-root " in line]
    if len(options) != 1:
        raise AssertionError("runner assembly slice must contain one Rustdoc root option")
    continuation = " \\" if lines[options[0]].rstrip().endswith("\\") else ""
    lines[options[0]] = f"  --rustdoc-root {expression}{continuation}"
    return "\n".join(lines) + "\n"


def _replace_with_symlink(path: Path) -> None:
    backing = path.with_name(f"{path.name}-backing")
    path.rename(backing)
    path.symlink_to(backing, target_is_directory=True)


def _make_missing(path: Path) -> None:
    path.rename(path.with_name(f"{path.name}-missing"))


def _make_non_directory(path: Path) -> None:
    backing = path.with_name(f"{path.name}-directory")
    path.rename(backing)
    path.write_text("not a directory\n", encoding="utf-8")


class _Run:
    def __init__(
        self,
        *,
        result: subprocess.CompletedProcess[str] | None,
        stage: Path,
        artifact: Path,
        selected_root: str | None,
        builder_output: str,
        gate_error: str | None = None,
    ) -> None:
        self.result = result
        self.stage = stage
        self.artifact = artifact
        self.selected_root = selected_root
        self.builder_output = builder_output
        self.gate_error = gate_error


class RustdocAssemblyHandoffTests(unittest.TestCase):
    maxDiff = None

    @staticmethod
    def _temporary_directory() -> tempfile.TemporaryDirectory[str]:
        parent = Path.home().resolve() / ".cache/eqiora/oracle-tests"
        parent.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(dir=parent)

    @staticmethod
    def _reference_gate(stage: Path) -> str | None:
        """Independent contract projection used only to prove mutant causality."""
        if stage.is_symlink() or not stage.is_dir():
            return "stage root is not a real directory"
        handoff = stage.joinpath(*HANDOFF_SUFFIX.parts)
        current = stage
        for component in HANDOFF_SUFFIX.parts:
            current = current / component
            if current.is_symlink() or not current.is_dir():
                return f"handoff component is not a real directory: {component}"
        try:
            if handoff.resolve(strict=True).relative_to(stage.resolve(strict=True)) != Path(
                *HANDOFF_SUFFIX.parts
            ):
                return "handoff resolved to a different stage child"
        except (OSError, ValueError):
            return "handoff escaped or did not resolve"
        entry = handoff / "eqiora/index.html"
        if entry.is_symlink() or not entry.is_file():
            return "Eqiora entry is not a regular non-symlink file"
        if (handoff / "eqiora_mcp").exists():
            return "private crate root exists"
        return None

    def _invoke(
        self,
        root: Path,
        *,
        mutate_stage: Callable[[Path], None] | None = None,
        selected_expression: str | None = None,
        reference_gate: bool = False,
        source_segment: str | None = None,
    ) -> _Run:
        raw = root / "rustdoc-target/doc"
        stage = root / "rustdoc-stage"
        astro = root / "astro"
        artifact = root / "build/site"
        _make_compiler_projection(raw)
        _make_astro(astro)
        artifact.parent.mkdir()
        stage.mkdir()
        builder = subprocess.run(
            [
                sys.executable,
                str(BUILDER),
                "--rustdoc-root",
                str(raw),
                "--output",
                str(stage),
            ],
            cwd=REPOSITORY,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.assertEqual(builder.returncode, 0, builder.stderr)
        landing = LANDING.read_text(encoding="utf-8")
        target_count = len(RUSTDOC_LINK.findall(landing))
        self.assertIn(f"{target_count} paths", builder.stdout)
        if mutate_stage is not None:
            mutate_stage(stage)

        gate_error = self._reference_gate(stage) if reference_gate else None
        if gate_error is not None:
            return _Run(
                result=None,
                stage=stage,
                artifact=artifact,
                selected_root=None,
                builder_output=builder.stdout,
                gate_error=gate_error,
            )

        segment = source_segment or _runner_post_builder_slice(
            RUNNER.read_text(encoding="utf-8")
        )
        if selected_expression is not None:
            segment = _replace_assembler_root(segment, selected_expression)
        script = root / "runner-handoff.sh"
        _write(
            script,
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            "cargo_version=0.1.0\n"
            "python_version=0.1.0\n"
            + segment,
        )
        script.chmod(0o755)

        trace = root / "assembler-arguments"
        shim = root / "bin"
        shim.mkdir()
        _write(shim / "npm", "#!/bin/sh\nexit 0\n")
        (shim / "npm").chmod(0o755)
        python = shlex.quote(sys.executable)
        _write(
            shim / "python3",
            "#!/bin/sh\n"
            "if [ \"${1:-}\" = tools/site/assemble_site.py ]; then\n"
            "  printf '%s\\0' \"$@\" > \"$EQIORA_ORACLE_ASSEMBLER_TRACE\"\n"
            "fi\n"
            f"exec {python} \"$@\"\n",
        )
        (shim / "python3").chmod(0o755)
        environment = os.environ.copy()
        environment.update(
            {
                "PATH": f"{shim}{os.pathsep}{environment['PATH']}",
                "EQIORA_API_SCRATCH": str(root),
                "EQIORA_SITE_ASTRO_OUT_DIR": str(astro),
                "EQIORA_SITE_RUSTDOC_STAGE": str(stage),
                "EQIORA_SITE_ARTIFACT": str(artifact),
                "EQIORA_SITE_SOURCE_SHA": "a" * 40,
                "EQIORA_ORACLE_ASSEMBLER_TRACE": str(trace),
            }
        )
        result = subprocess.run(
            [str(script)],
            cwd=REPOSITORY,
            env=environment,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        selected_root = None
        if trace.is_file():
            arguments = trace.read_bytes().rstrip(b"\0").split(b"\0")
            decoded = [argument.decode("utf-8") for argument in arguments]
            option = decoded.index("--rustdoc-root")
            selected_root = decoded[option + 1]
        return _Run(
            result=result,
            stage=stage,
            artifact=artifact,
            selected_root=selected_root,
            builder_output=builder.stdout,
        )

    def _acceptance_errors(self, run: _Run) -> list[str]:
        if run.gate_error is not None:
            return [run.gate_error]
        assert run.result is not None
        errors: list[str] = []
        if run.result.returncode != 0:
            errors.append(
                f"runner handoff returned {run.result.returncode}: {run.result.stderr}"
            )
        expected_root = run.stage.joinpath(*HANDOFF_SUFFIX.parts)
        if run.selected_root != str(expected_root):
            errors.append(
                f"assembler received {run.selected_root!r}, expected {str(expected_root)!r}"
            )
        entry = run.artifact / PUBLIC_PREFIX / "eqiora/index.html"
        if entry.is_symlink() or not entry.is_file():
            errors.append("final artifact lacks the regular Eqiora Rustdoc entry")
        if (run.artifact / "eqiora").exists():
            errors.append("final artifact exposes an unprefixed Eqiora crate root")
        doubled = run.artifact / PUBLIC_PREFIX / PUBLIC_PREFIX
        if doubled.exists():
            errors.append("final artifact contains a doubled Rustdoc prefix")
        api = run.artifact / PUBLIC_PREFIX
        if api.is_dir() and any(path.name == "eqiora_mcp" for path in api.rglob("*")):
            errors.append("final artifact contains a private eqiora_mcp root")
        if not errors:
            source = expected_root
            source_files = {
                path.relative_to(source): hashlib.sha256(path.read_bytes()).digest()
                for path in source.rglob("*")
                if path.is_file() and not path.is_symlink()
            }
            installed_files = {
                path.relative_to(api): hashlib.sha256(path.read_bytes()).digest()
                for path in api.rglob("*")
                if path.is_file() and not path.is_symlink()
            }
            if source_files != installed_files:
                errors.append("assembled Rustdoc is not the exact accepted builder subtree")
        return errors

    def _assert_accepted(self, run: _Run) -> None:
        self.assertEqual(self._acceptance_errors(run), [])

    def _assert_rejected(self, run: _Run, label: str) -> None:
        self.assertNotEqual(self._acceptance_errors(run), [], label)

    def _exercise_positive_then_mutants(
        self, *, reference_gate: bool, source_segment: str | None = None
    ) -> None:
        # The ordinary path is intentionally first. If it fails, no negative
        # result can be credited as evidence for this invocation.
        with self._temporary_directory() as temporary:
            self._assert_accepted(
                self._invoke(
                    Path(temporary),
                    reference_gate=reference_gate,
                    source_segment=source_segment,
                )
            )

        input_mutants: tuple[tuple[str, Callable[[Path], None]], ...] = (
            ("stage-root symlink", _replace_with_symlink),
            (
                "reference symlink",
                lambda stage: _replace_with_symlink(stage / "reference"),
            ),
            (
                "rust symlink",
                lambda stage: _replace_with_symlink(stage / "reference/rust"),
            ),
            (
                "api symlink",
                lambda stage: _replace_with_symlink(stage / PUBLIC_PREFIX),
            ),
            (
                "entry symlink",
                lambda stage: _replace_with_symlink(
                    stage / PUBLIC_PREFIX / "eqiora/index.html"
                ),
            ),
            ("missing reference", lambda stage: _make_missing(stage / "reference")),
            (
                "missing rust",
                lambda stage: _make_missing(stage / "reference/rust"),
            ),
            ("missing api", lambda stage: _make_missing(stage / PUBLIC_PREFIX)),
            (
                "missing eqiora",
                lambda stage: _make_missing(stage / PUBLIC_PREFIX / "eqiora"),
            ),
            (
                "missing entry",
                lambda stage: _make_missing(
                    stage / PUBLIC_PREFIX / "eqiora/index.html"
                ),
            ),
            (
                "non-directory api",
                lambda stage: _make_non_directory(stage / PUBLIC_PREFIX),
            ),
            (
                "private crate root",
                lambda stage: _write(
                    stage / PUBLIC_PREFIX / "eqiora_mcp/index.html",
                    "<!doctype html><title>private</title>\n",
                ),
            ),
        )
        for label, mutate in input_mutants:
            with self.subTest(label=label), self._temporary_directory() as temporary:
                self._assert_rejected(
                    self._invoke(
                        Path(temporary),
                        mutate_stage=mutate,
                        reference_gate=reference_gate,
                        source_segment=source_segment,
                    ),
                    label,
                )

        selector_mutants = (
            ("wrong parent root", '"$EQIORA_SITE_RUSTDOC_STAGE"'),
            ("too-shallow Rust root", '"$EQIORA_SITE_RUSTDOC_STAGE/reference/rust"'),
            (
                "too-deep crate root",
                '"$EQIORA_SITE_RUSTDOC_STAGE/reference/rust/api/eqiora"',
            ),
            (
                "nested Rustdoc root",
                '"$EQIORA_SITE_RUSTDOC_STAGE/reference/rust/api/eqiora/api"',
            ),
        )
        for label, expression in selector_mutants:
            with self.subTest(label=label), self._temporary_directory() as temporary:
                self._assert_rejected(
                    self._invoke(
                        Path(temporary),
                        selected_expression=expression,
                        reference_gate=reference_gate,
                        source_segment=source_segment,
                    ),
                    label,
                )

        def doubled(stage: Path) -> None:
            # Make the wrong parent otherwise acceptable to the assembler, so
            # rejection is caused by exact selection/final topology rather than
            # its earlier missing-entry guard.
            shutil.copytree(stage / PUBLIC_PREFIX / "eqiora", stage / "eqiora")

        with self.subTest(label="doubled public prefix"), self._temporary_directory() as temporary:
            self._assert_rejected(
                self._invoke(
                    Path(temporary),
                    mutate_stage=doubled,
                    selected_expression='"$EQIORA_SITE_RUSTDOC_STAGE"',
                    reference_gate=reference_gate,
                    source_segment=source_segment,
                ),
                "doubled public prefix",
            )

    def test_00_actual_runner_positive_precedes_causal_mutants(self) -> None:
        self._exercise_positive_then_mutants(reference_gate=False)

    def test_99_independent_contract_projection_proves_mutant_harness(self) -> None:
        # This is an oracle self-check, not product acceptance: the independent
        # gate proves that the same post-builder mutants are distinguishable
        # only after its ordinary exact-subtree path succeeds.
        source = RUNNER.read_text(encoding="utf-8")
        segment = _runner_post_builder_slice(source)
        corrected = _replace_assembler_root(
            segment, '"$EQIORA_SITE_RUSTDOC_STAGE/reference/rust/api"'
        )
        self._exercise_positive_then_mutants(
            reference_gate=True, source_segment=corrected
        )


if __name__ == "__main__":
    unittest.main()
