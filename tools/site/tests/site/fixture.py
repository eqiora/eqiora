from __future__ import annotations

import importlib.util
import json
import os
import re
import selectors
import signal
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

SOURCE_SHA = "a" * 40
REPOSITORY = Path(__file__).resolve().parents[4]
EDITOR_CHECK_INPUTS = (
    "crates/eqiora-lang/src/parser.rs",
    "crates/eqiora-lang/src/parser/compile_time.rs",
    "crates/eqiora-lang/src/parser/document.rs",
    "crates/eqiora-lang/src/parser/domain.rs",
    "crates/eqiora-lang/src/parser/formulation.rs",
    "crates/eqiora-lang/src/parser/instance.rs",
    "crates/eqiora-lang/src/parser/property.rs",
    "crates/eqiora-lang/src/parser/relation.rs",
    "docs/site/astro.config.mjs",
    "docs/site/ec.config.mjs",
    "editor/eqiora/README.md",
    "editor/eqiora/bundle.json",
    "editor/eqiora/language-configuration.json",
    "editor/eqiora/snippets/eqiora.json",
    "editor/eqiora/syntaxes/eqiora.tmLanguage.json",
    "tools/editor/check_syntax_bundle.py",
    "tools/editor/tests/test_syntax_bundle.py",
)
PRESSURE_ALT = "Steady Stokes pressure around a cylinder, with the current mesh and pressure scale in pascals."
WAKE_ALT = "Cell-average vorticity in a Kármán vortex street behind a circular cylinder."

SITE_ROUTES = (
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
    "/404.html",
)


def copy_editor_check_inputs(destination: Path) -> None:
    for relative in EDITOR_CHECK_INPUTS:
        source = REPOSITORY / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)


GIT_OBJECT_REPOSITORY_VARIABLE = "EQIORA_SITE_GIT_OBJECT_REPOSITORY"
SOURCE_SHA_VARIABLE = "EQIORA_SITE_SOURCE_SHA"
GIT_TIMEOUT_SECONDS = 30
GIT_IDENTITY_OUTPUT_LIMIT = 65_536
GIT_OBJECT_OUTPUT_LIMIT = 34_088_961
GIT_STDERR_LIMIT = 65_536
_LOWER_OBJECT_ID = re.compile(rb"[0-9a-f]{40}\n")
_GIT_EXECUTABLE = Path(shutil.which("git", path=os.defpath) or "/usr/bin/git").resolve()
_GIT_ENVIRONMENT = {
    "LC_ALL": "C",
    "LANG": "C",
    "TZ": "UTC",
    "GIT_CONFIG_NOSYSTEM": "1",
    "GIT_CONFIG_GLOBAL": os.devnull,
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_NO_REPLACE_OBJECTS": "1",
}
_HISTORICAL_COMMITS = {
    "5d2a9bef58c2df32cd6b14c5b6dd876beac7144f",
    "68c9c2fe245ac52cc20dcf5a65a2455de507f0dc",
    "19968da984c16e718baeb9faa5aae04260896c29",
}
_HISTORICAL_OBJECTS = {
    "3237f739098498ac46bfdd6a993c00b0575900f3",
    "57f8b9b476c04b8103b5a43c8a30504c0e2fa1fb",
    "47dc3e3d863cfb5727b87d785d09abf9743c0a72",
    "61c1bbede492aef4a9c85fa364d031e012621809",
    "20701fe8909295b980c1da7cf3eab366f8d5f27c",
    "6e685495bf6989e1ad902a7e88c199557285cbee",
    "1d19473c487b8035608cc88cbd99757f2b95865a",
    "21d5f0bc5213bca02336040f1085c7d52c63588f",
}
_HISTORICAL_QUERIES = {
    ("rev-parse", "--verify", "HEAD^{commit}"): GIT_IDENTITY_OUTPUT_LIMIT,
    ("rev-parse", "--verify", "HEAD^{tree}"): GIT_IDENTITY_OUTPUT_LIMIT,
    (
        "rev-parse",
        "5d2a9bef58c2df32cd6b14c5b6dd876beac7144f^{tree}",
    ): GIT_IDENTITY_OUTPUT_LIMIT,
    (
        "rev-parse",
        "5d2a9bef58c2df32cd6b14c5b6dd876beac7144f:.github/workflows/pages.yml",
    ): GIT_IDENTITY_OUTPUT_LIMIT,
    (
        "rev-parse",
        "5d2a9bef58c2df32cd6b14c5b6dd876beac7144f:CLAUDE.md",
    ): GIT_IDENTITY_OUTPUT_LIMIT,
    (
        "rev-parse",
        "5d2a9bef58c2df32cd6b14c5b6dd876beac7144f:AGENTS.md",
    ): GIT_IDENTITY_OUTPUT_LIMIT,
    (
        "ls-tree",
        "5d2a9bef58c2df32cd6b14c5b6dd876beac7144f",
        "--",
        "AGENTS.md",
        "CLAUDE.md",
    ): GIT_IDENTITY_OUTPUT_LIMIT,
    (
        "rev-parse",
        "68c9c2fe245ac52cc20dcf5a65a2455de507f0dc^{tree}",
    ): GIT_IDENTITY_OUTPUT_LIMIT,
    (
        "rev-parse",
        "68c9c2fe245ac52cc20dcf5a65a2455de507f0dc:.github/workflows/pages.yml",
    ): GIT_IDENTITY_OUTPUT_LIMIT,
    (
        "rev-parse",
        "19968da984c16e718baeb9faa5aae04260896c29^{tree}",
    ): GIT_IDENTITY_OUTPUT_LIMIT,
    (
        "archive",
        "--format=tar",
        "19968da984c16e718baeb9faa5aae04260896c29",
    ): GIT_OBJECT_OUTPUT_LIMIT,
    (
        "ls-tree",
        "-r",
        "-z",
        "19968da984c16e718baeb9faa5aae04260896c29",
    ): GIT_OBJECT_OUTPUT_LIMIT,
    (
        "show",
        "19968da984c16e718baeb9faa5aae04260896c29:docs/site/package-lock.json",
    ): GIT_OBJECT_OUTPUT_LIMIT,
    (
        "show",
        "19968da984c16e718baeb9faa5aae04260896c29:CLAUDE.md",
    ): GIT_IDENTITY_OUTPUT_LIMIT,
    (
        "show",
        "19968da984c16e718baeb9faa5aae04260896c29:AGENTS.md",
    ): GIT_IDENTITY_OUTPUT_LIMIT,
}
_HISTORICAL_QUERIES.update(
    {
        ("cat-file", "-e", f"{commit}^{{commit}}"): GIT_IDENTITY_OUTPUT_LIMIT
        for commit in _HISTORICAL_COMMITS
    }
)
_HISTORICAL_QUERIES.update(
    {
        ("cat-file", "-e", object_id): GIT_IDENTITY_OUTPUT_LIMIT
        for object_id in _HISTORICAL_OBJECTS
    }
)
_HISTORICAL_QUERIES.update(
    {
        ("cat-file", "blob", object_id): GIT_IDENTITY_OUTPUT_LIMIT
        for object_id in {
            "57f8b9b476c04b8103b5a43c8a30504c0e2fa1fb",
            "47dc3e3d863cfb5727b87d785d09abf9743c0a72",
            "61c1bbede492aef4a9c85fa364d031e012621809",
            "6e685495bf6989e1ad902a7e88c199557285cbee",
        }
    }
)


class GitObjectAuthorityError(AssertionError):
    pass


@dataclass(frozen=True)
class GitObjectAuthority:
    root: Path
    head: str
    tree: str


def git_read_environment() -> dict[str, str]:
    return {**_GIT_ENVIRONMENT, "PATH": os.defpath}


def _bounded_command(
    command: Sequence[str],
    *,
    output_limit: int,
    executable: Path = _GIT_EXECUTABLE,
) -> bytes:
    process = subprocess.Popen(
        [str(executable), *command],
        env=_GIT_ENVIRONMENT,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        start_new_session=True,
    )
    assert process.stdout is not None and process.stderr is not None
    streams = selectors.DefaultSelector()
    streams.register(process.stdout, selectors.EVENT_READ, ("stdout", output_limit))
    streams.register(process.stderr, selectors.EVENT_READ, ("stderr", GIT_STDERR_LIMIT))
    captured = {"stdout": bytearray(), "stderr": bytearray()}
    deadline = time.monotonic() + GIT_TIMEOUT_SECONDS
    try:
        while streams.get_map():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise GitObjectAuthorityError("Git object authority command timed out")
            ready = streams.select(remaining)
            if not ready:
                raise GitObjectAuthorityError("Git object authority command timed out")
            for key, _ in ready:
                label, limit = key.data
                chunk = os.read(key.fd, min(65_536, limit + 1 - len(captured[label])))
                if not chunk:
                    streams.unregister(key.fileobj)
                    continue
                captured[label].extend(chunk)
                if len(captured[label]) > limit:
                    raise GitObjectAuthorityError(
                        f"Git object authority {label} exceeded its bound"
                    )
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise GitObjectAuthorityError("Git object authority command timed out")
        returncode = process.wait(timeout=remaining)
    except (GitObjectAuthorityError, subprocess.TimeoutExpired):
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait()
        raise GitObjectAuthorityError(
            "Git object authority command failed or timed out"
        )
    finally:
        streams.close()
        process.stdout.close()
        process.stderr.close()
    stdout, stderr = bytes(captured["stdout"]), bytes(captured["stderr"])
    if returncode != 0 or stderr:
        raise GitObjectAuthorityError(
            "Git object authority command failed or emitted stderr"
        )
    return stdout


def _git_read(
    root: Path,
    arguments: Sequence[str],
    *,
    output_limit: int = GIT_IDENTITY_OUTPUT_LIMIT,
    executable: Path = _GIT_EXECUTABLE,
) -> bytes:
    return _bounded_command(
        [
            "--no-replace-objects",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "gc.auto=0",
            "-c",
            "maintenance.auto=false",
            "-C",
            str(root),
            *arguments,
        ],
        output_limit=output_limit,
        executable=executable,
    )


def _canonical_object_id(output: bytes, label: str) -> str:
    if _LOWER_OBJECT_ID.fullmatch(output) is None:
        raise GitObjectAuthorityError(
            f"Git object authority returned malformed {label} identity"
        )
    return output[:-1].decode("ascii")


def git_object_authority(
    *,
    repository: Path = REPOSITORY,
    environment: Mapping[str, str] | None = None,
    executable: Path = _GIT_EXECUTABLE,
) -> GitObjectAuthority:
    values = os.environ if environment is None else environment
    supplied = GIT_OBJECT_REPOSITORY_VARIABLE in values
    raw = values.get(GIT_OBJECT_REPOSITORY_VARIABLE, "")
    repository = repository.resolve(strict=True)
    if supplied:
        if not raw:
            raise GitObjectAuthorityError("Git object authority is empty")
        candidate = Path(raw)
        if not candidate.is_absolute():
            raise GitObjectAuthorityError("Git object authority is not absolute")
        try:
            root = candidate.resolve(strict=True)
        except OSError as error:
            raise GitObjectAuthorityError(
                "Git object authority is unavailable"
            ) from error
        if str(candidate) != str(root):
            raise GitObjectAuthorityError("Git object authority is not canonical")
        if not root.is_dir() or root.is_symlink():
            raise GitObjectAuthorityError(
                "Git object authority is not a real non-symlink directory"
            )
        if os.path.lexists(repository / ".git"):
            raise GitObjectAuthorityError("direct archive unexpectedly contains .git")
        if root == repository or root.is_relative_to(repository):
            raise GitObjectAuthorityError("Git object authority overlaps the archive")
    else:
        if not os.path.lexists(repository / ".git"):
            raise GitObjectAuthorityError(
                "a .git-absent archive requires EQIORA_SITE_GIT_OBJECT_REPOSITORY"
            )
        root = repository

    try:
        path_limit = os.pathconf(root, "PC_PATH_MAX")
    except (OSError, ValueError):
        path_limit = 4096
    if len(os.fsencode(root)) >= path_limit:
        raise GitObjectAuthorityError("Git object authority exceeds the path bound")

    top_level_output = _git_read(
        root, ["rev-parse", "--show-toplevel"], executable=executable
    )
    try:
        top_level_text = top_level_output.decode("utf-8")
    except UnicodeDecodeError as error:
        raise GitObjectAuthorityError(
            "Git object authority top level is malformed"
        ) from error
    if top_level_text != f"{root}\n":
        raise GitObjectAuthorityError("Git object authority top level differs")

    head = _canonical_object_id(
        _git_read(
            root, ["rev-parse", "--verify", "HEAD^{commit}"], executable=executable
        ),
        "HEAD",
    )
    expected_head = values.get(SOURCE_SHA_VARIABLE, head)
    if re.fullmatch(r"[0-9a-f]{40}", expected_head) is None:
        raise GitObjectAuthorityError("source SHA is missing or malformed")
    if head != expected_head:
        raise GitObjectAuthorityError(
            "Git object authority HEAD differs from source SHA"
        )
    tree = _canonical_object_id(
        _git_read(
            root, ["rev-parse", "--verify", "HEAD^{tree}"], executable=executable
        ),
        "HEAD tree",
    )
    return GitObjectAuthority(root=root, head=head, tree=tree)


def historical_git(
    *arguments: str,
    repository: Path = REPOSITORY,
    environment: Mapping[str, str] | None = None,
) -> bytes:
    command = tuple(arguments)
    output_limit = _HISTORICAL_QUERIES.get(command)
    if output_limit is None:
        raise GitObjectAuthorityError(
            "Git object authority command is not a frozen read-only object query"
        )
    authority = git_object_authority(repository=repository, environment=environment)
    return _git_read(authority.root, arguments, output_limit=output_limit)


def git_object_authority_status(
    *,
    repository: Path = REPOSITORY,
    environment: Mapping[str, str] | None = None,
) -> bytes:
    authority = git_object_authority(repository=repository, environment=environment)
    return _git_read(
        authority.root,
        ["status", "--porcelain=v1", "--untracked-files=all"],
    )


def _load_checker():
    path = REPOSITORY / "tools/site/check_site.py"
    spec = importlib.util.spec_from_file_location("release_site_check", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


checker = _load_checker()


def pinned_node_path(root: Path) -> str:
    directory = root / "pinned-node"
    directory.mkdir(parents=True, exist_ok=True)
    for name, version in (
        ("node", "v24.18.1"),
        ("npm", "11.16.0"),
        ("uv", "uv 0.12.1 (x86_64-unknown-linux-musl)"),
    ):
        executable = shutil.which(name)
        if executable is None:
            raise AssertionError(f"{name} is required by the site oracle")
        wrapper = directory / name
        wrapper.write_text(
            "#!/bin/sh\n"
            f"if [ \"${{1:-}}\" = --version ]; then printf '%s\\n' '{version}'; exit 0; fi\n"
            f'exec "{executable}" "$@"\n',
            encoding="utf-8",
        )
        wrapper.chmod(0o755)
    return f"{directory}{os.pathsep}{os.environ.get('PATH', '')}"


def _write_rustc_preflight_double(directory: Path) -> None:
    rustc = directory / "rustc"
    rustc.write_text(
        f"""#!{sys.executable}
from pathlib import Path
import os
import sys

args = sys.argv[1:]
if args == ["+stable", "-Vv"]:
    release = os.environ.get("FIXTURE_STABLE_RUST_RELEASE", "1.97.1")
    sys.stdout.write(f"fixture stable rustc\\nrelease: {{release}}\\n")
    raise SystemExit(0)
if args == ["-Vv"]:
    selected = os.environ.get("RUSTUP_TOOLCHAIN")
    if selected:
        trace = os.environ.get("FIXTURE_RUSTC_TRACE")
        if trace:
            with open(trace, "a", encoding="utf-8") as stream:
                stream.write(f"{{selected}}\\n")
        if selected in ("stable", "stable-x86_64-unknown-linux-gnu"):
            release = os.environ.get("FIXTURE_STABLE_RUST_RELEASE", "1.97.1")
        elif selected == "1.97.1-x86_64-unknown-linux-gnu":
            release = os.environ.get("FIXTURE_PINNED_RUST_RELEASE", "1.97.1")
        else:
            raise SystemExit(64)
        sys.stdout.write(f"fixture {{selected}} rustc\\nrelease: {{release}}\\n")
        raise SystemExit(0)
    try:
        toolchain = (Path.cwd() / "rust-toolchain.toml").read_bytes()
    except OSError:
        toolchain = None
    output = (
        (
            "fixture stable rustc\\nrelease: "
            + os.environ.get("FIXTURE_STABLE_RUST_RELEASE", "1.97.1")
            + "\\n"
        ).encode()
        if toolchain == {b'[toolchain]\nchannel = "stable"\ncomponents = ["rustfmt", "clippy"]\n'!r}
        else b"fixture selected rustc\\n"
    )
    sys.stdout.buffer.write(output)
    raise SystemExit(0)
raise SystemExit(64)
""",
        encoding="utf-8",
    )
    rustc.chmod(0o755)
    rustup = directory / "rustup"
    rustup.write_text(
        "#!/bin/sh\n"
        "test \"$#\" = 2\n"
        "test \"$1\" = toolchain\n"
        "test \"$2\" = list\n"
        "if test -n \"${FIXTURE_RUSTUP_TRACE:-}\"; then\n"
        "  printf 'toolchain list\\n' >> \"$FIXTURE_RUSTUP_TRACE\"\n"
        "fi\n"
        "printf '%s\\n' 'stable-x86_64-unknown-linux-gnu (active, default)' "
        "'1.97.1-x86_64-unknown-linux-gnu'\n",
        encoding="utf-8",
    )
    rustup.chmod(0o755)


def _write(path: Path, value: str | bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(value, encoding="utf-8")


def _workflow() -> str:
    return (REPOSITORY / ".github/workflows/pages.yml").read_text(encoding="utf-8")


def _runner() -> str:
    return """#!/usr/bin/env bash
export npm_config_offline=true CARGO_NET_OFFLINE=true
export EQIORA_SITE_CARGO_VERSION=0.1.0 EQIORA_SITE_PYTHON_VERSION=0.1.0
python3 tools/site/check_site.py source-topology --root "$EQIORA_SITE_SOURCE_ROOT"
python3 tools/site/check_site.py browser-supply --site-root docs/site --browser-cache "$PLAYWRIGHT_BROWSERS_PATH" --expected-executable-sha256 "$EQIORA_SITE_BROWSER_SHA256" --expected-executable-bytes "$EQIORA_SITE_BROWSER_BYTES"
python3 -m unittest tools.site.tests.test_site_tools -v
python3 tools/site/build_products.py
python3 tools/docs/generate_interface_reference.py --repository . --eqiora-binary bin/eqiora --check
python3 tools/site/build_rust_reference.py --rustdoc-root rustdoc/doc --output rustdoc-stage
python3 tools/site/check_site.py check
python3 tools/site/check_site.py serve
python3 tools/site/check_site.py source-topology --root "$EQIORA_SITE_SOURCE_ROOT"
"""


def _package_files(root: Path) -> None:
    dependencies = checker.RUNTIME_PINS
    dev_dependencies = checker.DEVELOPMENT_PINS
    package = {
        "engines": {"node": "24.18.1", "npm": "11.16.0"},
        "dependencies": dependencies,
        "devDependencies": dev_dependencies,
    }
    lock_packages = {
        "": {
            "dependencies": dependencies,
            "devDependencies": dev_dependencies,
            "engines": package["engines"],
        }
    }
    lock_packages.update(
        {
            f"node_modules/{name}": {
                "version": version,
                "integrity": "sha512-fixture",
            }
            for name, version in checker.DIRECT_PINS.items()
        }
    )
    lock = {"packages": lock_packages}
    _write(root / "docs/site/package.json", json.dumps(package))
    _write(root / "docs/site/package-lock.json", json.dumps(lock))


def _head(route: str) -> str:
    return f"""<head>
<meta charset="utf-8">
<meta property="og:image" content="https://eqiora.org/social-card.svg">
<link rel="canonical" href="https://eqiora.org{route}">
<link rel="icon" href="/favicon.svg">
<link rel="apple-touch-icon" href="/apple-touch-icon.png">
<link rel="stylesheet" href="/assets/site.css">
</head>"""


def _nav() -> str:
    return ('<a class="site-title" href="/"><img src="/assets/brand.svg" alt=""><span>Eqiora</span></a>'
            '<nav class="sidebar"><a href="/learn/">Learn</a><a href="/guides/">Guides</a>'
            '<a href="/gallery/">Gallery</a><a href="/reference/">Reference</a><a href="/get-started/">Get started</a>'
            '<a href="/release-notes/">Releases</a>'
            '<a href="/contributing/">Contributing</a>'
            '<a href="https://github.com/nkiyohara/eqiora">GitHub</a></nav>')


def _page(route: str, body: str) -> str:
    return f"<!doctype html><html>{_head(route)}<body><header>{_nav()}</header><main>{body}</main></body></html>"


def _exact_links() -> str:
    links = []
    for relative in checker.CASE_SOURCE_PATHS:
        url = f"https://github.com/nkiyohara/eqiora/blob/{SOURCE_SHA}/{relative}"
        label = Path(relative).parent.name + " " + Path(relative).name
        links.append(f'<a href="{url}">{label}</a>')
    return "".join(links)


def _case_body() -> str:
    # The synthetic artifact exercises both the current and legacy HTML-shape readers.
    return f"""<h1>Exact-cylinder steady Stokes</h1>
<p>Build a steady Stokes flow model and plot pressure.</p>
<section><h2>Problem setup</h2><p>2.2m x 0.41m channel; centre [0.2,0.2]m; radius 0.05m.</p>
<span class="katex"><span class="katex-mathml"><math><mi>H</mi></math></span><span class="katex-html">H</span></span></section>
<section><h2>Eqiora model definition</h2>
<div class="eqiora-math-region" role="region" aria-label="Displayed equation" tabindex="0"><span class="katex-display"><span class="katex"><span class="katex-mathml"><math display="block"><mi>sigma</mi></math></span><span class="katex-html">sigma</span></span></span></div>
<p>Eqiora source form</p><pre>sigma(u,p) = 2 mu sym(grad(u)) - p I
-div(sigma(u,p)) - grad(phi) = 0
div(u) = 0</pre></section>
<section><h2>Mesh and boundaries</h2></section>
<section><h2>Submit and result</h2><p>One immutable common Plan and direct Result carrier.</p><a href="https://github.com/nkiyohara/eqiora/blob/{SOURCE_SHA}/examples/python/exact_cylinder_stokes.py#L51-L67">Eqiora source form: Python resolve/run path</a></section>
<section><h2>Pressure visualization</h2><figure><img src="/assets/pressure.png" alt="{PRESSURE_ALT}"><figcaption>{checker.PRESSURE_CAPTION}</figcaption></figure></section>
<section><h2>Reading the pressure plot</h2>{_exact_links()}<a href="/learn/fluid-mechanics/">Continue with fluid mechanics</a></section>"""


def _home_body() -> str:
    return f"""<h1>Model meaning once. Realize it many ways.</h1>
<p>Eqiora is an open-source, meaning-first foundation for scientific modeling, simulation, differentiation, and execution.</p>
<p>Its central boundary is simple:</p>
<p>A model states typed mathematical relations. A realization chooses how those relations are discretized, solved, and executed.</p>
<p>That separation lets block diagrams, acausal physical networks, PDE fields, hybrid dynamics, and reusable components share one canonical meaning without making a numerical method or hardware backend part of the model.</p>
<a href="/get-started/">Get started</a><a href="/gallery/">Explore gallery</a>
<article><p>Featured walkthrough</p><h2>Kármán vortex street</h2><img src="/assets/wake.png" alt="{WAKE_ALT}"><p>Follow one transient cylinder flow from exact geometry through an accepted vorticity field.</p><p>Python</p><p>2D</p><p>transient Navier–Stokes</p><a href="/gallery/karman-vortex-street/">View the wake walkthrough</a></article>
<article><h2>Get started</h2><p>Learn the Model–Realization boundary and start from bounded examples.</p></article>
<article><h2>Textbooks</h2><p>Follow the planned path from mathematics and physics to Eqiora models, numerical realization, and interpretation.</p></article>
<article><h2>Reference</h2><p>Browse exact-commit Python, Rust, and CLI surfaces.</p></article>
<p>Docs explains how to use Eqiora. Textbooks teach the mathematics, physics, and numerics. Gallery presents complete simulations. Reference records exact APIs and protocols.</p>
<p>Release {{python_version}}</p><p>Eqiora is pre-1.0 research software under active development. The capability matrix and verification guide bound what is currently supported; this site does not widen those claims.</p>
<h2>One source of truth</h2><p>This website is a curated projection, not a parallel specification. Detailed contracts remain in the repository's architecture, RFCs, capability matrix, and validated verify manifests.</p>"""


def _artifact(root: Path, blobs: dict[str, bytes], python_version: str) -> Path:
    artifact = root / "artifact"
    _write(artifact / "social-card.svg", blobs["social"])
    _write(artifact / "favicon.svg", blobs["favicon"])
    _write(artifact / "apple-touch-icon.png", blobs["apple"])
    _write(artifact / "assets/pressure.png", blobs["pressure"])
    _write(artifact / "assets/wake.png", blobs["wake"])
    _write(artifact / "assets/brand.svg", blobs["favicon"])
    _write(artifact / "assets/KaTeX_Main-Regular.woff2", b"font")
    _write(
        artifact / "assets/site.css",
        "@font-face{font-family:KaTeX;src:url('./KaTeX_Main-Regular.woff2')}\n",
    )
    _write(
        artifact / "pagefind/pagefind.js",
        "export async function search(){return {results:[]}}\n",
    )

    pages = {
        "/": _home_body().format(python_version=python_version),
        "/contributing/architecture/": "<h1>Architecture</h1><p>Eqiora architecture.</p>",
        "/guides/how-eqiora-fits-together/": "<h1>Concepts</h1><p>Eqiora concepts.</p>",
        "/contributing/": "<h1>Contributing</h1><p>Contribute to Eqiora.</p>",
        "/evidence/": '<h1>Checking a claim</h1><h2>Find the source</h2><h2>Run a selected check</h2>',
        "/gallery/": '<h1>Gallery</h1><a href="/gallery/exact-cylinder-steady-stokes/">Exact-cylinder steady Stokes</a><a href="/gallery/mixed-boundary-elasticity/">Mixed-boundary linear elasticity</a>',
        "/gallery/exact-cylinder-steady-stokes/": _case_body(),
        "/gallery/mixed-boundary-elasticity/": "<h1>Mixed-boundary linear elasticity</h1><p>Static caller-owned displacement presentation.</p>",
        "/gallery/karman-vortex-street/": "<h1>Kármán vortex street</h1>",
        "/get-started/": "<h1>Get started</h1>",
        "/guides/": "<h1>Python</h1><p>Eqiora Python reference.</p>",
        "/guides/run-and-inspect/": "<h1>Run a model and inspect its result</h1><p>Installed release</p>",
        "/guides/differentiation/": "<h1>Differentiation</h1><p>View guide source</p><h2>Evaluate a point and its derivatives</h2>",
        "/guides/execution-and-arrays/": "<h1>Execution and arrays</h1><p>View guide source</p><h2>Structured failures</h2>",
        "/guides/modeling/": "<h1>Modeling</h1><p>View guide source</p><h2>Native declarations</h2>",
        "/reference/": '<h1>Reference</h1><p>Python Rust CLI</p><p>API presence is not verification or maturity.</p><form action="/reference/"><input aria-label="Search"></form>',
        "/reference/language/": "<h1>Language</h1>",
        "/reference/language/composition/": "<h1>Component composition</h1>",
        "/reference/language/declarations/": "<h1>Declarations</h1>",
        "/reference/language/equations/": "<h1>Equations and state</h1>",
        "/reference/language/units/": "<h1>Quantities and units</h1>",
        "/reference/standard-packages/": "<h1>Standard sources</h1>",
        "/reference/standard-packages/continuum/": "<h1>Continuum laws and boundaries</h1>",
        "/reference/standard-packages/electrical/": "<h1>Electrical components</h1>",
        "/reference/cli/": "<h1>CLI</h1><p>eqiora check</p>",
        "/reference/python/": "<h1>Python reference</h1><p>Python API families.</p>",
        "/reference/python/diff/": "<h1>Diff reference</h1><p>Differentiation API.</p>",
        "/reference/python/eqiora/": "<h1>eqiora Python module</h1><p>Diagnostic</p>",
        "/reference/python/fem/": "<h1>fem Python reference</h1>",
        "/reference/python/formulation/": "<h1>formulation Python reference</h1>",
        "/reference/python/fvm/": "<h1>fvm Python reference</h1>",
        "/reference/python/lang/": "<h1>lang Python reference</h1>",
        "/reference/python/solve/": "<h1>solve Python reference</h1>",
        "/reference/python/time/": "<h1>time Python reference</h1>",
        "/reference/python/viewer/": "<h1>viewer Python reference</h1>",
        "/reference/python/fluid/": "<h1>Fluid reference</h1><p>Fluid API.</p>",
        "/reference/python/fsi/": "<h1>FSI reference</h1><p>FSI API.</p>",
        "/reference/python/geometry/": "<h1>Geometry reference</h1><p>Geometry API.</p>",
        "/reference/python/jax/": "<h1>JAX reference</h1><p>JAX API.</p>",
        "/reference/python/matplotlib/": "<h1>Matplotlib reference</h1><p>Matplotlib API.</p>",
        "/reference/python/meshing/": "<h1>Meshing reference</h1><p>Meshing API.</p>",
        "/reference/python/solid/": "<h1>Solid reference</h1><p>Solid API.</p>",
        "/reference/python/torch/": "<h1>Torch reference</h1><p>Torch API.</p>",
        "/reference/python/trajectory/": "<h1>Trajectory reference</h1><p>Trajectory API.</p>",
        "/reference/rust/": '<h1>Rust reference</h1><p>eqiora::Diagnostic stable eqiora::api::CadBoxIntentV1 transitional eqiora::api module</p><a href="/reference/rust/api/eqiora/struct.Diagnostic.html">Diagnostic</a>',
        "/release-notes/": "<h1>Release notes</h1><p>Eqiora release notes.</p>",
        "/learn/": '<h1>Learn</h1><h2>Start a learning path</h2><h2>Browse by topic</h2><a href="/learn/mathematical-modeling/">Mathematical modeling</a><a href="/gallery/exact-cylinder-steady-stokes/">Steady flow investigation</a>',
        "/learn/mathematical-modeling/": "<h1>Mathematical Modeling with Eqiora</h1><h2>Build a model</h2><h2>Extend it through space</h2><a href=\"/learn/mathematical-modeling/models-not-simulations/\">Models are not simulations</a><a href=\"/learn/mathematical-modeling/quantities-dimensions-units/\">Quantities, dimensions, and units</a><a href=\"/learn/mathematical-modeling/algebraic-relations-networks/\">Algebraic relations and networks</a><a href=\"/learn/mathematical-modeling/ordinary-differential-equations/\">Ordinary differential equations</a><a href=\"/learn/mathematical-modeling/fields-spatial-domains/\">Fields and spatial domains</a><a href=\"/learn/mathematical-modeling/conservation-laws/\">Conservation laws</a><a href=\"/learn/mathematical-modeling/constitutive-laws/\">Constitutive laws</a><a href=\"/learn/mathematical-modeling/boundary-interface-conditions/\">Boundary and interface conditions</a><h2>Put it to use</h2>",
        "/learn/mathematical-modeling/algebraic-relations-networks/": "<h1>Algebraic relations and networks</h1><h2>Learning outcomes</h2><h2>Read the result</h2><h2>Deliberate failure</h2><h2>Exercises</h2><a href=\"/learn/mathematical-modeling/\">Back to the series map</a>",
        "/learn/mathematical-modeling/boundary-interface-conditions/": "<h1>Boundary and interface conditions</h1><h2>Learning outcomes</h2><h2>Deliberate failure</h2><h2>Exercises</h2><a href=\"/learn/mathematical-modeling/\">Back to the series map</a>",
        "/learn/mathematical-modeling/conservation-laws/": "<h1>Conservation laws</h1><h2>Learning outcomes</h2><h2>Deliberate failure</h2><h2>Exercises</h2><a href=\"/learn/mathematical-modeling/\">Back to the series map</a>",
        "/learn/mathematical-modeling/constitutive-laws/": "<h1>Constitutive laws</h1><h2>Learning outcomes</h2><h2>Deliberate failure</h2><h2>Exercises</h2><a href=\"/learn/mathematical-modeling/\">Back to the series map</a>",
        "/learn/mathematical-modeling/fields-spatial-domains/": "<h1>Fields and spatial domains</h1><h2>Learning outcomes</h2><h2>Deliberate failure</h2><h2>Exercises</h2><a href=\"/learn/mathematical-modeling/\">Back to the series map</a>",
        "/learn/mathematical-modeling/models-not-simulations/": "<h1>Models are not simulations</h1><h2>Learning outcomes</h2><h2>Read the result</h2><h2>Deliberate failure</h2><h2>Exercises</h2><a href=\"/learn/mathematical-modeling/\">Back to the series map</a>",
        "/learn/mathematical-modeling/ordinary-differential-equations/": "<h1>Ordinary differential equations</h1><p>Compare observations with the independently derived closed form</p><h2>Learning outcomes</h2><h2>Read the result</h2><h2>Deliberate failure</h2><h2>Exercises</h2><a href=\"/learn/mathematical-modeling/\">Back to the series map</a>",
        "/learn/mathematical-modeling/quantities-dimensions-units/": "<h1>Quantities, dimensions, and units</h1><h2>Learning outcomes</h2><h2>Read the result</h2><h2>Deliberate failure</h2><h2>Exercises</h2><a href=\"/learn/mathematical-modeling/\">Back to the series map</a>",
        '/reference/standard-packages/controls/': '<h1>Control components</h1>',
        '/reference/python/units/': '<h1>Units</h1>',
        '/reference/python/colab/': '<h1>Colab</h1>',
        '/learn/circuits-dynamics/component-laws/': '<h1>4. Open the components</h1>',
        '/learn/circuits-dynamics/conserving-networks/': '<h1>2. Conserving networks</h1>',
        '/learn/circuits-dynamics/decay-and-storage/': '<h1>5. Storage and decay</h1>',
        '/learn/circuits-dynamics/': '<h1>Circuits and dynamics</h1>',
        '/learn/circuits-dynamics/resistor-divider/': '<h1>3. Build and run a divider</h1>',
        '/learn/circuits-dynamics/voltage-current-power/': '<h1>1. Voltage, current, and power</h1>',
        '/learn/fluid-mechanics/computing-incompressible-flow/': '<h1>5. Computing incompressible flow</h1>',
        '/learn/fluid-mechanics/continuum-and-statics/': '<h1>1. Continuum and statics</h1>',
        '/learn/fluid-mechanics/cylinder-wakes/': '<h1>6. Cylinder wakes</h1>',
        '/learn/fluid-mechanics/exact-parallel-flows/': '<h1>3. Exact parallel flows</h1>',
        '/learn/fluid-mechanics/': '<h1>Fluid mechanics and CFD</h1>',
        '/learn/fluid-mechanics/mass-and-momentum/': '<h1>2. Mass, momentum and stress</h1>',
        '/learn/fluid-mechanics/references/': '<h1>References and further study</h1>',
        '/learn/fluid-mechanics/stokes-and-boundaries/': '<h1>4. Stokes flow and boundaries</h1>',
        '/learn/heat-transfer/assessing-a-thermal-calculation/': '<h1>6. Assessing a thermal calculation</h1>',
        '/learn/heat-transfer/boundaries-and-interfaces/': '<h1>3. Boundaries and interfaces</h1>',
        '/learn/heat-transfer/conservation-and-fourier-law/': "<h1>1. Conservation and Fourier's law</h1>",
        '/learn/heat-transfer/heated-body-in-eqiora/': '<h1>4. A heated body in Eqiora</h1>',
        '/learn/heat-transfer/': '<h1>Heat transfer</h1>',
        '/learn/heat-transfer/steady-conduction/': '<h1>2. Steady conduction</h1>',
        '/learn/heat-transfer/transient-storage/': '<h1>5. Transient storage</h1>',
        '/learn/inverse-problems/differentiating-a-solved-model/': '<h1>4. Differentiating a solved model</h1>',
        '/learn/inverse-problems/heat-source-recovery/': '<h1>2. Recovering a heat source</h1>',
        '/learn/inverse-problems/': '<h1>Inverse problems and differentiable design</h1>',
        '/learn/inverse-problems/measurements-to-parameters/': '<h1>1. Measurements to parameters</h1>',
        '/learn/inverse-problems/noise-scaling-and-design/': '<h1>5. Noise, scaling, and design</h1>',
        '/learn/inverse-problems/sensitivity-and-identifiability/': '<h1>3. Sensitivity and identifiability</h1>',
        '/learn/numerical-simulation/errors-and-residuals/': '<h1>1. Errors and residuals</h1>',
        '/learn/numerical-simulation/finite-volume-balance/': '<h1>4. Finite-volume balance</h1>',
        '/learn/numerical-simulation/': '<h1>Numerical simulation</h1>',
        '/learn/numerical-simulation/refinement-and-reproducibility/': '<h1>5. Refinement and reproducibility</h1>',
        '/learn/numerical-simulation/time-integration/': '<h1>2. Time integration</h1>',
        '/learn/numerical-simulation/weak-forms-and-finite-elements/': '<h1>3. Weak forms and finite elements</h1>',
        '/learn/structural-mechanics/': '<h1>Structural mechanics and finite elements</h1>',
        '/learn/structural-mechanics/interpreting-results/': '<h1>5. Reading displacement, reactions and error</h1>',
        '/learn/structural-mechanics/loads-energy/': '<h1>1. Loads, supports and elastic energy</h1>',
        '/learn/structural-mechanics/mixed-boundary-square/': '<h1>4. A square with mixed boundaries</h1>',
        '/learn/structural-mechanics/strain-stress/': '<h1>2. Displacement, strain and stress</h1>',
        '/learn/structural-mechanics/virtual-work/': '<h1>3. Virtual work and finite elements</h1>',
        "/404.html": "<h1>Page not found</h1>",
    }
    assert tuple(pages) == SITE_ROUTES
    for route, body in pages.items():
        if route == "/":
            relative = Path("index.html")
        elif route == "/404.html":
            relative = Path("404.html")
        else:
            relative = Path(route.removeprefix("/")) / "index.html"
        canonical = "/404/" if route == "/404.html" else route
        _write(artifact / relative, _page(canonical, body))
    _write(
        artifact / "reference/rust/api/eqiora/struct.Diagnostic.html",
        '<!doctype html><html><body><main><h1>Struct Diagnostic</h1><a href="index.html">eqiora</a></main></body></html>',
    )
    _write(
        artifact / "reference/rust/api/eqiora/index.html",
        '<!doctype html><html><body><main><h1>Crate eqiora</h1><a href="struct.Diagnostic.html">Diagnostic</a></main></body></html>',
    )
    urls = "".join(
        f"<url><loc>https://eqiora.org{route}</loc></url>" for route in SITE_ROUTES[:-1]
    )
    _write(
        artifact / "sitemap-index.xml", f'<?xml version="1.0"?><urlset>{urls}</urlset>'
    )
    _write(
        artifact / "robots.txt",
        "User-agent: *\nAllow: /\nSitemap: https://eqiora.org/sitemap-index.xml\n",
    )
    return artifact


def make_fixture(root: Path, cargo_version: str = "0.1.0"):
    python_version = cargo_version
    blobs = {
        "wake": b"fixture admitted wake poster",
        "pressure": b"fixture admitted pressure",
        "social": b"<svg><title>Eqiora</title></svg>\n",
        "favicon": b"<svg><title>Eqiora mark</title></svg>\n",
        "apple": b"fixture apple png",
    }
    _package_files(root)
    _write(
        root / "Cargo.toml",
        f'[workspace]\nmembers = []\n[workspace.package]\nversion = "{cargo_version}"\n',
    )
    _write(
        root / "tools/release/python_candidate_common.py",
        "def python_distribution_version(version):\n"
        "    return version\n",
    )
    _write(
        root / "tools/ci/classify_changes.py",
        (REPOSITORY / "tools/ci/classify_changes.py").read_text(encoding="utf-8"),
    )
    _write(root / ".github/workflows/pages.yml", _workflow())
    _write(root / "tools/site/run_offline_site_checks.sh", _runner())
    _write(
        root / "docs/site/src/components/site/ExactSourceLink.astro",
        "---\nconst source = import.meta.env.EQIORA_SITE_SOURCE_SHA;\n"
        "const kinds = ['blob', 'tree'];\n---\n<a href={source}>{Astro.slots}</a>\n",
    )
    _write(
        root / "docs/site/src/components/site/ReleaseIdentity.astro",
        "---\nconst cargo = import.meta.env.EQIORA_SITE_CARGO_VERSION;\n"
        "const python = import.meta.env.EQIORA_SITE_PYTHON_VERSION;\n---\n"
        "<span>{cargo}{python}</span>\n",
    )
    _write(
        root / "docs/site/src/content/docs/index.mdx",
        "import ExactSourceLink from '@components/site/ExactSourceLink.astro';\n"
        "import ReleaseIdentity from '@components/site/ReleaseIdentity.astro';\n"
        '<ExactSourceLink path="AGENTS.md" kind="blob">AGENTS</ExactSourceLink>\n'
        "<ReleaseIdentity />\n",
    )
    _write(
        root / "docs/site/src/content/docs/evidence/index.mdx",
        "import ExactSourceLink from '@components/site/ExactSourceLink.astro';\n"
        '<ExactSourceLink path="verify/example/case.toml" kind="blob">case</ExactSourceLink>\n',
    )
    _write(
        root / "docs/site/astro.config.mjs",
        "const required = [\n"
        "  'src/components/site/ExactSourceLink.astro',\n"
        "  'src/components/site/ReleaseIdentity.astro',\n"
        "];\n",
    )
    _write(
        root / "docs/site/src/assets/gallery/karman-vortex-street-poster.png",
        blobs["wake"],
    )
    _write(
        root / "docs/site/src/assets/gallery/exact-cylinder-pressure-presentation.png",
        blobs["pressure"],
    )
    _write(root / "docs/site/public/social-card.svg", blobs["social"])
    _write(root / "docs/site/public/favicon.svg", blobs["favicon"])
    _write(root / "docs/site/public/apple-touch-icon.png", blobs["apple"])
    identities = checker.SiteIdentities(
        wake=checker.sha256(
            root / "docs/site/src/assets/gallery/karman-vortex-street-poster.png"
        ),
        pressure=checker.sha256(
            root / "docs/site/src/assets/gallery/exact-cylinder-pressure-presentation.png"
        ),
        social=checker.sha256(root / "docs/site/public/social-card.svg"),
        favicon=checker.sha256(root / "docs/site/public/favicon.svg"),
        apple_touch=checker.sha256(root / "docs/site/public/apple-touch-icon.png"),
    )
    artifact = _artifact(root, blobs, python_version)
    return artifact, identities
