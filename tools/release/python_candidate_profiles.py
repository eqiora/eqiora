"""Isolated validation profiles for one immutable Python candidate family."""

from __future__ import annotations

import os
import shutil
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

from python_candidate_common import (
    CandidateError,
    DistributionConfig,
    checked_run,
)


ROOT = Path(__file__).resolve().parents[2]
CI_ROOT = ROOT / "tools/ci"
if str(CI_ROOT) not in sys.path:
    sys.path.insert(0, str(CI_ROOT))

from resource_scheduler import (  # noqa: E402
    ResourceBudget,
    ResourceRequest,
    ScheduledTask,
    TaskOutcome,
    run_tasks,
)


EXACT_CYLINDER_DEMO = Path("examples/python/exact_cylinder_stokes.py")
EXACT_CYLINDER_REPOSITORY_SOURCE = Path("examples/steady-flow-past-cylinder.eqi")
MIXED_BOUNDARY_ELASTICITY_DEMO = Path("examples/python/mixed_boundary_elasticity.py")
MIXED_BOUNDARY_REPOSITORY_SOURCE = Path(
    "verify/solid/mixed-boundary-elasticity-2d/models/direct.eqi"
)
PYTHON_TEST_FIXTURES = (
    Path("examples/property-composition"),
    Path("examples/heated-body"),
    Path("verify/interfaces/control-plane-compile-check"),
    Path("verify/interfaces/current-authoring-profile"),
    Path("crates/eqiora-api/packages/org.example.poisson"),
    Path("verify/packages/offline-model-package"),
    Path("verify/interfaces/python-package-conformance"),
    Path(
        "verify/interfaces/python-offline-model-package/models/typed-execution-lineage"
    ),
)
PYTHON_TEST_RESOURCES = (
    Path("crates/eqiora-api/schemas/compile-v2.schema.json"),
    Path("docs/site/src/content/docs/reference/language/index.mdx"),
    Path("docs/site/src/content/docs/reference/language/declarations.mdx"),
    Path("docs/site/src/content/docs/reference/language/units.mdx"),
    Path("docs/site/src/content/docs/reference/language/equations.mdx"),
    Path("docs/site/src/content/docs/reference/language/composition.mdx"),
    Path("docs/site/src/content/docs/reference/language/_examples/declarations.eqi"),
    Path("docs/site/src/content/docs/reference/language/_examples/units.eqi"),
    Path("docs/site/src/content/docs/reference/language/_examples/clocked.eqi"),
    Path("docs/site/src/content/docs/reference/language/_examples/composition.eqi"),
    Path("docs/site/src/content/docs/reference/standard-packages/index.mdx"),
    Path("docs/site/src/content/docs/reference/standard-packages/electrical.mdx"),
    Path("docs/site/src/content/docs/reference/standard-packages/continuum.mdx"),
    Path("docs/site/src/content/docs/reference/standard-packages/controls.mdx"),
    Path("docs/site/src/content/docs/reference/standard-packages/_examples/electrical.eqi"),
    Path("docs/site/src/content/docs/reference/standard-packages/_examples/elastic-body.eqi"),
    Path("docs/site/src/content/docs/reference/standard-packages/_examples/inertial-stokes.eqi"),
    Path("examples/decay.eqi"),
    Path("examples/standard-sampled-components/src/main.eqi"),
    Path("examples/python/coupled_scalar.py"),
    Path("examples/python/textbook_decay.py"),
    Path("crates/eqiora-api/packages/Eqiora.Electrical.Basic/src/basic.eqi"),
    Path("crates/eqiora-api/packages/Eqiora.Controls.Sampled/src/sampled.eqi"),
    Path("crates/eqiora-api/packages/Eqiora.Solid.LinearElasticity/src/linear_elasticity.eqi"),
    Path("crates/eqiora-api/packages/Eqiora.Fluid.InertialStokes/src/inertial_stokes.eqi"),
)

COMPLETE_PROFILE_NAMES = (
    "base-3.11",
    "base-3.12",
    "base-3.13",
    "base-3.14",
    "numpy-floor-3.12",
    "generated-public-api",
    "torch-3.13",
    "jax-3.13",
    "matplotlib-3.13",
    "typing-3.13",
)

DEVELOPMENT_PROFILE_NAMES = COMPLETE_PROFILE_NAMES[:6]

_HEAVY = ResourceRequest(1, 3 * 1024, locks=("python-heavy-profile",))
_BASE = ResourceRequest(1, 1024)
_MATPLOTLIB = ResourceRequest(1, 1024)
_DOCS = ResourceRequest(1, 256)


@dataclass(frozen=True)
class ProfileWorkspace:
    name: str
    resources: ResourceRequest
    root: Path
    environment: Path
    consumer: Path
    temporary: Path
    log: Path
    matplotlib_config: Path | None
    environment_variables: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class ProfileReceipt:
    name: str
    checks: tuple[str, ...]
    dependency_profiles: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]
    diagnostics: tuple[str, ...]
    log: str


@dataclass(frozen=True)
class MergedProfileReceipts:
    receipts: tuple[ProfileReceipt, ...]
    checks: tuple[str, ...]
    dependency_profiles: tuple[tuple[str, tuple[tuple[str, str], ...]], ...]
    diagnostics: tuple[tuple[str, str], ...]
    logs: tuple[tuple[str, str], ...]


def _profile_environment(
    name: str,
    temporary: Path,
    matplotlib_config: Path | None,
    config: DistributionConfig,
) -> tuple[tuple[str, str], ...]:
    environment = {"TMPDIR": str(temporary)}
    if name == "torch-3.13":
        environment["EQIORA_TEST_TORCH_VERSION"] = config.torch.split("==", maxsplit=1)[
            1
        ]
    elif name == "jax-3.13":
        environment.update(
            {
                "EQIORA_REQUIRE_JAX_ABI_PROBE": "1",
                "EQIORA_TEST_JAX_VERSION": config.jax[0].split("==", maxsplit=1)[1],
                "EQIORA_TEST_PYTHON_VERSION": config.extras_interpreter,
                "JAX_ENABLE_X64": "1",
                "XLA_FLAGS": "--xla_force_host_platform_device_count=2",
            }
        )
    elif name == "matplotlib-3.13":
        assert matplotlib_config is not None
        environment.update(
            {
                "EQIORA_TEST_MATPLOTLIB_VERSION": config.matplotlib.split(
                    "==", maxsplit=1
                )[1],
                "MPLBACKEND": "Agg",
                "MPLCONFIGDIR": str(matplotlib_config),
            }
        )
    return tuple(sorted(environment.items()))


def _request(name: str) -> ResourceRequest:
    if name in {"torch-3.13", "jax-3.13", "typing-3.13"}:
        return _HEAVY
    if name == "matplotlib-3.13":
        return _MATPLOTLIB
    if name == "generated-public-api":
        return _DOCS
    return _BASE


def build_profile_plan(
    scratch: Path, config: DistributionConfig, *, skip_extras: bool
) -> tuple[ProfileWorkspace, ...]:
    """Describe every profile-owned writable path before work is submitted."""

    names = DEVELOPMENT_PROFILE_NAMES if skip_extras else COMPLETE_PROFILE_NAMES
    workspaces: list[ProfileWorkspace] = []
    roots: set[Path] = set()
    for name in names:
        root = scratch / "profiles" / name
        if root in roots:  # pragma: no cover - frozen names are unique
            raise CandidateError(f"duplicate profile writable root: {root}")
        roots.add(root)
        matplotlib_config = (
            root / "matplotlib-config" if name == "matplotlib-3.13" else None
        )
        temporary = root / "tmp"
        workspaces.append(
            ProfileWorkspace(
                name=name,
                resources=_request(name),
                root=root,
                environment=root / "environment",
                consumer=root / "consumer",
                temporary=temporary,
                log=root / "profile.log",
                matplotlib_config=matplotlib_config,
                environment_variables=_profile_environment(
                    name, temporary, matplotlib_config, config
                ),
            )
        )
    if config.interpreters != ("3.11", "3.12", "3.13", "3.14"):
        raise CandidateError("candidate profile plan requires CPython 3.11-3.14")
    return tuple(workspaces)


def profile_budget() -> ResourceBudget:
    """Use an enclosing lane budget, with the admitted direct-run default."""

    def value(name: str, default: int) -> int:
        raw = os.environ.get(name)
        if raw is None:
            return default
        try:
            return int(raw)
        except ValueError as error:
            raise CandidateError(f"{name} must be an integer") from error

    return ResourceBudget(
        value("EQIORA_PYTHON_CANDIDATE_CPU_SLOTS", 2),
        value("EQIORA_PYTHON_CANDIDATE_MEMORY_MIB", 4096),
        value("EQIORA_PYTHON_CANDIDATE_GPU_SLOTS", 0),
    )


def scheduled_profile_tasks(
    plan: Sequence[ProfileWorkspace],
    execute: Callable[[ProfileWorkspace], ProfileReceipt],
) -> tuple[ScheduledTask[ProfileReceipt], ...]:
    """Prioritize two base witnesses, then interleave heavy and light work."""

    by_name = {item.name: item for item in plan}
    priority = (
        "base-3.11",
        "base-3.12",
        "torch-3.13",
        "base-3.13",
        "jax-3.13",
        "base-3.14",
        "typing-3.13",
        "numpy-floor-3.12",
        "matplotlib-3.13",
        "generated-public-api",
    )
    scheduled = tuple(
        ScheduledTask(
            item.name,
            item.resources,
            lambda item=item: execute(item),
        )
        for name in priority
        if (item := by_name.get(name)) is not None
    )
    if len(scheduled) != len(plan):
        raise CandidateError(
            "candidate profile plan contains a duplicate or unscheduled identity"
        )
    return scheduled


def run_profile_tasks(
    plan: Sequence[ProfileWorkspace],
    execute: Callable[[ProfileWorkspace], ProfileReceipt],
    *,
    budget: ResourceBudget | None = None,
) -> tuple[TaskOutcome[ProfileReceipt], ...]:
    """Run one immutable profile plan under the enclosing resource budget."""

    try:
        return run_tasks(
            scheduled_profile_tasks(plan, execute),
            profile_budget() if budget is None else budget,
        )
    except ValueError as error:
        raise CandidateError(
            f"candidate profile schedule is invalid: {error}"
        ) from error


def merge_profile_receipts(
    expected_names: Sequence[str], receipts: Sequence[ProfileReceipt]
) -> MergedProfileReceipts:
    """Validate exact receipt ownership and merge in frozen profile order."""

    expected = tuple(expected_names)
    by_name: dict[str, ProfileReceipt] = {}
    for receipt in receipts:
        if receipt.name in by_name or receipt.name not in expected:
            raise ValueError("profile receipt identity is duplicate or unexpected")
        by_name[receipt.name] = receipt
    if set(by_name) != set(expected) or len(receipts) != len(expected):
        raise ValueError("profile receipt inventory differs from the selected plan")
    ordered = tuple(by_name[name] for name in expected)

    dependencies: dict[str, tuple[tuple[str, str], ...]] = {}
    for receipt in ordered:
        for name, values in receipt.dependency_profiles:
            if name in dependencies:
                raise ValueError("profile receipt repeats a dependency profile")
            dependencies[name] = values
    return MergedProfileReceipts(
        receipts=ordered,
        checks=tuple(check for receipt in ordered for check in receipt.checks),
        dependency_profiles=tuple(dependencies.items()),
        diagnostics=tuple(
            (receipt.name, diagnostic)
            for receipt in ordered
            for diagnostic in receipt.diagnostics
        ),
        logs=tuple((receipt.name, receipt.log) for receipt in ordered),
    )


def venv_python(environment: Path) -> Path:
    return environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def install_environment(
    *,
    uv: str,
    interpreter: str,
    environment: Path,
    requirements: list[str],
    run: Callable[..., str] = checked_run,
) -> Path:
    run([uv, "venv", "--python", interpreter, str(environment)])
    python = venv_python(environment)
    run(
        [uv, "pip", "install", "--python", str(python), *requirements],
        cwd=environment.parent,
    )
    return python


def assert_installed_origin(
    python: Path,
    wheel: Path,
    run_root: Path,
    expected_version: str,
    *,
    run: Callable[..., str] = checked_run,
) -> None:
    script = """
import importlib.metadata as metadata
from pathlib import Path
import eqiora

root = Path.cwd().resolve()
module = Path(eqiora.__file__).resolve()
assert root not in module.parents, (root, module)
distribution = metadata.distribution("eqiora")
assert distribution.version == __import__("os").environ["EQIORA_EXPECTED_VERSION"]
assert eqiora.__version__ == distribution.version
files = {str(path) for path in distribution.files or ()}
assert "eqiora/py.typed" in files
assert "eqiora/__init__.pyi" in files
assert "eqiora/fsi.pyi" in files
assert "eqiora/examples/fixed-reference-fsi.eqi" in files
assert "eqiora/examples/transient-flow-past-cylinder.eqi" in files
for retired in (
    "FixedReferenceFsiStep",
    "FixedReferenceFsiResult",
    "solve_fixed_reference_fsi",
):
    assert not hasattr(eqiora.fsi, retired)
    assert retired not in eqiora.fsi.__all__
for optional in ("torch", "jax", "jaxlib", "matplotlib", "gmsh"):
    assert optional not in __import__("sys").modules
"""
    run(
        [str(python), "-I", "-c", script],
        cwd=run_root,
        extra_environment={"EQIORA_EXPECTED_VERSION": expected_version},
    )


def prepare_base_consumer_tree(extracted: Path, run_root: Path) -> tuple[Path, Path]:
    tests = run_root / "bindings/python/tests"
    typecheck = run_root / "bindings/python/typecheck"
    shutil.copytree(extracted / "bindings/python/tests", tests)
    shutil.copytree(extracted / "bindings/python/typecheck", typecheck)
    for relative in PYTHON_TEST_FIXTURES:
        shutil.copytree(extracted / relative, run_root / relative)
    for relative in PYTHON_TEST_RESOURCES:
        destination = run_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(extracted / relative, destination)
    return tests, typecheck


def _copy_demo(
    extracted: Path,
    run_root: Path,
    source: Path,
    forbidden_repository_source: Path,
    description: str,
) -> Path:
    authored = extracted / source
    if not authored.is_file():
        raise CandidateError(f"source distribution omits checked-in demo {source}")
    forbidden = run_root / forbidden_repository_source
    if forbidden.exists():
        raise CandidateError(f"{description} consumer tree carries repository source")
    destination = run_root / source
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(authored, destination)
    if forbidden.exists():  # pragma: no cover
        raise CandidateError(f"{description} demo copy introduced repository source")
    return destination


def prepare_exact_cylinder_demo_consumer(extracted: Path, run_root: Path) -> Path:
    return _copy_demo(
        extracted,
        run_root,
        EXACT_CYLINDER_DEMO,
        EXACT_CYLINDER_REPOSITORY_SOURCE,
        "exact-cylinder",
    )


def prepare_mixed_boundary_elasticity_demo_consumer(
    extracted: Path, run_root: Path
) -> Path:
    return _copy_demo(
        extracted,
        run_root,
        MIXED_BOUNDARY_ELASTICITY_DEMO,
        MIXED_BOUNDARY_REPOSITORY_SOURCE,
        "mixed-boundary",
    )


def assert_matplotlib_is_optional(
    python: Path,
    run_root: Path,
    *,
    run: Callable[..., str] = checked_run,
) -> None:
    script = """
import importlib.util
import sys
assert importlib.util.find_spec("matplotlib") is None
import eqiora
assert "matplotlib" not in sys.modules
try:
    import eqiora.matplotlib
except ImportError as error:
    assert str(error) == ("eqiora.matplotlib requires the optional 'matplotlib' dependency; install eqiora[matplotlib]")
else:
    raise AssertionError("the base environment unexpectedly imported Matplotlib")
"""
    run([str(python), "-I", "-c", script], cwd=run_root)


def run_public_smoke(
    *,
    python: Path,
    extracted: Path,
    run_root: Path,
    expected_version: str,
    profile: str,
    run: Callable[..., str] = checked_run,
) -> None:
    run(
        [
            str(python),
            "-I",
            str(extracted / "tools/release/python_public_smoke.py"),
            "--expected-version",
            expected_version,
            "--profile",
            profile,
        ],
        cwd=run_root,
    )


def run_base_profile(
    *,
    uv: str,
    interpreter: str,
    python_version: str,
    wheel: Path,
    extracted: Path,
    workspace: ProfileWorkspace,
    config: DistributionConfig,
    run: Callable[..., str] = checked_run,
) -> list[str]:
    python = install_environment(
        uv=uv,
        interpreter=interpreter,
        environment=workspace.environment,
        requirements=[str(wheel), config.pytest, config.mypy],
        run=run,
    )
    workspace.consumer.mkdir(parents=True)
    tests, typecheck = prepare_base_consumer_tree(extracted, workspace.consumer)
    prepare_exact_cylinder_demo_consumer(extracted, workspace.consumer)
    prepare_mixed_boundary_elasticity_demo_consumer(extracted, workspace.consumer)
    assert_installed_origin(
        python, wheel, workspace.consumer, config.python_version, run=run
    )
    assert_matplotlib_is_optional(python, workspace.consumer, run=run)
    gmsh_tests = tuple(
        tests / name
        for name in (
            "test_gmsh_meshing.py",
            "test_exact_cylinder_stokes_result.py",
        )
    )
    run(
        [
            str(python),
            "-I",
            "-m",
            "pytest",
            "-q",
            str(tests),
            *(argument for test in gmsh_tests for argument in ("--ignore", str(test))),
        ],
        cwd=workspace.consumer,
    )
    run(
        [str(python), "-I", "-m", "mypy", "--strict", str(typecheck / "base.py")],
        cwd=workspace.consumer,
    )
    run_public_smoke(
        python=python,
        extracted=extracted,
        run_root=workspace.consumer,
        expected_version=config.python_version,
        profile="base",
        run=run,
    )
    run(
        [
            uv,
            "pip",
            "install",
            "--python",
            str(python),
            f"{wheel}[gmsh]",
        ],
        cwd=workspace.environment.parent,
    )
    gmsh_path = str(python.parent)
    if inherited_path := os.environ.get("PATH"):
        gmsh_path = os.pathsep.join((gmsh_path, inherited_path))
    run(
        [
            str(python),
            "-I",
            "-m",
            "pytest",
            "-q",
            *(str(test) for test in gmsh_tests),
        ],
        cwd=workspace.consumer,
        extra_environment={
            "EQIORA_GMSH": str(
                python.parent / ("gmsh.exe" if os.name == "nt" else "gmsh")
            ),
            "PATH": gmsh_path,
        },
    )
    compact = python_version.replace(".", "")
    return [
        f"cp{compact}:installed-wheel",
        f"cp{compact}:base-and-numpy",
        f"cp{compact}:packaged-exact-cylinder-model-demo",
        f"cp{compact}:packaged-mixed-boundary-elasticity-demo",
        f"cp{compact}:async-and-cancellation",
        f"cp{compact}:strict-base-typing",
        f"cp{compact}:public-smoke-base",
        f"cp{compact}:matplotlib-free-base",
    ]


def run_optional_profile(
    *,
    name: str,
    uv: str,
    interpreter: str,
    wheel: Path,
    extracted: Path,
    workspace: ProfileWorkspace,
    config: DistributionConfig,
    run: Callable[..., str] = checked_run,
) -> list[str]:
    if name == "torch":
        exact, test = [config.torch], "test_torch.py"
    elif name == "jax":
        exact, test = list(config.jax), "test_jax.py"
    elif name == "matplotlib":
        exact, test = [config.matplotlib], "test_matplotlib.py"
    else:  # pragma: no cover
        raise CandidateError(f"unknown optional profile: {name}")
    python = install_environment(
        uv=uv,
        interpreter=interpreter,
        environment=workspace.environment,
        requirements=[
            f"{wheel}[gmsh,{name}]" if name == "matplotlib" else f"{wheel}[{name}]",
            config.pytest,
            *exact,
        ],
        run=run,
    )
    workspace.consumer.mkdir(parents=True)
    test_path = workspace.consumer / test
    shutil.copy2(extracted / f"bindings/python/tests/{test}", test_path)
    if name != "matplotlib":
        run(
            [str(python), "-I", "-m", "pytest", "-q", str(test_path)],
            cwd=workspace.consumer,
        )
        run_public_smoke(
            python=python,
            extracted=extracted,
            run_root=workspace.consumer,
            expected_version=config.python_version,
            profile=name,
            run=run,
        )
        compact = config.extras_interpreter.replace(".", "")
        return [f"cp{compact}:{name}", f"cp{compact}:public-smoke-{name}"]

    gmsh_path = str(python.parent)
    if inherited_path := os.environ.get("PATH"):
        gmsh_path = os.pathsep.join((gmsh_path, inherited_path))
    gmsh_environment = {
        "EQIORA_GMSH": str(python.parent / ("gmsh.exe" if os.name == "nt" else "gmsh")),
        "PATH": gmsh_path,
    }
    run(
        [str(python), "-I", "-m", "pytest", "-q", str(test_path)],
        cwd=workspace.consumer,
        extra_environment=gmsh_environment,
    )
    destinations = (
        (
            prepare_exact_cylinder_demo_consumer(extracted, workspace.consumer),
            "exact-cylinder-pressure.png",
            ["--pressure-png"],
            "installed exact-cylinder Matplotlib demo",
        ),
        (
            prepare_mixed_boundary_elasticity_demo_consumer(
                extracted, workspace.consumer
            ),
            "mixed-boundary-displacement.png",
            ["--displacement-png", "{destination}", "--scale", "1"],
            "installed mixed-boundary Matplotlib demo",
        ),
    )
    for demo, filename, arguments, description in destinations:
        destination = workspace.consumer / filename
        rendered = [
            str(destination) if value == "{destination}" else value
            for value in arguments
        ]
        if rendered == ["--pressure-png"]:
            rendered.append(str(destination))
        run(
            [str(python), "-I", str(demo), *rendered],
            cwd=workspace.consumer,
            extra_environment=gmsh_environment,
        )
        if not destination.is_file() or not destination.read_bytes().startswith(
            b"\x89PNG\r\n\x1a\n"
        ):
            raise CandidateError(f"{description} did not write a PNG")
    compact = config.extras_interpreter.replace(".", "")
    return [
        f"cp{compact}:matplotlib",
        f"cp{compact}:packaged-exact-cylinder-pressure-demo",
        f"cp{compact}:packaged-mixed-boundary-displacement-demo",
    ]


def run_numpy_floor_profile(
    *,
    uv: str,
    interpreter: str,
    wheel: Path,
    extracted: Path,
    workspace: ProfileWorkspace,
    config: DistributionConfig,
    run: Callable[..., str] = checked_run,
) -> tuple[list[str], dict[str, str]]:
    python = install_environment(
        uv=uv,
        interpreter=interpreter,
        environment=workspace.environment,
        requirements=[str(wheel), config.numpy_floor, config.pytest],
        run=run,
    )
    workspace.consumer.mkdir(parents=True)
    test_path = workspace.consumer / "test_array_transport.py"
    shutil.copy2(extracted / "bindings/python/tests/test_array_transport.py", test_path)
    run(
        [str(python), "-I", "-m", "pytest", "-q", str(test_path)],
        cwd=workspace.consumer,
    )
    run_public_smoke(
        python=python,
        extracted=extracted,
        run_root=workspace.consumer,
        expected_version=config.python_version,
        profile="base",
        run=run,
    )
    observed = run(
        [
            str(python),
            "-I",
            "-c",
            "import importlib.metadata as m; print(m.version('numpy'))",
        ],
        cwd=workspace.consumer,
        capture=True,
    )
    expected = config.numpy_floor.split("==", maxsplit=1)[1]
    if observed != expected:
        raise CandidateError(
            f"NumPy floor profile expected {expected}, observed {observed!r}"
        )
    compact = config.numpy_floor_interpreter.replace(".", "")
    profile = f"cp{compact}:numpy-{observed}-floor"
    return [profile], {
        "python": config.numpy_floor_interpreter,
        "requirement": config.numpy_floor,
        "observed": observed,
        "profile": profile,
    }


def run_full_typing_profile(
    *,
    uv: str,
    interpreter: str,
    wheel: Path,
    extracted: Path,
    workspace: ProfileWorkspace,
    config: DistributionConfig,
    run: Callable[..., str] = checked_run,
) -> str:
    python = install_environment(
        uv=uv,
        interpreter=interpreter,
        environment=workspace.environment,
        requirements=[
            f"{wheel}[torch,jax,matplotlib]",
            config.mypy,
            config.torch,
            *config.jax,
            config.matplotlib,
        ],
        run=run,
    )
    workspace.consumer.mkdir(parents=True)
    typecheck = workspace.consumer / "typecheck"
    shutil.copytree(extracted / "bindings/python/typecheck", typecheck)
    run(
        [
            str(python),
            "-I",
            "-m",
            "mypy.stubtest",
            "eqiora",
            "--concise",
            "--ignore-disjoint-bases",
        ],
        cwd=workspace.consumer,
    )
    run(
        [
            str(python),
            "-I",
            "-m",
            "mypy",
            "--strict",
            str(typecheck / "diff.py"),
            str(typecheck / "torch_adapter.py"),
            str(typecheck / "jax_adapter.py"),
            str(typecheck / "matplotlib_adapter.py"),
        ],
        cwd=workspace.consumer,
    )
    return f"cp{config.extras_interpreter.replace('.', '')}:complete-public-typing"
