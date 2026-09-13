from __future__ import annotations

import ast
import base64
import hashlib
import importlib.metadata
import inspect
import json
import os
import shutil
import stat
import tempfile
from pathlib import Path
from typing import Callable, get_type_hints

import pytest

import eqiora


ROOT = Path(__file__).resolve().parents[3]
HOME_SCRATCH = Path.home() / ".cache/eqiora/oracle-tests"

CURRENT_COMPILER_VERSION = "0.1.0-alpha.11"
CURRENT_SEMANTIC_CANONICALIZATION_VERSION = 2
CONFORMANCE = ROOT / "verify/interfaces/python-package-conformance"
FALSE_CLAIM = CONFORMANCE / "models/false-scientific-claim"
FALSE_CLAIM_STORE = FALSE_CLAIM / "store"
FALSE_CLAIM_RESOLUTION_FILE = FALSE_CLAIM / "resolution.json"
CONFORMANCE_IDENTITIES = json.loads(
    (CONFORMANCE / "expected/identities.json").read_text(encoding="utf-8")
)
CONFORMANCE_PROFILE = "eqiora.package.structural-conformance-v1"
CONFORMANCE_PACKAGE_FIELDS = (
    "name",
    "version",
    "semantic_digest",
    "source_digest",
)
CONFORMANCE_REPORT_FIELDS = (
    "profile",
    "eqiora_version",
    "compiler",
    "compiler_version",
    "semantic_canonicalization_version",
    "source_bundle_version",
    "resolution_version",
    "root_package",
    "packages",
    "entry_model",
    "resolution_digest",
    "package_compilation_digest",
    "model_id",
    "model_revision",
    "model_digest",
    "deterministic_replay_agreement",
)
EXPECTED_EQIORA_ALL = [
    "Module",
    "equal",
    "not_equal",
    "less",
    "less_equal",
    "greater",
    "greater_equal",
    "logical_not",
    "logical_and",
    "logical_or",

    "__version__",
    "Array",
    "AuthoredFormulation",
    "BoundarySide",
    "CancellationError",
    "CapabilityError",
    "CompatibilityError",
    "Connection",
    "ClockDomain",
    "ExecutionSession",
    "ExecutionCheckpoint",
    "ConservingPort",
    "ConvergenceReason",
    "DerivativeImplementation",
    "Diagnostic",
    "DifferentiableEvaluation",
    "DifferentiableJvp",
    "DifferentiablePrimal",
    "DifferentiableProgram",
    "DifferentiableVjp",
    "CompleteEvaluationMap",
    "EvaluationMapPlan",
    "EvaluationMapTerminalReport",
    "EvaluationMapCancellation",
    "EvaluationMapJvp",
    "EvaluationMapVjp",
    "DifferentiationEvidence",
    "DifferentiationMode",
    "Dimension",
    "ValueType",
    "Enum",
    "EnumValue",
    "FiniteSpace",
    "IndexSet",
    "DomainRef",
    "Domain",
    "EqioraError",
    "ExecutionError",
    "Expression",
    "Field",
    "FieldOutput",
    "FieldRef",
    "FormulationView",
    "FormulationKind",
    "FormulationSelectionMode",
    "InitialField",
    "InternalError",
    "LinearSolveSummary",
    "LinearizationState",
    "MathReference",
    "MathRendering",
    "Model",
    "PackageConformancePackage",
    "PackageConformanceReport",
    "Parameter",
    "Observable",
    "ObservableRef",
    "Observation",
    "TrajectoryObservation",
    "ObservableStateTangent",
    "integral",
    "measure",
    "ParameterRef",
    "PhysicalDomain",
    "PropertyBinding",
    "QuantityLabel",
    "Plan",
    "FieldRole",
    "Initial",
    "Relation",
    "Result",
    "Profile",
    "ProfileEvent",
    "ProfilePhase",
    "Revision",
    "ResolvedExecution",
    "ScalarPlanView",
    "Run",
    "RunStatus",
    "Series",
    "State",
    "TransientRunCancellation",
    "TransientRunProgress",
    "StructuralSemanticFingerprint",
    "ValidationError",
    "ValueEdit",
    "View",
    "across",
    "check_package_conformance",
    "compile",
    "compile_package",
    "connect",
    "derivative",
    "div",
    "grad",
    "lang",
    "units",
    "resolve",
    "ProjectUpdate",
    "preview_local_project",
    "resolve_local_project",
    "add_local_dependency",
    "remove_local_dependency",
    "run",
    "submit",
    "through",
    "trace",
    "diff",
    "fem",
    "fluid",
    "formulation",
    "fsi",
    "fvm",
    "geometry",
    "meshing",
    "solid",
    "solve",
    "time",
    "trajectory",
    "add_bundled_dependency",
    "add_git_dependency",
    "fetch_project",
    "open_project",
    "update_project",
    "vendor_project",
]


def canonical_fixture(path: Path) -> bytes:
    stored = path.read_bytes()
    assert stored.endswith(b"\n")
    canonical = stored.removesuffix(b"\n")
    assert not canonical.endswith(b"\n")
    return canonical


def canonical_json(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def source_bundle_digest(source: object) -> str:
    digest = hashlib.sha256()
    digest.update(b"eqiora.source-bundle.sha256.v1\0")
    digest.update(canonical_json(source))
    return digest.hexdigest()


FALSE_CLAIM_RESOLUTION = canonical_fixture(FALSE_CLAIM_RESOLUTION_FILE)


def assert_compatibility(error: eqiora.CompatibilityError) -> None:
    assert error.category == "compatibility"
    assert error.diagnostics
    assert error.diagnostics[0].code == "EQ0901"


def check_conformance(
    store: object,
    resolution: bytes,
    *,
    entry_model: str = "Main",
    profile: object = CONFORMANCE_PROFILE,
) -> eqiora.PackageConformanceReport:
    return eqiora.check_package_conformance(
        store,
        resolution,
        entry_model=entry_model,
        profile=profile,
    )


def assert_conformance_rejection(
    store: Path,
    *,
    resolution: bytes = FALSE_CLAIM_RESOLUTION,
    entry_model: str = "Main",
    profile: object = CONFORMANCE_PROFILE,
    expected: type[eqiora.EqioraError] = eqiora.CompatibilityError,
) -> eqiora.EqioraError:
    before = tree_snapshot(store)
    result: object = object()
    sentinel = result
    with pytest.raises(expected) as caught:
        result = check_conformance(
            store,
            resolution,
            entry_model=entry_model,
            profile=profile,
        )
    assert result is sentinel, "a rejected check exposed a partial report"
    assert tree_snapshot(store) == before, "the structural check mutated its store"
    return caught.value


def expected_conformance_report(
    label: str, compilation_digest: str = "0" * 64
) -> eqiora.PackageConformanceReport:
    facts = CONFORMANCE_IDENTITIES[label]
    package = eqiora.PackageConformancePackage(
        facts["name"],
        facts["version"],
        facts["semantic_identity"],
        facts["source_identity"],
    )
    return eqiora.PackageConformanceReport(
        CONFORMANCE_IDENTITIES["profile"],
        eqiora.__version__,
        CONFORMANCE_IDENTITIES["compiler"],
        CURRENT_COMPILER_VERSION,
        CURRENT_SEMANTIC_CANONICALIZATION_VERSION,
        CONFORMANCE_IDENTITIES["source_bundle_version"],
        CONFORMANCE_IDENTITIES["resolution_version"],
        package,
        (package,),
        "Main",
        facts["resolution_identity"],
        compilation_digest,
        facts["object_id"],
        facts["revision"],
        facts["canonical_identity"],
        True,
    )


def tree_snapshot(root: Path) -> tuple[tuple[object, ...], ...]:
    snapshot: list[tuple[object, ...]] = []
    for path in sorted(
        root.rglob("*"), key=lambda item: item.relative_to(root).as_posix()
    ):
        relative = path.relative_to(root).as_posix()
        metadata = path.lstat()
        mode = stat.S_IMODE(metadata.st_mode)
        if path.is_symlink():
            snapshot.append((relative, "symlink", mode, os.readlink(path)))
        elif path.is_dir():
            snapshot.append((relative, "directory", mode))
        elif path.is_file():
            content = path.read_bytes()
            snapshot.append(
                (
                    relative,
                    "file",
                    mode,
                    len(content),
                    hashlib.sha256(content).hexdigest(),
                )
            )
        else:
            snapshot.append((relative, "nonregular", mode))
    return tuple(snapshot)


def with_scratch(callback: Callable[[Path], None]) -> None:
    HOME_SCRATCH.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="python-offline-model-package-",
        dir=HOME_SCRATCH,
    ) as directory:
        callback(Path(directory))


def current_package_fixture(
    parent: Path,
    name: str = "org.example.structural_false_claim",
) -> tuple[Path, bytes]:
    project = parent / "project"
    package = project / "package/src"
    package.mkdir(parents=True)
    shutil.copyfile(
        Path(__file__).parent / "fixtures/package_conformance.eqi",
        package / "main.eqi",
    )
    (project / "README.md").write_text(
        "This package proves that every physical prediction is exact.\n",
        encoding="utf-8",
    )
    (project / "eqiora.toml").write_text(
        f'''[package]
name = "{name}"
version = "0.1.0"
source = "package/src"
entry = "main"
''',
        encoding="utf-8",
    )
    store = parent / "store"
    store.mkdir()
    return store, eqiora.resolve_local_project(project, store)


def test_package_conformance_public_signature_named_tuples_and_stub_are_exact() -> None:
    assert eqiora.__all__ == EXPECTED_EQIORA_ALL
    assert len(eqiora.__all__) == len(set(eqiora.__all__))
    assert "_check_package_conformance" not in eqiora.__all__

    signature = inspect.signature(eqiora.check_package_conformance)
    assert tuple(signature.parameters) == (
        "store_root",
        "resolution_bytes",
        "entry_model",
        "profile",
    )
    assert (
        signature.parameters["store_root"].kind
        is inspect.Parameter.POSITIONAL_OR_KEYWORD
    )
    assert (
        signature.parameters["resolution_bytes"].kind
        is inspect.Parameter.POSITIONAL_OR_KEYWORD
    )
    for name in ("entry_model", "profile"):
        parameter = signature.parameters[name]
        assert parameter.kind is inspect.Parameter.KEYWORD_ONLY
        assert parameter.default is inspect.Parameter.empty

    hints = get_type_hints(eqiora.check_package_conformance)
    assert hints == {
        "store_root": str | os.PathLike[str],
        "resolution_bytes": bytes,
        "entry_model": str,
        "profile": str,
        "return": eqiora.PackageConformanceReport,
    }

    assert eqiora.PackageConformancePackage.__bases__ == (tuple,)
    assert eqiora.PackageConformancePackage._fields == CONFORMANCE_PACKAGE_FIELDS
    assert eqiora.PackageConformancePackage.__match_args__ == CONFORMANCE_PACKAGE_FIELDS
    assert get_type_hints(eqiora.PackageConformancePackage) == {
        "name": str,
        "version": str,
        "semantic_digest": str,
        "source_digest": str,
    }
    assert eqiora.PackageConformanceReport.__bases__ == (tuple,)
    assert eqiora.PackageConformanceReport._fields == CONFORMANCE_REPORT_FIELDS
    assert eqiora.PackageConformanceReport.__match_args__ == CONFORMANCE_REPORT_FIELDS
    assert get_type_hints(eqiora.PackageConformanceReport) == {
        "profile": str,
        "eqiora_version": str,
        "compiler": str,
        "compiler_version": str,
        "semantic_canonicalization_version": int,
        "source_bundle_version": int,
        "resolution_version": int,
        "root_package": eqiora.PackageConformancePackage,
        "packages": tuple[eqiora.PackageConformancePackage, ...],
        "entry_model": str,
        "resolution_digest": str,
        "package_compilation_digest": str,
        "model_id": str,
        "model_revision": int,
        "model_digest": str,
        "deterministic_replay_agreement": bool,
    }

    caller_value = expected_conformance_report("false_claim")
    with pytest.raises(AttributeError):
        caller_value.profile = "changed"
    with pytest.raises(AttributeError):
        del caller_value.root_package
    with pytest.raises(TypeError):
        caller_value[0] = "changed"
    assert not hasattr(caller_value, "trust_status")
    assert not hasattr(caller_value, "attestation")
    assert not hasattr(caller_value, "to_json")

    stub_path = Path(inspect.getfile(eqiora)).with_name("__init__.pyi")
    stub = ast.parse(stub_path.read_text(encoding="utf-8"))
    classes = {node.name: node for node in stub.body if isinstance(node, ast.ClassDef)}
    for name, fields in [
        ("PackageConformancePackage", CONFORMANCE_PACKAGE_FIELDS),
        ("PackageConformanceReport", CONFORMANCE_REPORT_FIELDS),
    ]:
        assert [ast.unparse(base) for base in classes[name].bases] == ["NamedTuple"]
        declarations = {
            node.target.id: ast.unparse(node.annotation)
            for node in classes[name].body
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
        }
        assert tuple(declarations) == fields
        assert (
            declarations
            == {
                "PackageConformancePackage": {
                    "name": "str",
                    "version": "str",
                    "semantic_digest": "str",
                    "source_digest": "str",
                },
                "PackageConformanceReport": {
                    "profile": "str",
                    "eqiora_version": "str",
                    "compiler": "str",
                    "compiler_version": "str",
                    "semantic_canonicalization_version": "int",
                    "source_bundle_version": "int",
                    "resolution_version": "int",
                    "root_package": "PackageConformancePackage",
                    "packages": "tuple[PackageConformancePackage, ...]",
                    "entry_model": "str",
                    "resolution_digest": "str",
                    "package_compilation_digest": "str",
                    "model_id": "str",
                    "model_revision": "int",
                    "model_digest": "str",
                    "deterministic_replay_agreement": "bool",
                },
            }[name]
        )
    function = next(
        node
        for node in stub.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "check_package_conformance"
    )
    assert [argument.arg for argument in function.args.args] == [
        "store_root",
        "resolution_bytes",
    ]
    assert [argument.arg for argument in function.args.kwonlyargs] == [
        "entry_model",
        "profile",
    ]
    assert {
        argument.arg: ast.unparse(argument.annotation)
        for argument in [*function.args.args, *function.args.kwonlyargs]
    } == {
        "store_root": "str | os.PathLike[str]",
        "resolution_bytes": "bytes",
        "entry_model": "str",
        "profile": "str",
    }
    assert function.args.defaults == []
    assert function.args.kw_defaults == [None, None]
    assert ast.unparse(function.returns) == "PackageConformanceReport"
    stub_exports = next(
        ast.literal_eval(node.value)
        for node in stub.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "__all__"
            for target in node.targets
        )
    )
    assert stub_exports == EXPECTED_EQIORA_ALL


def test_structural_reports_match_current_packages_without_scientific_inference() -> None:
    def run(parent: Path) -> None:
        false_store, false_resolution = current_package_fixture(parent / "false")
        before = tree_snapshot(false_store)
        false_report = check_conformance(false_store, false_resolution)
        assert false_report.root_package.name == "org.example.structural_false_claim"
        assert len(false_report.package_compilation_digest) == 64
        assert false_report.deterministic_replay_agreement is True
        assert false_report.eqiora_version == eqiora.__version__
        assert false_report.eqiora_version == importlib.metadata.version("eqiora")
        assert tree_snapshot(false_store) == before
        assert check_conformance(false_store, false_resolution) == false_report

        release_path = false_store / f"{false_report.root_package.source_digest}.json"
        release = json.loads(release_path.read_bytes())
        documentation = next(
            file
            for file in release["source"]["files"]
            if file["role"] == "documentation"
        )
        decoded_documentation = base64.b64decode(documentation["bytes"]).decode("utf-8")
        assert "every physical prediction is exact" in decoded_documentation
        assert set(false_report._asdict()) == set(CONFORMANCE_REPORT_FIELDS)
        assert all("evidence" not in field for field in false_report._fields)
        assert all("physics" not in field for field in false_report._fields)

        reordered_store = parent / "reordered-creation"
        reordered_store.mkdir()
        (reordered_store / "unlisted-before.json").write_text(
            "not part of the exact closure\n", encoding="utf-8"
        )
        for source in sorted(false_store.iterdir(), reverse=True):
            shutil.copy2(source, reordered_store / source.name)
        reordered_before = tree_snapshot(reordered_store)
        assert check_conformance(reordered_store, false_resolution) == false_report
        assert tree_snapshot(reordered_store) == reordered_before

        poisson_store, poisson_resolution = current_package_fixture(
            parent / "poisson",
            "org.example.poisson",
        )
        poisson_before = tree_snapshot(poisson_store)
        poisson_report = check_conformance(poisson_store, poisson_resolution)
        assert poisson_report.root_package.name == "org.example.poisson"
        assert tree_snapshot(poisson_store) == poisson_before
        assert poisson_report.root_package != false_report.root_package
        assert poisson_report.resolution_digest != false_report.resolution_digest
        assert (
            poisson_report.package_compilation_digest
            != false_report.package_compilation_digest
        )
        assert poisson_report.model_digest != false_report.model_digest

        caller_copy = eqiora.PackageConformanceReport(*false_report)
        assert caller_copy == false_report
        assert check_conformance(false_store, false_resolution) == false_report

    with_scratch(run)


def test_profile_and_argument_shapes_fail_before_filesystem_authority() -> None:
    class UntouchedPath:
        calls = 0

        def __fspath__(self) -> str:
            self.calls += 1
            raise AssertionError("the profile rejection touched os.fspath")

    for rejected in [
        "EQIORA.PACKAGE.STRUCTURAL-CONFORMANCE-V1",
        "eqiora.package.structural-conformance-v1 ",
        "eqiora.package.structural-conformance-v2",
        "eqiora.package.structural-conformance",
        "structural-conformance-v1",
        "",
    ]:
        path = UntouchedPath()
        with pytest.raises(eqiora.CompatibilityError) as caught:
            check_conformance(path, b"not a resolution", profile=rejected)
        assert_compatibility(caught.value)
        assert path.calls == 0

    path = UntouchedPath()
    with pytest.raises(TypeError):
        check_conformance(path, b"not a resolution", profile=1)
    assert path.calls == 0

    with pytest.raises(TypeError):
        check_conformance(os.fsencode(FALSE_CLAIM_STORE), FALSE_CLAIM_RESOLUTION)

    class BytesPath:
        def __fspath__(self) -> bytes:
            return os.fsencode(FALSE_CLAIM_STORE)

    with pytest.raises(TypeError):
        check_conformance(BytesPath(), FALSE_CLAIM_RESOLUTION)

    class ResolutionBytes(bytes):
        pass

    def check_resolution_subclass(parent: Path) -> None:
        store, resolution = current_package_fixture(parent)
        report = check_conformance(store, ResolutionBytes(resolution))
        assert report.root_package.name == "org.example.structural_false_claim"

    with_scratch(check_resolution_subclass)

    with pytest.raises(TypeError):
        eqiora.check_package_conformance(
            FALSE_CLAIM_STORE,
            FALSE_CLAIM_RESOLUTION,
            entry_model="Main",
        )
    with pytest.raises(TypeError):
        eqiora.check_package_conformance(
            FALSE_CLAIM_STORE,
            FALSE_CLAIM_RESOLUTION,
            "Main",
            CONFORMANCE_PROFILE,
        )
    with pytest.raises(TypeError):
        eqiora.check_package_conformance(
            FALSE_CLAIM_STORE,
            FALSE_CLAIM_RESOLUTION,
            entry_model=1,
            profile=CONFORMANCE_PROFILE,
        )
    for invalid in [
        bytearray(FALSE_CLAIM_RESOLUTION),
        memoryview(FALSE_CLAIM_RESOLUTION),
        FALSE_CLAIM_RESOLUTION.decode("utf-8"),
    ]:
        with pytest.raises(TypeError):
            check_conformance(FALSE_CLAIM_STORE, invalid)


def test_resolution_wire_is_exact_before_store_and_rejects_stale_or_foreign_inputs() -> (
    None
):
    false_decoded = json.loads(FALSE_CLAIM_RESOLUTION)
    key_reordered = json.dumps(
        {
            "nodes": false_decoded["nodes"],
            "schema": false_decoded["schema"],
            "root": false_decoded["root"],
            "edges": false_decoded["edges"],
        },
        separators=(",", ":"),
    ).encode("utf-8")

    class UntouchedPath:
        calls = 0

        def __fspath__(self) -> str:
            self.calls += 1
            raise AssertionError("noncanonical resolution touched os.fspath")

    for invalid in [
        b" " + FALSE_CLAIM_RESOLUTION,
        FALSE_CLAIM_RESOLUTION_FILE.read_bytes(),
        key_reordered,
        b"{not-json",
        b"{}",
    ]:
        path = UntouchedPath()
        with pytest.raises(eqiora.CompatibilityError) as caught:
            check_conformance(path, invalid)
        assert_compatibility(caught.value)
        assert path.calls == 0

    def run(parent: Path) -> None:
        store, current_resolution = current_package_fixture(parent / "false")

        digest_mismatch = json.loads(current_resolution)
        digest_mismatch["nodes"][0]["source_digest"] = "0" * 64
        mismatch_bytes = json.dumps(digest_mismatch, separators=(",", ":")).encode(
            "utf-8"
        )
        assert_compatibility(
            assert_conformance_rejection(store, resolution=mismatch_bytes)
        )

        stale_identity = json.loads(current_resolution)
        stale_identity["root"]["semantic_digest"] = "1" * 64
        stale_identity["nodes"][0]["identity"]["semantic_digest"] = "1" * 64
        stale_bytes = json.dumps(stale_identity, separators=(",", ":")).encode("utf-8")
        assert_compatibility(
            assert_conformance_rejection(store, resolution=stale_bytes)
        )

        _, foreign_resolution = current_package_fixture(
            parent / "poisson",
            "org.example.poisson",
        )
        assert_compatibility(assert_conformance_rejection(store, resolution=foreign_resolution))

    with_scratch(run)


def test_release_normalization_accepts_representation_but_rejects_semantic_changes_and_roles() -> (
    None
):
    def run(parent: Path) -> None:
        current_store, current_resolution = current_package_fixture(parent / "current")
        expected = check_conformance(current_store, current_resolution)
        normalized = parent / "normalized-release"
        shutil.copytree(current_store, normalized)
        release_path = normalized / f"{expected.root_package.source_digest}.json"
        release = json.loads(release_path.read_bytes())
        source = release["source"]
        manifest = source["manifest"]
        represented = {
            "source": {
                "files": list(reversed(source["files"])),
                "manifest": {
                    "bundle": list(reversed(manifest["bundle"])),
                    "dependencies": list(reversed(manifest["dependencies"])),
                    "version": manifest["version"],
                    "name": manifest["name"],
                    "schema": manifest["schema"],
                    "entry": manifest["entry"],
                },
                "package": source["package"],
                "schema": source["schema"],
            },
            "semantic": {
                "declarations": list(reversed(release["semantic"]["declarations"])),
                "schema": release["semantic"]["schema"],
            },
            "schema": release["schema"],
        }
        release_path.write_text(
            json.dumps(represented, indent=2) + "\n", encoding="utf-8"
        )
        represented_before = tree_snapshot(normalized)
        normalized_report = check_conformance(normalized, current_resolution)
        assert normalized_report == expected
        assert tree_snapshot(normalized) == represented_before

        source_changed = parent / "source-changed"
        shutil.copytree(current_store, source_changed)
        source_path = source_changed / release_path.name
        source_release = json.loads(source_path.read_bytes())
        model_source = next(
            file
            for file in source_release["source"]["files"]
            if file["role"] == "model_source"
        )
        decoded = base64.b64decode(model_source["bytes"])
        model_source["bytes"] = base64.b64encode(decoded + b"\n").decode("ascii")
        source_path.write_bytes(
            json.dumps(source_release, separators=(",", ":")).encode("utf-8")
        )
        assert_compatibility(
            assert_conformance_rejection(
                source_changed,
                resolution=current_resolution,
            )
        )

        foreign = parent / "foreign-release"
        shutil.copytree(current_store, foreign)
        foreign_path = foreign / release_path.name
        foreign_store, foreign_resolution = current_package_fixture(
            parent / "foreign-package", "org.example.poisson"
        )
        foreign_report = check_conformance(foreign_store, foreign_resolution)
        typed_release = foreign_store / f"{foreign_report.root_package.source_digest}.json"
        foreign_path.write_bytes(typed_release.read_bytes())
        assert_compatibility(
            assert_conformance_rejection(foreign, resolution=current_resolution)
        )

        for role in ["executable", "plugin"]:
            hostile = parent / f"role-{role}"
            shutil.copytree(current_store, hostile)
            hostile_path = hostile / release_path.name
            hostile_release = json.loads(hostile_path.read_bytes())
            assert source_bundle_digest(hostile_release["source"]) == hostile_path.stem
            hostile_file = hostile_release["source"]["files"][0]
            hostile_file["role"] = role
            hostile_bundle = next(
                entry
                for entry in hostile_release["source"]["manifest"]["bundle"]
                if entry["path"] == hostile_file["path"]
            )
            hostile_bundle["role"] = role
            hostile_source = hostile_release["source"]
            hostile_digest = source_bundle_digest(hostile_source)
            hostile_path.unlink()
            hostile_path = hostile / f"{hostile_digest}.json"
            hostile_path.write_bytes(canonical_json(hostile_release))
            hostile_resolution = json.loads(current_resolution)
            hostile_resolution["nodes"][0]["source_digest"] = hostile_digest
            assert hostile_bundle["role"] == hostile_file["role"] == role
            assert hostile_path.stem == hostile_resolution["nodes"][0]["source_digest"]
            assert_compatibility(
                assert_conformance_rejection(
                    hostile,
                    resolution=canonical_json(hostile_resolution),
                )
            )

    with_scratch(run)


@pytest.mark.parametrize(
    "selector",
    ["", "Missing", "dep.Main", "org.example.poisson.Main", "wave_number"],
)
def test_conformance_selector_is_bare_root_local_without_a_visibility_policy(
    selector: str, tmp_path: Path
) -> None:
    store, resolution = current_package_fixture(tmp_path)
    error = assert_conformance_rejection(
        store,
        resolution=resolution,
        entry_model=selector,
        expected=eqiora.ValidationError,
    )
    assert error.category == "validation"
    assert error.diagnostics
    assert (
        check_conformance(store, resolution).entry_model == "Main"
    )


def test_conformance_filesystem_authority_and_atomicity_are_exact() -> None:
    def run(parent: Path) -> None:
        current_store, current_resolution = current_package_fixture(parent / "current")
        accepted = parent / "accepted"
        shutil.copytree(current_store, accepted)
        (accepted / "unlisted.json").write_text("ignored\n", encoding="utf-8")
        before = tree_snapshot(accepted)
        report = check_conformance(accepted, current_resolution)
        assert report.root_package.name == "org.example.structural_false_claim"
        assert tree_snapshot(accepted) == before

        missing = parent / "missing"
        shutil.copytree(current_store, missing)
        next(missing.glob("*.json")).unlink()
        assert_compatibility(
            assert_conformance_rejection(missing, resolution=current_resolution)
        )

        nonregular = parent / "nonregular"
        shutil.copytree(current_store, nonregular)
        nonregular_entry = next(nonregular.glob("*.json"))
        nonregular_entry.unlink()
        nonregular_entry.mkdir()
        assert_compatibility(
            assert_conformance_rejection(nonregular, resolution=current_resolution)
        )

        absent = parent / "absent"
        with pytest.raises(eqiora.CompatibilityError) as missing_root:
            check_conformance(absent, current_resolution)
        assert_compatibility(missing_root.value)

        regular = parent / "regular-root"
        regular.write_text("not a directory\n", encoding="utf-8")
        with pytest.raises(eqiora.CompatibilityError) as regular_root:
            check_conformance(regular, current_resolution)
        assert_compatibility(regular_root.value)

        if os.name == "posix":
            exact_link = parent / "exact-link"
            shutil.copytree(current_store, exact_link)
            linked_entry = next(exact_link.glob("*.json"))
            outside = parent / "outside-release.json"
            linked_entry.replace(outside)
            linked_entry.symlink_to(outside)
            assert_compatibility(
                assert_conformance_rejection(exact_link, resolution=current_resolution)
            )

            real = parent / "real-store"
            shutil.copytree(current_store, real)
            root_link = parent / "root-link"
            root_link.symlink_to(real, target_is_directory=True)
            real_before = tree_snapshot(real)
            with pytest.raises(eqiora.CompatibilityError) as linked_root:
                check_conformance(root_link, current_resolution)
            assert_compatibility(linked_root.value)
            assert tree_snapshot(real) == real_before

    with_scratch(run)
