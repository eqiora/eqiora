import ast
import inspect
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import pytest


SOURCE = """
model decay() {
  state x: 1;
  initial { x = 1; }
  parameter rate: 1 / s = 1;
  relation flow {
    derivative(x) + rate * x = 0;
  }
}
"""

MALFORMED_RETIRED_SCHEMA_SPECIMEN = b'{"schema":"eqiora.model-envelope/v7"}'


def test_distribution_version_is_native_and_matches_installed_metadata() -> None:
    import eqiora

    assert eqiora.__version__ == metadata.version("eqiora") == "0.1.0a10"


def test_base_import_does_not_load_optional_numerical_frameworks() -> None:
    script = """
import sys
import eqiora

for name in ("numpy", "torch", "jax", "cupy"):
    assert name not in sys.modules, (name, sorted(sys.modules))
"""
    subprocess.run(
        [sys.executable, "-I", "-c", script],
        check=True,
        text=True,
        capture_output=True,
    )


def test_compile_contract_is_claim_local_at_runtime_and_in_the_stub() -> None:
    import eqiora

    assert str(inspect.signature(eqiora.compile)) == (
        "(*, path=None, source=None, filename=None, geometry=None, "
        "bindings=None, entry=None)"
    )

    stub = Path(eqiora.__file__).with_name("__init__.pyi")
    syntax = ast.parse(stub.read_text(encoding="utf-8"), filename=str(stub))
    declarations = [
        node
        for node in syntax.body
        if isinstance(node, ast.FunctionDef) and node.name == "compile"
    ]
    assert len(declarations) == 1
    declaration = declarations[0]
    assert [argument.arg for argument in declaration.args.posonlyargs] == []
    assert declaration.args.args == []
    assert [argument.arg for argument in declaration.args.kwonlyargs] == [
        "path",
        "source",
        "filename",
        "geometry",
        "bindings",
        "entry",
    ]
    assert declaration.args.vararg is None
    assert declaration.args.kwarg is None
    assert len(declaration.args.kw_defaults) == 6
    assert all(ast.literal_eval(default) is None for default in declaration.args.kw_defaults)
    assert ast.unparse(declaration.returns) == "Model"


def test_revision_identity_is_exact_across_artifact_bytes() -> None:
    import eqiora

    model = eqiora.compile(source=SOURCE, filename="decay.eqi")
    revision = model.revision
    replay = eqiora.Model.from_bytes(model.to_bytes())

    assert revision.number == 1
    assert revision.model_id == model.model_id
    assert revision.digest == model.digest
    assert replay.revision == revision
    assert replay == model
    assert hash(replay) == hash(model)
    assert hash(replay.revision) == hash(revision)


def test_current_only_surface_rejects_retired_selectors_and_malformed_bytes() -> None:
    import eqiora

    for retired in (
        "compatibility",
        "ExactModelCodec",
        "compile_exact",
        "define_exact",
        "replay",
        "replay_exact",
    ):
        assert not hasattr(eqiora, retired)

    with pytest.raises(eqiora.CompatibilityError) as caught:
        eqiora.Model.from_bytes(MALFORMED_RETIRED_SCHEMA_SPECIMEN)
    assert [diagnostic.code for diagnostic in caught.value.diagnostics] == ["EQ0901"]


def test_model_bytes_and_eqmodel_files_are_symmetric_and_exact(tmp_path: Path) -> None:
    import eqiora

    model = eqiora.compile(source=SOURCE, filename="decay.eqi")
    encoded = model.to_bytes()
    restored = eqiora.Model.from_bytes(encoded)

    assert restored == model
    assert restored.revision == model.revision
    assert restored.to_bytes() == encoded

    path = tmp_path / "decay.eqmodel"
    path.write_bytes(b"not a Model")
    model.write(path)
    assert path.read_bytes() == encoded
    assert not list(tmp_path.glob(".eqiora-model-*.tmp"))

    reopened = eqiora.Model.read(path)
    assert reopened == model
    assert reopened.revision == model.revision
    assert reopened.to_bytes() == encoded


def test_model_artifact_io_fails_closed_at_format_and_file_boundaries(
    tmp_path: Path,
) -> None:
    import eqiora

    model = eqiora.compile(source=SOURCE, filename="decay.eqi")
    encoded = model.to_bytes()

    rejected_bytes = (
        encoded[:-1],
        encoded + b"\n",
        b"foreign",
        b"[" * 65 + b"0" + b"]" * 65,
        b"x" * (16 * 1024 * 1024 + 1),
    )
    for rejected in rejected_bytes:
        with pytest.raises(eqiora.CompatibilityError) as caught:
            eqiora.Model.from_bytes(rejected)
        assert caught.value.category == "compatibility"
        assert [diagnostic.code for diagnostic in caught.value.diagnostics] == ["EQ0901"]

    wrong_suffix = tmp_path / "decay.eqi"
    with pytest.raises(eqiora.CompatibilityError):
        model.write(wrong_suffix)
    assert not wrong_suffix.exists()

    with pytest.raises(eqiora.CompatibilityError):
        eqiora.Model.read(wrong_suffix)

    directory = tmp_path / "directory.eqmodel"
    directory.mkdir()
    with pytest.raises(eqiora.CompatibilityError):
        eqiora.Model.read(directory)
    with pytest.raises(eqiora.CompatibilityError):
        model.write(directory)

    target = tmp_path / "target.eqmodel"
    target.write_bytes(encoded)
    symlink = tmp_path / "symlink.eqmodel"
    symlink.symlink_to(target)
    with pytest.raises(eqiora.CompatibilityError):
        eqiora.Model.read(symlink)
    with pytest.raises(eqiora.CompatibilityError):
        model.write(symlink)
    assert target.read_bytes() == encoded

    missing_parent = tmp_path / "missing" / "decay.eqmodel"
    with pytest.raises(eqiora.CompatibilityError):
        model.write(missing_parent)
    assert not missing_parent.exists()
    assert not list(tmp_path.glob(".eqiora-model-*.tmp"))

    oversized = tmp_path / "oversized.eqmodel"
    oversized.write_bytes(b"x" * (16 * 1024 * 1024 + 1))
    with pytest.raises(eqiora.CompatibilityError) as caught:
        eqiora.Model.read(oversized)
    assert caught.value.diagnostics[0].code == "EQ0901"


def test_value_edit_is_atomic_immutable_and_stale_base_safe() -> None:
    import eqiora

    base = eqiora.compile(source=SOURCE, filename="decay.eqi")
    edit = base.preview_value_edit("rate", 2.0)
    child = base.commit(edit)

    assert base.revision.number == 1
    assert child.revision.number == 2
    assert child.model_id == base.model_id
    assert child.digest != base.digest
    assert edit.base_digest == base.digest
    assert edit.base_revision == base.revision.number
    assert edit.target_id in base.parameter_ids
    assert edit == base.preview_value_edit("rate", 2.0)
    assert hash(edit) == hash(base.preview_value_edit("rate", 2.0))

    replay = eqiora.Model.from_bytes(child.to_bytes())
    assert replay == child

    with pytest.raises(eqiora.ValidationError) as caught:
        child.commit(edit)
    assert caught.value.category == "validation"
    assert [diagnostic.code for diagnostic in caught.value.diagnostics] == ["EQ0106"]

    replay_edit = replay.preview_value_edit(replay.parameter_ids[0], 3.0)
    grandchild = replay.commit(replay_edit)
    assert grandchild.revision.number == 3

    sibling = base.commit(base.preview_value_edit("rate", 3.0))
    child_parameter_edit = child.preview_value_edit("rate", 4.0)
    sibling_parameter_edit = sibling.preview_value_edit("rate", 4.0)
    assert child_parameter_edit.key != sibling_parameter_edit.key
    assert child_parameter_edit != sibling_parameter_edit
    assert len({child_parameter_edit, sibling_parameter_edit}) == 2


def test_direct_exception_construction_has_the_stubbed_attributes() -> None:
    import eqiora

    error = eqiora.ValidationError("manually constructed")
    assert error.category == "validation"
    assert error.diagnostics == ()


def test_exception_taxonomy_keeps_structured_diagnostics() -> None:
    import eqiora

    with pytest.raises(eqiora.ValidationError) as validation:
        eqiora.compile(source='model broken() { field ; }', filename="broken.eqi")
    assert isinstance(validation.value, eqiora.EqioraError)
    assert validation.value.category == "validation"
    assert validation.value.diagnostics[0].source_span is not None

    with pytest.raises(eqiora.CompatibilityError) as compatibility:
        eqiora.Model.from_bytes(b"{}")
    assert compatibility.value.category == "compatibility"
    assert compatibility.value.diagnostics[0].code == "EQ0901"

    execution = eqiora.ExecutionError("manually constructed")
    assert execution.category == "execution"
    assert execution.diagnostics == ()

    assert issubclass(eqiora.CapabilityError, eqiora.EqioraError)
    assert issubclass(eqiora.CancellationError, eqiora.EqioraError)
    assert issubclass(eqiora.InternalError, eqiora.EqioraError)
