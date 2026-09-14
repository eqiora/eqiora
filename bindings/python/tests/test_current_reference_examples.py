"""Execute the displayed Reference programs through the installed public package."""

from pathlib import Path
import re
import shutil

import pytest

import eqiora


ROOT = Path(__file__).resolve().parents[3]
REFERENCE = ROOT / "docs/site/src/content/docs/reference"
PAGES = sorted((REFERENCE / "language").glob("*.mdx")) + sorted(
    (REFERENCE / "standard-packages").glob("*.mdx")
)


def test_current_reference_examples_and_displayed_python(tmp_path, monkeypatch):
    examples = sorted(REFERENCE.glob("*/_examples/*.eqi"))
    assert examples
    assert PAGES
    for source in examples:
        shutil.copyfile(source, tmp_path / source.name)
    shutil.copyfile(
        ROOT / "examples/voltage-divider/src/main.eqi", tmp_path / "electrical.eqi"
    )
    shutil.copyfile(tmp_path / "declarations.eqi", tmp_path / "model.eqi")
    for name, filename in (
        ("Eqiora.Electrical.Basic", "basic.eqi"),
        ("Eqiora.Solid.LinearElasticity", "linear_elasticity.eqi"),
        ("Eqiora.Fluid.InertialStokes", "inertial_stokes.eqi"),
    ):
        relative = Path("crates/eqiora-api/packages") / name / "src" / filename
        destination = tmp_path / "eqiora-source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, destination)
    monkeypatch.chdir(tmp_path)
    for page in PAGES:
        for code in re.findall(r"```python\n(.*?)```", page.read_text(), re.DOTALL):
            namespace = {}
            exec(compile(code, str(page), "exec"), namespace)
            assert isinstance(namespace["model"], eqiora.Model), page
    for source in sorted((REFERENCE / "language/_examples").glob("*.eqi")):
        assert eqiora.compile(path=source).field_ids


def test_reference_rejects_wrong_units_and_missing_required_binding():
    declaration = (REFERENCE / "language/_examples/declarations.eqi").read_text()
    composition = (REFERENCE / "language/_examples/composition.eqi").read_text()
    wrong_unit = declaration.replace("variable current: A;", "variable current: m;")
    missing_input = composition.replace("input = 2", "offset = 2")
    assert wrong_unit != declaration and missing_input != composition
    with pytest.raises(eqiora.ValidationError) as units:
        eqiora.compile(source=wrong_unit, filename="wrong-unit.eqi")
    assert "dimension" in str(units.value).lower()
    with pytest.raises(eqiora.ValidationError) as binding:
        eqiora.compile(source=missing_input, filename="missing-input.eqi")
    assert "input" in str(binding.value)
    assert any(word in str(binding.value).lower() for word in ("required", "missing"))


def test_reference_accepts_compatible_explicit_input_units():
    declaration = (REFERENCE / "language/_examples/declarations.eqi").read_text()
    explicit = declaration.replace("= 12,", "= 12 [V],")
    assert explicit != declaration
    assert isinstance(eqiora.compile(source=explicit, filename="input-units.eqi"), eqiora.Model)
