"""Execute the displayed Reference programs through the installed public package."""

from pathlib import Path
import json
import re
import shutil

import pytest

import eqiora


ROOT = Path(__file__).resolve().parents[3]
REFERENCE = ROOT / "docs/site/src/content/docs/reference"
PAGES = sorted((REFERENCE / "language").glob("*.mdx")) + sorted(
    (REFERENCE / "standard-packages").glob("*.mdx")
)


def _linear_result(model):
    plan = eqiora.resolve(model, solve=eqiora.solve.Linear(
        algorithm=eqiora.solve.LinearSolver.SparseLu,
        preconditioner=eqiora.solve.Preconditioner.Identity,
        reduction=eqiora.solve.Reduction.Fast,
        provider=eqiora.solve.SolverProvider.faer(),
        relative_tolerance=1e-12, absolute_tolerance=1e-14, maximum_iterations=100,
    ))
    return eqiora.run(plan, state=eqiora.State.initial(plan))


def _assert_divider(model, result):
    # Ohm's law and power conservation for the displayed 12 V, 1/2 kohm circuit.
    current = 12 / (1000 + 2000)
    expected = {
        "current": current,
        "midpoint": current * 2000,
        "upper_power": current**2 * 1000,
        "lower_power": current**2 * 2000,
        "source_power": -12 * current,
    }
    for name, value in expected.items():
        tolerance = 1e-10 if name == "midpoint" else 1e-12
        assert result.observe(model.observable(name)).value == pytest.approx(
            value, abs=tolerance, rel=0
        )


def test_current_reference_examples_and_displayed_python(tmp_path, monkeypatch):
    assert PAGES
    executed = 0
    for page in PAGES:
        text = page.read_text()
        blocks = re.findall(r"```python\n(.*?)```", text, re.DOTALL)
        if not blocks:
            continue
        relative = page.relative_to(REFERENCE).as_posix()
        work = tmp_path / page.parent.name / page.stem
        work.mkdir(parents=True)
        monkeypatch.chdir(work)
        # Prepare only the files the prose asks the reader to save. Execute the
        # displayed Python unchanged, retaining earlier blocks' names on this page.
        if relative == "language/index.mdx":
            source, = re.findall(r"```eqi\n(.*?)```", text, re.DOTALL)
            (work / "current.eqi").write_text(source)
        elif relative == "standard-packages/continuum.mdx":
            shutil.copyfile(
                REFERENCE / "standard-packages/_examples/elastic-body.eqi",
                work / "model.eqi",
            )
        elif relative == "standard-packages/electrical.mdx":
            shutil.copyfile(
                ROOT / "examples/voltage-divider/src/main.eqi",
                work / "electrical.eqi",
            )
        elif relative == "standard-packages/index.mdx":
            project = work / "divider"
            (project / "src").mkdir(parents=True)
            manifest, = re.findall(
                r'```toml title="eqiora.toml"\n(.*?)```', text, re.DOTALL
            )
            (project / "eqiora.toml").write_text(manifest)
            shutil.copyfile(
                ROOT / "examples/voltage-divider/src/main.eqi",
                project / "src/main.eqi",
            )
        else:
            pytest.fail(f"Add the documented file setup and assertions for {relative}")
        namespace = {}
        models = []
        for code in blocks:
            exec(compile(code, str(page), "exec"), namespace)
            model = namespace["model"]
            assert isinstance(model, eqiora.Model), page
            models.append(model)
            executed += 1
        if relative == "language/index.mdx":
            assert model.field_ids == [model.field("current").id]
            assert len(model.parameter_ids) == 2
            assert namespace["measurement"] == model.observable("measured_current")
            assert namespace["measurement"].model_digest == model.digest
        elif relative == "standard-packages/continuum.mdx":
            assert model.field("displacement").id in model.field_ids
            assert model.field("load_potential").id in model.field_ids
            names = {
                node["identity"]["name"]
                for node in json.loads(namespace["resolution"])["nodes"]
            }
            assert names == {
                "org.example.Continuum", "Eqiora.Solid.LinearElasticity",
                "Eqiora.Mechanics.Interfaces",
            }
        elif relative == "standard-packages/electrical.mdx":
            _assert_divider(model, namespace["result"])
        else:
            # Both displayed blocks must retain the same model through vendoring.
            assert len(models) == 2
            assert models[0].digest == models[1].digest
            _assert_divider(model, _linear_result(model))
            shutil.rmtree(namespace["store"])
            resolution = eqiora.open_project(namespace["project"], namespace["vendor"])
            offline = eqiora.compile_package(
                namespace["vendor"], resolution, entry="VoltageDivider"
            )
            assert offline.digest == model.digest
    assert executed


def test_current_reference_language_sources_compile():
    sources = sorted((REFERENCE / "language/_examples").glob("*.eqi"))
    assert sources
    for source in sources:
        assert eqiora.compile(path=source).field_ids


@pytest.mark.parametrize("filename,package,version,entry,fields", [
    ("elastic-body.eqi", "Eqiora.Solid.LinearElasticity", "0.6.0",
     "ReferenceElasticBody", ("displacement", "load_potential")),
    ("stokes.eqi", "Eqiora.Fluid.Incompressible", "0.6.0",
     "ReferenceStokes", ("velocity", "pressure", "force_potential")),
    ("sampled.eqi", "Eqiora.Controls.Sampled", "0.1.0",
     "SampledSignals", ("delay.memory", "integral.memory")),
])
def test_current_reference_package_sources_compile(
    tmp_path, filename, package, version, entry, fields
):
    project = tmp_path / "project"
    (project / "src").mkdir(parents=True)
    shutil.copyfile(
        REFERENCE / "standard-packages/_examples" / filename,
        project / "src/main.eqi",
    )
    (project / "eqiora.toml").write_text(
        '[package]\nname = "org.example.Reference"\nversion = "0.1.0"\nentry = "main"\n'
    )
    store = tmp_path / "store"
    store.mkdir()
    resolution = eqiora.add_bundled_dependency(project, store, package, version=version)
    model = eqiora.compile_package(store, resolution, entry=entry)
    assert isinstance(model, eqiora.Model)
    assert model.package_compilation_digest is not None
    for name in fields:
        assert model.field(name).id in model.field_ids


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
