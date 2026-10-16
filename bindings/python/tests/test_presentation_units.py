"""Presentation uses the native unit catalog without editing scientific quantities."""
import csv
import dataclasses
import json
from itertools import repeat
import pytest
import eqiora as q


def pressure_result():
    model = q.compile(source="""model Stress() {
        variable stress:Pa;
        relation prescribed {stress=210000000000[Pa];}
        observable value:Pa=stress;
    }""")
    plan = q.resolve(model, solve=q.solve.Linear(
        algorithm=q.solve.LinearSolver.SparseLu, preconditioner=q.solve.Preconditioner.Identity,
        reduction=q.solve.Reduction.Fast, provider=q.solve.SolverProvider.faer(),
        relative_tolerance=1e-12, absolute_tolerance=1e-14, maximum_iterations=10))
    return model, plan, q.submit(plan, state=q.State.initial(plan))


def test_pressure_presentation_preserves_the_complete_result():
    model, plan, run = pressure_result()
    result = run.result()
    before = (model.to_bytes(), plan.to_bytes(), result.to_bytes(), run.plan_key, run.model_digest, run.history)
    observation = result.observe(model.observable("value"))
    displayed = q.units.Pa.prefixed("G").present(observation.value, observation.value_type.dimension,
                                                decimal_places=3)
    assert displayed.value == 210.
    assert displayed.unit == "GPa"
    assert displayed.text == "210.000 GPa"
    assert displayed.stored_value == 210000000000.
    assert displayed.dimension == q.Dimension(mass=1, length=-1, time=-2)
    assert displayed.rounding == "nearest-ties-even"
    assert displayed.decimal_places == 3 and displayed.locale == "C"
    with pytest.raises(dataclasses.FrozenInstanceError):
        displayed.value = 211.
    assert before == (model.to_bytes(), plan.to_bytes(), result.to_bytes(), run.plan_key, run.model_digest, run.history)
    with pytest.raises(ValueError, match="incompatible"):
        q.units.m.present(observation.value, observation.value_type.dimension)


def test_exact_conversion_and_explicit_formatting_reject_guesses():
    meter = q.Dimension(length=1)
    assert q.units.m.prefixed("n").present(1., meter).value == 1_000_000_000.
    assert q.units.m.prefixed("n").present(1e-10, meter).value.hex() == "0x1.999999999999ap-4"
    for locale in ("en_US", "de_DE", ""):
        with pytest.raises(ValueError, match="locale"):
            q.units.m.present(1., meter, locale=locale)
    for value in (float("inf"), float("nan")):
        with pytest.raises(ValueError, match="finite"):
            q.units.m.present(value, meter)
    with pytest.raises(ValueError, match="precision"):
        q.units.m.present(1., meter, decimal_places=18)
    with pytest.raises(TypeError):
        q.units.m.present(1., meter, decimal_places=True)
    with pytest.raises(TypeError):
        q.units.m.present(True, meter)
    for value, text in ((2.5, "2 m"), (3.5, "4 m")):
        assert q.units.m.present(value, meter, decimal_places=0).text == text


def test_csv_and_accessible_text_share_units_and_preserve_existing_files(tmp_path):
    model, plan, run = pressure_result()
    result = run.result()
    observation = result.observe(model.observable("value"))
    before = (model.to_bytes(), plan.to_bytes(), result.to_bytes(), run.plan_key, run.model_digest, run.history)
    unit = q.units.Pa.prefixed("G")
    path = tmp_path / "pressure.csv"
    q.data.export_observations_csv(path, [observation, observation], unit=unit, decimal_places=3)
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["occurrence"] for row in rows] == ["0", "1"]
    assert all(row["display_value"] == "210" and row["display_unit"] == "GPa" for row in rows)
    displayed = unit.present(observation.value, observation.value_type.dimension, decimal_places=3)
    assert all(row["display_text"] == displayed.text for row in rows)
    assert all(row["rounding"] == displayed.rounding and row["locale"] == "C" for row in rows)
    assert all(row["source_real_si"] == "210000000000" and row["source_imaginary_si"] == "0" for row in rows)
    assert all(row["result_identity"] == observation.result_identity for row in rows)
    assert all(row["observable_id"] == observation.observable_id for row in rows)
    assert all(row["time_s"] == "" and row["state_identity"] == "" for row in rows)
    original = path.read_bytes()
    for arguments in ({"unit": q.units.m}, {"unit": unit, "locale": "de_DE"},
                      {"unit": unit, "component": 1}, {"unit": unit, "projection": "guess"}):
        with pytest.raises(q.ValidationError):
            q.data.export_observations_csv(path, [observation], **arguments)
        assert path.read_bytes() == original
    with pytest.raises(ValueError, match="budget"):
        q.data.export_observations_csv(path, repeat(observation, 65_537), unit=unit)
    assert path.read_bytes() == original
    assert before == (model.to_bytes(), plan.to_bytes(), result.to_bytes(), run.plan_key, run.model_digest, run.history)
    assert not list(tmp_path.glob(".eqiora-*.tmp"))


def test_csv_retains_complex_component_shape_time_and_undefined_phase(tmp_path):
    model = q.compile(source="""model Channels() {
        state z:array<complex<m>,2>;
        initial {z=[math.complex(3[m],4[m]),math.complex(0[m],0[m])];}
        relation flow {derivative(z)=[math.complex(0[m/s],0[m/s]),math.complex(0[m/s],0[m/s])];}
        observable value:array<complex<m>,2>=z;
    }""")
    field = model.field("z")
    plan = q.resolve(model, temporal=q.time.ImplicitMidpoint(step_s=0.25,
        relative_tolerance=1e-11, absolute_tolerances={
            (field, 0, component, imaginary): 1e-13
            for component in range(2) for imaginary in (False, True)}))
    result = q.run(plan, state=q.State.initial(plan), until_s=0.5, output_times_s=(0.25, 0.5))
    before = result.to_bytes()
    observations = [result.observe(model.observable("value"), output_index=index) for index in range(2)]
    path = tmp_path / "complex.csv"
    q.data.export_observations_csv(path, observations, projection="magnitude", unit=q.units.m.prefixed("m"))
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [float(row["time_s"]) for row in rows] == [0.25, 0.5]
    assert len({row["state_identity"] for row in rows}) == 2
    for row, observation in zip(rows, observations, strict=True):
        ty = json.loads(row["value_type"])
        assert ty["shape"] == [2] and ty["array_rank"] == 1
        assert ty["frame"] == "invariant" and ty["scalar_domain"] == "complex"
        assert row["source_real_si"] == "3" and row["source_imaginary_si"] == "4"
        assert float(row["display_value"]) == 5000.
        assert row["state_identity"] == observation.output[2]
    for projection, unit, expected in (
        ("real", q.units.m.prefixed("m"), 3000.),
        ("imaginary", q.units.m.prefixed("m"), 4000.),
        ("squared_magnitude", q.units.m.prefixed("m") ** 2, 25_000_000.),
        ("phase", q.units.one, 0.9272952180016122),
    ):
        q.data.export_observations_csv(path, observations, projection=projection, unit=unit)
        with path.open(newline="") as stream:
            projected = list(csv.DictReader(stream))
        assert all(float(row["display_value"]) == pytest.approx(expected, rel=0, abs=2e-16) for row in projected)
    q.data.export_observations_csv(path, observations, projection="phase", component=1, unit=q.units.one)
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert all(row["status"] == "undefined" and row["angular_convention"] == "radians" for row in rows)
    assert all(row["display_value"] == row["projected_value_si"] == "" for row in rows)
    assert result.to_bytes() == before
