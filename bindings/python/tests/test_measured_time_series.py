"""Measured forcing through the existing table, Model and common ODE owners."""
import csv
import io
import json
from decimal import Decimal

import eqiora as q
import pytest


@pytest.fixture
def measured_model(tmp_path):
    # Explicit column units normalize through the compiler; the table owns interpolation.
    measured = "elapsed_ms,force_mN\n0,0\n500,2000\n1000,0\n"
    reader = csv.DictReader(io.StringIO(measured))
    columns = (("elapsed_ms", q.units.s.prefixed("m"), q.Dimension(time=1)),
               ("force_mN", q.units.N.prefixed("m"), q.Dimension(mass=1, length=1, time=-2)))
    values = []
    for row in reader:
        for column, unit, dimension in columns:
            value, normalized_dimension = unit.normalize(Decimal(row[column]))
            assert normalized_dimension == dimension
            values.append(value)
    project = tmp_path / "project"
    for directory in ("src", "data", "docs"):
        (project / directory).mkdir(parents=True)
    (project / "eqiora.toml").write_text(
        '[package]\nname="org.example.MeasuredForce"\nversion="1.0.0"\nentry="main"\n'
    )
    (project / "src/main.eqi").write_text("""
public property contract ForceHistory(input elapsed:s): N {
    derivatives first_open_intervals;
}
public property release Samples: ForceHistory {
    table {
        data force_samples;
        axis elapsed:s;
        value force:N;
        interpolation piecewise_affine;
        preprocessing identity;
        missing reject;
        knot_derivative reject;
        endpoint_derivative reject;
    }
    validity elapsed in [0[s],1[s]];
    outside reject;
    branch single;
    citation synthetic_definition;
    license repository_license;
}
public property contract SpeedHistory(input elapsed:s): m/s {
    derivatives first_open_intervals;
}
public property release SpeedSamples: SpeedHistory {
    table {
        data speed_samples;
        axis elapsed:s;
        value speed:m/s;
        interpolation piecewise_affine;
        preprocessing identity;
        missing reject;
        knot_derivative reject;
        endpoint_derivative reject;
    }
    validity elapsed in [0[s],1[s]];
    outside reject;
    branch single;
    citation synthetic_definition;
    license repository_license;
}
component Driven(parameter mass:kg, property force:ForceHistory, property measured:SpeedHistory) {
    state velocity:m/s;
    initial { velocity=0[m/s]; }
    relation motion { derivative(velocity)=force(elapsed=time())/mass; }
    observable speed:m/s=velocity;
    observable residual:m/s=velocity-measured(elapsed=time());
    observable normalized:1=residual / 0.01[m/s];
    observable loss:1=normalized*normalized/2;
}
public model Main() {
    parameter mass:kg=2;
    instance body:Driven(mass=mass, force=Samples, measured=SpeedSamples);
}
""")
    (project / "data/force_samples.json").write_text(json.dumps({
        "schema": "eqiora.resolved-array/v1", "encoding": "eqiora.canonical-json/v1",
        "scalar": "f64", "shape": [3, 2], "values": values,
    }))
    (project / "data/speed_samples.json").write_text(json.dumps({
        "schema": "eqiora.resolved-array/v1", "encoding": "eqiora.canonical-json/v1",
        "scalar": "f64", "shape": [5, 2],
        "values": [0., 0., .125, .015625, .5, .25, .875, .484375, 1., .5],
    }))
    (project / "docs/synthetic_definition.md").write_text(
        "Synthetic force history, milliseconds and millinewtons:\n\n" + measured
    )
    (project / "docs/repository_license.md").write_text("CC0-1.0 synthetic data.")
    store = tmp_path / "store"
    store.mkdir()
    resolution = q.resolve_local_project(project, store)
    model = q.compile_package(store, resolution, entry="Main")
    return model


@pytest.mark.parametrize("step", [0.25, 0.125])
def test_measured_force_drives_common_trajectory_and_replays(measured_model, step):
    model = measured_model
    velocity = model.field("body.velocity")
    speed = model.observable("body.speed")
    model = q.Model.from_bytes(model.to_bytes())
    velocity = model.field(velocity.id)
    plan = q.resolve(model, temporal=q.time.ImplicitMidpoint(
        step_s=step, relative_tolerance=1e-11,
        absolute_tolerances={(velocity, 0, 0, False): 1e-13},
    ))
    times = tuple(step * index for index in range(1, round(1 / step) + 1))
    result = q.run(plan, state=q.State.initial(plan), until_s=1.0, output_times_s=times)
    encoded = result.to_bytes()
    reopened = q.Result.from_bytes(q.Plan.from_bytes(plan.to_bytes()), encoded)
    # F/m = 2t until 1/2, then 2-2t. Its integral is t², then 2t-t²-1/2.
    # Midpoint is exact on each linear segment; both schedules contain the knot.
    for candidate in (result, reopened):
        assert candidate.to_bytes() == encoded
        assert candidate.series(velocity).dimension == (0, 1, -1, 0, 0, 0, 0)
        for index, (time, actual) in enumerate(candidate.series(velocity)):
            expected = time * time if time <= 0.5 else 2 * time - time * time - 0.5
            assert actual == pytest.approx(expected, abs=2e-12, rel=0)
            observed = candidate.observe(speed, output_index=index)
            assert observed.value == pytest.approx(expected, abs=2e-12, rel=0)
            assert observed.value_type == q.ValueType.real(q.Dimension(length=1, time=-1))
            assert observed.output == (index, time, q.State.from_result(plan, candidate, time_s=time).digest)
            assert observed.result_identity == json.loads(encoded)["identity"]


@pytest.mark.parametrize("times", [(0.125, 0.5, 0.875), (0.125, 0.875)])
def test_measured_residual_products_and_model_edits(measured_model, times):
    model = measured_model
    original = model.to_bytes()
    initial_parameter = model.parameter("mass")
    assert model.parameter("body.mass").id == initial_parameter.id
    assert initial_parameter.value_type == q.ValueType.real(q.Dimension(mass=1))
    velocity_id = model.field("body.velocity").id
    observables = {name: model.observable("body." + name).id
                   for name in ("speed", "residual", "normalized", "loss")}
    # Reopening retains exact identities; source aliases are authoring metadata.
    current = q.Model.from_bytes(original)
    for mass_value in (1., 2., 3.):
        current = current.commit(current.preview_value_edit(initial_parameter.id, mass_value))
        mass = current.parameter(initial_parameter.id)
        velocity = current.field(velocity_id)
        speed = current.observable(observables["speed"])
        forward = q.time.ForwardSensitivity(relative_tolerance=1e-10, absolute_tolerances=(
            q.time.SensitivityTolerance((velocity, 0, 0, False), mass, 1e-12,
                                        q.Dimension(mass=-1, length=1, time=-1)),
        ))
        plan = q.resolve(current, temporal=q.time.Tsitouras45(
            initial_step_s=1e-3, relative_tolerance=1e-11,
            absolute_tolerances={(velocity, 0, 0, False): 1e-13},
            forward_sensitivities=forward,
        ))
        result = q.run(plan, state=q.State.initial(plan), until_s=1., output_times_s=times)
        encoded = result.to_bytes()
        for candidate in (result, q.Result.from_bytes(q.Plan.from_bytes(plan.to_bytes()), encoded)):
            for index, time in enumerate(times):
                # Independently integrate F=4t before .5 and F=4-4t after it.
                impulse = 2*time*time if time <= .5 else 4*time-2*time*time-1
                expected = impulse/mass_value
                value = candidate.observe(speed, output_index=index)
                product = candidate.observe_parameter_jvp(speed,
                    {mass: (q.Dimension(mass=1), 1.)}, output_index=index)
                assert value.value == pytest.approx(expected, abs=1e-8, rel=0)
                assert product.value == pytest.approx(-expected/mass_value, abs=1e-8, rel=0)
                assert product.value_type == q.ValueType.real(q.Dimension(length=1,time=-1))
                residual = expected - impulse / 2
                derivative = -expected / mass_value
                for name, expected_value, expected_product, dimension in (
                    ("residual", residual, derivative, q.Dimension(length=1, time=-1)),
                    ("normalized", residual / .01, derivative / .01, q.Dimension()),
                    ("loss", .5 * (residual / .01)**2, residual * derivative / .01**2, q.Dimension()),
                ):
                    observable = current.observable(observables[name])
                    value = candidate.observe(observable, output_index=index)
                    product = candidate.observe_parameter_jvp(observable,
                        {mass: (q.Dimension(mass=1), 1.)}, output_index=index)
                    assert value.value_type == product.value_type == q.ValueType.real(dimension)
                    assert value.value == pytest.approx(expected_value, abs=1e-5, rel=0)
                    assert product.value == pytest.approx(expected_product, abs=1e-5, rel=0)
            assert candidate.to_bytes() == encoded
    assert model.to_bytes() == original
