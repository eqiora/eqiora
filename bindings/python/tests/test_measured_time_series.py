"""Measured forcing through the existing table, Model and common ODE owners."""
import csv
import io
import json

import eqiora as q
import pytest


@pytest.mark.parametrize("step", [0.25, 0.125])
def test_measured_force_drives_common_trajectory_and_replays(tmp_path, step):
    # Explicit coherent-SI column mapping; the table owns interpolation and validity.
    measured = "elapsed_s,force_N\n0,0\n0.5,2\n1,0\n"
    reader = csv.DictReader(io.StringIO(measured))
    values = [float(row[column]) for row in reader for column in ("elapsed_s", "force_N")]
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
component Driven(parameter mass:kg, property force:ForceHistory) {
    state velocity:m/s;
    initial { velocity=0[m/s]; }
    relation motion { derivative(velocity)=force(elapsed=time())/mass; }
    observable speed:m/s=velocity;
}
public model Main() {
    instance body:Driven(mass=2[kg],force=Samples);
}
""")
    (project / "data/force_samples.json").write_text(json.dumps({
        "schema": "eqiora.resolved-array/v1", "encoding": "eqiora.canonical-json/v1",
        "scalar": "f64", "shape": [3, 2], "values": values,
    }))
    (project / "docs/synthetic_definition.md").write_text(
        "Synthetic force history, seconds and newtons:\n\n" + measured
    )
    (project / "docs/repository_license.md").write_text("CC0-1.0 synthetic data.")
    store = tmp_path / "store"
    store.mkdir()
    resolution = q.resolve_local_project(project, store)
    model = q.compile_package(store, resolution, entry="Main")
    velocity = model.field("body.velocity")
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
        for time, actual in candidate.series(velocity):
            expected = time * time if time <= 0.5 else 2 * time - time * time - 0.5
            assert actual == pytest.approx(expected, abs=2e-12, rel=0)
