"""Compare a CSV temperature sensor with exact retained heat-solve outputs.

Requires the development Eqiora checkout installed in the Python environment.
Run: python measured_heat_sensor.py --output measured-heat
Optional --csv and --license read the same mapped columns and their attribution.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import eqiora as q

COLUMNS = (("time_ms", q.units.s.prefixed("m")),
           ("x_mm", q.units.m.prefixed("m")),
           ("y_mm", q.units.m.prefixed("m")),
           ("temperature_K", q.units.K))
DIMENSIONS = (q.Dimension(time=1), q.Dimension(length=1),
              q.Dimension(length=1), q.Dimension(temperature=1))
CSV = "time_ms,x_mm,y_mm,temperature_K\n0,250,500,300\n250,250,500,300.04017857142856\n750,250,500,300.0467383381924\n"
STEP_S = 0.25
END_S = 0.75

SOURCE = r"""
public property contract TemperatureHistory(input elapsed:s): K { derivatives first_open_intervals; }
public property release Samples: TemperatureHistory {
 table { data temperature_samples; axis elapsed:s; value temperature:K;
 interpolation piecewise_affine; preprocessing identity; missing reject;
 knot_derivative reject; endpoint_derivative reject; }
 validity elapsed in [0[s],0.75[s]]; outside reject; branch single;
 citation measurement_source; license repository_license;
}
component Heat(support body:volume(ambient_dimension=2),
 support left:boundary(parent=body), support right:boundary(parent=body),
 support bottom:boundary(parent=body), support top:boundary(parent=body),
 property measured:TemperatureHistory) {
 state u:K on body in h1;
 initial {u=300[K];}
 law balance on body {storage 1[s/m^2]*u; flux -grad(u); source 1[K/m^2];}
 relation left_fixed on left {trace(u)=300[K];}
 relation right_fixed on right {trace(u)=300[K];}
 relation bottom_fixed on bottom {trace(u)=300[K];}
 relation top_fixed on top {trace(u)=300[K];}
 observable profile:K on body=u;
 observable residual:K on body=profile-measured(elapsed=time());
 observable normalized:1 on body=residual/0.01[K];
 observable loss:1 on body=normalized*normalized/2;
}
public model Main(support body:volume(ambient_dimension=2),
 support left:boundary(parent=body), support right:boundary(parent=body),
 support bottom:boundary(parent=body), support top:boundary(parent=body)) {
 instance plate:Heat(body=body,left=left,right=right,bottom=bottom,top=top,measured=Samples);
}
"""

def prepare(root: Path, text: str, *, license_text: str, columns=COLUMNS):
    """Bind one stationary sensor to the explicit body support and time grid."""
    array, dimensions = q.data.csv_array(text, columns=columns)
    if dimensions != DIMENSIONS:
        raise ValueError("Sensor columns require time, length, length and temperature dimensions")
    rows = tuple(zip(*(iter(array.values),) * 4, strict=True))
    if not rows or rows[0][0] != 0 or rows[-1][0] != END_S:
        raise ValueError("This experiment requires baseline 0 s and final 0.75 s records")
    if any(row[1:3] != rows[0][1:3] for row in rows):
        raise ValueError("This example has one stationary sensor on the body support")
    # Preserve recorded values and order. The property owner admits the time axis.
    samples = q.data.ResolvedArray.from_f64(
        (len(rows), 2), [value for row in rows for value in (row[0], row[3])],
    )
    project = root / "project"
    for name in ("src", "data", "docs"):
        (project / name).mkdir(parents=True)
    (project / "eqiora.toml").write_text(
        '[package]\nname="org.example.MeasuredHeat"\nversion="1.0.0"\nentry="main"\n'
    )
    (project / "src/main.eqi").write_text(SOURCE)
    (project / "data/temperature_samples.json").write_bytes(samples.to_bytes())
    mapping = "\n".join(
        f"{name}: normalized value and dimension of one input unit = {unit.normalize(1)}"
        for name, unit in columns
    )
    (project / "docs/measurement_source.md").write_text(
        "Stationary sensor on body; explicit column mapping to coherent SI:\n"
        + mapping + "\n\n" + text
    )
    (project / "docs/repository_license.md").write_text(license_text)
    store = root / "store"
    store.mkdir()
    resolution = q.resolve_local_project(project, store)

    graph = q.geometry.GeometryGraph()
    rectangle = graph.rectangle(x_bounds=(0., 1.), y_bounds=(0., 1.))
    sides = ("left", "right", "bottom", "top")
    geometry = graph.build(rectangle, named_topology={
        "body": rectangle.region, **dict(zip(sides, rectangle.boundaries, strict=True)),
    })
    body = geometry.selection("body")
    model = q.compile_package(store, resolution, entry="Main", geometry=geometry, bindings={
        "body": body, **{side: (geometry.selection(side), body) for side in sides},
    })
    mesh = q.meshing.generate(q.meshing.resolve(
        geometry, q.meshing.CartesianMesher(cells=(2, 2)),
    ))
    plan = q.resolve(model, mesh=mesh, spatial=q.fem.Q1(),
        temporal=q.time.BackwardEuler(STEP_S), solve=q.solve.Linear(
            algorithm=q.solve.LinearSolver.SparseLu,
            preconditioner=q.solve.Preconditioner.Identity,
            reduction=q.solve.Reduction.Fast, provider=q.solve.SolverProvider.faer(),
            relative_tolerance=1e-12, absolute_tolerance=1e-14, maximum_iterations=100,
        ))
    result = q.run(plan, state=q.State.initial(plan), until_s=END_S,
                   output_times_s=tuple(row[0] for row in rows[1:]))
    return model, plan, result, rows


def compare(model, result, rows, observable_ids):
    """Query native typed residuals at each recorded position and accepted time."""
    observations = []
    for index, (time, x, y, measured) in enumerate(rows[1:]):
        point = [(x, DIMENSIONS[1]), (y, DIMENSIONS[2])]
        values = {name: result.observe_at(model.observable(identity), point, output_index=index)
                  for name, identity in observable_ids.items()}
        if values["profile"].output[1] != time:
            raise ValueError("Recorded time differs from the exact retained output")
        observations.append(values)
    return observations


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("measured-heat"))
    parser.add_argument("--csv", type=Path)
    parser.add_argument("--license", type=Path, help="Attribution/license text for supplied CSV data")
    args = parser.parse_args()
    if args.csv and not args.license:
        parser.error("--csv requires --license with the data's attribution/license text")
    text = args.csv.read_text() if args.csv else CSV
    license_text = args.license.read_text() if args.license else "CC0-1.0 synthetic data."
    model, plan, result, rows = prepare(args.output, text, license_text=license_text)
    ids = {name: model.observable("plate." + name).id
           for name in ("profile", "residual", "normalized", "loss")}
    before = result.to_bytes()
    observed = compare(model, result, rows, ids)
    for values in observed:
        print(f"t={values['profile'].output[1]:g} s, "
              f"T={values['profile'].value:.12g} K, "
              f"residual={values['residual'].value:.6g} K, "
              f"loss={values['loss'].value:.6g}")
    for name, artifact in (("model", model), ("plan", plan), ("result", result)):
        (args.output / (name + ".json")).write_bytes(artifact.to_bytes())
    reopened_model = q.Model.from_bytes((args.output / "model.json").read_bytes())
    reopened_plan = q.Plan.from_bytes((args.output / "plan.json").read_bytes())
    reopened = q.Result.from_bytes(reopened_plan, (args.output / "result.json").read_bytes())
    assert reopened.to_bytes() == before
    replay = compare(reopened_model, reopened, rows, ids)
    assert [[item.value for item in row.values()] for row in replay] == [
        [item.value for item in row.values()] for row in observed]


if __name__ == "__main__":
    main()
