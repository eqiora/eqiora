"""Run maintained steady and transient heat with installed Eqiora."""
from pathlib import Path
import tempfile

import eqiora


def geometry_and_bindings(length=1.0, height=1.0):
    graph = eqiora.geometry.GeometryGraph()
    square = graph.rectangle(x_bounds=(0.0, length), y_bounds=(0.0, height))
    names = ("x_lower", "x_upper", "y_lower", "y_upper")
    geometry = graph.build(square, named_topology={
        "body": square.region,
        **dict(zip(names, square.boundaries)),
    })
    body = geometry.selection("body")
    bindings = {"body": body, "conductivity": 1.0, "heating": 1.0, **{name: (geometry.selection(name), body) for name in names}}
    return geometry, bindings


def convection_geometry_and_bindings(length=2.0, height=1.0, conductivity=3.0,
                                     coefficient=2.0, fixed=280.0, ambient=320.0):
    geometry, bindings = geometry_and_bindings(length, height)
    bindings.pop("heating")
    bindings.update(conductivity=conductivity, coefficient=coefficient,
                    fixed_temperature=fixed, ambient_temperature=ambient)
    return geometry, bindings


def resolve(model, geometry, temporal=None):
    mesh = eqiora.meshing.generate(eqiora.meshing.resolve(
        geometry, eqiora.meshing.CartesianMesher(cells=(2, 2))))
    linear = eqiora.solve.Linear(
        algorithm=eqiora.solve.LinearSolver.BiConjugateGradientStabilized,
        preconditioner=eqiora.solve.Preconditioner.Identity,
        reduction=eqiora.solve.Reduction.Reproducible,
        provider=eqiora.solve.SolverProvider.reference(),
        relative_tolerance=1e-10, absolute_tolerance=1e-12, maximum_iterations=1000,
    )
    return eqiora.resolve(model, mesh=mesh, spatial=eqiora.fem.Q1(), solve=linear, temporal=temporal)


def main():
    project = Path(__file__).resolve().parent
    geometry, bindings = geometry_and_bindings()
    with tempfile.TemporaryDirectory(prefix="eqiora-heated-body-") as scratch:
        store = Path(scratch) / "store"
        store.mkdir()
        lock = eqiora.resolve_local_project(project, store)
        model = eqiora.compile_package(store, lock, entry="HeatedBody",
                                      geometry=geometry, bindings=bindings)
        plan = resolve(model, geometry)
        result = eqiora.run(plan)
        field = model.field("definition.temperature")
        print("Steady temperature coefficients [K]:")
        print(result.output(field).values("vertex").numpy())
        transient = eqiora.compile_package(
            store, lock, entry="TransientHeatedBody", geometry=geometry,
            bindings={**bindings, "capacity": 1.0},
        )
        plan = resolve(transient, geometry, eqiora.time.BackwardEuler(step_s=1/24))
        result = eqiora.run(plan, state=eqiora.State.initial(plan), steps=3, output_steps=(1, 2, 3))
        temperature = transient.field("definition.temperature")
        print("Transient temperature coefficients [K]:")
        for state in result.trajectory.states:
            print(state.time_s, state.field(temperature).values("vertex"))

        geometry, bindings = convection_geometry_and_bindings()
        convection = eqiora.compile_package(
            store, lock, entry="ConvectiveHeatedBody", geometry=geometry, bindings=bindings)
        result = eqiora.run(resolve(convection, geometry))
        print("Steady convective temperature coefficients [K]:")
        print(result.output(convection.field("definition.temperature")).values("vertex").numpy())


if __name__ == "__main__":
    main()
