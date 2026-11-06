"""Run the imported conservative scalar law with installed development Eqiora."""
from pathlib import Path
import eqiora


def geometry_and_bindings():
    graph = eqiora.geometry.GeometryGraph()
    square = graph.rectangle(x_bounds=(0., 1.), y_bounds=(0., 1.))
    names = ("left", "right", "bottom", "top")
    geometry = graph.build(square, named_topology={"body": square.region, **dict(zip(names, square.boundaries))})
    body = geometry.selection("body")
    return geometry, {"body": body, **{name: (geometry.selection(name), body) for name in names}}


def resolve(model, geometry):
    mesh = eqiora.meshing.generate(eqiora.meshing.resolve(geometry, eqiora.meshing.CartesianMesher(cells=(2, 2))))
    return eqiora.resolve(model, mesh=mesh, spatial=eqiora.fem.Q1(), temporal=eqiora.time.BackwardEuler(step_s=0.25), solve=eqiora.solve.Linear(
        algorithm=eqiora.solve.LinearSolver.BiConjugateGradientStabilized,
        preconditioner=eqiora.solve.Preconditioner.Identity,
        reduction=eqiora.solve.Reduction.Reproducible,
        provider=eqiora.solve.SolverProvider.reference(),
        relative_tolerance=1e-12, absolute_tolerance=1e-14, maximum_iterations=1000))


def main():
    geometry, bindings = geometry_and_bindings()
    model = eqiora.compile(path=Path(__file__).with_name("model.eqi"), entry="Main", geometry=geometry, bindings=bindings)
    plan = resolve(model, geometry)
    result = eqiora.run(plan, state=eqiora.State.initial(plan), steps=3, output_steps=(1, 2, 3))
    field = model.field("definition.concentration")
    for state in result.trajectory.states:
        print(state.time_s, state.field(field).values("vertex"))


if __name__ == "__main__":
    main()
