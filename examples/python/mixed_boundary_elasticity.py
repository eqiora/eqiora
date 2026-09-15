"""Run and optionally plot the mixed-boundary elasticity case."""

import argparse
from importlib.resources import files
from pathlib import Path

import eqiora


def solve() -> tuple[eqiora.Plan, eqiora.Result]:
    graph = eqiora.geometry.GeometryGraph()
    rectangle = graph.rectangle(x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
    geometry = graph.build(
        rectangle,
        named_topology={
            "body": rectangle.region,
            "x_lower": rectangle.boundaries[0],
            "x_upper": rectangle.boundaries[1],
            "y_lower": rectangle.boundaries[2],
            "y_upper": rectangle.boundaries[3],
        },
    )
    mesh_request = eqiora.meshing.CartesianMesher(cells=(16, 16))
    mesh_plan = eqiora.meshing.resolve(geometry, mesh_request)
    mesh = eqiora.meshing.generate(mesh_plan)
    model = eqiora.compile(
        path=files(eqiora).joinpath("examples", "mixed-boundary-elasticity.eqi"),
        geometry=geometry,
        entry="MixedBoundaryElasticity2d",
        bindings={
            "body": geometry.selection("body"),
            **{
                side: (geometry.selection(side), geometry.selection("body"))
                for side in ("x_lower", "x_upper", "y_lower", "y_upper")
            },
            "mu": 3.0, "lambda": 0.0, "length_scale": 1.0,
        },
    )
    plan = eqiora.resolve(
        model,
        mesh=mesh,
        spatial=eqiora.fem.Q1(),
        solve=eqiora.solve.Linear(
            algorithm=eqiora.solve.LinearSolver.ConjugateGradient,
            preconditioner=eqiora.solve.Preconditioner.Identity,
            reduction=eqiora.solve.Reduction.Reproducible,
            provider=eqiora.solve.SolverProvider.reference(),
            relative_tolerance=1.0e-10,
            absolute_tolerance=1.0e-12,
            maximum_iterations=10_000,
        ),
    )
    return plan, eqiora.run(plan)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--displacement-png",
        type=Path,
        help="save the displacement image (requires eqiora[matplotlib])",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="visible displacement scale used only by the optional still",
    )
    arguments = parser.parse_args()

    plan, result = solve()
    evidence = eqiora.solid.linear_elasticity_evidence(result)
    print(result.plan_key)
    print(evidence.solve)
    print("constrained reaction", evidence.constrained_reaction, "N/m")
    print("integrated body force", evidence.integrated_body_force, "N/m")
    if arguments.displacement_png is not None:
        import eqiora.matplotlib as eqplot

        figure = eqplot.plot_deformed_field(
            result,
            field=plan.capability.displacement,
            scale=arguments.scale,
        )
        figure.savefig(arguments.displacement_png, dpi=160)
        print("displacement still", arguments.displacement_png)


if __name__ == "__main__":
    main()
