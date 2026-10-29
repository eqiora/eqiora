"""Solve and optionally plot the Python-authored exact-cylinder Stokes case."""

import argparse
from importlib.resources import files
from pathlib import Path

import eqiora


def solve() -> tuple[eqiora.Result, eqiora.FieldRef, eqiora.geometry.Geometry]:
    graph = eqiora.geometry.GeometryGraph()
    rectangle = graph.rectangle(x_bounds=(0.0, 2.2), y_bounds=(0.0, 0.41))
    circle = graph.circle(center=(0.2, 0.2), radius=0.05)
    fluid = graph.subtract(rectangle, circle)
    geometry = graph.build(
        fluid,
        named_topology={
            "fluid": fluid.region,
            "inlet": rectangle.boundaries[0],
            "outlet": rectangle.boundaries[1],
            "walls": rectangle.boundaries[2:4],
            "cylinder": circle.boundaries[0],
        },
    )
    mesh_request = eqiora.meshing.GmshMesher(
        maximum_boundary_error=1e-4,
        maximum_target_size=0.025,
        minimum_mean_ratio=1e-5,
        maximum_boundary_facets=50,
    )
    mesh = eqiora.meshing.generate(geometry, mesh_request)
    source_path = files(eqiora).joinpath("examples", "steady-flow-past-cylinder.eqi")
    channel_height = geometry.bounds[1][1] - geometry.bounds[1][0]
    model = eqiora.compile(
        path=source_path,
        geometry=geometry,
        entry="SteadyFlowPastCylinder",
        bindings={
            "fluid": geometry.selection("fluid"),
            **{
                side: (geometry.selection(side), geometry.selection("fluid"))
                for side in ("inlet", "outlet", "walls", "cylinder")
            },
            "dynamic_viscosity": 1.0e-3,
            "zero_pressure": 0.0,
            "inlet_speed": 0.3,
            "channel_height": channel_height,
        },
    )
    linear = eqiora.solve.Linear(
        algorithm=eqiora.solve.LinearSolver.SparseLu,
        preconditioner=eqiora.solve.Preconditioner.Identity,
        reduction=eqiora.solve.Reduction.Fast,
        provider=eqiora.solve.SolverProvider.faer(),
        relative_tolerance=1e-6,
        absolute_tolerance=1e-13,
        maximum_iterations=10_000,
    )
    plan = eqiora.resolve(
        model,
        mesh=mesh,
        spatial=eqiora.fem.MiniP1(),
        solve=linear,
        scaling=None,
    )
    return eqiora.run(plan), plan.capability.pressure, geometry


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pressure-png",
        type=Path,
        help="save the accepted P1 pressure still (requires eqiora[matplotlib])",
    )
    arguments = parser.parse_args()

    result, pressure_field, geometry = solve()
    pressure = result.output(pressure_field)
    pressure_values = pressure.values("vertex")
    cylinder_force = result.boundary_force(geometry.selection("cylinder"))
    inlet_flux = result.boundary_flux(geometry.selection("inlet"))
    outlet_flux = result.boundary_flux(geometry.selection("outlet"))
    print(result.plan_key)
    print(result.solve)
    print(
        "pressure",
        min(pressure_values),
        max(pressure_values),
        "Pa on",
        pressure.coefficient_count("vertex"),
        "vertices",
    )
    print("cylinder force on fluid", cylinder_force.on_domain, "N/m")
    print("net flux", inlet_flux.value + outlet_flux.value, "m^2/s")
    if arguments.pressure_png is not None:
        import eqiora.matplotlib as eqplot

        figure = eqplot.plot_scalar_field(result, field=pressure_field)
        figure.savefig(arguments.pressure_png, dpi=180)
        print("pressure still", arguments.pressure_png)


if __name__ == "__main__":
    main()
