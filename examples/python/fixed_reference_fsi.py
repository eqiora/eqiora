"""Compile, resolve, and run the common fixed-reference FSI path."""

from importlib.resources import files

import numpy as np
import eqiora


def geometry_and_mesh() -> tuple[eqiora.geometry.Geometry, eqiora.meshing.Mesh]:
    graph = eqiora.geometry.GeometryGraph()
    fluid = graph.rectangle(x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
    solid = graph.rectangle(x_bounds=(1.0, 2.0), y_bounds=(0.0, 1.0))
    partition = graph.partition(
        fluid, solid, interface=(fluid.boundaries[1], solid.boundaries[0])
    )
    geometry = graph.build(
        partition,
        named_topology={
            "fluid": fluid.region,
            "fluid_x_lower": fluid.boundaries[0],
            "fluid_x_upper": fluid.boundaries[1],
            "fluid_y_lower": fluid.boundaries[2],
            "fluid_y_upper": fluid.boundaries[3],
            "solid": solid.region,
            "solid_x_lower": solid.boundaries[0],
            "solid_x_upper": solid.boundaries[1],
            "solid_y_lower": solid.boundaries[2],
            "solid_y_upper": solid.boundaries[3],
        },
    )
    request = eqiora.meshing.AffineTriangleMesher(cells=(2, 2))
    mesh_plan = eqiora.meshing.resolve(geometry, request)
    return geometry, eqiora.meshing.generate(mesh_plan)


def solve() -> eqiora.Result:
    geometry, mesh = geometry_and_mesh()
    model = eqiora.compile(
        path=files(eqiora).joinpath("examples", "fixed-reference-fsi.eqi"),
        geometry=geometry,
        entry="FixedReferenceFsi2d",
        bindings={
            **{region: geometry.selection(region) for region in ("fluid", "solid")},
            **{
                f"{region}_{side}": (
                    geometry.selection(f"{region}_{side}"), geometry.selection(region)
                )
                for region in ("fluid", "solid")
                for side in ("x_lower", "x_upper", "y_lower", "y_upper")
            },
            "fluid_density": 2.0,
            "fluid_viscosity": 0.5,
            "solid_density": 3.0,
            "solid_mu": 4.0,
            "solid_lambda": 2.0,
            "zero_pressure": 0.0,
        },
    )
    fluid = model.domain("fluid")
    solid = model.domain("solid")
    plan = eqiora.resolve(
        model,
        mesh=mesh,
        spatial=(eqiora.fem.MiniP1().at(fluid), eqiora.fem.P1().at(solid)),
        temporal=eqiora.time.BackwardEuler(step_s=0.05),
        solve=eqiora.solve.Linear(
            algorithm=eqiora.solve.LinearSolver.MinimumResidual,
            preconditioner=eqiora.solve.Preconditioner.Identity,
            reduction=eqiora.solve.Reduction.Reproducible,
            provider=eqiora.solve.SolverProvider.reference(),
            relative_tolerance=1.0e-11,
            absolute_tolerance=1.0e-13,
            maximum_iterations=20_000,
        ),
        scaling=None,
    )
    coordinates = np.asarray(mesh.coordinates)
    cells = np.asarray(mesh.cells)
    fluid_vertices = np.flatnonzero(coordinates[:, 0] <= 1.0)
    solid_vertices = np.flatnonzero(coordinates[:, 0] >= 1.0)
    flow_cell_indices = np.flatnonzero(coordinates[cells, 0].mean(axis=1) < 1.0)
    solid_displacement = np.zeros((solid_vertices.size, 2))
    interface_midpoint = np.flatnonzero(
        (coordinates[solid_vertices, 0] == 1.0)
        & (coordinates[solid_vertices, 1] == 0.5)
    )
    assert interface_midpoint.size == 1
    solid_displacement[interface_midpoint[0], 0] = 0.02
    fluid_velocity = model.field("definition.fluid_velocity")
    fluid_pressure = model.field("definition.fluid_pressure")
    solid_velocity = model.field("definition.solid_velocity")
    solid_displacement_field = model.field("definition.solid_displacement")
    state = eqiora.State.initial(
        plan,
        time_s=0.0,
        fields=(
            eqiora.InitialField(
                fluid_velocity,
                vertex_values=np.zeros((fluid_vertices.size, 2)),
                cell_values=np.zeros((flow_cell_indices.size, 2)),
            ),
            eqiora.InitialField(
                fluid_pressure,
                vertex_values=np.full(fluid_vertices.size, 0.25),
            ),
            eqiora.InitialField(
                solid_velocity,
                vertex_values=np.zeros((solid_vertices.size, 2)),
            ),
            eqiora.InitialField(
                solid_displacement_field,
                vertex_values=solid_displacement,
            ),
        ),
    )
    return eqiora.run(plan, state=state, steps=2, output_steps=(1, 2))


if __name__ == "__main__":
    result = solve()
    print(result.trajectory.digest)
    print(eqiora.fsi.evidence(result).states[-1].solve)
