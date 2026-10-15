"""Physical integrals share exact support, quadrature and immutable Result ownership."""
import pytest

import eqiora as q


@pytest.mark.parametrize("mesh_kind", ["Q1", "P1 rectangle", "P1 partition"])
def test_physical_affine_integrals_through_installed_result(mesh_kind):
    simplex = mesh_kind != "Q1"
    partition = mesh_kind == "P1 partition"
    graph = q.geometry.GeometryGraph()
    names = ("left", "right", "bottom", "top")
    if partition:
        west = graph.rectangle(x_bounds=(0.0, 2.0), y_bounds=(0.0, 3.0))
        east = graph.rectangle(x_bounds=(2.0, 4.0), y_bounds=(0.0, 3.0))
        operation = graph.partition(west, east, interface=(west.boundaries[1], east.boundaries[0]))
        regions = {"west": west, "east": east}
    else:
        operation = graph.rectangle(x_bounds=(0.0, 4.0), y_bounds=(0.0, 3.0))
        regions = {"whole": operation}
    topology = {name: region.region for name, region in regions.items()}
    topology.update({f"{name}_{side}": boundary for name, region in regions.items()
                     for side, boundary in zip(names, region.boundaries, strict=True)})
    geometry = graph.build(operation, named_topology=topology)
    provider = q.meshing.AffineTriangleMesher if simplex else q.meshing.CartesianMesher
    mesh = q.meshing.generate(q.meshing.resolve(geometry, provider(cells=(2, 2))))

    # Author each Region's Field and reductions in the same lexical component.
    # Each mesh production retains its own exact geometry and Region closure.
    bindings = {"auxiliary": q.CoordinateInterval(0, 2, dimension=q.Dimension(length=1))}
    supports = ["support auxiliary:interval(m)"]
    declarations = []
    for name in regions:
        bindings[name] = geometry.selection(name)
        supports.append(f"support {name}:volume(ambient_dimension=2)")
        temperature = f"{name}_temperature"
        declarations += [
            f"variable {temperature}:K on {name} in smooth;",
            f"relation {name}_balance on {name} {{-div(grad({temperature}))=0[K/m^2];}}",
        ]
        for side in names:
            boundary = f"{name}_{side}"
            bindings[boundary] = (geometry.selection(boundary), bindings[name])
            supports.append(f"support {boundary}:boundary(parent={name})")
            if not partition or boundary not in ("west_right", "east_left"):
                declarations.append(
                    f"relation {boundary}_fixed on {boundary} {{trace({temperature})=300[K]+2[K/m]*coordinate(0)-3[K/m]*coordinate(1);}}"
                )
            declarations.append(
                f"observable {boundary}_flux:W/m=integral(normal(-2[W/(m*K)]*grad({temperature})),measure({boundary}));"
            )
        declarations += [
            f"observable {name}_top_temperature:K*m=integral(trace({temperature}),measure({name}_top));",
            f"observable {name}_volume_temperature:K*m^2=integral({temperature},measure({name}));",
        ]
    first, last = next(iter(regions)), next(reversed(regions))
    if partition:
        declarations += [
            "instance outgoing:TemperatureInterface(body=west,face=west_right,value=west_temperature);",
            "instance incoming:TemperatureInterface(body=east,face=east_left,value=east_temperature);",
            "connect outgoing.edge,incoming.edge;",
        ]
    declarations += [
        f"observable left_flux:W/m={first}_left_flux;",
        f"observable right_flux:W/m={last}_right_flux;",
        "observable net_flux:W/m=left_flux+right_flux+bottom_flux+top_flux;",
        "observable auxiliary_length:m=integral(1,measure(auxiliary));",
    ]
    for output, unit in (("bottom_flux", "W/m"), ("top_flux", "W/m"),
                         ("top_temperature", "K*m"), ("volume_temperature", "K*m^2")):
        expression = "+".join(f"{name}_{output}" for name in regions)
        declarations.append(f"observable {output}:{unit}={expression};")
    source = """
public connector TemperatureBoundary {
    trace value:K;
    flux outward_flux:K/m;
    shape [];
    frame invariant;
    pairing euclidean_boundary_duality;
    orientation parent_outward;
}
public component TemperatureInterface(
    support body:volume(ambient_dimension=2),
    support face:boundary(parent=body),
    variable value:K on body in smooth,
    port edge:TemperatureBoundary over face
) {
    relation carrier on face {
        trace(value)=edge.value;
        normal(grad(value))=edge.outward_flux;
    }
}
""" + "public component Heat(" + ",".join(supports) + ") {" + "\n".join(declarations) + "}"
    model = q.compile(source=source, entry="Heat", geometry=geometry, bindings=bindings)
    spatial = (tuple(q.fem.P1().at(model.domain(name)) for name in regions)
               if simplex else q.fem.Q1())
    plan = q.resolve(model, mesh=mesh, spatial=spatial, solve=q.solve.Linear(
        algorithm=q.solve.LinearSolver.BiConjugateGradientStabilized,
        preconditioner=q.solve.Preconditioner.Identity,
        reduction=q.solve.Reduction.Reproducible, provider=q.solve.SolverProvider.reference(),
        relative_tolerance=1e-11, absolute_tolerance=1e-13, maximum_iterations=1000,
    ))
    result = q.run(plan)
    accepted = result.to_bytes()
    replay = q.Result.from_bytes(q.Plan.from_bytes(plan.to_bytes()), accepted)
    flux_type = q.ValueType.real(q.Dimension(mass=1, length=1, time=-3))
    # Harmonic T=300+2*x-3*y gives heat flux (-4,6) W/m². Multiply
    # each signed normal component by the edge length, 3 or 4 metres.
    # On the top edge mean T=295 K; over area 12 m² mean T=299.5 K.
    for candidate in (result, replay):
        for name, expected, value_type, dimension in (
            ("left_flux", 12, flux_type, 1),
            ("right_flux", -12, flux_type, 1),
            ("bottom_flux", -24, flux_type, 1),
            ("top_flux", 24, flux_type, 1),
            ("net_flux", 0, flux_type, 1),
            ("top_temperature", 1180, q.ValueType.real(q.Dimension(temperature=1, length=1)), 1),
            ("volume_temperature", 3594, q.ValueType.real(q.Dimension(temperature=1, length=2)), 2),
        ):
            observation = candidate.observe(model.observable(f"definition.{name}"), quadrature_points=2)
            assert observation.value == pytest.approx(expected, abs=1e-9, rel=0)
            assert observation.value_type == value_type
            kind = "SimplexDuffyGaussLegendre" if simplex else "GaussLegendre"
            assert set(observation.quadratures.values()) == {(kind, dimension, 2)}
        # An independent coordinate factor keeps its own reference family even
        # when this Result's physical Mesh consists of simplices.
        auxiliary = candidate.observe(model.observable("definition.auxiliary_length"), quadrature_points=2)
        assert auxiliary.value == pytest.approx(2, abs=1e-12, rel=0)
        assert auxiliary.value_type == q.ValueType.real(q.Dimension(length=1))
        assert set(auxiliary.quadratures.values()) == {("GaussLegendre", 1, 2)}
        assert candidate.to_bytes() == accepted
