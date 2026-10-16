"""Physical integrals share exact support, quadrature and immutable Result ownership."""
import pytest

import eqiora as q


def mini_result():
    graph = q.geometry.GeometryGraph()
    body = graph.rectangle(x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
    sides = ("left", "right", "bottom", "top")
    geometry = graph.build(body, named_topology={
        "body": body.region,
        **dict(zip(sides, body.boundaries, strict=True)),
    })
    bindings = {"body": geometry.selection("body")}
    bindings.update({side: (geometry.selection(side), bindings["body"]) for side in sides})
    source = """
public model Flow(
    support body:volume(ambient_dimension=2),
    support left:boundary(parent=body), support right:boundary(parent=body),
    support bottom:boundary(parent=body), support top:boundary(parent=body),
    support path:interval(m)
) {
    variable u:vector<m/s,2> on body in smooth;
    variable p:kg/(m*s^2) on body in h1;
    variable g:m^2/s on body in smooth;
    variable f:kg/(m*s^2) on body in smooth;
    relation prescribed_velocity on body {g=1[m/s]*coordinate(0);}
    relation force on body {f=1[kg/(m^2*s^2)]*coordinate(0)-1[kg/(m*s^2)];}
    relation momentum on body {-div(2[kg/(m*s)]*symmetric_part(grad(u))-isotropic_lift(p))-grad(f)=0;}
    relation constraint on body {div(u)=0;}
    relation inlet on left {trace(u)=trace(grad(g));}
    relation lower on bottom {trace(u)=trace(grad(g));}
    relation upper on top {trace(u)=trace(grad(g));}
    relation outlet on right {normal(2[kg/(m*s)]*symmetric_part(grad(u))-isotropic_lift(p))=0;}
    coordinate x:m on body from body[0];
    coordinate y:m on body from body[1];
    coordinate s:m on path from path;
    observable line:kg/(m*s^2) on path=pullback(p,from=(s),at=(x=s,y=0.5[m]));
    observable pressure_integral:kg*m/s^2=integral(p,measure(body));
    observable left_pressure:kg/s^2=integral(trace(p),measure(left));
    observable area:m^2=integral(1,measure(body));
    observable prescribed_integral:kg*m/s^2=integral(f,measure(body));
    observable velocity_integral:m^3/s=integral(component(u,indices=(0,)),measure(body));
}
"""
    bindings["path"] = q.CoordinateInterval(0, 1, dimension=q.Dimension(length=1))
    model = q.compile(source=source, entry="Flow", geometry=geometry, bindings=bindings)
    mesh = q.meshing.generate(q.meshing.resolve(geometry, q.meshing.AffineTriangleMesher(cells=(2, 3))))
    # Point values expose iterative error that a domain integral can cancel.
    # Resolve the linear system more tightly than the fixed observation bound.
    plan = q.resolve(model, mesh=mesh, spatial=q.fem.MiniP1(), solve=q.solve.Linear(
        algorithm=q.solve.LinearSolver.BiConjugateGradientStabilized,
        preconditioner=q.solve.Preconditioner.Identity,
        reduction=q.solve.Reduction.Reproducible, provider=q.solve.SolverProvider.reference(),
        relative_tolerance=1e-13, absolute_tolerance=1e-15, maximum_iterations=1000,
    ))
    return model, plan, q.run(plan)


def test_pressure_queries_do_not_reconstruct_unrequested_mini_velocity():
    model, plan, result = mini_result()
    accepted = result.to_bytes()
    replay = q.Result.from_bytes(q.Plan.from_bytes(plan.to_bytes()), accepted)
    force = q.Dimension(mass=1, length=1, time=-2)
    # Constant u=(1,0) has zero strain/divergence; grad(p)=grad(f) and
    # zero right traction give p=x-1. Unit-square and left-edge integrals
    # are -1/2 and -1; the prescribed f has the same volume integral.
    for candidate in (result, replay):
        for name, expected, dimension in (
            ("pressure_integral", -0.5, force),
            ("left_pressure", -1.0, q.Dimension(mass=1, time=-2)),
            ("area", 1.0, q.Dimension(length=2)),
            ("prescribed_integral", -0.5, force),
        ):
            observed = candidate.observe(model.observable(name), quadrature_points=2)
            assert observed.value == pytest.approx(expected, abs=1e-9, rel=0)
            assert observed.value_type == q.ValueType.real(dimension)
        with pytest.raises(q.ValidationError, match="continuous linear nodal Field space"):
            candidate.observe(model.observable("velocity_integral"), quadrature_points=2)
        # The same P1 reconstruction restricts p=x-1 to the path y=1/2.
        for coordinate in (0., .125, .5, .875, 1.):
            sample = candidate.observe_at(model.observable("line"), [(coordinate, q.Dimension(length=1))])
            assert sample.value == pytest.approx(coordinate-1., abs=1e-9, rel=0)
            assert sample.value_type == q.ValueType.real(q.Dimension(mass=1, length=-1, time=-2))
        assert candidate.to_bytes() == accepted


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
