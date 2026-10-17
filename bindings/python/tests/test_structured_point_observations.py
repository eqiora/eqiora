"""Complete typed point queries use the installed ordinary spatial lifecycle."""
import pytest
import eqiora as q


@pytest.mark.parametrize("complex_domain", [False, True])
@pytest.mark.parametrize("method", [q.fem.Q1, q.fvm.CellCenteredTpfa])
def test_channel_point_values_partials_permutations_and_replay(complex_domain, method):
    length = q.Dimension(length=1)
    scalar = q.ValueType.complex if complex_domain else q.ValueType.real
    kind = q.ValueType.array(scalar(), 2)
    rate = q.ValueType.array(scalar(q.Dimension(length=-1)), 2)
    module = q.Module("main")
    owner = module.component("Channels")
    body = owner.volume("body", dimensions=1)
    left = owner.boundary("left", parent=body)
    right = owner.boundary("right", parent=body)
    x = owner.coordinate("x", value_type=q.ValueType.real(length), on=body, factor=body, axis=0)
    u = owner.field("u", value_type=kind, on=body, role=q.FieldRole.Variable,
                    spatial_regularity=q.SpatialRegularity.Smooth)
    phase = 2+3j if complex_domain else 2.
    owner.relation("balance", q.lang.equation(-q.lang.div(phase*q.lang.grad(u)), 0), on=body)
    for name, boundary, values in (
        ("left_value", left, (1+1j, -2+3j) if complex_domain else (1., -2.)),
        ("right_value", right, (3+0j, -1+5j) if complex_domain else (3., -1.)),
    ):
        owner.relation(name, q.lang.equation(q.lang.trace(u), q.lang.array(values)), on=boundary)
    owner.observable("profile", u, on=body, value_type=kind)
    owner.observable("permuted", q.lang.array((u[1], u[0])), on=body, value_type=kind)
    owner.observable("slope", q.lang.partial(u, wrt=x), on=body, value_type=rate)
    owner.observable("reflected", 2*q.lang.math.conj(u) if complex_domain else 2*u,
                     on=body, value_type=kind)
    energy = (q.lang.math.abs2(u[0]) + 2*q.lang.math.abs2(u[1]) if complex_domain
              else u[0]**2 + 2*u[1]**2)
    reflected = 2*q.lang.math.conj(u[1]) if complex_domain else 2*u[1]
    integral_type = q.ValueType.array(scalar(length), 4)
    integrated = owner.observable("integrated", q.lang.array((u[0], u[1], reflected, energy)),
                     value_type=integral_type, integrate_over=body)
    weights = q.lang.array((2+3j, -1+2j) if complex_domain else (2., -3.))
    composed = q.lang.array((weights[0]*integrated[1], weights[1]*integrated[0]))
    if complex_domain:
        composed = q.lang.math.conj(composed)
    composed = owner.observable("composed", composed, value_type=q.ValueType.array(scalar(length), 2))
    norm = (q.lang.math.abs2(composed[0])+q.lang.math.abs2(composed[1]) if complex_domain
            else composed[0]**2+composed[1]**2)
    owner.observable("norm", norm, value_type=q.ValueType.real(q.Dimension(length=2)))
    graph = q.geometry.GeometryGraph()
    interval = graph.interval(bounds=(0., 1.))
    geometry = graph.build(interval, named_topology={"body": interval.region,
        "left": interval.boundaries[0], "right": interval.boundaries[1]})
    parent = geometry.selection("body")
    bindings = {"body": parent, "left": (geometry.selection("left"), parent),
                "right": (geometry.selection("right"), parent)}
    direct = q.compile(source=module, entry="Channels", geometry=geometry, bindings=bindings)
    emitted = q.compile(source=module.to_eqi(), entry="Channels", geometry=geometry, bindings=bindings)
    assert direct.to_bytes() == emitted.to_bytes()
    mesh = q.meshing.generate(q.meshing.resolve(geometry, q.meshing.CartesianMesher(cells=(4,))))
    plan = q.resolve(direct, mesh=mesh, spatial=method(), solve=q.solve.Linear(
        algorithm=q.solve.LinearSolver.BiConjugateGradientStabilized,
        preconditioner=q.solve.Preconditioner.Identity, reduction=q.solve.Reduction.Reproducible,
        provider=q.solve.SolverProvider.reference(), relative_tolerance=0.,
        absolute_tolerance=1e-12, maximum_iterations=128))
    plan = q.Plan.from_bytes(plan.to_bytes())
    result = q.run(plan)
    for candidate in (result, q.Result.from_bytes(plan, result.to_bytes())):
        before = candidate.to_bytes()
        for position in (0.31, 0.44):
            # P0 reconstructs the cell average at x=3/8 throughout this cell;
            # Q1 reconstructs the affine Field at the requested point.
            at = position if method is q.fem.Q1 else 0.375
            expected = [1+2*at, -2+at]
            slope = [2., 1.] if method is q.fem.Q1 else [0., 0.]
            if complex_domain:
                expected = [expected[0]+1j*(1-at), expected[1]+1j*(3+2*at)]
                slope = [2-1j, 1+2j] if method is q.fem.Q1 else [0j, 0j]
            for name, values, value_type in (
                ("profile", expected, kind), ("permuted", expected[::-1], kind),
                ("slope", slope, rate),
                ("reflected", [2*value.conjugate() for value in expected], kind),
            ):
                observation = candidate.observe_at(direct.observable(f"definition.{name}"),
                                                    [(position, length)])
                assert observation.value_type == value_type
                assert observation.value == pytest.approx(values, abs=1e-9, rel=0)
                assert all(type(value) is (complex if complex_domain else float)
                           for value in observation.value)
        energy = (42. if complex_domain else 9.) - (0. if method is q.fem.Q1 else
                  (15./192. if complex_domain else 6./192.))
        expected = [2+0.5j, -1.5+4j, -3-8j, complex(energy)] if complex_domain else [2., -1.5, -3., energy]
        integrated = candidate.observe(direct.observable("definition.integrated"), quadrature_points=2)
        assert integrated.value_type == integral_type
        assert integrated.value == pytest.approx(expected, abs=1e-9, rel=0)
        entries = ([3., -2., -1., 4.] if complex_domain else [3., -1.]) * (5 if method is q.fem.Q1 else 4)
        direction = candidate.observable_state_tangent({direct.field("definition.u"): (q.Dimension(), entries)})
        for name, value, tangent in (
            ("composed", [-15-3.5j, -3-3.5j] if complex_domain else [-3., -6.],
             [-14-5j, 1-8j] if complex_domain else [-2., -9.]),
            ("norm", 258.5 if complex_domain else 45., 505. if complex_domain else 120.),
        ):
            observable = direct.observable(f"definition.{name}")
            primal = candidate.observe(observable, quadrature_points=2)
            jvp = candidate.observe_state_jvp(observable, direction, quadrature_points=2)
            expected_type = (q.ValueType.array(scalar(length), 2) if name == "composed"
                             else q.ValueType.real(q.Dimension(length=2)))
            assert primal.value_type == jvp.value_type == expected_type
            assert primal.value == pytest.approx(value, abs=1e-8, rel=0)
            assert jvp.value == pytest.approx(tangent, abs=1e-8, rel=0)
        assert candidate.to_bytes() == before
