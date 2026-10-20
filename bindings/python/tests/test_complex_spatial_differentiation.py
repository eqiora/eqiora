"""Complex spatial coefficients and derivatives retain their mathematical axes."""
import numpy as np
import pytest
import eqiora as q


@pytest.mark.parametrize("method", [q.fem.Q1, q.fvm.CellCenteredTpfa])
def test_complex_spatial_parameter_actions_use_real_pairing(method):
    graph = q.geometry.GeometryGraph()
    interval = graph.interval(bounds=(0., 1.))
    geometry = graph.build(interval, named_topology={"body": interval.region,
        "left": interval.boundaries[0], "right": interval.boundaries[1]})
    parent = geometry.selection("body")
    p = np.array([5+7j, -1+4j])
    load = np.array([11-2j, 3+8j])
    model = q.compile(source="""
      public component Channels(support body:volume(ambient_dimension=1),
        support left:boundary(parent=body), support right:boundary(parent=body),
        parameter a:complex<1>, parameter p:array<complex<1>,2>,
        parameter load:array<complex<1>,2>) {
        variable u:array<complex<1>,2> on body in h1;
        relation balance on body {-div(math.conj(a)*grad(u))=load/1[m^2];}
        relation left_value on left {trace(u)=p;}
        relation right_value on right {trace(u)=2*p;}
      }""", entry="Channels", geometry=geometry, bindings={
        "body": parent, "left": (geometry.selection("left"), parent),
        "right": (geometry.selection("right"), parent), "a": 2+3j,
        "p": p.tolist(), "load": load.tolist()})
    mesh = q.meshing.generate(q.meshing.resolve(geometry, q.meshing.CartesianMesher(cells=(4,))))
    plan = q.resolve(model, mesh=mesh, spatial=method(), solve=q.solve.Linear(
        algorithm=q.solve.LinearSolver.BiConjugateGradientStabilized,
        preconditioner=q.solve.Preconditioner.Identity, reduction=q.solve.Reduction.Reproducible,
        provider=q.solve.SolverProvider.reference(), relative_tolerance=1e-12,
        absolute_tolerance=1e-14, maximum_iterations=100))
    program = q.diff.compile(plan, inputs=tuple(model.parameter(name) for name in ("a", "p", "load")),
                             outputs=(plan.fields[0],))
    nodal = method is q.fem.Q1
    x = (np.arange(5 if nodal else 4) + (0 if nodal else .5)) / 4
    if nodal:
        # A complete complex seed need not solve the equations; only its exact
        # physical type, coefficient layout and essential trace are admitted.
        seed = (1+x[:, None])*p + x[:, None]*(1-x[:, None])*(3-5j)
        field = plan.fields[0]
        initial = q.State.initial(plan, fields=(q.InitialField(field, vertex_values=seed),))
        before = initial.to_bytes()
        for typed_seed in (seed.astype(np.complex64),
                           [[np.complex64(value) for value in row] for row in seed]):
            typed = q.State.initial(plan, fields=(q.InitialField(field, vertex_values=typed_seed),))
            assert typed.to_bytes() == before
        expected_seed = seed.copy()
        seed[1, 0] += 100+200j
        for state in (initial, q.State.from_bytes(plan, before)):
            assert state.time_s is None
            assert state.to_bytes() == before
            actual = state.field(field).values("vertex")
            assert actual.dtype == np.dtype("complex128")
            assert actual.shape == (5, 2)
            assert not actual.flags.writeable
            assert state.field(field).values("vertex") is actual
            np.testing.assert_array_equal(actual, expected_seed)
            with pytest.raises(ValueError):
                actual[0, 0] = 0
        wrong = expected_seed.copy()
        wrong[0, 0] += 1j
        for invalid in (wrong, expected_seed[:-1], expected_seed[:, 0]):
            with pytest.raises(q.ValidationError):
                q.State.initial(plan, fields=(q.InitialField(field, vertex_values=invalid),))
    # Directly integrate -conj(a)*u''=load. TPFA's half-cell Dirichlet
    # distances add load*h²/(8*conj(a)); Q1 has exact quadratic nodal values.
    f = x*(1-x)/2 + (0 if nodal else 1/128)
    da, dp, df = .5-.25j, np.array([2-3j, -4+1j]), np.array([5+6j, -2+7j])
    for a in (2+3j, 3-2j):
        point = program.evaluate((a, p, load))
        expected = (1+x[:, None])*p + f[:, None]*load/a.conjugate()
        tangent = (1+x[:, None])*dp + f[:, None]*(df/a.conjugate()-load*da.conjugate()/a.conjugate()**2)
        primal = np.asarray(point.primal().output[0])
        assert primal.shape == (len(x), 2)
        assert primal.dtype == np.dtype("complex128")
        np.testing.assert_allclose(primal, expected, rtol=0, atol=2e-10)
        np.testing.assert_allclose(point.jvp((da, dp, df)).tangent[0], tangent, rtol=0, atol=2e-10)
        weights = (np.arange(primal.size)+1).reshape(primal.shape)*(1-.5j)
        reverse = point.vjp((weights,))
        # Re <w,du> defines the cotangent. The a action is conjugate-linear,
        # whereas the p and load actions are complex-linear.
        ga = -np.sum(np.conjugate(weights)*f[:, None]*load/a.conjugate()**2)
        gp = np.sum((1+x[:, None])*weights, axis=0)
        gf = np.sum(f[:, None]*weights/a, axis=0)
        for actual, oracle in zip(reverse.input_cotangent, (ga, gp, gf), strict=True):
            np.testing.assert_allclose(actual, oracle, rtol=0, atol=2e-10)
        assert reverse.evidence.primal_solve is not None
