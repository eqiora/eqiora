"""Reduced spatial outputs share the solved Field and explicit integration rule."""
import numpy as np
import pytest
import eqiora as q


def _fixture(method, complex_values, extra_source="", *, solve=None):
    graph = q.geometry.GeometryGraph()
    interval = graph.interval(bounds=(0., 3.))
    geometry = graph.build(interval, named_topology={"body": interval.region,
        "left": interval.boundaries[0], "right": interval.boundaries[1]})
    parent = geometry.selection("body")
    source = """
      public component Channels(support body:volume(ambient_dimension=1),
        support left:boundary(parent=body), support right:boundary(parent=body),
        parameter a:complex<1>, parameter p:array<complex<1>,2>,
        parameter load:array<complex<1>,2>) {
        variable u:array<complex<1>,2> on body in h1;
        relation balance on body {-div(math.conj(a)*grad(u))=load/1[m^2];}
        relation left_value on left {trace(u)=p;}
        relation right_value on right {trace(u)=2*p;}
        observable totals:array<complex<m>,2> = integral(u, measure(body));
        observable energy:m = integral(math.abs2(u[0])+2*math.abs2(u[1]), measure(body));
        observable combined:array<complex<m>,2> = [math.conj(a*totals[1]), totals[0]+3[m]*p[1]];
        observable loss:m = energy+math.real(combined[0]);
      }"""
    if not complex_values:
        source = (source.replace("complex<1>", "1").replace("complex<m>", "m")
                  .replace("math.conj(a)", "a").replace("math.conj(a*totals[1])", "a*totals[1]")
                  .replace("math.real(combined[0])", "combined[0]")
                  .replace("math.abs2(u[0])", "u[0]*u[0]")
                  .replace("math.abs2(u[1])", "u[1]*u[1]"))
    source = source.rsplit("}", 1)[0] + extra_source + "}"
    values = np.array([2+3j, 5+7j, -1+4j, 11-2j, 3+8j])
    direction = np.array([.5-.25j, 2-3j, -4+1j, 5+6j, -2+7j])
    if not complex_values:
        values, direction = values.real.copy(), direction.real.copy()
    model = q.compile(source=source, entry="Channels", geometry=geometry, bindings={
        "body": parent, "left": (geometry.selection("left"), parent),
        "right": (geometry.selection("right"), parent), "a": values[0].item(),
        "p": values[1:3].tolist(), "load": values[3:].tolist()})
    mesh = q.meshing.generate(q.meshing.resolve(geometry, q.meshing.CartesianMesher(cells=(4,))))
    plan = q.resolve(model, mesh=mesh, spatial=method(), solve=solve or q.solve.Linear(
        algorithm=q.solve.LinearSolver.BiConjugateGradientStabilized,
        preconditioner=q.solve.Preconditioner.Identity, reduction=q.solve.Reduction.Reproducible,
        provider=q.solve.SolverProvider.reference(), relative_tolerance=1e-14,
        absolute_tolerance=1e-14, maximum_iterations=100))
    return model, plan, values, direction


@pytest.mark.parametrize("method", [q.fem.Q1, q.fvm.CellCenteredTpfa])
@pytest.mark.parametrize("complex_values", [False, True])
@pytest.mark.parametrize("solve", [
    None,
    q.solve.Linear(algorithm=q.solve.LinearSolver.SparseLu,
        preconditioner=q.solve.Preconditioner.Identity, reduction=q.solve.Reduction.Fast,
        provider=q.solve.SolverProvider.faer(), relative_tolerance=1e-14,
        absolute_tolerance=1e-14, maximum_iterations=100),
])
def test_spatial_observable_total_derivatives(method, complex_values, solve):
    model, plan, values, direction = _fixture(method, complex_values, solve=solve)
    inputs = tuple(model.parameter(name) for name in ("a", "p", "load"))
    outputs = (model.observable("definition.loss"), plan.fields[0],
               model.observable("definition.combined"))
    with pytest.raises(q.ValidationError, match="quadrature"):
        q.diff.compile(plan, inputs=inputs, outputs=outputs)
    program = q.diff.compile(plan, inputs=inputs, outputs=outputs, quadrature_points=2)
    nodal = method is q.fem.Q1
    x = (np.arange(5 if nodal else 4) + (0 if nodal else .5)) * .75
    alpha = 1+x/3
    f = x*(3-x)/2 + (0 if nodal else 9/128)
    # Exact moments of the represented Q1 interpolant / TPFA constants on [0,3].
    ia, iff, iaa, iaf, ifff = ((4.5, 135/64, 7., 405/128, 1863/1024) if nodal
                              else (4.5, 81/32, 447/64, 243/64, 1215/512))

    def parts(v):
        return v[0].item(), v[1:3], v[3:]

    def oracle(v, d):
        a, p, load = parts(v)
        da, dp, dl = parts(d)
        c = load/np.conjugate(a)
        dc = dl/np.conjugate(a)-load*np.conjugate(da)/np.conjugate(a)**2
        total, dt = ia*p+iff*c, ia*dp+iff*dc
        weights = np.array([1., 2.])
        energy = np.sum(weights*(iaa*abs(p)**2+2*iaf*(np.conjugate(p)*c).real+ifff*abs(c)**2))
        de = np.sum(weights*(2*iaa*(np.conjugate(p)*dp).real
                    +2*iaf*(np.conjugate(dp)*c+np.conjugate(p)*dc).real
                    +2*ifff*(np.conjugate(c)*dc).real))
        combined = np.array([np.conjugate(a*total[1]), total[0]+3*p[1]])
        derivative = np.array([np.conjugate(da*total[1]+a*dt[1]), dt[0]+3*dp[1]])
        return ((energy+combined[0].real, alpha[:, None]*p+f[:, None]*c, combined),
                (de+derivative[0].real, alpha[:, None]*dp+f[:, None]*dc, derivative))

    for a in (values[0], 3-2j if complex_values else 3.):
        values[0] = a
        point = program.evaluate(parts(values))
        assert point.primal().evidence.steps[0].primal_solves[0].backend == plan.solve.backend
        expected, tangent = oracle(values, direction)
        for actual, wanted in zip(point.primal().output, expected, strict=True):
            assert np.asarray(actual).shape == np.asarray(wanted).shape
            assert np.asarray(actual).dtype == np.asarray(wanted).dtype
            np.testing.assert_allclose(actual, wanted, rtol=0, atol=2e-10)
        for actual, wanted in zip(point.jvp(parts(direction)).tangent, tangent, strict=True):
            np.testing.assert_allclose(actual, wanted, rtol=0, atol=2e-10)
        weights = tuple((np.arange(np.size(v))+1).reshape(np.shape(v))*.25
                        * (1-.5j if np.iscomplexobj(v) else 1.) for v in expected)
        gradient = np.zeros_like(values)
        for index in range(len(values)):
            for unit in ((1., 1j) if complex_values else (1.,)):
                seed = np.zeros_like(values)
                seed[index] = unit
                gradient[index] += unit*sum(np.vdot(w, dy).real for w, dy in
                                             zip(weights, oracle(values, seed)[1], strict=True))
        for actual, wanted in zip(point.vjp(weights).input_cotangent, parts(gradient), strict=True):
            np.testing.assert_allclose(actual, wanted, rtol=0, atol=2e-10)


@pytest.mark.parametrize("method", [q.fem.Q1, q.fvm.CellCenteredTpfa])
@pytest.mark.parametrize("complex_values", [False, True])
def test_spatial_observable_frozen_inputs_respect_selected_parameters(method, complex_values):
    real = lambda expression: f"math.real({expression})" if complex_values else expression
    for active in (False, True):
        source = real("p[0]") + "-5" if active else real("load[0]") + "-11"
        extra = f"""
            variable fixed:1 on body;
            relation prescribed on body {{fixed={source};}}
            observable output:m = integral(math.sqrt(fixed)+{real('u[0]')}, measure(body));
        """
        model, plan, values, direction = _fixture(method, complex_values, extra)
        inputs = tuple(model.parameter(name) for name in ("a", "p"))
        outputs = (model.observable("definition.output"),)
        if active:
            with pytest.raises(q.ValidationError, match="derivative"):
                q.diff.compile(plan, inputs=inputs, outputs=outputs, quadrature_points=2)
            continue
        program = q.diff.compile(plan, inputs=inputs, outputs=outputs, quadrature_points=2)
        point = program.evaluate((values[0].item(), values[1:3]))
        a, load = values[0], values[3]
        moment = 135/64 if method is q.fem.Q1 else 81/32
        expected = 4.5*values[1].real+moment*(load/np.conjugate(a)).real
        derivative = 4.5*direction[1].real-moment*(
            load*np.conjugate(direction[0])/np.conjugate(a)**2).real
        np.testing.assert_allclose(point.primal().output[0], expected, rtol=0, atol=2e-10)
        np.testing.assert_allclose(point.jvp((direction[0].item(), direction[1:3])).tangent[0],
                                   derivative, rtol=0, atol=2e-10)
        gradient = point.vjp((1.,)).input_cotangent
        np.testing.assert_allclose(gradient[0], -moment*load/np.conjugate(a)**2, rtol=0, atol=2e-10)
        np.testing.assert_allclose(gradient[1], [4.5, 0.], rtol=0, atol=2e-10)
