"""Ordered coupled Fields preserve separate primal and derivative leaves."""
import numpy as np
import pytest
import eqiora as q


@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("complex_values", [False, True])
@pytest.mark.parametrize("method", [q.fem.Q1, q.fvm.CellCenteredTpfa])
def test_coupled_spatial_field_selection_and_real_pairing(method, complex_values, mixed):
    graph = q.geometry.GeometryGraph()
    interval = graph.interval(bounds=(0., 3.))
    geometry = graph.build(interval, named_topology={
        "body": interval.region, "left": interval.boundaries[0], "right": interval.boundaries[1]})
    parent = geometry.selection("body")
    scalar = "complex<1>" if complex_values else "1"
    coefficient = "math.conj(a)" if complex_values else "a"
    a = 2+3j if complex_values else 2.
    p = np.array([5+7j, -1+4j]) if complex_values else np.array([5., -1.])
    load = np.array([11-2j, 3+8j]) if complex_values else np.array([11., 3.])
    source = """
      public component Coupled(support body:volume(ambient_dimension=1),
        support left:boundary(parent=body), support right:boundary(parent=body),
        parameter a:VALUE, parameter p:array<VALUE,2>, parameter load:array<VALUE,2>) {
        variable u:array<VALUE,2> on body in h1;
        variable v:array<VALUE,2> on body in h1;
        relation first on body {-div(COEFFICIENT*grad(u)+grad(v))=load/1[m^2];}
        relation second on body {-div(grad(v))=2*load/1[m^2];}
        relation left_u on left {trace(u)=p;}
        relation right_u on right {trace(u)=2*p;}
        relation left_v on left {trace(v)=3*p;}
        relation right_v on right {trace(v)=6*p;}
      }""".replace("VALUE", scalar).replace("COEFFICIENT", coefficient)
    if mixed:
        source = (source.replace("on body in h1", "on body in smooth")
                  .replace("-div(grad(v))", "-div(a*grad(v))")
                  .replace("trace(v)=6*p;", "normal(a*grad(v))=a*p/1[m]-3*load/1[m];"))
    model = q.compile(source=source, entry="Coupled", geometry=geometry, bindings={
        "body": parent, "left": (geometry.selection("left"), parent),
        "right": (geometry.selection("right"), parent), "a": a,
        "p": p.tolist(), "load": load.tolist()})
    mesh = q.meshing.generate(q.meshing.resolve(geometry, q.meshing.CartesianMesher(cells=(4,))))
    plan = q.resolve(model, mesh=mesh, spatial=method(), solve=q.solve.Linear(
        algorithm=q.solve.LinearSolver.BiConjugateGradientStabilized,
        preconditioner=q.solve.Preconditioner.Identity, reduction=q.solve.Reduction.Reproducible,
        provider=q.solve.SolverProvider.reference(), relative_tolerance=1e-12,
        absolute_tolerance=1e-14, maximum_iterations=200))
    nodal = method is q.fem.Q1
    x = (np.arange(5 if nodal else 4) + (0 if nodal else .5)) * .75
    # Integrating the second equation first gives v''=-2*load and u''=load/conj(a).
    # The same centred TPFA balance adds h²/8 to the quadratic shape factor.
    f = (x*(3-x)/2 + (0 if nodal else .75**2/8))[:, None]
    lift = (1+x/3)[:, None]
    da = .5-.25j if complex_values else .5
    dp = np.array([2-3j, -4+1j]) if complex_values else np.array([2., -4.])
    dq = np.array([5+6j, -2+7j]) if complex_values else np.array([5., -2.])
    for names in [("u",), ("v",), ("v", "u"), ("u", "v")]:
        program = q.diff.compile(plan, inputs=tuple(model.parameter(name) for name in ("a", "p", "load")),
                                 outputs=tuple(model.field(f"definition.{name}") for name in names))
        for current_a in (a, 3-2j if complex_values else 3.):
            point = program.evaluate((current_a, p, load))
            c = np.conjugate(current_a)
            b = current_a if mixed else 1.
            db = da if mixed else 0.
            factor = 1-2/b
            values = {"u": lift*p+f*load*factor/c, "v": 3*lift*p+2*f*load/b}
            tangents = {"u": lift*dp+f*((dq*factor+2*load*db/b**2)/c
                                       -load*factor*np.conjugate(da)/c**2),
                        "v": 3*lift*dp+2*f*(dq/b-load*db/b**2)}
            weights = tuple((np.arange(values[name].size)+1+i*7).reshape(values[name].shape)
                            * (1-.5j if complex_values else 1.) for i, name in enumerate(names))
            for name, value, tangent in zip(names, point.primal().output, point.jvp((da, dp, dq)).tangent, strict=True):
                assert np.asarray(value).shape == (len(x), 2)
                assert np.asarray(value).dtype == np.dtype("complex128" if complex_values else "float64")
                np.testing.assert_allclose(value, values[name], rtol=0, atol=2e-9)
                np.testing.assert_allclose(tangent, tangents[name], rtol=0, atol=2e-9)
            by_field = dict(zip(names, weights, strict=True))
            wu, wv = by_field.get("u", 0.), by_field.get("v", 0.)
            # The u coefficient contributes both a holomorphic and an
            # antiholomorphic action when the second diffusion is a Parameter.
            hu = 2*f*load/(b**2*c) if mixed else 0.
            hv = -2*f*load/b**2 if mixed else 0.
            au = -f*load*factor/c**2
            expected = (np.sum(np.conjugate(hu)*wu+au*np.conjugate(wu)+np.conjugate(hv)*wv),
                        np.sum(lift*(wu+3*wv), axis=0),
                        np.sum(f*(np.conjugate(factor/c)*wu+2/np.conjugate(b)*wv), axis=0))
            for actual, oracle in zip(point.vjp(weights).input_cotangent, expected, strict=True):
                np.testing.assert_allclose(actual, oracle, rtol=0, atol=2e-8)
