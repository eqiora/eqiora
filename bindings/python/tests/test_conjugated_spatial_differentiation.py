"""Independent inverse and real adjoint of a U + b conjugate(U)^T = P."""
import numpy as np
import pytest

import eqiora as q


@pytest.mark.parametrize("tpfa", [False, True])
@pytest.mark.parametrize("native", [False, True])
@pytest.mark.parametrize("faer", [False, True])
def test_tensor_conjugation_shares_primal_partials_and_observation_products(native, faer, tpfa):
    graph = q.geometry.GeometryGraph()
    rectangle = graph.rectangle(x_bounds=(0., 2.), y_bounds=(0., 3.))
    sides = ("left", "right", "bottom", "top")
    geometry = graph.build(rectangle, named_topology={
        "body": rectangle.region,
        **dict(zip(sides, rectangle.boundaries, strict=True)),
    })
    body = geometry.selection("body")
    b = 0j if native else 1-2j
    model = q.compile(source=f"""
    public model Tensor(support body:volume(ambient_dimension=2),
        support left:boundary(parent=body), support right:boundary(parent=body),
        support bottom:boundary(parent=body), support top:boundary(parent=body)) {{
        parameter a:complex<1>=math.complex(2,3);
        parameter b:complex<1>=math.complex({b.real},{b.imag});
        parameter p:tensor<complex<1>,2,2>=tensor_value(frame=body,
            components=[[math.complex(1,2),math.complex(3,-1)],
                        [math.complex(-2,4),math.complex(5,3)]]);
        variable u:tensor<complex<1>,2,2> on body in h1;
        relation balance on body {{a*u+b*transpose(math.conj(u))=p;}}
        observable integrated:tensor<complex<m^2>,2,2>=integral(u,measure(body));
        observable total:tensor<complex<m^2>,2,2>=integrated+p*1[m^2];
    }}""", entry="Tensor", geometry=geometry, bindings={
        "body": body, **{side: (geometry.selection(side), body) for side in sides},
    })
    mesh = q.meshing.generate(q.meshing.resolve(
        geometry, q.meshing.CartesianMesher(cells=(2, 3))))
    # For this unit-cell Q1 mass matrix, ||M^-1||_2 <= 24. With
    # sigma_min(a I+b J) >= sqrt(13)-sqrt(5), the two solve residuals
    # bound coefficient-gradient error by 7.83e4 * relative_tolerance.
    # TPFA has six unit-area cells, so M=I and the same bound is conservative.
    # Request 1e-15 to leave rounding margin inside the fixed 2e-10 oracle.
    # Budget four recurrences of the 96 real coordinates, including residual corrections.
    plan = q.resolve(model, mesh=mesh, spatial=q.fvm.CellCenteredTpfa() if tpfa else q.fem.Q1(), solve=q.solve.Linear(
        algorithm=q.solve.LinearSolver.SparseLu if faer else q.solve.LinearSolver.BiConjugateGradientStabilized,
        preconditioner=q.solve.Preconditioner.Identity,
        reduction=q.solve.Reduction.Fast if faer else q.solve.Reduction.Reproducible,
        provider=q.solve.SolverProvider.faer() if faer else q.solve.SolverProvider.reference(),
        relative_tolerance=1e-15, absolute_tolerance=1e-15, maximum_iterations=384))
    plan = q.Plan.from_bytes(plan.to_bytes())
    inputs = tuple(plan.model.parameter(model.parameter(name).id) for name in ("a", "b", "p"))
    total = plan.model.observable(model.observable("total").id)
    program = q.diff.compile(plan, inputs=inputs, outputs=(total, plan.fields[0]), quadrature_points=2)
    p = np.array([[1+2j, 3-1j], [-2+4j, 5+3j]])
    dp = np.array([[2+1j, -1+3j], [4-2j, -3-1j]])
    da, db = .5-.25j, -.75+.5j
    w = np.array([[2-1j, -3+2j], [1+4j, -2-3j]])
    count = 6 if tpfa else 12
    field_w = np.arange(1, count+1)[:, None, None]*w
    # The same immutable Program also crosses the numerical-domain boundary.
    for a, b, parameter in ((2+3j, b, p), (3+2j, b, p+dp),
                            (2+3j, 1-2j if native else 0j, p)):
        # J(U)=conj(U)^T is an involution. Multiplying by conj(a)-b*J
        # gives U=(conj(a)*P-b*J(P))/(|a|^2-|b|^2).
        denominator = (a.conjugate()*a-b.conjugate()*b).real
        numerator = a.conjugate()*parameter-b*parameter.conj().T
        u = numerator/denominator
        # Differentiate this explicit quotient, independently of residual AD.
        ddenominator = 2*(a.conjugate()*da-b.conjugate()*db).real
        dnumerator = da.conjugate()*parameter+a.conjugate()*dp-db*parameter.conj().T-b*dp.conj().T
        du = (dnumerator*denominator-numerator*ddenominator)/denominator**2
        point = program.evaluate((a, b, parameter))
        expected = (6*u+parameter, np.broadcast_to(u, (count, 2, 2)))
        tangent = (6*du+dp, np.broadcast_to(du, (count, 2, 2)))
        for actual, wanted in zip(point.primal().output, expected, strict=True):
            assert actual.shape == wanted.shape
            np.testing.assert_allclose(actual, wanted, rtol=0, atol=2e-10)
        for actual, wanted in zip(point.jvp((da, db, dp)).tangent, tangent, strict=True):
            np.testing.assert_allclose(actual, wanted, rtol=0, atol=2e-10)
        # L* = conj(a) I + b J under Re(sum(conj(W)*U)).
        combined = 6*w+field_w.sum(axis=0)
        adjoint = (a*combined-b*combined.conj().T)/denominator
        expected_reverse = (-(adjoint*u.conj()).sum(), -(adjoint*u.T).sum(), adjoint+w)
        reverse = point.vjp((w, field_w)).input_cotangent
        for actual, wanted in zip(reverse, expected_reverse, strict=True):
            np.testing.assert_allclose(actual, wanted, rtol=0, atol=2e-10)
        lhs = sum(np.vdot(weight, value).real for weight, value in zip((w, field_w), tangent, strict=True))
        rhs = sum(np.vdot(weight, value).real for weight, value in zip(reverse, (da, db, dp), strict=True))
        assert lhs == pytest.approx(rhs, rel=0, abs=2e-10)


@pytest.mark.parametrize("faer", [False, True])
@pytest.mark.parametrize("tpfa", [False, True])
@pytest.mark.parametrize("dimension", [1, 2])
def test_scalar_affine_flux_preserves_natural_trace_and_total_products(faer, tpfa, dimension):
    graph = q.geometry.GeometryGraph()
    shape_geometry = (graph.interval(bounds=(0., 2.)) if dimension == 1
                      else graph.rectangle(x_bounds=(0., 2.), y_bounds=(0., 3.)))
    sides = ("left", "right") if dimension == 1 else ("left", "right", "bottom", "top")
    geometry = graph.build(shape_geometry, named_topology={
        "body": shape_geometry.region,
        **dict(zip(sides, shape_geometry.boundaries, strict=True)),
    })
    body = geometry.selection("body")
    profile = "1+coordinate(0)/1[m]" + ("+2*coordinate(1)/1[m]" if dimension == 2 else "")
    flux = "math.complex(4,3)*grad(u)+p*math.conj(grad(u))+g"
    extra_supports = ("support bottom:boundary(parent=body), support top:boundary(parent=body),"
                      if dimension == 2 else "")
    extra_data = ("parameter e1:vector<1,2>=tensor_value(frame=body,components=[0,1]);"
                 if dimension == 2 else "")
    extra_flux = "+2*coordinate(1)*e1" if dimension == 2 else ""
    extra_boundaries = (f"""
        relation bottom_value on bottom {{trace(u)=p*({profile});}}
        relation top_flux on top {{normal({flux})=
            (2*math.complex(4,3)*p+2*p*math.conj(p)+6*p)/1[m];}}
    """ if dimension == 2 else "")
    model = q.compile(source=f"""
    public component AffineFlux(support body:volume(ambient_dimension={dimension}),
        support left:boundary(parent=body), support right:boundary(parent=body),
        {extra_supports} parameter p:complex<1>) {{
        parameter e:vector<1,{dimension}>=tensor_value(frame=body,components={"[1]" if dimension == 1 else "[1,0]"});
        {extra_data}
        variable g:vector<complex<1/m>,{dimension}> on body in h1;
        relation drift on body {{g=p*(coordinate(0)*e{extra_flux})/1[m^2];}}
        variable u:complex<1> on body in smooth;
        relation balance on body {{-div({flux})=-{1 if dimension == 1 else 3}*p/1[m^2];}}
        relation left_value on left {{trace(u)=p*({profile});}}
        relation right_flux on right {{normal({flux})=
            (math.complex(4,3)*p+p*math.conj(p)+2*p)/1[m];}}
        {extra_boundaries}
    }}
    """, entry="AffineFlux", geometry=geometry, bindings={
        "body": body, **{side: (geometry.selection(side), body) for side in sides}, "p": 2+3j,
    })
    counts = (2,) if dimension == 1 else (2, 3)
    mesh = q.meshing.generate(q.meshing.resolve(
        geometry, q.meshing.CartesianMesher(cells=counts)))
    plan = q.resolve(model, mesh=mesh, spatial=q.fvm.CellCenteredTpfa() if tpfa else q.fem.Q1(), solve=q.solve.Linear(
        algorithm=q.solve.LinearSolver.SparseLu if faer else q.solve.LinearSolver.BiConjugateGradientStabilized,
        preconditioner=q.solve.Preconditioner.Identity,
        reduction=q.solve.Reduction.Fast if faer else q.solve.Reduction.Reproducible,
        provider=q.solve.SolverProvider.faer() if faer else q.solve.SolverProvider.reference(),
        relative_tolerance=1e-12, absolute_tolerance=1e-14, maximum_iterations=100))
    plan = q.Plan.from_bytes(plan.to_bytes())
    result = q.run(plan)
    assert q.Result.from_bytes(plan, result.to_bytes()).to_bytes() == result.to_bytes()
    program = q.diff.compile(plan, inputs=(plan.model.parameter(model.parameter("p").id),),
                             outputs=(plan.fields[0],))
    # u=p*(1+x+2y), with y omitted in 1D; g=p*(x,2y).
    # The Cartesian coefficient inventory uses its last axis fastest.
    # This affine solution and its cell averages are exact for both methods.
    # Its real adjoint is sum((1+x+2y)*w), independently of residual AD.
    axes = [np.arange(count if tpfa else count+1)+(0.5 if tpfa else 0.) for count in counts]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, dimension)
    shape = 1+points @ np.arange(1, dimension+1)
    direction = 2-1j
    weights = np.arange(1, shape.size+1)+1j*np.arange(shape.size, 0, -1)
    for p in (2+3j, 0j, -1+4j):
        point = program.evaluate((p,))
        np.testing.assert_allclose(point.primal().output[0], shape*p, rtol=0, atol=2e-10)
        tangent = point.jvp((direction,)).tangent[0]
        np.testing.assert_allclose(tangent, shape*direction, rtol=0, atol=2e-10)
        reverse = point.vjp((weights,)).input_cotangent[0]
        np.testing.assert_allclose(reverse, np.sum(shape*weights), rtol=0, atol=2e-10)
        assert np.vdot(weights, tangent).real == pytest.approx(
            np.vdot(reverse, direction).real, rel=0, abs=2e-10)
