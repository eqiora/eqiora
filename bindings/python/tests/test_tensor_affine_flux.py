"""An affine tensor solution checks composed gradients, traces and real adjoints."""
import numpy as np
import pytest

import eqiora as q


@pytest.mark.parametrize("faer", [False, True])
@pytest.mark.parametrize("tpfa", [False, True])
def test_tensor_affine_flux_preserves_component_permutation_and_total_derivatives(faer, tpfa):
    graph = q.geometry.GeometryGraph()
    rectangle = graph.rectangle(x_bounds=(0., 2.), y_bounds=(0., 3.))
    sides = ("left", "right", "bottom", "top")
    geometry = graph.build(rectangle, named_topology={
        "body": rectangle.region,
        **dict(zip(sides, rectangle.boundaries, strict=True)),
    })
    body = geometry.selection("body")
    flux = "math.complex(4,3)*grad(u)+grad(p*transpose(math.conj((1+math.sqrt(s))*u)))+g"
    model = q.compile(source=f"""
    public component TensorFlux(support body:volume(ambient_dimension=2),
        support left:boundary(parent=body), support right:boundary(parent=body),
        support bottom:boundary(parent=body), support top:boundary(parent=body),
        parameter p:complex<1>, parameter s:1) {{
        parameter c:tensor<complex<1>,2,2>=tensor_value(frame=body,
            components=[[math.complex(1,2),math.complex(3,-1)],
                        [math.complex(-2,4),math.complex(5,3)]]);
        parameter e0:vector<1,2>=tensor_value(frame=body,components=[1,0]);
        parameter e1:vector<1,2>=tensor_value(frame=body,components=[0,1]);
        variable g:tensor<complex<1/m>,2,2,2> on body in h1;
        relation drift on body {{g=p*outer(c,coordinate(0)*e0+2*coordinate(1)*e1)/1[m^2];}}
        variable u:tensor<complex<1>,2,2> on body in smooth;
        relation balance on body {{-div({flux})=-3*p*c/1[m^2];}}
        relation left_value on left {{trace(u)=p*c*(1+coordinate(0)/1[m]+2*coordinate(1)/1[m]);}}
        relation bottom_value on bottom {{trace(u)=p*c*(1+coordinate(0)/1[m]+2*coordinate(1)/1[m]);}}
        relation right_flux on right {{normal({flux})=
            (math.complex(4,3)*p*c+p*math.conj(p)*transpose(math.conj(c))+2*p*c)/1[m];}}
        relation top_flux on top {{normal({flux})=
            (2*math.complex(4,3)*p*c+2*p*math.conj(p)*transpose(math.conj(c))+6*p*c)/1[m];}}
    }}
    """, entry="TensorFlux", geometry=geometry, bindings={
        "body": body, **{side: (geometry.selection(side), body) for side in sides}, "p": 2+3j, "s": 0.,
    })
    mesh = q.meshing.generate(q.meshing.resolve(
        geometry, q.meshing.CartesianMesher(cells=(2, 3))))
    # s is held fixed at zero; sqrt(s) has a value but no derivative there.
    # Its presence must not affect differentiation with respect to p.
    # A=L⊗K: L is the scalar Q1/TPFA stiffness, K(U)=(4+3i)*U+p*conj(U)^T.
    # Exact rational elimination of these small blocks bounds
    # ||w||*||du||+||lambda||*||A*du|| below 4e4 at all selected points.
    # The paired residual terms thus budget <4e-11 at 1e-15, leaving margin
    # inside the unchanged 2e-10 oracle. Four full 96-coordinate recurrences
    # also budget the reference solver's residual corrections.
    plan = q.resolve(model, mesh=mesh, spatial=q.fvm.CellCenteredTpfa() if tpfa else q.fem.Q1(), solve=q.solve.Linear(
        algorithm=q.solve.LinearSolver.SparseLu if faer else q.solve.LinearSolver.BiConjugateGradientStabilized,
        preconditioner=q.solve.Preconditioner.Identity,
        reduction=q.solve.Reduction.Fast if faer else q.solve.Reduction.Reproducible,
        provider=q.solve.SolverProvider.faer() if faer else q.solve.SolverProvider.reference(),
        relative_tolerance=1e-15, absolute_tolerance=1e-15, maximum_iterations=384))
    plan = q.Plan.from_bytes(plan.to_bytes())
    result = q.run(plan)
    assert q.Result.from_bytes(plan, result.to_bytes()).to_bytes() == result.to_bytes()
    program = q.diff.compile(plan, inputs=(plan.model.parameter(model.parameter("p").id),),
                             outputs=(plan.fields[0],))
    # u=p*C*(1+x+2y), g=p*outer(C,(x,2y)). Its divergence is 3*p*C.
    # Component transposition is independent of the derivative axis.
    axes = [np.arange(n if tpfa else n+1)+(0.5 if tpfa else 0.) for n in (2, 3)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 2)
    c = np.array([[1+2j, 3-1j], [-2+4j, 5+3j]])
    response = (1+points @ np.array([1., 2.]))[:, None, None]*c
    direction = 2-1j
    w = np.array([[2-1j, -3+2j], [1+4j, -2-3j]])/16
    weights = np.arange(1, len(points)+1)[:, None, None]*w
    for p in (2+3j, 0j, -1+4j):
        point = program.evaluate((p,))
        primal = point.primal().output[0]
        assert primal.shape == response.shape
        np.testing.assert_allclose(primal, response*p, rtol=0, atol=2e-10)
        tangent = point.jvp((direction,)).tangent[0]
        np.testing.assert_allclose(tangent, response*direction, rtol=0, atol=2e-10)
        reverse = point.vjp((weights,)).input_cotangent[0]
        np.testing.assert_allclose(reverse, np.sum(response.conj()*weights), rtol=0, atol=2e-10)
        assert np.vdot(weights, tangent).real == pytest.approx(
            np.vdot(reverse, direction).real, rel=0, abs=2e-10)
