"""Equivalent decompositions of -u'' + u'/m = p/m² have u=p*(1+x/m)."""
import numpy as np
import pytest

import eqiora as q


@pytest.mark.parametrize("complex_values", [False, True])
@pytest.mark.parametrize("faer", [False, True])
@pytest.mark.parametrize("equation", [
    "-div(grad(u)-e*u/1[m])=p/1[m^2]",
    "-div(grad(u))+div(e*u/1[m])=p/1[m^2]",
    "div(e*u/1[m])-div(grad(u))=p/1[m^2]",
    "div(grad(u))-div(e*u/1[m])=-p/1[m^2]",
    "-div(grad(u))+div(p*coordinate(0)*e/1[m^2])=p/1[m^2]",
    "-div(2*grad(u))+div(grad(u))+div(e*u/1[m])=p/1[m^2]",
    "div(grad(u))-div(2*grad(u))+div(e*u/1[m])=p/1[m^2]",
    "div(2*grad(u))-div(grad(u))-div(e*u/1[m])=-p/1[m^2]",
    "-div((1+p)*grad(u))+div(p*grad(u))+div(e*u/1[m])=p/1[m^2]",
])
def test_scalar_principal_orientation_is_independent_of_other_fluxes(equation, faer, complex_values):
    graph = q.geometry.GeometryGraph()
    interval = graph.interval(bounds=(0., 2.))
    geometry = graph.build(interval, named_topology={
        "body": interval.region, "left": interval.boundaries[0], "right": interval.boundaries[1],
    })
    body = geometry.selection("body")
    domain = "complex<1>" if complex_values else "1"
    model = q.compile(source=f"""
      public component Transport(support body:volume(ambient_dimension=1),
        support left:boundary(parent=body), support right:boundary(parent=body),
        parameter p:{domain}) {{
        parameter e:vector<1,1>=tensor_value(frame=body,components=[1]);
        variable u:{domain} on body in smooth;
        relation balance on body {{{equation};}}
        relation left_value on left {{trace(u)=p;}}
        relation right_value on right {{trace(u)=3*p;}}
      }}
    """, entry="Transport", geometry=geometry, bindings={
        "body": body, "left": (geometry.selection("left"), body),
        "right": (geometry.selection("right"), body), "p": 2+3j if complex_values else 2.,
    })
    mesh = q.meshing.generate(q.meshing.resolve(geometry, q.meshing.CartesianMesher(cells=(4,))))
    plan = q.resolve(model, mesh=mesh, spatial=q.fem.Q1(), solve=q.solve.Linear(
        algorithm=q.solve.LinearSolver.SparseLu if faer else q.solve.LinearSolver.BiConjugateGradientStabilized,
        provider=q.solve.SolverProvider.faer() if faer else q.solve.SolverProvider.reference(),
        preconditioner=q.solve.Preconditioner.Identity,
        reduction=q.solve.Reduction.Fast if faer else q.solve.Reduction.Reproducible,
        relative_tolerance=1e-14, absolute_tolerance=1e-14, maximum_iterations=100))
    plan = q.Plan.from_bytes(plan.to_bytes())
    program = q.diff.compile(plan, inputs=(plan.model.parameter(model.parameter("p").id),),
                             outputs=(plan.fields[0],))
    parameter = -1+4j if complex_values else -1.
    direction = 1-2j if complex_values else 1.
    weights = np.arange(1., 6.) * (1+.5j if complex_values else 1.)
    point = program.evaluate((parameter,))
    shape = 1+np.arange(5)*.5
    np.testing.assert_allclose(point.primal().output[0], parameter*shape, rtol=0, atol=2e-10)
    np.testing.assert_allclose(point.jvp((direction,)).tangent[0], direction*shape, rtol=0, atol=2e-10)
    np.testing.assert_allclose(point.vjp((weights,)).input_cotangent[0], np.dot(shape, weights),
                               rtol=0, atol=2e-10)
