"""Tensor component axes survive a spatial solve and total derivative products."""
import numpy as np
import pytest

import eqiora as q


@pytest.mark.parametrize("complex_values", [False, True])
@pytest.mark.parametrize("method", [q.fem.Q1, q.fvm.CellCenteredTpfa])
def test_tensor_residual_preserves_transpose_phase_and_total_products(complex_values, method):
    graph = q.geometry.GeometryGraph()
    rectangle = graph.rectangle(x_bounds=(0., 2.), y_bounds=(0., 3.))
    sides = ("left", "right", "bottom", "top")
    geometry = graph.build(rectangle, named_topology={
        "body": rectangle.region, **dict(zip(sides, rectangle.boundaries, strict=True)),
    })
    body = geometry.selection("body")
    scalar = "complex<1>" if complex_values else "1"
    integral = "complex<m^2>" if complex_values else "m^2"
    phase = 2+3j if complex_values else 2.
    source = f"""
    public model Tensor(support body:volume(ambient_dimension=2),
        support left:boundary(parent=body), support right:boundary(parent=body),
        support bottom:boundary(parent=body), support top:boundary(parent=body)) {{
        parameter p:tensor<{scalar},2,2>=tensor_value(frame=body,components=[[1,2],[3,5]]);
        variable u:tensor<{scalar},2,2> on body in h1;
        relation balance on body {{ {"math.complex(2,3)" if complex_values else "2"}*(u+2*transpose(u))=p; }}
        observable total:tensor<{integral},2,2>=integral(u,measure(body));
    }}
    """
    model = q.compile(source=source, entry="Tensor", geometry=geometry, bindings={
        "body": body, **{side: (geometry.selection(side), body) for side in sides},
    })
    mesh = q.meshing.generate(q.meshing.resolve(geometry, q.meshing.CartesianMesher(cells=(2, 3))))
    plan = q.resolve(model, mesh=mesh, spatial=method(), solve=q.solve.Linear(
        algorithm=q.solve.LinearSolver.SparseLu,
        preconditioner=q.solve.Preconditioner.Identity, reduction=q.solve.Reduction.Fast,
        provider=q.solve.SolverProvider.faer(), relative_tolerance=1e-12,
        absolute_tolerance=1e-12, maximum_iterations=32))
    plan = q.Plan.from_bytes(plan.to_bytes())
    parameter_ref = plan.model.parameter(model.parameter("p").id)
    observable_ref = plan.model.observable(model.observable("total").id)
    assert parameter_ref == model.parameter("p")
    assert parameter_ref.value_type == model.parameter("p").value_type
    assert parameter_ref.value == model.parameter("p").value
    assert plan.model.parameter_ids == [parameter_ref.id]
    with pytest.raises(KeyError):
        plan.model.parameter(plan.fields[0].id)
    result = q.run(plan)
    scalar_type = q.ValueType.complex if complex_values else q.ValueType.real
    integral_type = q.ValueType.tensor(scalar_type(q.Dimension(length=2)), 2, 2)
    # Solve a*(U+2*U^T)=P by subtracting its transpose:
    # U=(2*P^T-P)/(3*a). The area is six; both bases reproduce constants.
    initial = np.array([[1., 2.], [3., 5.]])
    for candidate in (result, q.Result.from_bytes(plan, result.to_bytes())):
        observed = candidate.observe(model.observable("total"), quadrature_points=2)
        assert observed.value_type == integral_type
        np.testing.assert_allclose(observed.value, 2*(2*initial.T-initial)/phase, rtol=0, atol=2e-10)

    program = q.diff.compile(plan, inputs=(parameter_ref,),
                             outputs=(observable_ref, plan.fields[0]), quadrature_points=2)
    foreign = model.commit(model.preview_value_edit(parameter_ref.id, [[3., -2.], [7., 1.]]))
    for inputs, outputs in (
        ((foreign.parameter(parameter_ref.id),), (observable_ref,)),
        ((parameter_ref,), (foreign.observable(observable_ref.id),)),
        ((parameter_ref,), (foreign.field(plan.fields[0].id),)),
    ):
        with pytest.raises(q.ValidationError, match="exact Model artifact"):
            q.diff.compile(plan, inputs=inputs, outputs=outputs, quadrature_points=2)
    dtype = np.complex128 if complex_values else np.float64
    direction = np.array([[2., -1.], [4., 3.]], dtype=dtype)
    weights = np.array([[1., 3.], [-2., 4.]], dtype=dtype)
    changed = np.array([[3., -2.], [7., 1.]], dtype=dtype)
    if complex_values:
        direction += 1j*np.array([[1., 4.], [-2., 3.]])
        weights += 1j*np.array([[2., -1.], [3., 5.]])
        changed += 1j*np.array([[-1., 5.], [2., -4.]])
    count = 12 if method is q.fem.Q1 else 6
    for parameter in (initial.astype(dtype), changed):
        point = program.evaluate((parameter,))
        unknown = (2*parameter.T-parameter)/(3*phase)
        tangent = (2*direction.T-direction)/(3*phase)
        expected = (6*unknown, np.broadcast_to(unknown, (count, 2, 2)))
        expected_tangent = (6*tangent, np.broadcast_to(tangent, (count, 2, 2)))
        for actual, wanted in zip(point.primal().output, expected, strict=True):
            assert actual.shape == wanted.shape
            assert actual.dtype == np.dtype(dtype)
            np.testing.assert_allclose(actual, wanted, rtol=0, atol=2e-10)
        for actual, wanted in zip(point.jvp((direction,)).tangent, expected_tangent, strict=True):
            np.testing.assert_allclose(actual, wanted, rtol=0, atol=2e-10)
        field_weights = np.arange(1, count+1)[:, None, None]*weights
        # Real pairing uses the adjoint 1/conj(a); transpose acts on tensor
        # components only, never on the entity axis of the complete Field.
        total_weights = 6*weights+field_weights.sum(axis=0)
        gradient = (2*total_weights.T-total_weights)/(3*np.conjugate(phase))
        reverse = point.vjp((weights, field_weights))
        assert reverse.input_cotangent[0].shape == (2, 2)
        np.testing.assert_allclose(reverse.input_cotangent[0], gradient, rtol=0, atol=2e-10)
