"""Ordinary Plans preserve complex Fields with either numerical scalar domain."""
import numpy as np
import pytest
import eqiora as q


@pytest.mark.parametrize("conjugate", [False, True])
@pytest.mark.parametrize("faer", [False, True])
def test_complex_channel_equations_select_exact_storage_and_replay(conjugate, faer):
    graph = q.geometry.GeometryGraph()
    interval = graph.interval(bounds=(0., 1.))
    geometry = graph.build(interval, named_topology={
        "body": interval.region, "left": interval.boundaries[0],
        "right": interval.boundaries[1],
    })
    parent = geometry.selection("body")
    term = "+math.complex(1,2)*math.conj(u)" if conjugate else ""
    # Independently expand (3-2i)*u + (1+2i)*conj(u), or just the
    # first term, for the affine field checked below. Its curvature is zero.
    forcing = (
        "[math.complex(12+8*coordinate(0)/1[m],4-2*coordinate(0)/1[m]),"
        "math.complex(4+12*coordinate(0)/1[m],6+8*coordinate(0)/1[m])]"
        if conjugate else
        "[math.complex(7+7*coordinate(0)/1[m],4-9*coordinate(0)/1[m]),"
        "math.complex(5*coordinate(0)/1[m],13+14*coordinate(0)/1[m])]"
    )
    model = q.compile(source=f"""
      public component Channels(support body:volume(ambient_dimension=1),
        support left:boundary(parent=body), support right:boundary(parent=body)) {{
        variable u:array<complex<1>,2> on body in h1;
        relation balance on body {{
          -div(math.complex(2,3)*grad(u))+(math.complex(3,-2)*u{term}-{forcing})/1[m^2]=0;
        }}
        relation left_value on left {{trace(u)=[math.complex(1,2),math.complex(-2,3)];}}
        relation right_value on right {{trace(u)=[math.complex(4,1),math.complex(-3,7)];}}
      }}""", entry="Channels", geometry=geometry, bindings={
        "body": parent, "left": (geometry.selection("left"), parent),
        "right": (geometry.selection("right"), parent),
    })
    mesh = q.meshing.generate(q.meshing.resolve(
        geometry, q.meshing.CartesianMesher(cells=(4,))))
    provider = q.solve.SolverProvider.faer() if faer else q.solve.SolverProvider.reference()
    solve = q.solve.Linear(
        algorithm=q.solve.LinearSolver.SparseLu if faer else q.solve.LinearSolver.BiConjugateGradientStabilized,
        preconditioner=q.solve.Preconditioner.Identity,
        reduction=q.solve.Reduction.Fast if faer else q.solve.Reduction.Reproducible,
        provider=provider, relative_tolerance=1e-12, absolute_tolerance=1e-14,
        maximum_iterations=100,
    )
    plan = q.Plan.from_bytes(q.resolve(model, mesh=mesh, spatial=q.fem.Q1(), solve=solve).to_bytes())
    result = q.run(plan)
    x = np.arange(5)/4
    expected = np.stack((1+3*x+1j*(2-x), -2-x+1j*(3+4*x)), axis=1)
    for candidate in (result, q.Result.from_bytes(plan, result.to_bytes())):
        field = candidate.output(plan.fields[0])
        assert field.value_type == q.ValueType.array(q.ValueType.complex(), 2)
        shape = field.logical_shape("vertex")
        assert shape == (5, 2)
        assert field.coefficient_count("vertex") == 5
        actual = np.asarray(field.values("vertex")).reshape(shape)
        assert actual.dtype == np.dtype("complex128")
        np.testing.assert_allclose(actual, expected, rtol=0, atol=2e-10)
