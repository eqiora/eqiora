"""Typed spatial history retains physical values through one temporal lifecycle."""
import numpy as np
import pytest
import eqiora as q


@pytest.mark.parametrize("complex_field", [False, True])
@pytest.mark.parametrize("law", [False, True])
@pytest.mark.parametrize("nonlinear", [False, True])
def test_typed_spatial_history_replays_and_restarts(complex_field, law, nonlinear):
    graph = q.geometry.GeometryGraph()
    interval = graph.interval(bounds=(0., 1.))
    geometry = graph.build(interval, named_topology={
        "body": interval.region, "left": interval.boundaries[0],
        "right": interval.boundaries[1],
    })
    body = geometry.selection("body")
    scalar = "complex<1>" if complex_field else "1"
    datum = ("math.complex(1,2)" if complex_field else "1") if nonlinear else (
        "math.complex(2,4)" if complex_field else "2")
    capacity = 1 if nonlinear else 3
    source = "-0.1[1/m^2]*math.abs2(u)*u" if nonlinear else (
        "math.complex(4,8)/1[m^2]" if complex_field else "4[1/m^2]")
    balance = (
        f"law balance on body {{storage {capacity}[s/m^2]*u; flux -grad(u); source {source};}}"
        if law else f"relation balance on body {{{capacity}[s/m^2]*derivative(u)-div(grad(u))={source};}}"
    )
    boundary = "normal(grad(u))=0" if nonlinear else f"trace(u)={datum}"
    model = q.compile(source=f"""public component Heat(
      support body:volume(ambient_dimension=1), support left:boundary(parent=body),
      support right:boundary(parent=body)) {{
      state u:{scalar} on body in smooth;
      initial {{u={datum};}}
      {balance}
      relation left_value on left {{{boundary};}}
      relation right_value on right {{{boundary};}}
    }}""", entry="Heat", geometry=geometry, bindings={
        "body": body, "left": (geometry.selection("left"), body),
        "right": (geometry.selection("right"), body),
    })
    mesh = q.meshing.generate(q.meshing.resolve(
        geometry, q.meshing.CartesianMesher(cells=(2,))))
    linear = q.solve.Linear(
        algorithm=q.solve.LinearSolver.BiConjugateGradientStabilized,
        preconditioner=q.solve.Preconditioner.Identity,
        reduction=q.solve.Reduction.Reproducible,
        provider=q.solve.SolverProvider.reference(),
        relative_tolerance=1e-12, absolute_tolerance=1e-14, maximum_iterations=100,
    )
    solve = q.solve.Newton(linear=linear, relative_tolerance=0., absolute_tolerance=1e-12,
                           maximum_iterations=20, maximum_line_search_steps=16) if nonlinear else linear
    plan = q.resolve(model, mesh=mesh, spatial=q.fem.Q1(),
                     temporal=q.time.BackwardEuler(0.25), solve=solve)
    plan = q.Plan.from_bytes(plan.to_bytes())
    initial = q.State.initial(plan)
    before = initial.to_bytes()
    initial = q.State.from_bytes(plan, before)
    result = q.run(plan, state=initial, steps=3, output_steps=(1, 2, 3))
    amplitude = 1.
    multiplier = 1+2j if complex_field else 1.
    for index, state in enumerate(result.trajectory.states):
        if nonlinear:
            # A spatially uniform value is multiplier*a, with
            # a_next + 0.1*|multiplier|²*dt*a_next³ = a_previous.
            lo, hi = 0., amplitude
            coefficient = (0.5 if complex_field else 0.1)*0.25
            for _ in range(80):
                mid = (lo+hi)/2
                if mid+coefficient*mid**3 < amplitude:
                    lo = mid
                else:
                    hi = mid
            amplitude = (lo+hi)/2
            expected = np.full(3, multiplier*amplitude)
        else:
            # Two Q1 cells: midpoint excess satisfies e_next=e/2+1/4.
            expected = multiplier*np.array([2., (2.25, 2.375, 2.4375)[index], 2.])
        np.testing.assert_allclose(state.fields[0].values("vertex"), expected, rtol=0, atol=1e-10)
    restarted = q.run(plan, state=q.State.from_bytes(plan, result.trajectory.states[0].to_bytes()),
                     steps=2, output_steps=(2,))
    assert restarted.trajectory.states[-1].to_bytes() == result.trajectory.states[-1].to_bytes()
    replay = q.Result.from_bytes(plan, result.to_bytes())
    assert replay.to_bytes() == result.to_bytes()
    assert initial.to_bytes() == before
