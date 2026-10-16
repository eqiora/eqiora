"""Space/time reductions use exact accepted steps and ordinary typed observations."""

import pytest
import eqiora as q


def test_spatial_time_integrals_use_covered_steps_and_observable_interpolation():
    graph = q.geometry.GeometryGraph()
    rectangle = graph.rectangle(x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
    sides = ("left", "right", "bottom", "top")
    geometry = graph.build(
        rectangle,
        named_topology={
            "body": rectangle.region,
            **dict(zip(sides, rectangle.boundaries, strict=True)),
        },
    )
    model = q.compile(
        source="""public component Ramp(support body:volume(ambient_dimension=2),
        support left:boundary(parent=body), support right:boundary(parent=body),
        support bottom:boundary(parent=body), support top:boundary(parent=body)) {
        state u:K on body in smooth;
        initial {u=0[K];}
        law balance on body {storage 1[s/m^2]*u; flux -grad(u); source 1[K/m^2];}
        relation left_flux on left {normal(grad(u))=0;}
        relation right_flux on right {normal(grad(u))=0;}
        relation bottom_flux on bottom {normal(grad(u))=0;}
        relation top_flux on top {normal(grad(u))=0;}
        observable amount:K^2*m^2=integral(u*u,measure(body));
        observable profile:K on body=u;
        observable clock:s=time();
    }""",
        entry="Ramp",
        geometry=geometry,
        bindings={
            "body": geometry.selection("body"),
            **{
                side: (geometry.selection(side), geometry.selection("body"))
                for side in sides
            },
        },
    )
    mesh = q.meshing.generate(
        q.meshing.resolve(geometry, q.meshing.CartesianMesher(cells=(2, 2)))
    )
    plan = q.resolve(
        model,
        mesh=mesh,
        spatial=q.fem.Q1(),
        temporal=q.time.BackwardEuler(0.25),
        solve=q.solve.Linear(
            algorithm=q.solve.LinearSolver.SparseLu,
            preconditioner=q.solve.Preconditioner.Identity,
            reduction=q.solve.Reduction.Fast,
            provider=q.solve.SolverProvider.faer(),
            relative_tolerance=1e-12,
            absolute_tolerance=1e-14,
            maximum_iterations=100,
        ),
    )
    amount = model.observable("definition.amount")
    clock = model.observable("definition.clock")
    profile = model.observable("definition.profile")
    rule = q.time.TimeObservationInterpolation.EndpointLinear
    dense = q.run(plan, state=q.State.initial(plan), steps=4, output_steps=(1, 2, 3, 4))
    sparse = q.run(plan, state=q.State.initial(plan), steps=4, output_steps=(1, 3))
    # u=t is spatially constant and exactly advanced by backward Euler. For its
    # squared spatial integral, endpoint interpolation at h=1/4 yields 11/32
    # on [0,1] and 59/256 on [1/8,7/8], rather than a dense-State integral.
    for original in (dense, sparse):
        before = original.to_bytes()
        replay = q.Result.from_bytes(q.Plan.from_bytes(plan.to_bytes()), before)
        assert replay.initial_state.to_bytes() == q.State.initial(plan).to_bytes()
        assert replay.initial_state.time_s == 0.
        for result in (original, replay):
            for window, expected in (((0.0, 0.25), 1 / 128), ((0.5, 0.75), 13 / 128)):
                value = result.observe_time_integral(
                    amount, interpolation=rule, interval_s=window, quadrature_points=2
                )
                assert value.value == pytest.approx(expected, abs=1e-11, rel=0)
                assert value.value_type == q.ValueType.real(
                    q.Dimension(temperature=2, length=2, time=1)
                )
                for operation, endpoint in ((result.observe_time_minimum, window[0]), (result.observe_time_maximum, window[1])):
                    extreme = operation(amount, interpolation=rule, interval_s=window, quadrature_points=2)
                    assert extreme.value == pytest.approx(endpoint**2, abs=1e-11, rel=0)
                    assert extreme.extremum_time_s == endpoint
                    assert extreme.value_type == q.ValueType.real(q.Dimension(temperature=2, length=2))
                    assert extreme.spatial_quadratures == value.spatial_quadratures
                assert value.interval_s == window
                assert value.interpolation == rule
                assert (
                    value.spatial_quadratures
                    == result.observe(
                        amount, output_index=0, quadrature_points=2
                    ).quadratures
                )
                assert (
                    value.result_identity
                    == result.observe(clock, output_index=0).result_identity
                )
            if len(result.trajectory.states) == 4:
                for window, expected in ((None, 11 / 32), ((0.125, 0.875), 59 / 256)):
                    value = result.observe_time_integral(
                        amount, interpolation=rule, interval_s=window, quadrature_points=2
                    )
                    assert value.value == pytest.approx(expected, abs=1e-11, rel=0)
                # At t=1/8 the interpolant of t^2 is 1/32; at 7/8 it is 25/32.
                for operation, expected, time in ((result.observe_time_minimum, 1/32, 0.125), (result.observe_time_maximum, 25/32, 0.875)):
                    extreme = operation(amount, interpolation=rule, interval_s=(0.125,0.875), quadrature_points=2)
                    assert extreme.value == pytest.approx(expected, abs=1e-11, rel=0)
                    assert extreme.extremum_time_s == time
                assert result.observe_time_integral(
                    clock, interpolation=rule
                ).value == pytest.approx(0.5)
                with pytest.raises(q.ValidationError, match="no native midpoint"):
                    result.observe_time_integral(
                        amount,
                        interpolation=q.time.TimeObservationInterpolation.NativeMidpointQuadratic,
                        quadrature_points=2,
                    )
                with pytest.raises(q.ValidationError):
                    result.observe_time_integral(amount, interpolation=rule)
                with pytest.raises(q.ValidationError, match="finite Observable"):
                    result.observe_time_integral(profile, interpolation=rule)
            else:
                for window in (None, (0.0, 0.5), (0.25, 0.5), (0.5, 1.0)):
                    with pytest.raises(
                        q.ValidationError,
                        match="not covered by retained accepted steps",
                    ):
                        result.observe_time_integral(
                            amount,
                            interpolation=rule,
                            interval_s=window,
                            quadrature_points=2,
                        )
                    with pytest.raises(q.ValidationError, match="not covered by retained accepted steps"):
                        result.observe_time_minimum(amount, interpolation=rule, interval_s=window, quadrature_points=2)
            assert result.to_bytes() == before
