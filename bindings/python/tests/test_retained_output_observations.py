"""Queries bind the exact retained State without inventing omitted time history."""
import pytest

import eqiora as q


def test_retained_heat_outputs_share_point_integral_time_and_replay():
    graph = q.geometry.GeometryGraph()
    rectangle = graph.rectangle(x_bounds=(0.0, 1.0), y_bounds=(0.0, 1.0))
    sides = ("left", "right", "bottom", "top")
    geometry = graph.build(rectangle, named_topology={
        "body": rectangle.region, **dict(zip(sides, rectangle.boundaries, strict=True)),
    })
    body = geometry.selection("body")
    model = q.compile(source="""public component Heat(
        support body:volume(ambient_dimension=2),
        support left:boundary(parent=body), support right:boundary(parent=body),
        support bottom:boundary(parent=body), support top:boundary(parent=body)
    ) {
        state u:K on body in h1;
        initial {u=300[K];}
        law balance on body {storage 1[s/m^2]*u; flux -grad(u); source 1[K/m^2];}
        relation left_fixed on left {trace(u)=300[K];}
        relation right_fixed on right {trace(u)=300[K];}
        relation bottom_fixed on bottom {trace(u)=300[K];}
        relation top_fixed on top {trace(u)=300[K];}
        observable profile:K on body=u;
        observable volume:K*m^2=integral(u,measure(body));
        observable boundary:K*m=integral(trace(u),measure(left));
        observable clock:s=time();
        observable weighted:K*m^2*s=integral(time()*u,measure(body));
    }""", entry="Heat", geometry=geometry,
        bindings={"body": body, **{side: (geometry.selection(side), body) for side in sides}})
    mesh = q.meshing.generate(q.meshing.resolve(geometry, q.meshing.CartesianMesher(cells=(2, 2))))
    plan = q.resolve(model, mesh=mesh, spatial=q.fem.Q1(), temporal=q.time.BackwardEuler(1/24),
        solve=q.solve.Linear(algorithm=q.solve.LinearSolver.SparseLu,
            preconditioner=q.solve.Preconditioner.Identity, reduction=q.solve.Reduction.Fast,
            provider=q.solve.SolverProvider.faer(), relative_tolerance=1e-12,
            absolute_tolerance=1e-14, maximum_iterations=100))
    result = q.run(plan, state=q.State.initial(plan), steps=3, output_steps=(1, 3))
    before = result.to_bytes()
    replay = q.Result.from_bytes(q.Plan.from_bytes(plan.to_bytes()), before)
    outputs = {name: model.observable(f"definition.{name}")
               for name in ("profile", "volume", "boundary", "clock", "weighted")}
    length = q.Dimension(length=1)
    temperature = q.Dimension(temperature=1)
    # Four Q1 cells: M_cc=1/9, K_cc=8/3, F_c=1/4. With dt=1/24,
    # e_n=(3/32)*(1-2^-n), and the center hat integrates to 1/4.
    for candidate in (result, replay):
        for index, step in enumerate((1, 3)):
            state = candidate.trajectory.states[index]
            excess = 3/32*(1-2**(-step))
            sample = candidate.observe_at(outputs["profile"], [(0.5, length), (0.5, length)], output_index=index)
            assert sample.value == pytest.approx(300+excess, abs=1e-10, rel=0)
            assert sample.value_type == q.ValueType.real(temperature)
            assert sample.point is not None
            assert sample.output == (index, state.time_s, state.digest)
            for name, expected, dimension in (
                ("volume", 300+excess/4, q.Dimension(temperature=1, length=2)),
                ("boundary", 300, q.Dimension(temperature=1, length=1)),
                ("clock", step/24, q.Dimension(time=1)),
                ("weighted", (step/24)*(300+excess/4), q.Dimension(temperature=1, length=2, time=1)),
            ):
                observed = candidate.observe(outputs[name], output_index=index,
                                             quadrature_points=None if name == "clock" else 2)
                assert observed.value == pytest.approx(expected, abs=1e-10, rel=0)
                assert observed.value_type == q.ValueType.real(dimension)
                assert observed.output == sample.output
                assert observed.result_identity == sample.result_identity
        for index, message in ((None, "explicit output index"), (2, "outside the retained schedule")):
            with pytest.raises(q.ValidationError, match=message):
                candidate.observe(outputs["volume"], quadrature_points=2, output_index=index)
            with pytest.raises(q.ValidationError, match=message):
                candidate.observe_at(outputs["profile"], [(0.5, length), (0.5, length)], output_index=index)
        with pytest.raises(q.ValidationError):
            candidate.observable_state_tangent({model.field("definition.u"): (temperature, [0.0]*9)})
        assert candidate.to_bytes() == before
