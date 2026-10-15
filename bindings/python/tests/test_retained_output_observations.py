"""Queries bind the exact retained State without inventing omitted time history."""
import json
import pytest

import eqiora as q


def test_retained_ode_outputs_preserve_complex_shape_rates_time_and_replay():
    model = q.compile(source="""model Channels() {
        state z:array<complex<m>,2>;
        initial {z=[math.complex(1[m],2[m]),math.complex(-3[m],4[m])];}
        relation flow {derivative(z)=[math.complex(2[m/s],-1[m/s]),math.complex(-1[m/s],3[m/s])];}
        observable snapshot:array<complex<m>,2>=z;
        observable rate:array<complex<m/s>,2>=derivative(z);
        observable clock:s=time();
    }""")
    field = model.field("z")
    plan = q.resolve(model, temporal=q.time.ImplicitMidpoint(
        step_s=0.25, relative_tolerance=1e-11,
        absolute_tolerances={(field, 0, component, imaginary): 1e-13
                             for component in range(2) for imaginary in (False, True)},
    ))
    plan = q.Plan.from_bytes(plan.to_bytes())
    times = (0.125, 0.3, 0.875)
    result = q.run(plan, state=q.State.initial(plan), until_s=1., output_times_s=times)
    before = result.to_bytes()
    replay = q.Result.from_bytes(plan, before)
    snapshot, rate, clock = (model.observable(name) for name in ("snapshot", "rate", "clock"))
    for candidate in (result, replay):
        for index, time in enumerate(times):
            sample = candidate.observe(snapshot, output_index=index)
            assert sample.value == pytest.approx(((1+2j)+(2-1j)*time, (-3+4j)+(-1+3j)*time), abs=2e-12, rel=0)
            assert sample.value_type == q.ValueType.array(q.ValueType.complex(q.Dimension(length=1)), 2)
            assert sample.output == (index, time, q.State.from_result(plan, candidate, time_s=time).digest)
            assert sample.result_identity == json.loads(before)["identity"]
            rates = candidate.observe(rate, output_index=index)
            assert rates.value == pytest.approx((2-1j, -1+3j), abs=2e-12, rel=0)
            assert rates.value_type == q.ValueType.array(q.ValueType.complex(q.Dimension(length=1, time=-1)), 2)
            assert candidate.observe(clock, output_index=index).value == time
        for index in (None, len(times)):
            with pytest.raises(q.ValidationError):
                candidate.observe(snapshot, output_index=index)
        with pytest.raises(q.ValidationError):
            candidate.observe(snapshot, output_index=0, quadrature_points=1)
        with pytest.raises(q.ValidationError):
            candidate.observe_at(snapshot, [(0., q.Dimension(length=1))], output_index=0)
        assert candidate.to_bytes() == before


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
        support bottom:boundary(parent=body), support top:boundary(parent=body),
        support path:interval(m)
    ) {
        state u:K on body in h1;
        initial {u=300[K];}
        law balance on body {storage 1[s/m^2]*u; flux -grad(u); source 1[K/m^2];}
        relation left_fixed on left {trace(u)=300[K];}
        relation right_fixed on right {trace(u)=300[K];}
        relation bottom_fixed on bottom {trace(u)=300[K];}
        relation top_fixed on top {trace(u)=300[K];}
        coordinate x:m on body from body[0];
        coordinate y:m on body from body[1];
        coordinate s:m on path from path;
        observable line:K on path=pullback(u,from=(s),at=(x=s,y=0.5[m]));
        observable outside_line:K on path=pullback(u,from=(s),at=(x=s+1[m],y=0.5[m]));
        observable profile:K on body=u;
        observable volume:K*m^2=integral(u,measure(body));
        observable boundary:K*m=integral(trace(u),measure(left));
        observable clock:s=time();
        observable weighted:K*m^2*s=integral(time()*u,measure(body));
    }""", entry="Heat", geometry=geometry,
        bindings={"body": body, "path": q.CoordinateInterval(0, 1, dimension=q.Dimension(length=1)),
                  **{side: (geometry.selection(side), body) for side in sides}})
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
               for name in ("profile", "volume", "boundary", "clock", "weighted", "line", "outside_line")}
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
            # Restrict the same Q1 center hat to y=1/2. Its independently known
            # values at x=0,1/4,1/2,3/4,1 are 0,1/2,1,1/2,0.
            for position, hat in ((0., 0.), (0.25, 0.5), (0.5, 1.), (0.75, 0.5), (1., 0.)):
                path_sample = candidate.observe_at(outputs["line"], [(position, length)], output_index=index)
                assert path_sample.value == pytest.approx(300 + excess*hat, abs=1e-10, rel=0)
                assert path_sample.value_type == q.ValueType.real(temperature)
                assert path_sample.output == sample.output
                assert path_sample.result_identity == sample.result_identity
                assert path_sample.point == (model.domain("path").id, [(position, length)])
            # A valid path coordinate does not authorize an exterior physical point.
            with pytest.raises(q.ValidationError, match="outside"):
                candidate.observe_at(outputs["outside_line"], [(0.25, length)], output_index=index)
            with pytest.raises(q.ValidationError):
                candidate.observe_at(outputs["line"], [(0.25, q.Dimension(time=1))], output_index=index)
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
