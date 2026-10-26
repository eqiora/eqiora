"""Midpoint interpolation and constrained recovery use the ordinary Run clock."""
import eqiora as q
import numpy as np
import pytest
import subprocess
import sys

from test_nonlinear_spatial_differentiation import newton_policy


def constrained_program(complex_field, spatial):
    graph = q.geometry.GeometryGraph()
    interval = graph.interval(bounds=(0., 1.))
    geometry = graph.build(interval, named_topology={
        "body": interval.region, "left": interval.boundaries[0], "right": interval.boundaries[1],
    })
    kind = "complex<1>" if complex_field else "1"
    seed = "math.complex(1,2)" if complex_field else "1"
    square = "math.abs2(u)" if complex_field else "u*u"
    source = f"""public model Main(support body:volume(ambient_dimension=1),
        support left:boundary(parent=body),support right:boundary(parent=body)) {{
        parameter p:1=2; state u:{kind} on body in smooth;
        variable z:{kind} on body in smooth;
        initial {{u={seed}; z={5 if complex_field else 1};}}
        relation evolution on body {{1[s]*derivative(u)=p;}}
        relation constraint on body {{z={square};}}
        observable total:{'complex<m>' if complex_field else 'm'}=integral(time()/1[s]*(z+p),measure(body));
    }}"""
    model = q.compile(source=source, entry="Main", geometry=geometry,
                      bindings={"body": geometry.selection("body"),
                          "left": (geometry.selection("left"), geometry.selection("body")),
                          "right": (geometry.selection("right"), geometry.selection("body"))})
    mesh = q.meshing.generate(q.meshing.resolve(geometry, q.meshing.CartesianMesher(cells=(2,))))
    fields = tuple(model.field(name) for name in ("u", "z"))
    plan = q.resolve(model, mesh=mesh, spatial=spatial(), solve=newton_policy(),
                     temporal=q.time.ImplicitMidpoint(step_s=1., relative_tolerance=1e-10,
                        absolute_tolerances={(field, 0, 0, imaginary): 1e-12
                            for field in fields
                            for imaginary in ((False, True) if complex_field else (False,))}))
    plan = q.Plan.from_bytes(plan.to_bytes())
    state = q.State.initial(plan, time_s=0.)
    times = (.2, .5, 1., 2.25)
    program = q.diff.compile(plan, inputs=(plan.model.parameter(model.parameter("p").id),),
        outputs=(plan.model.field(fields[0].id), plan.model.field(fields[1].id),
                 plan.model.observable(model.observable("total").id)),
        state=state, until_s=2.5, output_times_s=times, quadrature_points=2)
    return program, plan, state, np.array(times)


@pytest.mark.parametrize("complex_field", [False, True])
@pytest.mark.parametrize("spatial", [q.fem.Q1, q.fvm.CellCenteredTpfa])
def test_midpoint_interpolation_recovery_products_and_record(complex_field, spatial, tmp_path):
    program, plan, state, times = constrained_program(complex_field, spatial)
    before = state.to_bytes()
    resting = program.evaluate((0.,)).primal()
    # The real resting equations assemble exact zeros. Complex Q1 may make a
    # roundoff correction when evaluating |1+2i|² in its real coordinate blocks.
    if not complex_field:
        assert all(group.completed_iterations == 0
                   for step in resting.evidence.steps for group in step.primal_solves)
    point = program.evaluate((3.,))
    width = 3 if spatial is q.fem.Q1 else 2
    # Uniform exact ODE u=1+p*t+(2i), algebraic z=|u|². This also
    # distinguishes observation recovery from interpolating endpoint z values.
    u = 1 + 3*times + (2j if complex_field else 0)
    z = np.abs(u)**2
    du, dz = times, 2*u.real*times
    expected = (np.repeat(u[:, None], width, axis=1),
                np.repeat(z[:, None], width, axis=1), times*(z+3))
    tangent = (np.repeat(du[:, None], width, axis=1),
               np.repeat(dz[:, None], width, axis=1), times*(dz+1))
    primal = point.primal()
    for actual, wanted in zip(primal.output, expected, strict=True):
        np.testing.assert_allclose(actual, wanted, rtol=0, atol=2e-9)
    for actual, wanted in zip(point.jvp((1.,)).tangent, tangent, strict=True):
        np.testing.assert_allclose(actual, wanted, rtol=0, atol=2e-9)
    weights = tuple(np.full_like(value, 1+2j if complex_field else 2) for value in primal.output)
    pairing = sum(np.vdot(w, d).real for w, d in zip(weights, tangent, strict=True))
    np.testing.assert_allclose(point.vjp(weights).input_cotangent, [pairing], rtol=0, atol=2e-8)
    # First step owns three observations, second none, last is shortened.
    # Every constrained step retains its stage and endpoint even if unobserved.
    assert [len(step.primal_solves) for step in primal.evidence.steps] == [5, 2, 3]
    assert all(isinstance(group, q.NonlinearSolveSummary)
               for step in primal.evidence.steps for group in step.primal_solves)
    collection = program.map((np.array([3., 3.]),), point_shape=(2,)).execute_recorded()
    encoded = collection.to_bytes()
    (tmp_path / "accepted.bin").write_bytes(encoded)
    np.savez(tmp_path / "expected.npz", *expected)
    subprocess.run([sys.executable, "-I", "-c", """
from pathlib import Path
import eqiora as q
import numpy as np
data = Path('accepted.bin').read_bytes()
collection = q.EvaluationCollection.from_bytes(data)
assert collection.to_bytes() == data
with np.load('expected.npz', allow_pickle=False) as expected:
    for member in collection[0]:
        for index, actual in enumerate(member.result.output):
            np.testing.assert_allclose(actual, expected[f'arr_{index}'], rtol=0, atol=2e-9)
"""], cwd=tmp_path, check=True, capture_output=True, text=True)
    del point, collection, program
    reopened = q.EvaluationCollection.from_bytes(encoded)
    assert reopened.to_bytes() == encoded
    for member in reopened[0]:
        for actual, wanted in zip(member.result.output, expected, strict=True):
            np.testing.assert_allclose(actual, wanted, rtol=0, atol=2e-9)
    assert state.to_bytes() == before
