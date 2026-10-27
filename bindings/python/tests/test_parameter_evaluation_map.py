"""Ordered primal frequency Runs share ordinary map and exact collection owners."""
import json
import os
import subprocess
import sys

import eqiora as q
import numpy as np
import pytest

# Derived before execution: V=1/(1-i omega RC), I=-i omega C V.
# One Q1 wave element: (-1-omega²/6)u0+(1-omega²/3)u1=0, u0=1.
RC_VALUES = [(0.8 + 0.4j, 0.0002 - 0.0004j),
             (0.5 + 0.5j, 0.0005 - 0.0005j),
             (0.8 + 0.4j, 0.0002 - 0.0004j)]
WAVE_VALUES = [(1, 1.75), (1, -5), (1, 1.75)]
TOLERANCE = 1e-10  # Existing harmonic-response acceptance boundary.

RC = """model RC(
 parameter resistance:Ohm=1000[Ohm], parameter capacitance:F=1e-6[F],
 parameter omega:1/s=1000[1/s], parameter drive:complex<V>=math.complex(1[V],0[V]),
 input source:V
) {
 state voltage:V; variable current:A;
 initial {voltage=0[V];}
 relation network {source-voltage=resistance*current;current=capacitance*derivative(voltage);}
 form response for network {
  harmonic(angular_frequency=omega,convention=negative_exponential,normalization=peak);
  excitation source=drive;
  amplitude voltage_hat:complex<V> for voltage;
  amplitude current_hat:complex<A> for current;
 }
}"""
WAVE = """public model Wave(
 support body:volume(ambient_dimension=1),support left:boundary(parent=body),
 support right:boundary(parent=body),
 input force:1/s^2,input boundary_value:1
) {
 parameter omega:1/s=1[1/s];
 state u:1 on body in smooth;
 initial {u=0;derivative(u)=0[1/s];}
 relation balance on body {derivative(derivative(u))-div(1[m^2/s^2]*grad(u))=force;}
 relation fixed on left {trace(u)=boundary_value;}
 relation flux on right {normal(1[m^2/s^2]*grad(u))=0[m/s^2];}
 form response for balance,fixed,flux {
  harmonic(angular_frequency=omega,convention=negative_exponential,normalization=peak);
  excitation force=math.complex(0[1/s^2],0[1/s^2]);
  excitation boundary_value=math.complex(1,0);
  amplitude u_hat:complex<1> on body for u;
 }
}"""


def linear():
    return q.solve.Linear(algorithm=q.solve.LinearSolver.SparseLu,
        provider=q.solve.SolverProvider.faer(), reduction=q.solve.Reduction.Fast,
        preconditioner=q.solve.Preconditioner.Identity, relative_tolerance=1e-12,
        absolute_tolerance=1e-14, maximum_iterations=32)


def problem(kind):
    if kind == "rc":
        model = q.compile(source=RC)
        return q.resolve(model, solve=linear()), model.parameter("omega"), [500., 1000., 500.], RC_VALUES
    graph = q.geometry.GeometryGraph()
    interval = graph.interval(bounds=(0., 1.))
    geometry = graph.build(interval, named_topology={"body": interval.region,
        "left": interval.boundaries[0], "right": interval.boundaries[1]})
    body = geometry.selection("body")
    model = q.compile(source=WAVE, entry="Wave", geometry=geometry, bindings={
        "body": body, "left": (geometry.selection("left"), body),
        "right": (geometry.selection("right"), body)})
    mesh = q.meshing.generate(q.meshing.resolve(geometry, q.meshing.CartesianMesher(cells=(1,))))
    return q.resolve(model, mesh=mesh, spatial=q.fem.Q1(), solve=linear()), model.parameter("omega"), [1., 2., 1.], WAVE_VALUES


def mapped(plan, omega, frequencies, **options):
    return q.EvaluationMapPlan.from_parameters(plan, [omega], [(value,) for value in frequencies], **options)


def values(member, kind):
    plan, result = member.plan, member.result
    if kind == "wave":
        return np.asarray(result.output(plan.harmonic_amplitudes[0][2]).values("vertex"))
    outputs = [result.output(amplitude) for _, _, amplitude in plan.harmonic_amplitudes]
    for output in outputs:
        assert output.mesh is None and output.space == "finite"
        assert output.associations == ("global",) and output.logical_shape("global") == ()
        assert output.coefficient_count("global") == 1
    with pytest.raises(TypeError, match="no Mesh"):
        result.mesh(plan.harmonic_amplitudes[0][2])
    return tuple(complex(output.values("global")[0]) for output in outputs)


@pytest.mark.parametrize("kind", ["rc", "wave"])
def test_ordered_parameter_runs_retain_exact_member_plans_and_reopen_offline(kind, tmp_path):
    plan, omega, frequencies, expected = problem(kind)
    request = mapped(plan, omega, frequencies)
    assert request.parameter_template is plan
    assert tuple(parameter.id for parameter in request.parameters) == (omega.id,)
    assert request.points == tuple((value,) for value in frequencies)
    assert request[-1] == (frequencies[-1],)
    collection = request.execute_recorded()
    group = collection[0]
    assert collection.complete and len(group) == 3
    assert group[0].occurrence_id != group[2].occurrence_id
    assert group[0].plan.identity == group[2].plan.identity
    assert group[0].plan.identity != group[1].plan.identity
    assert group[1 if kind == "rc" else 0].plan.identity == plan.identity
    changed = group[0 if kind == "rc" else 1]
    with pytest.raises(ValueError, match="different exact Model"):
        changed.result.output(plan.harmonic_amplitudes[0][2])
    for member, wanted, frequency in zip(group, expected, frequencies):
        assert member.status == "accepted" and member.point == (frequency,)
        assert member.plan.harmonic_angular_frequency == frequency
        np.testing.assert_allclose(values(member, kind), wanted, rtol=0, atol=TOLERANCE)
    # A cold independent Run at the same point reproduces the declared response.
    cold = mapped(plan, omega, [frequencies[0]]).execute_recorded()[0][0]
    np.testing.assert_allclose(values(cold, kind), expected[0], rtol=0, atol=TOLERANCE)
    path = tmp_path / "study.eqcollection"
    collection.write(path)
    code = """import eqiora as q,sys
q.run=lambda *a,**kw: (_ for _ in ()).throw(AssertionError('offline solve'))
q.resolve=q.run
c=q.EvaluationCollection.read(sys.argv[1])
assert c.complete and len(c[0])==3
assert c[0][0].plan.identity==c[0][2].plan.identity
assert c.to_bytes()==open(sys.argv[1],'rb').read()
"""
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    subprocess.run([sys.executable, "-I", "-c", code, str(path)], env=environment, check=True)
    reopened = q.EvaluationCollection.read(path)
    assert reopened.to_bytes() == collection.to_bytes()
    for member, wanted in zip(reopened[0], expected):
        np.testing.assert_allclose(values(member, kind), wanted, rtol=0, atol=TOLERANCE)
    # A stale changed-frequency factor/Plan or mislabeled Result cannot meet this request.
    wire = json.loads(collection.to_bytes())
    wire["groups"][0]["members"][1]["outcome"] = wire["groups"][0]["members"][0]["outcome"]
    with pytest.raises(q.ValidationError):
        q.EvaluationCollection.from_bytes(json.dumps(wire, separators=(",", ":")).encode())


def test_failed_and_cancelled_frequency_occurrences_remain_inspectable_and_retry_is_new_history():
    plan, omega, _, _ = problem("rc")
    request = mapped(plan, omega, [500., 0., 1000.])
    original = request.execute_recorded()
    assert [member.status for member in original[0]] == ["accepted", "failed", "not_started"]
    assert original[0][1].diagnostics and original[0][1].point == (0.,)
    assert q.EvaluationCollection.from_bytes(original.to_bytes()).to_bytes() == original.to_bytes()
    unsuccessful = [member.point if member.status == "not_started" else (1000.,)
                    for member in original[0] if member.status != "accepted"]
    retry = q.EvaluationMapPlan.from_parameters(plan, [omega], unsuccessful).execute_recorded()
    combined = q.EvaluationCollection.combine([original, retry])
    assert combined[0][0].reference == original[0][0].reference
    assert combined[0].execution_id != combined[1].execution_id
    assert len(combined[1]) == 2 and combined[1].complete
    assert original[0][1].point == (0.,) and combined[1][0].point == (1000.,)
    siblings = mapped(plan, omega, [0., 1000.], chunk_size=2, workers=2).execute_recorded()
    assert [member.status for member in siblings[0]] == ["failed", "accepted"]
    assert mapped(plan, omega, [-1.]).execute().statuses == ["failed"]
    cancellation = q.EvaluationMapCancellation()
    cancellation.cancel()
    cancelled = mapped(plan, omega, [500., 1000.]).execute_recorded(cancellation=cancellation)
    assert [member.status for member in cancelled[0]] == ["cancelled", "not_started"]
    retried = mapped(plan, omega, [member.point[0] for member in cancelled[0]]).execute_recorded()
    assert retried.complete and retried[0].execution_id != cancelled[0].execution_id
    with pytest.raises(TypeError, match="original Model"):
        q.EvaluationMapPlan.from_parameters(plan, [plan.model.parameter(omega.id)], [(500.,)])
    with pytest.raises(q.ValidationError):
        q.diff.compile(plan, inputs=(plan.model.parameter(omega.id),), outputs=tuple(plan.fields))


def test_finite_field_output_preserves_real_and_complex_intrinsic_shape_units_and_no_mesh():
    model = q.compile(source="""model Finite() {
      variable voltage:V;
      variable currents:array<complex<A>,2>;
      relation values {
        voltage=3[V];
        currents=[math.complex(1[A],2[A]),math.complex(-3[A],4[A])];
      }
    }""")
    plan = q.resolve(model, solve=linear())
    result = q.run(plan, state=q.State.initial(plan))
    voltage = plan.model.field(model.field("voltage").id)
    currents = plan.model.field(model.field("currents").id)
    for field, expected, shape in ((voltage, [3.], ()), (currents, [1+2j, -3+4j], (2,))):
        output = result.output(field)
        assert output.mesh is None and output.space == "finite"
        assert output.associations == ("global",) and output.logical_shape("global") == shape
        assert output.coefficient_count("global") == 1
        wanted_type = q.ValueType.real(q.Dimension(length=2, mass=1, time=-3, current=-1)) if field == voltage else q.ValueType.array(q.ValueType.complex(q.Dimension(current=1)), 2)
        assert output.value_type == wanted_type
        np.testing.assert_allclose(output.values("global"), expected, rtol=0, atol=TOLERANCE)
        with pytest.raises(TypeError, match="no Mesh"):
            result.mesh(field)
    replayed = q.Result.from_bytes(plan, result.to_bytes())
    np.testing.assert_array_equal(replayed.output(currents).values("global"), result.output(currents).values("global"))


@pytest.mark.parametrize("automatic", [False, True])
def test_reference_selected_parameter_collection_reopens_with_standard_optional_provider(automatic):
    model = q.compile(source="model M(){parameter p:1=2;variable u:1;relation r{u=p;}}")
    if automatic:
        solve = q.solve.Linear(objective=q.solve.SolverPlanningObjective.Robust,
            relative_tolerance=1e-12, absolute_tolerance=1e-14, maximum_iterations=32)
    else:
        solve = q.solve.Linear(algorithm=q.solve.LinearSolver.BiConjugateGradientStabilized,
            provider=q.solve.SolverProvider.reference(), reduction=q.solve.Reduction.Reproducible,
            preconditioner=q.solve.Preconditioner.Identity, relative_tolerance=1e-12,
            absolute_tolerance=1e-14, maximum_iterations=32)
    plan = q.resolve(model, solve=solve)
    request = q.EvaluationMapPlan.from_parameters(plan, [model.parameter("p")], [(2.,), (3.,), (2.,)])
    collection = request.execute_recorded()
    assert collection.complete
    assert collection[0][0].plan.identity == plan.identity
    reopened = q.EvaluationCollection.from_bytes(collection.to_bytes())
    assert reopened.to_bytes() == collection.to_bytes()
    for member, wanted in zip(reopened[0], [2., 3., 2.]):
        assert list(member.result.output(member.plan.fields[0]).values("global")) == pytest.approx([wanted], abs=TOLERANCE)


def test_reopened_multi_parameter_conditions_keep_selected_ids_types_and_column_order():
    model = q.compile(source="""model Conditions(){
      parameter voltage:V=2[V];parameter current:A=3[A];
      variable u:V;variable i:A;relation r{u=voltage;i=current;}
    }""")
    plan = q.resolve(model, solve=linear())
    selected = [model.parameter("current"), model.parameter("voltage")]
    points = [(3., 2.), (5., 7.), (3., 2.)]
    request = q.EvaluationMapPlan.from_parameters(plan, selected, points)
    assert request.parameters == tuple(selected)
    collection = q.EvaluationCollection.from_bytes(request.execute_recorded().to_bytes())
    group = collection[0]
    assert tuple(parameter.id for parameter in group.parameters) == tuple(parameter.id for parameter in selected)
    assert tuple(parameter.value_type for parameter in group.parameters) == tuple(parameter.value_type for parameter in selected)
    assert all(parameter.model_digest == model.digest for parameter in group.parameters)
    for member, (current, voltage) in zip(group, points):
        assert member.point == (current, voltage)
        for name, wanted in (("u", voltage), ("i", current)):
            field = member.plan.model.field(model.field(name).id)
            assert list(member.result.output(field).values("global")) == pytest.approx([wanted], abs=TOLERANCE)
