"""Exact typed derivative policy controls and ordinary Plan persistence."""
import pytest
import eqiora

SOURCE = """
model Ramp() {
 state x: m; parameter rate: m/s = 1;
 initial { x = 0[m]; }
 relation flow { derivative(x) = rate; }
}
"""


def policy(model, dimension=None):
    return eqiora.time.ForwardSensitivity(
        relative_tolerance=1e-9,
        absolute_tolerances=(eqiora.time.SensitivityTolerance(
            model.field("x"), model.parameter("rate"), 1e-11,
            dimension or eqiora.Dimension(time=1),
        ),),
    )


def resolve(model, forward):
    return eqiora.resolve(model, temporal=eqiora.time.Tsitouras45(
        initial_step_s=0.001, relative_tolerance=1e-9,
        absolute_tolerances={model.field("x"): 1e-11},
        forward_sensitivities=forward,
    ))


def test_forward_controls_replay_exact_references_and_dimensioned_tolerances():
    model = eqiora.compile(source=SOURCE, filename="forward-policy.eqi")
    forward = policy(model)
    plan = resolve(model, forward)
    assert plan.temporal.forward_sensitivities.model_digest == model.digest
    plain = resolve(model, None)
    assert plan.identity != plain.identity
    reopened = eqiora.Plan.from_bytes(plan.to_bytes())
    assert reopened.to_bytes() == plan.to_bytes()
    entry = reopened.temporal.forward_sensitivities.absolute_tolerances[0]
    assert entry.field == model.field("x")
    assert entry.parameter == model.parameter("rate")
    assert entry.dimension == eqiora.Dimension(time=1)
    assert entry.value == 1e-11
    with pytest.raises(AttributeError):
        forward.relative_tolerance = 1e-6


def test_wrong_units_duplicates_and_foreign_references_fail_closed():
    model = eqiora.compile(source=SOURCE, filename="forward-policy.eqi")
    foreign = eqiora.compile(source=SOURCE.replace("m/s = 1", "m/s = 2"), filename="foreign-policy.eqi")
    with pytest.raises(eqiora.ValidationError):
        resolve(model, policy(model, eqiora.Dimension(length=1)))
    with pytest.raises((TypeError, eqiora.ValidationError)):
        resolve(model, policy(foreign))
    with pytest.raises(TypeError):
        eqiora.time.SensitivityTolerance(model.field("x"), foreign.parameter("rate"), 1e-11, eqiora.Dimension(time=1))
    entry = policy(model).absolute_tolerances[0]
    with pytest.raises(eqiora.ValidationError):
        eqiora.time.ForwardSensitivity(relative_tolerance=1e-9, absolute_tolerances=(entry, entry))
