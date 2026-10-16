"""Typed trajectory statistics reuse native observations and finite-moment arithmetic."""
from fractions import Fraction
import math

import eqiora as q
import pytest


def fixture():
    model = q.compile(source="""
    model Process() {
      noise W:wiener(covariance=1);
      state x:m;
      initial {x=0[m];}
      law evolution for x {calculus ito; drift 0[m/s]; diffusion W=1[m/s^(1/2)];}
      observable position:m=x;
      observable duration:s=2[s/m]*x;
    }
    """)
    plan = q.resolve(model, temporal=q.time.EulerMaruyama(depth=3))
    outputs = [model.observable("position"), model.observable("duration")]
    results = []
    identities = []
    for member in range(3):
        identity = q.SamplingIdentity(bytes([member]), b"path")
        identities.append(identity)
        result = q.run(
            plan, state=q.State.initial(plan), until_s=1.,
            output_times_s=(.5, 1.) if member == 0 else (1.,),
            master_seed=bytes(32), sampling=identity,
        )
        # The reduction must work with persisted Results, without source aliases
        # or an original live Run object as hidden authority.
        results.append(q.Result.from_bytes(plan, result.to_bytes()))
    return plan, outputs, identities, results


def test_time_result_statistics_retain_units_complete_membership_and_convention():
    plan, outputs, identities, results = fixture()
    field = plan.model.field("x")
    values = [Fraction(float(result.series(field).values[-1])) for result in results]
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    estimator = q.MomentEstimator.from_time_results(
        results, outputs[:1], time_s=1., assume_iid=True,
    )
    assert estimator.value == float(mean)
    assert estimator.value_type == q.ValueType.real(q.Dimension(length=1))
    assert estimator.standard_error() == pytest.approx(math.sqrt(float(variance / 3)), rel=2e-15)
    assert estimator.reduction == "iid-mean"
    assert estimator.weights == [1., 1., 1.]
    assert estimator.sampling_identities == identities
    assert estimator.observables == outputs[:1]
    assert len(estimator.result_identities) == 3
    assert estimator.time_s == 1.
    assert estimator.model_digest == outputs[0].model_digest
    assert "members=3" in repr(estimator)
    covariance = q.MomentEstimator.from_time_results(
        results, outputs, time_s=1., moment="covariance", assume_iid=True,
    )
    assert covariance.value == float(2 * variance)
    assert covariance.value_type == q.ValueType.real(q.Dimension(length=1, time=1))
    with pytest.raises(q.ValidationError):
        covariance.standard_error()


def test_sample_replay_is_an_empirical_occurrence_but_not_an_iid_replica():
    _, outputs, _, results = fixture()
    repeated = [results[0], results[0]]
    empirical = q.MomentEstimator.from_time_results(
        repeated, outputs[:1], time_s=1., moment="variance",
    )
    assert empirical.value == 0.
    assert len(empirical.result_identities) == 2
    with pytest.raises(q.ValidationError):
        q.MomentEstimator.from_time_results(repeated, outputs[:1], time_s=1., assume_iid=True)
    with pytest.raises(q.ValidationError):
        empirical.standard_error()
    with pytest.raises(q.ValidationError):
        q.MomentEstimator.from_time_results(
            results, outputs[:1], time_s=1., assume_iid=True, weights=[1., 2., 1.],
        )


def test_missing_members_and_unretained_times_are_never_silently_filtered():
    _, outputs, _, results = fixture()
    with pytest.raises(TypeError):
        q.MomentEstimator.from_time_results([results[0], None], outputs[:1], time_s=1.)
    with pytest.raises(q.ValidationError):
        q.MomentEstimator.from_time_results(results, outputs[:1], time_s=.5, weights=[1., 0., 0.])
    with pytest.raises(q.ValidationError):
        q.MomentEstimator.from_time_results(results, outputs[:1], time_s=1., weights=[1.])
    with pytest.raises(ValueError):
        q.MomentEstimator.from_time_results(results, outputs[:1], time_s=1., moment="sum", assume_iid=True)
