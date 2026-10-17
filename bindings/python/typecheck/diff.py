from typing import assert_type

import numpy as np
import numpy.typing as npt

import eqiora
from eqiora import diff
from eqiora.diff import (
    DerivativeImplementation,
    DifferentiableEvaluation,
    DifferentiableJvp,
    DifferentiablePrimal,
    DifferentiableProgram,
    DifferentiableVjp,
    DifferentiationEvidence,
    DifferentiationMode,
    FieldRef,
    LinearizationState,
    ParameterRef,
)


def check_diff_exports(
    plan: eqiora.Plan,
    parameter: ParameterRef,
    field: FieldRef,
    evaluation: DifferentiableEvaluation,
    primal: DifferentiablePrimal,
    jvp: DifferentiableJvp,
    vjp: DifferentiableVjp,
    evidence: DifferentiationEvidence,
) -> None:
    program = diff.compile(
        plan,
        inputs=(parameter,),
        outputs=(field,),
    )
    assert_type(program, DifferentiableProgram)
    assert_type(evaluation.primal(), DifferentiablePrimal)
    assert_type(primal.evidence, DifferentiationEvidence)
    assert_type(jvp.evidence, DifferentiationEvidence)
    assert_type(vjp.evidence, DifferentiationEvidence)
    assert_type(jvp.tangent, tuple[npt.NDArray[np.float64 | np.complex128], ...])
    assert_type(vjp.input_cotangent, tuple[npt.NDArray[np.float64 | np.complex128], ...])
    assert_type(program.jvp((1.0,)), DifferentiableJvp)
    assert_type(evaluation.vjp((np.ones(program.outputs[0].shape),)), DifferentiableVjp)
    assert_type(evidence.mode, DifferentiationMode)
    assert_type(evidence.implementation, DerivativeImplementation)
    assert_type(evidence.linearization_state, LinearizationState)
    assert_type(evidence.plan_identity, str)
    assert_type(program.plan_identity, str)
    batch = program.map((np.ones(3, dtype=np.float64),), point_shape=(3,))
    assert_type(batch, eqiora.EvaluationMapPlan)
    token = eqiora.EvaluationMapCancellation()
    assert_type(token.requested, bool)
    outcome = batch.execute(cancellation=token)
    assert_type(outcome, eqiora.CompleteEvaluationMap | eqiora.EvaluationMapTerminalReport)
    if isinstance(outcome, eqiora.CompleteEvaluationMap):
        assert_type(outcome.member(0), DifferentiableEvaluation | eqiora.Result)
        assert_type(outcome.jvp((np.ones(3, dtype=np.float64),)), eqiora.EvaluationMapJvp)
        assert_type(outcome.vjp((np.ones((*batch.point_shape, *program.outputs[0].shape), dtype=np.float64),)), eqiora.EvaluationMapVjp)
    else:
        assert_type(outcome.stopped_index, int)
        assert_type(outcome.member(0), DifferentiableEvaluation | eqiora.Result | None)
