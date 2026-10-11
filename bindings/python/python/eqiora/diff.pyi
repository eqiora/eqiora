"""Implicit differentiation of Eqiora programs.

Authority: ``bindings/python/python/eqiora/diff.py``.
"""

from collections.abc import Sequence

from . import (
    DerivativeImplementation as DerivativeImplementation,
    DifferentiableEvaluation as DifferentiableEvaluation,
    DifferentiableJvp as DifferentiableJvp,
    DifferentiablePrimal as DifferentiablePrimal,
    DifferentiableProgram as DifferentiableProgram,
    DifferentiableVjp as DifferentiableVjp,
    CompleteEvaluationMap as CompleteEvaluationMap,
    EvaluationMapPlan as EvaluationMapPlan,
    EvaluationMapTerminalReport as EvaluationMapTerminalReport,
    EvaluationMapCancellation as EvaluationMapCancellation,
    EvaluationMapJvp as EvaluationMapJvp,
    EvaluationMapVjp as EvaluationMapVjp,
    DifferentiationEvidence as DifferentiationEvidence,
    DifferentiationMode as DifferentiationMode,
    FieldRef as FieldRef,
    LinearizationState as LinearizationState,
    Plan,
    ObservableRef,
    State,
    ParameterRef as ParameterRef,
)

def compile(
    plan: Plan,
    *,
    inputs: Sequence[ParameterRef],
    output: FieldRef | ObservableRef,
    state: State | None = None,
) -> DifferentiableProgram:
    """Compile a differentiable program over an ordered set of parameters.

    For a finite affine Plan, the inputs are real or complex Parameters, scalar or
    shaped, and the output is a real or complex Field or a real scalar Observable;
    the Plan must not require state. Complex derivatives use the real differential
    with the pairing Re(sum(conj(a)*b)), so dependence on the conjugate is kept.
    Input coordinates are ordered by selection, then row-major by component, then
    real before imaginary part. The input identities count Parameters; the input
    shape counts real coordinates. A finite nonlinear Plan also needs its exact seed.

    Authority: ``bindings/python/python/eqiora/diff.py::compile``.
    """

    ...

__all__ = [
    "DerivativeImplementation",
    "DifferentiableEvaluation",
    "DifferentiableJvp",
    "DifferentiablePrimal",
    "DifferentiableProgram",
    "DifferentiableVjp",
    "CompleteEvaluationMap",
    "EvaluationMapPlan",
    "EvaluationMapTerminalReport",
    "EvaluationMapCancellation",
    "EvaluationMapJvp",
    "EvaluationMapVjp",
    "DifferentiationEvidence",
    "DifferentiationMode",
    "FieldRef",
    "LinearizationState",
    "ParameterRef",
    "compile",
]
