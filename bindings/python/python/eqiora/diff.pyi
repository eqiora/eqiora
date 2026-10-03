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
    """Compile a program over an ordered parameter-coordinate set.

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
