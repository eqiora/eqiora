"""Accepted implicit differentiation over exact Eqiora programs."""

from ._eqiora import (
    DerivativeImplementation,
    DifferentiableEvaluation,
    DifferentiableJvp,
    DifferentiablePrimal,
    DifferentiableProgram,
    DifferentiableVjp,
    CompleteEvaluationMap,
    EvaluationMapPlan,
    EvaluationMapTerminalReport,
    EvaluationMapCancellation,
    EvaluationMapJvp,
    EvaluationMapVjp,
    DifferentiationEvidence,
    DifferentiationMode,
    FieldRef,
    LinearizationState,
    ParameterRef,
    _compile_differentiable,
)

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


def compile(plan, *, inputs, output, state=None):
    """Compile one immutable program over an ordered Parameter coordinate set.

    ``program.evaluate(parameters)`` accepts another complete numerical point
    without mutating the Model or Plan. Parameters form an ordered sequence of
    value leaves: scalars or exact CPU ``float64``/``complex128`` arrays in each
    selected Parameter's intrinsic shape. Each array leaf also admits an Eqiora
    Array or no-transfer CPU DLPack producer. Tangents and cotangents currently
    use flat real-coordinate CPU ``float64`` arrays. Finite nonlinear Plans require
    their initial ``state`` and an Observable output; spatial scalar Plans
    select a Field output. Finite affine Plans need no seed, and support real
    Observable sensitivities to typed real or complex Parameters with real or complex
    Field coordinates, including conjugate dependence. Complex actions use the
    real differential and pairing ``Re(sum(conj(a)*b))``; they do not assume
    holomorphic dependence. Derivative arrays concatenate selected Parameters in
    selection order, with row-major components and real then imaginary parts
    for each complex component. ``input_ids`` lists Parameter identities;
    ``input_shape`` counts their real numerical coordinates.
    """

    return _compile_differentiable(
        plan,
        inputs=inputs,
        output=output,
        state=state,
    )
