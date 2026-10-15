"""Accepted implicit differentiation over exact Eqiora programs."""

from ._eqiora import (
    DerivativeImplementation,
    DifferentiableEvaluation,
    DifferentiableJvp,
    DifferentiablePrimal,
    DifferentiableProgram,
    DifferentiableVjp,
    DifferentiableValueSignature,
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
    "DifferentiableValueSignature",
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


def compile(plan, *, inputs, outputs, state=None):
    """Compile one immutable program over an ordered Parameter coordinate set.

    ``program.evaluate(parameters)`` accepts another complete numerical point
    without mutating the Model or Plan. Parameters form an ordered sequence of
    value leaves: scalars or exact CPU ``float64``/``complex128`` arrays in each
    selected Parameter's intrinsic shape. Each array leaf also admits an Eqiora
    Array or no-transfer CPU DLPack producer. Tangents and cotangents use the same ordered leaf structure,
    bound to the input and output signatures respectively. Finite nonlinear Plans require
    their initial ``state``; ``outputs`` selects ordered exact Field or Observable
    references. Spatial scalar Plans currently select one Field. Finite affine Plans need no seed, and support real
    Observable sensitivities to typed real or complex Parameters with real or complex
    Field coordinates, including conjugate dependence. Complex actions use the
    real differential and pairing ``Re(sum(conj(a)*b))``; they do not assume
    holomorphic dependence. Complex derivative leaves keep complex components;
    cotangents identify the base spaces without assigning primal units to dual values. ``input_ids`` lists Parameter identities;
    ``input_shape`` counts their real numerical coordinates.

    Accepted ``point``, ``output``, ``tangent`` and ``input_cotangent`` values are tuples of immutable NumPy leaves,
    retaining each value's native shape and real/complex dtype. Scalars are
    zero-dimensional arrays. ``program.inputs`` and ``program.outputs`` expose
    the native signatures, including units and spatial coefficient associations.
    Maps use the same ordered leaves, with explicitly declared point/seed axes
    before each intrinsic shape. Partial residual/unknown actions and the PyTorch
    adapter still use real coordinates. JAX projects the same ordered native
    leaves as pytrees.
    """

    return _compile_differentiable(
        plan,
        inputs=inputs,
        outputs=outputs,
        state=state,
    )
