"""Accepted implicit differentiation over exact Eqiora programs."""

from ._eqiora import (
    DerivativeImplementation,
    DifferentiableEvaluation,
    DifferentiableJvp,
    DifferentiablePrimal,
    DifferentiableProgram,
    DifferentiableVjp,
    DifferentiableValueSignature,
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
    "DifferentiationEvidence",
    "DifferentiationMode",
    "FieldRef",
    "LinearizationState",
    "ParameterRef",
    "compile",
]


def compile(
    plan, *, inputs, outputs, state=None, quadrature_points=None,
    until_s=None, output_times_s=None, steps=None, output_steps=None,
):
    """Compile one immutable program over an ordered Parameter coordinate set.

    ``program.evaluate(parameters)`` accepts another complete numerical point
    without mutating the Model or Plan. Parameters form an ordered sequence of
    value leaves: scalars or exact CPU ``float64``/``complex128`` arrays in each
    selected Parameter's intrinsic shape. Each array leaf also admits an Eqiora
    Array or no-transfer CPU DLPack producer. Tangents and cotangents use the same ordered leaf structure,
    bound to the input and output signatures respectively. Finite nonlinear Plans require
    their initial ``state``; ``outputs`` selects ordered exact Field or Observable
    references. Static spatial Plans select ordered complete real/complex Fields
    and reduced Observables, while retaining the complete coupled solve. Spatial
    integrals require explicit ``quadrature_points``; the resulting rules belong
    to the immutable Program and its saved identity.
    Spatial backward-Euler Plans take the ordinary Run controls: ``state`` and
    either ``steps``/``output_steps`` or ``until_s``/``output_times_s``. The current
    admitted request is one step with its endpoint and complete Field outputs.
    Parameter changes hold that exact preceding physical State and both endpoint
    times fixed; these are not derivatives through earlier steps. Compilation
    needs no previously solved canonical step. Maps and saved collections retain
    this same request meaning and reopen accepted candidates without solving.

    Finite affine Plans need no seed, and support real
    Observable sensitivities to typed real or complex Parameters with real or complex
    Field coordinates, including conjugate dependence. Complex actions use the
    real differential and pairing ``Re(sum(conj(a)*b))``; they do not assume
    holomorphic dependence. Complex derivative leaves keep complex components;
    cotangents identify the base spaces without assigning primal units to dual values.

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
        quadrature_points=quadrature_points,
        until_s=until_s,
        output_times_s=output_times_s,
        steps=steps,
        output_steps=output_steps,
    )
