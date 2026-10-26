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
    DifferentiableValueSignature as DifferentiableValueSignature,
    DifferentiationEvidence as DifferentiationEvidence,
    DifferentiationStepEvidence as DifferentiationStepEvidence,
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
    outputs: Sequence[FieldRef | ObservableRef],
    state: State | None = None,
    quadrature_points: int | None = None,
    until_s: float | None = None,
    output_times_s: Sequence[float] | None = None,
    steps: int | None = None,
    output_steps: Sequence[int] | None = None,
) -> DifferentiableProgram:
    """Compile a differentiable program over an ordered set of parameters.

    For a finite affine Plan, the inputs are real or complex Parameters, scalar or
    shaped, and outputs are ordered real/complex Fields or Observables;
    the Plan must not require state. Complex derivatives use the real differential
    with the pairing Re(sum(conj(a)*b)), so dependence on the conjugate is kept.
    Input coordinates are ordered by selection, then row-major by component, then
    real before imaginary part. The input identities count Parameters; the input
    shape counts real coordinates. A finite nonlinear Plan also needs its exact seed.

    Spatial backward-Euler Programs bind the complete Run request and initial State.
    Fields and reduced Observables have a leading physical-time axis; derivatives
    propagate through every accepted step, independently of the output schedule.
    Observables retain endpoint Time and direct Parameter dependence. Spatial
    integrals use the same explicit quadrature rules as stationary Programs.

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
    "DifferentiableValueSignature",
    "DifferentiationEvidence",
    "DifferentiationStepEvidence",
    "DifferentiationMode",
    "FieldRef",
    "LinearizationState",
    "ParameterRef",
    "compile",
]
