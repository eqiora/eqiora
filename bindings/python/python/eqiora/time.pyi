"""Time-integration methods and related settings.

Authority: ``crates/eqiora-python/src/common_plan/policy.rs::PyBackwardEuler``.
"""
from typing import ClassVar, Mapping, Sequence, Self, final
from . import ActivationRef, Dimension, FieldRef, ParameterRef

@final
class TimePlanView:
    """Resolved plan for finite time evolution (no mesh).

    Authority: ``crates/eqiora-python/src/common_plan/capability_view.rs::PyTimePlanView``.
    """
    @property
    def kind(self) -> str: ...
    @property
    def backend(self) -> str: ...
    @property
    def backend_version(self) -> str: ...
    def __repr__(self) -> str: ...

    @property
    def scalar_type(self) -> str: ...
    @property
    def value_representation(self) -> str: ...

@final
class BackwardEuler:
    """Backward Euler with a fixed positive step size.

    Authority: ``crates/eqiora-python/src/common_plan/policy.rs::PyBackwardEuler``.
    """
    def __new__(cls, step_s: float) -> Self: ...
    @property
    def step_s(self) -> float: ...
    def __repr__(self) -> str: ...

@final
class Tsitouras45:
    """Adaptive explicit Tsitouras 5(4) integration.

    Tolerances are in SI units and are set per Field, derivative order,
    component, and real or imaginary part.

    Authority: ``crates/eqiora-python/src/common_plan/policy.rs::PyTsitouras45``.
    """
    def with_conserved_norm(self, fields: Sequence[FieldRef], *, target: float, tolerance: float, dimension: Dimension) -> Self: ...
    def with_hermitian_parameter(self, parameter: ParameterRef) -> Self: ...
    def __new__(
        cls,
        *,
        initial_step_s: float,
        relative_tolerance: float,
        absolute_tolerances: Mapping[tuple[FieldRef, int, int, bool], float],
        events: EventPolicy | None = None,
        forward_sensitivities: ForwardSensitivity | None = None,
    ) -> Self: ...
    @property
    def initial_step_s(self) -> float: ...
    @property
    def relative_tolerance(self) -> float: ...
    @property
    def absolute_tolerances(self) -> dict[tuple[FieldRef, int, int, bool], float]: ...
    @property
    def events(self) -> EventPolicy | None: ...
    @property
    def forward_sensitivities(self) -> ForwardSensitivity | None: ...
    def __repr__(self) -> str: ...

@final
class ImplicitMidpoint:
    """Implicit midpoint rule with a fixed step and per-coordinate Newton tolerances.

    Resolution requires an explicit ``solve.Linear`` or ``solve.Newton`` policy.
    Linear requires structurally affine State dependence at each stage. Newton
    uses its iteration and line-search budgets with these per-coordinate
    correction tolerances; its residual tolerances do not control time stages.
    The tolerances control the Newton corrections, not the time-discretization
    error. Output samples are interpolated from the collocation polynomial of
    each completed step, so requesting outputs does not change the steps.

    Authority: ``crates/eqiora-python/src/common_plan/policy.rs::PyImplicitMidpoint``.
    """
    def with_conserved_norm(self, fields: Sequence[FieldRef], *, target: float, tolerance: float, dimension: Dimension) -> Self: ...
    def with_hermitian_parameter(self, parameter: ParameterRef) -> Self: ...
    def __new__(cls, *, step_s: float, relative_tolerance: float, absolute_tolerances: Mapping[tuple[FieldRef, int, int, bool], float]) -> Self: ...
    @property
    def step_s(self) -> float: ...
    @property
    def relative_tolerance(self) -> float: ...
    @property
    def absolute_tolerances(self) -> dict[tuple[FieldRef, int, int, bool], float]: ...
    def __repr__(self) -> str: ...

@final
class Bdf:
    """Adaptive residual-native BDF integration for admitted spatial systems.

    Tolerances control local time error in each real coordinate, including
    algebraic coordinates. Output cadence does not choose integration steps.
    Admitted real/complex index-one spatial systems retain the supplied initial
    State and derive consistent physical rates from the original equations.
    Requested outputs recover algebraic constraints from the accepted native
    interpolant. Resolution requires an explicit spatial solve policy.
    Finite models, events, sensitivities and durable adaptive checkpoints are
    not currently connected to this policy.

    Authority: ``crates/eqiora-python/src/common_plan/policy.rs::PyBdf``.
    """
    def __new__(cls, *, initial_step_s: float, relative_tolerance: float, absolute_tolerances: Mapping[tuple[FieldRef, int, int, bool], float]) -> Self: ...
    @property
    def initial_step_s(self) -> float: ...
    @property
    def relative_tolerance(self) -> float: ...
    @property
    def absolute_tolerances(self) -> dict[tuple[FieldRef, int, int, bool], float]: ...
    def __repr__(self) -> str: ...

__all__ = ["EulerMaruyama", "PoissonEvents", "BackwardEuler", "TimePlanView", "Tsitouras45", "ImplicitMidpoint", "Bdf", "TimeObservationInterpolation", "GuardTolerance", "EventPolicy", "ForwardSensitivity", "SensitivityTolerance"]

@final
class TimeObservationInterpolation:
    """Interpolation of Observable samples for time integrals and extrema.

    EndpointLinear uses accepted endpoints. NativeMidpointQuadratic also uses
    native accepted midpoint values; it is unavailable for spatial histories.
    Neither policy interpolates a restart State.

    Authority: ``crates/eqiora-python/src/result/time_observe.rs::PyTimeObservationInterpolation``.
    """
    NativeMidpointQuadratic: ClassVar[TimeObservationInterpolation]
    EndpointLinear: ClassVar[TimeObservationInterpolation]
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class GuardTolerance:
    """Positive tolerance, in coherent SI units, for one activation guard.

    Authority: ``crates/eqiora-python/src/common_plan/event_policy.rs::PyGuardTolerance``.
    """
    def __new__(cls, activation: ActivationRef, value: float, dimension: Dimension) -> Self: ...
    @property
    def activation(self) -> ActivationRef: ...
    @property
    def value(self) -> float: ...
    @property
    def dimension(self) -> Dimension: ...

@final
class EventPolicy:
    """Event-handling settings, with guard tolerances in each Activation's units.

    Events that must fire together as one root group need identical guard
    tolerances.

    Authority: ``crates/eqiora-python/src/common_plan/event_policy.rs::PyEventPolicy``.
    """
    def __new__(cls, *, max_events: int, guard_tolerances: tuple[GuardTolerance, ...]) -> Self: ...
    @property
    def max_events(self) -> int: ...
    @property
    def model_digest(self) -> str: ...
    @property
    def guard_tolerances(self) -> tuple[GuardTolerance, ...]: ...

@final
class SensitivityTolerance:
    """Positive tolerance for the derivative of a Field with respect to a Parameter, in the units of that derivative.

    Authority: ``crates/eqiora-python/src/common_plan/forward_policy.rs::PySensitivityTolerance``.
    """
    def __new__(cls, coordinate: tuple[FieldRef, int, int, bool], parameter: ParameterRef, value: float, dimension: Dimension) -> Self: ...
    @property
    def coordinate(self) -> tuple[FieldRef, int, int, bool]: ...
    @property
    def parameter(self) -> ParameterRef: ...
    @property
    def value(self) -> float: ...
    @property
    def dimension(self) -> Dimension: ...

@final
class ForwardSensitivity:
    """Error control for continuous forward sensitivities.

    Authority: ``crates/eqiora-python/src/common_plan/forward_policy.rs::PyForwardSensitivity``.
    """
    def __new__(cls, *, relative_tolerance: float, absolute_tolerances: tuple[SensitivityTolerance, ...]) -> Self: ...
    @property
    def relative_tolerance(self) -> float: ...
    @property
    def model_digest(self) -> str: ...
    @property
    def absolute_tolerances(self) -> tuple[SensitivityTolerance, ...]: ...

@final
class EulerMaruyama:
    """Ito diffusion on 2**depth equal steps over the original Run horizon.

    Output times must lie on that dyadic grid. Exact restart retains the original
    partition; this profile does not promise cross-platform random bytes.

    Authority: ``crates/eqiora-python/src/common_plan/policy/process.rs::PyEulerMaruyama``.
    """
    def __new__(cls, *, depth: int) -> Self: ...
    @property
    def depth(self) -> int: ...
    def __repr__(self) -> str: ...

@final
class PoissonEvents:
    """Constant-rate independent Poisson arrivals with a total event budget.

    Authority: ``crates/eqiora-python/src/common_plan/policy/process.rs::PyPoissonEvents``.
    """
    def __new__(cls, *, max_events: int) -> Self: ...
    @property
    def max_events(self) -> int: ...
    def __repr__(self) -> str: ...
