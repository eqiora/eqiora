"""Time-integration policies for numerical plans.

Authority: ``crates/eqiora-python/src/common_plan/policy.rs::PyBackwardEuler``.
"""
from typing import ClassVar, Mapping, Self, final
from . import ActivationRef, Dimension, FieldRef, ParameterRef

@final
class OdePlanView:
    """Resolved no-Mesh ODE capability.

    Authority: ``crates/eqiora-python/src/common_plan/capability_view.rs::PyOdePlanView``.
    """
    @property
    def kind(self) -> str: ...
    @property
    def backend(self) -> str: ...
    @property
    def backend_version(self) -> str: ...
    def __repr__(self) -> str: ...

@final
class BackwardEuler:
    """Positive Backward-Euler operator step.

    Authority: ``crates/eqiora-python/src/common_plan/policy.rs::PyBackwardEuler``.
    """
    def __new__(cls, step_s: float) -> Self: ...
    @property
    def step_s(self) -> float: ...
    def __repr__(self) -> str: ...

@final
class Tsitouras45:
    """Adaptive explicit ODE integration with exact Field-bound SI tolerances.

    Authority: ``crates/eqiora-python/src/common_plan/policy.rs::PyTsitouras45``.
    """
    def __new__(
        cls,
        *,
        initial_step_s: float,
        relative_tolerance: float,
        absolute_tolerances: Mapping[FieldRef, float],
        events: EventPolicy | None = None,
        forward_sensitivities: ForwardSensitivity | None = None,
    ) -> Self: ...
    @property
    def initial_step_s(self) -> float: ...
    @property
    def relative_tolerance(self) -> float: ...
    @property
    def absolute_tolerances(self) -> dict[FieldRef, float]: ...
    @property
    def events(self) -> EventPolicy | None: ...
    @property
    def forward_sensitivities(self) -> ForwardSensitivity | None: ...
    def __repr__(self) -> str: ...

__all__ = ["BackwardEuler", "OdePlanView", "Tsitouras45", "TimeFunctionalQuadrature", "GuardTolerance", "EventPolicy", "ForwardSensitivity", "SensitivityTolerance"]

@final
class TimeFunctionalQuadrature:
    """Explicit quadrature over native accepted-step integration history.

    Authority: ``crates/eqiora-python/src/result/time_observe.rs::PyTimeFunctionalQuadrature``.
    """
    AcceptedStepSimpson: ClassVar[TimeFunctionalQuadrature]
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class GuardTolerance:
    """Positive coherent-SI tolerance for one exact Activation guard.

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
    """Explicit bounded canonical event execution with per-Activation guard units.

    Atomic root group members require identical guard tolerances.

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
    """Positive Field/Parameter derivative tolerance in quotient physical units.

    Authority: ``crates/eqiora-python/src/common_plan/forward_policy.rs::PySensitivityTolerance``.
    """
    def __new__(cls, field: FieldRef, parameter: ParameterRef, value: float, dimension: Dimension) -> Self: ...
    @property
    def field(self) -> FieldRef: ...
    @property
    def parameter(self) -> ParameterRef: ...
    @property
    def value(self) -> float: ...
    @property
    def dimension(self) -> Dimension: ...

@final
class ForwardSensitivity:
    """Explicit continuous forward derivative error controls.

    Authority: ``crates/eqiora-python/src/common_plan/forward_policy.rs::PyForwardSensitivity``.
    """
    def __new__(cls, *, relative_tolerance: float, absolute_tolerances: tuple[SensitivityTolerance, ...]) -> Self: ...
    @property
    def relative_tolerance(self) -> float: ...
    @property
    def model_digest(self) -> str: ...
    @property
    def absolute_tolerances(self) -> tuple[SensitivityTolerance, ...]: ...
