"""Configure incompressible flow and inspect its results.

Authority: ``bindings/python/python/eqiora/fluid.py``.
"""

from typing import ClassVar, final
from fractions import Fraction

from . import FieldRef, LinearSolveSummary, Result

@final
class IncompressibleFlowPlanView:
    """Resolved incompressible-flow roles, spaces, gauge, and scales.

    Authority: ``crates/eqiora-python/src/common_plan/capability_view.rs::PyIncompressibleFlowPlanView``.
    """
    @property
    def kind(self) -> str: ...
    @property
    def velocity(self) -> FieldRef: ...
    @property
    def pressure(self) -> FieldRef: ...
    @property
    def velocity_space(self) -> str: ...
    @property
    def pressure_space(self) -> str: ...
    @property
    def pressure_gauge(self) -> PressureGauge2d | None: ...
    @property
    def scaling(self) -> IncompressibleScales: ...
    @property
    def scaling_receipt(self) -> IncompressibleScalingReceipt2d: ...

@final
class IncompressibleScaling:
    """Optional manual components for exact-cylinder incompressible scaling.

    ``None`` leaves that component under deterministic resolver ownership.

    Authority: ``crates/eqiora-python/src/common_plan/scaling.rs::PyIncompressibleScaling``.
    """

    def __new__(
        cls,
        *,
        length_m: float | None = None,
        velocity_m_per_s: float | None = None,
        pressure_pa: float | None = None,
    ) -> IncompressibleScaling: ...
    @property
    def length_m(self) -> float | None: ...
    @property
    def velocity_m_per_s(self) -> float | None: ...
    @property
    def pressure_pa(self) -> float | None: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class IncompressibleScales:
    """Immutable effective 2D incompressible scales for a resolved Plan.

    Authority: ``crates/eqiora-python/src/common_plan/scaling.rs::PyIncompressibleScales``.
    """

    @property
    def length_m(self) -> float: ...
    @property
    def velocity_m_per_s(self) -> float: ...
    @property
    def pressure_pa(self) -> float: ...
    @property
    def gauge_per_s(self) -> float: ...
    @property
    def weak_functional_w(self) -> float: ...

@final
class IncompressibleScalingComponent2d:
    """Closed intrinsic-2D scaling component.

    Authority: ``crates/eqiora-python/src/common_plan/scaling.rs::PyScalingComponent2d``.
    """
    Length: ClassVar[IncompressibleScalingComponent2d]
    Velocity: ClassVar[IncompressibleScalingComponent2d]
    Pressure: ClassVar[IncompressibleScalingComponent2d]
    Gauge: ClassVar[IncompressibleScalingComponent2d]
    WeakFunctional: ClassVar[IncompressibleScalingComponent2d]
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class IncompressibleScalingMode:
    """Closed provenance mode for one effective scaling component.

    Authority: ``crates/eqiora-python/src/common_plan/scaling.rs::PyScalingMode``.
    """
    Manual: ClassVar[IncompressibleScalingMode]
    Automatic: ClassVar[IncompressibleScalingMode]
    Derived: ClassVar[IncompressibleScalingMode]
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class IncompressibleScalingRule2d:
    """Closed rule used to resolve one intrinsic-2D component.

    Authority: ``crates/eqiora-python/src/common_plan/scaling.rs::PyScalingRule2d``.
    """
    ManualOverrideV1: ClassVar[IncompressibleScalingRule2d]
    ExactChannelHeightV1: ClassVar[IncompressibleScalingRule2d]
    ExactInletMaximumV1: ClassVar[IncompressibleScalingRule2d]
    ViscousStokesPressureV1: ClassVar[IncompressibleScalingRule2d]
    GaugeRateV1: ClassVar[IncompressibleScalingRule2d]
    WeakFunctionalV1: ClassVar[IncompressibleScalingRule2d]
    ExactPartitionLengthV1: ClassVar[IncompressibleScalingRule2d]
    SolidShearWaveVelocityV1: ClassVar[IncompressibleScalingRule2d]
    FluidDynamicPressureV1: ClassVar[IncompressibleScalingRule2d]
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class IncompressibleScalingAuthorityKind:
    """Closed kind of authoritative scaling observation.

    Authority: ``crates/eqiora-python/src/common_plan/scaling.rs::PyScalingAuthorityKind``.
    """
    ManualRequest: ClassVar[IncompressibleScalingAuthorityKind]
    ExactGeometrySpan: ClassVar[IncompressibleScalingAuthorityKind]
    ModelInletMaximum: ClassVar[IncompressibleScalingAuthorityKind]
    ModelDynamicViscosity: ClassVar[IncompressibleScalingAuthorityKind]
    ModelSolidShearWave: ClassVar[IncompressibleScalingAuthorityKind]
    ModelFluidMassDensity: ClassVar[IncompressibleScalingAuthorityKind]
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class PressureGauge2d:
    """Closed pressure representative selected by transient resolution.

    Authority: ``crates/eqiora-python/src/common_plan/policy.rs::PyPressureGauge2d``.
    """
    ZeroIntegral: ClassVar[PressureGauge2d]
    BoundaryTraction: ClassVar[PressureGauge2d]
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class IncompressibleScalingAuthority2d:
    """Immutable typed authoritative observation.

    Authority: ``crates/eqiora-python/src/common_plan/scaling.rs::PyScalingAuthority2d``.
    """
    @property
    def kind(self) -> IncompressibleScalingAuthorityKind: ...
    @property
    def axis(self) -> int | None: ...
    @property
    def bounds_m(self) -> tuple[float, float] | None: ...
    @property
    def coordinate_m(self) -> tuple[float, float] | None: ...
    @property
    def outward_normal(self) -> tuple[float, float] | None: ...
    @property
    def velocity_m_per_s(self) -> tuple[float, float] | None: ...
    @property
    def dynamic_viscosity_pa_s(self) -> float | None: ...
    @property
    def shear_modulus_pa(self) -> float | None: ...
    @property
    def mass_density_kg_per_m3(self) -> float | None: ...

@final
class IncompressibleScalingComponentRecord2d:
    """Immutable effective value and provenance for one component.

    Authority: ``crates/eqiora-python/src/common_plan/scaling.rs::PyScalingComponentRecord2d``.
    """
    @property
    def component(self) -> IncompressibleScalingComponent2d: ...
    @property
    def value(self) -> float: ...
    @property
    def dimension(self) -> tuple[Fraction, Fraction, Fraction, Fraction, Fraction, Fraction, Fraction]: ...
    @property
    def mode(self) -> IncompressibleScalingMode: ...
    @property
    def rule(self) -> IncompressibleScalingRule2d: ...
    @property
    def dependencies(self) -> tuple[IncompressibleScalingComponent2d, ...]: ...
    @property
    def authorities(self) -> tuple[IncompressibleScalingAuthority2d, ...]: ...

@final
class IncompressibleScalingReceipt2d:
    """Immutable five-component receipt with exact resource lineage.

    Authority: ``crates/eqiora-python/src/common_plan/scaling.rs::PyIncompressibleScalingReceipt2d``.
    """
    @property
    def provenance_digest(self) -> str: ...
    @property
    def model_digest(self) -> str: ...
    @property
    def geometry_digest(self) -> str: ...
    @property
    def correspondence_digest(self) -> str: ...
    @property
    def mesh_digest(self) -> str: ...
    @property
    def production_digest(self) -> str | None: ...
    @property
    def length(self) -> IncompressibleScalingComponentRecord2d: ...
    @property
    def velocity(self) -> IncompressibleScalingComponentRecord2d: ...
    @property
    def pressure(self) -> IncompressibleScalingComponentRecord2d: ...
    @property
    def gauge(self) -> IncompressibleScalingComponentRecord2d: ...
    @property
    def weak_functional(self) -> IncompressibleScalingComponentRecord2d: ...
    @property
    def components(self) -> tuple[IncompressibleScalingComponentRecord2d, ...]: ...

@final
class SteadyStokesEvidence:
    """Scientific evidence selected from an accepted steady-Stokes result.

    Authority: ``crates/eqiora-python/src/steady_stokes.rs::PySteadyStokesEvidence``.
    """

    @property
    def plan_key(self) -> str: ...
    @property
    def pressure_minimum(self) -> float: ...
    @property
    def pressure_maximum(self) -> float: ...
    @property
    def exact_bounds(self) -> tuple[tuple[float, float], tuple[float, float]]: ...
    @property
    def net_flux(self) -> float: ...
    @property
    def constrained_reaction(self) -> tuple[float, float]: ...
    @property
    def integrated_body_force(self) -> tuple[float, float]: ...
    @property
    def integrated_boundary_traction(self) -> tuple[float, float]: ...
    @property
    def momentum_closure(self) -> tuple[float, float]: ...
    @property
    def solve(self) -> LinearSolveSummary: ...
    @property
    def continuity_residual_norm(self) -> float: ...

def steady_stokes_evidence(result: Result, /) -> SteadyStokesEvidence:
    """Select typed steady-Stokes evidence from its accepted result.

    Authority: ``crates/eqiora-python/src/steady_stokes.rs::steady_stokes_evidence``.
    """

    ...

__all__ = [
    "IncompressibleFlowPlanView",
    "IncompressibleScales",
    "IncompressibleScaling",
    "IncompressibleScalingAuthority2d",
    "IncompressibleScalingAuthorityKind",
    "IncompressibleScalingComponent2d",
    "IncompressibleScalingComponentRecord2d",
    "IncompressibleScalingMode",
    "IncompressibleScalingReceipt2d",
    "IncompressibleScalingRule2d",
    "PressureGauge2d",
    "SteadyStokesEvidence",
    "steady_stokes_evidence",
]
