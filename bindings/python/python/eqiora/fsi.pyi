"""Inspect fluid–structure interaction results.

Authority: ``bindings/python/python/eqiora/fsi.py``.
"""

from typing import final
import numpy as np
import numpy.typing as npt
from . import LinearSolveSummary, Result, State, fluid

@final
class FixedReferenceFsiPlanView:
    """Resolved scales for fixed-reference FSI.

    Authority: ``crates/eqiora-python/src/common_plan/capability_view.rs::PyFixedReferenceFsiPlanView``.
    """
    @property
    def kind(self) -> str: ...
    @property
    def scaling(self) -> fluid.IncompressibleScales: ...
    @property
    def scaling_receipt(self) -> fluid.IncompressibleScalingReceipt2d: ...

@final
class FsiDomainEvidence:
    """Exact Domain and its owned mesh cells.

    Authority: ``crates/eqiora-python/src/fsi_evidence.rs::PyFsiDomainEvidence``.
    """
    @property
    def identity(self) -> str: ...
    @property
    def cells(self) -> npt.NDArray[np.uint32]: ...

@final
class FsiConnectionEvidence:
    """Exact Connection, endpoint identities, and trace facets.

    Authority: ``crates/eqiora-python/src/fsi_evidence.rs::PyFsiConnectionEvidence``.
    """
    @property
    def identity(self) -> str: ...
    @property
    def endpoint_domains(self) -> tuple[str, str]: ...
    @property
    def endpoint_fields(self) -> tuple[str, str]: ...
    @property
    def facets(self) -> npt.NDArray[np.uint32]: ...

@final
class FsiInterfaceActionEvidence:
    """Recovered action on one exact Connection entity and basis slot.

    Row ``i`` of ``endpoint_actions`` belongs to ``endpoint_domains[i]`` and
    ``endpoint_fields[i]``.

    Authority: ``crates/eqiora-python/src/fsi_evidence.rs::PyFsiInterfaceActionEvidence``.
    """
    @property
    def connection(self) -> str: ...
    @property
    def entity_dimension(self) -> int: ...
    @property
    def entity_index(self) -> int: ...
    @property
    def slot(self) -> int: ...
    @property
    def endpoint_domains(self) -> tuple[str, str]: ...
    @property
    def endpoint_fields(self) -> tuple[str, str]: ...
    @property
    def endpoint_actions(self) -> npt.NDArray[np.float64]: ...
    @property
    def imbalance(self) -> npt.NDArray[np.float64]: ...

@final
class FsiStateEvidence:
    """Numerical observations for one exact accepted common FSI State.

    Authority: ``crates/eqiora-python/src/fsi_evidence.rs::PyFsiStateEvidence``.
    """
    @property
    def state_digest(self) -> str: ...
    @property
    def interface_actions(self) -> tuple[FsiInterfaceActionEvidence, ...]: ...
    @property
    def previous_kinetic_energy_j_per_m(self) -> float: ...
    @property
    def next_kinetic_energy_j_per_m(self) -> float: ...
    @property
    def previous_elastic_energy_j_per_m(self) -> float: ...
    @property
    def next_elastic_energy_j_per_m(self) -> float: ...
    @property
    def kinetic_increment_j_per_m(self) -> float: ...
    @property
    def elastic_increment_j_per_m(self) -> float: ...
    @property
    def viscous_dissipation_j_per_m(self) -> float: ...
    @property
    def energy_defect_j_per_m(self) -> float: ...
    @property
    def numerical_residual_norm(self) -> float: ...
    @property
    def continuity_residual_norm(self) -> float: ...
    @property
    def kinematic_residual_norm(self) -> float: ...
    @property
    def interface_velocity_jump_norm(self) -> float: ...
    @property
    def interface_action_imbalance_n_per_m(self) -> float: ...
    @property
    def solve(self) -> LinearSolveSummary: ...
    @property
    def assembly_packets(self) -> int: ...
    @property
    def assembly_targets(self) -> int: ...

@final
class FsiEvidence:
    """Observation-only partition and per-State evidence for a common FSI Result.

    Authority: ``crates/eqiora-python/src/fsi_evidence.rs::PyFsiEvidence``.
    """
    @property
    def request_identity(self) -> str: ...
    @property
    def domains(self) -> tuple[FsiDomainEvidence, ...]: ...
    @property
    def connections(self) -> tuple[FsiConnectionEvidence, ...]: ...
    @property
    def states(self) -> tuple[FsiStateEvidence, ...]: ...
    def state(self, state: State) -> FsiStateEvidence: ...

def evidence(result: Result) -> FsiEvidence:
    """Select observation-only FSI evidence from an accepted common Result.

    Authority: ``crates/eqiora-python/src/fsi_evidence.rs::evidence``.
    """
    ...

__all__ = ["FixedReferenceFsiPlanView", "FsiConnectionEvidence", "FsiDomainEvidence", "FsiEvidence", "FsiInterfaceActionEvidence", "FsiStateEvidence", "evidence"]
