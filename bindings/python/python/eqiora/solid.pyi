"""Configure linear elasticity and inspect structural results.

Authority: ``bindings/python/python/eqiora/solid.py``.
"""

from typing import final

from . import LinearSolveSummary, Result

@final
class LinearElasticityEvidence:
    """Scientific evidence selected from an accepted structural result.

    Authority: ``crates/eqiora-python/src/elasticity.rs::PyLinearElasticityEvidence``.
    """

    @property
    def plan_key(self) -> str: ...
    @property
    def constrained_reaction(self) -> tuple[float, float]: ...
    @property
    def integrated_body_force(self) -> tuple[float, float]: ...
    @property
    def assembly_packets(self) -> int: ...
    @property
    def assembly_targets(self) -> int: ...
    @property
    def solve(self) -> LinearSolveSummary: ...
    @property
    def exact_bounds(self) -> tuple[tuple[float, float], tuple[float, float]]: ...

def linear_elasticity_evidence(result: Result, /) -> LinearElasticityEvidence:
    """Select typed linear-elasticity evidence from its result.

    Authority: ``crates/eqiora-python/src/elasticity.rs::linear_elasticity_evidence``.
    """

    ...

__all__ = [
    "LinearElasticityEvidence",
    "linear_elasticity_evidence",
]
