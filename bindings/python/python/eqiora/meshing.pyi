"""Plan and generate meshes from geometry.

Authority: ``bindings/python/python/eqiora/meshing.py``.
"""

from typing import Self, final

import numpy as np
import numpy.typing as npt

from .geometry import Geometry, GeometrySelection
from . import DomainRef, Model

@final
class AffineTriangleMesher:
    """Select deterministic rectangle affine-triangle meshing.

    Every structured cell uses the lower-left to upper-right
    diagonal; callers select only the positive subdivision counts.

    Authority: ``crates/eqiora-python/src/meshing/plan.rs::PyAffineTriangleMesher``.
    """
    def __new__(cls, *, cells: tuple[int, int]) -> Self: ...
    @property
    def cells(self) -> tuple[int, int]: ...
    @property
    def diagonal(self) -> str: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __repr__(self) -> str: ...

@final
class CartesianMesher:
    """Select deterministic structured Cartesian meshing.

    Authority: ``crates/eqiora-python/src/meshing/plan.rs::PyCartesianMesher``.
    """
    def __new__(
        cls,
        *,
        cells: tuple[int] | tuple[int, int] | tuple[int, int, int],
    ) -> Self: ...
    @property
    def cells(self) -> tuple[int] | tuple[int, int] | tuple[int, int, int]: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __repr__(self) -> str: ...

@final
class GmshMesher:
    """Select the exact external Gmsh provider.

    Authority: ``crates/eqiora-python/src/meshing/plan.rs::PyGmshMesher``.
    """
    def __new__(
        cls,
        *,
        maximum_boundary_error: float = ...,
        maximum_target_size: float | None = ...,
        minimum_mean_ratio: float = ...,
        maximum_boundary_facets: int = ...,
    ) -> Self: ...
    @property
    def maximum_boundary_error(self) -> float: ...
    @property
    def maximum_target_size(self) -> float | None: ...
    @property
    def minimum_mean_ratio(self) -> float: ...
    @property
    def maximum_boundary_facets(self) -> int: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __repr__(self) -> str: ...

@final
class MeshPlan:
    """Complete provider plan bound to one exact geometry.

    Authority: ``crates/eqiora-python/src/meshing/plan.rs::PyMeshPlan``.
    """

    @property
    def source_digest(self) -> str: ...
    @property
    def provider(self) -> AffineTriangleMesher | CartesianMesher | GmshMesher: ...
    def __repr__(self) -> str: ...

@final
class Mesh:
    """Immutable source-bound accepted mesh.

    Authority: ``crates/eqiora-python/src/meshing/mesh.rs::PyMesh``.
    """

    @staticmethod
    def coordinate_factors(model: Model, domain: DomainRef, cells_per_factor: list[int]) -> Mesh: ...
    @staticmethod
    def from_bytes(data: bytes) -> Mesh: ...
    def to_bytes(self) -> bytes: ...
    @property
    def source_digest(self) -> str: ...
    @property
    def realized_geometry_digest(self) -> str | None: ...
    @property
    def digest(self) -> str: ...
    @property
    def correspondence_digest(self) -> str | None: ...
    @property
    def production_lineage_bytes(self) -> bytes: ...
    @property
    def production_lineage_digest(self) -> str: ...
    @property
    def canonical_bytes(self) -> bytes: ...
    @property
    def dimension(self) -> int: ...
    @property
    def vertex_count(self) -> int: ...
    @property
    def cell_count(self) -> int: ...
    @property
    def coordinates(self) -> npt.NDArray[np.float64]: ...
    @property
    def cells(self) -> npt.NDArray[np.uint32]: ...
    def entity_vertices(self, entity: tuple[int, int]) -> tuple[int, ...]:
        """Ordered vertex indices for a (dimension, index) entity in this exact Mesh.

        Combine with coordinates in coherent SI units to interpret oriented line/area
        moments. The order belongs to the retained Mesh, including top-cell ordering.
        Unknown entities reject; the returned tuple is immutable.
        """
        ...
    @property
    def minimum_mean_ratio(self) -> float: ...
    @property
    def selection_names(self) -> tuple[str, ...]: ...
    def selection_entity_count(self, name: str | GeometrySelection) -> int: ...
    def __repr__(self) -> str: ...

def resolve(
    geometry: Geometry,
    provider: AffineTriangleMesher | CartesianMesher | GmshMesher,
    /,
) -> MeshPlan:
    """Resolve a provider plan for the exact supplied geometry.

    Authority: ``crates/eqiora-python/src/meshing/plan.rs::resolve``.
    """

    ...

def generate(plan: MeshPlan, /) -> Mesh:
    """Execute a resolved provider plan and publish its accepted mesh.

    Authority: ``crates/eqiora-python/src/meshing/mesh.rs::generate``.
    """

    ...

__all__ = ["AffineTriangleMesher", "CartesianMesher", "GmshMesher", "Mesh", "MeshPlan", "generate", "resolve"]
