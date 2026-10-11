"""Construct exact geometry values and select regions and boundaries.

Authority: ``bindings/python/python/eqiora/geometry.py``.
"""

from collections.abc import Mapping, Sequence
from typing import final, overload

@final
class GeometryRegionHandle:
    """Handle to one exact region returned by geometry construction.

    Authority: ``crates/eqiora-python/src/planar_operation.rs::PyGeometryRegionHandle``.
    """

    @property
    def dimension(self) -> int: ...
    def __eq__(self, other: object, /) -> bool: ...

@final
class GeometryBoundaryHandle:
    """Handle to one exact boundary returned by geometry construction.

    Authority: ``crates/eqiora-python/src/planar_operation.rs::PyGeometryBoundaryHandle``.
    """

    @property
    def dimension(self) -> int: ...
    def __eq__(self, other: object, /) -> bool: ...

@final
class GeometryOperation:
    """Immutable result of one exact primitive or Boolean operation.

    ``boundaries`` uses canonical construction order: an interval returns
    ``(lower, upper)``, a rectangle returns
    ``(x_lower, x_upper, y_lower, y_upper)``, a circle returns its sole curve,
    and subtract returns the four outer boundaries followed by the created cut.

    Authority: ``crates/eqiora-python/src/planar_operation.rs::PyGeometryOperation``.
    """

    @property
    def region(self) -> GeometryRegionHandle: ...
    @property
    def boundaries(self) -> tuple[GeometryBoundaryHandle, ...]: ...
    def __eq__(self, other: object, /) -> bool: ...

@final
class GeometryGraph:
    """Common owner of exact Cartesian, planar, and solid authoring operations.

    Authority: ``crates/eqiora-python/src/planar_operation.rs::PyGeometryGraph``.
    """

    def __init__(self) -> None: ...
    def interval(
        self,
        *,
        bounds: tuple[float, float],
    ) -> GeometryOperation: ...
    def rectangle(
        self,
        *,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
    ) -> GeometryOperation: ...
    def circle(
        self,
        *,
        center: tuple[float, float],
        radius: float,
    ) -> GeometryOperation: ...
    def subtract(
        self,
        rectangle: GeometryOperation,
        circle: GeometryOperation,
    ) -> GeometryOperation: ...
    def partition(
        self,
        left: GeometryOperation,
        right: GeometryOperation,
        /,
        *,
        interface: tuple[GeometryBoundaryHandle, GeometryBoundaryHandle],
    ) -> GeometryOperation: ...
    def rectangle_extrusion(
        self,
        *,
        x_bounds: tuple[float, float],
        y_bounds: tuple[float, float],
        plane_z: float,
        depth: float,
        modeling_tolerance: float,
    ) -> GeometrySolidOperation: ...
    def decode_solid(self, data: bytes) -> GeometrySolidOperation: ...
    def circular_through_cut(
        self,
        target: GeometrySolidOperation,
        /,
        *,
        center: tuple[float, float],
        radius: float,
        boolean_tolerance: float,
    ) -> GeometrySolidOperation: ...
    @overload
    def build(
        self,
        operation: GeometryOperation,
        /,
        *,
        named_topology: Mapping[
            str,
            GeometryRegionHandle
            | GeometryBoundaryHandle
            | Sequence[GeometryRegionHandle | GeometryBoundaryHandle],
        ],
    ) -> Geometry: ...
    @overload
    def build(
        self,
        operation: GeometrySolidOperation,
        /,
        *,
        named_topology: None = None,
    ) -> GeometryBuildReceipt: ...
    @overload
    def build(
        self,
        operation: GeometrySolidOperation,
        /,
        *,
        named_topology: Mapping[
            str, GeometryFaceHandle | Sequence[GeometryFaceHandle]
        ],
    ) -> Geometry: ...

@final
class GeometryFaceHandle:
    """Exact solid-face handle bound to one graph session and revision.

    Authority: ``crates/eqiora-python/src/cad_authored.rs::PyGeometryFaceHandle``.
    """

    @property
    def canonical_bytes(self) -> bytes: ...
    @property
    def graph_digest(self) -> str: ...
    @property
    def provenance_key(self) -> str: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class GeometryBuildReceipt:
    """Read-only result of a solid build.

    Authority: ``crates/eqiora-python/src/cad_authored.rs::PyGeometryBuildReceipt``.
    """

    @property
    def graph_digest(self) -> str: ...
    @property
    def provider_profile(self) -> str: ...
    @property
    def requested_modeling_tolerance(self) -> float: ...
    @property
    def requested_boolean_tolerance(self) -> float | None: ...
    @property
    def effective_boolean_tolerance(self) -> float | None: ...
    @property
    def maximum_position_discrepancy(self) -> float: ...
    @property
    def maximum_area_discrepancy(self) -> float: ...
    @property
    def maximum_volume_discrepancy(self) -> float: ...
    @property
    def repair(self) -> str: ...
    @property
    def retained_unchanged(self) -> tuple[GeometryFaceHandle, ...]: ...
    @property
    def retained_modified(self) -> tuple[GeometryFaceHandle, ...]: ...
    @property
    def created(self) -> tuple[GeometryFaceHandle, ...]: ...
    @property
    def deleted(self) -> tuple[GeometryFaceHandle, ...]: ...
    @property
    def split(self) -> tuple[GeometryFaceHandle, ...]: ...
    @property
    def merged(self) -> tuple[GeometryFaceHandle, ...]: ...
    def __eq__(self, other: object, /) -> bool: ...

@final
class GeometrySolidOperation:
    """Immutable solid operation owned by one ``GeometryGraph``.

    Authority: ``crates/eqiora-python/src/cad_authored.rs::PyGeometrySolidOperation``.
    """

    @property
    def canonical_bytes(self) -> bytes: ...
    @property
    def graph_digest(self) -> str: ...
    @property
    def x_bounds(self) -> tuple[float, float]: ...
    @property
    def y_bounds(self) -> tuple[float, float]: ...
    @property
    def plane_z(self) -> float: ...
    @property
    def extrusion_depth(self) -> float: ...
    @property
    def requested_modeling_tolerance(self) -> float: ...
    @property
    def requested_boolean_tolerance(self) -> float | None: ...
    @property
    def cut_center(self) -> tuple[float, float] | None: ...
    @property
    def cut_radius(self) -> float | None: ...
    @property
    def bounds(
        self,
    ) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]: ...
    @property
    def vertex_count(self) -> int | None: ...
    @property
    def edge_count(self) -> int | None: ...
    @property
    def face_count(self) -> int: ...
    @property
    def closed_shell_count(self) -> int: ...
    @property
    def body_count(self) -> int: ...
    @property
    def genus(self) -> int: ...
    @property
    def volume(self) -> float: ...
    @property
    def surface_area(self) -> float: ...
    @property
    def repair(self) -> str: ...
    @property
    def selection_names(self) -> tuple[str, ...]: ...
    def face_handle(self, name: str) -> GeometryFaceHandle: ...
    def resolve_face(self, handle: GeometryFaceHandle) -> str: ...
    def face_area(self, handle: GeometryFaceHandle) -> float: ...
    def face_boundary_loop_count(self, handle: GeometryFaceHandle) -> int: ...
    def rectangular_face_vertices(
        self, handle: GeometryFaceHandle
    ) -> tuple[tuple[float, float, float], ...] | None: ...
    def rectangular_face_centroid(
        self, handle: GeometryFaceHandle
    ) -> tuple[float, float, float] | None: ...
    def planar_face_outward_normal(
        self, handle: GeometryFaceHandle
    ) -> tuple[float, float, float] | None: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...

@final
class GeometrySelection:
    """Immutable named selection bound to one exact Geometry revision.

    Authority: ``crates/eqiora-python/src/geometry.rs::PyGeometrySelection``.
    """

    @property
    def source_digest(self) -> str: ...
    @property
    def name(self) -> str: ...
    @property
    def dimension(self) -> int: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

@final
class Geometry:
    """Immutable exact geometry produced by an accepted authored graph.

    Authority: ``crates/eqiora-python/src/geometry.rs::PyGeometry``.
    """

    @property
    def dimension(self) -> int: ...
    @property
    def bounds(
        self,
    ) -> (
        tuple[tuple[float, float]]
        | tuple[tuple[float, float], tuple[float, float]]
        | tuple[
            tuple[float, float], tuple[float, float], tuple[float, float]
        ]
    ): ...
    @property
    def classification_tolerance(self) -> float | None: ...
    @property
    def canonical_bytes(self) -> bytes: ...
    @property
    def digest(self) -> str: ...
    @property
    def selection_names(self) -> tuple[str, ...]: ...
    def selection_dimension(self, name: str) -> int: ...
    def selection(self, name: str) -> GeometrySelection: ...
    def __eq__(self, other: object, /) -> bool: ...
    def __hash__(self) -> int: ...
    def __repr__(self) -> str: ...

__all__ = [
    "Geometry",
    "GeometryBoundaryHandle",
    "GeometryBuildReceipt",
    "GeometryFaceHandle",
    "GeometryGraph",
    "GeometryOperation",
    "GeometryRegionHandle",
    "GeometrySelection",
    "GeometrySolidOperation",
]
