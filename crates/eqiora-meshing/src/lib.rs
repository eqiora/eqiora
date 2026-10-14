//! **eqiora-meshing** — backend-neutral topology and geometry contracts.
//!
//! This crate owns inspectable mesh revisions, reference cells, affine entity
//! maps, and acceptance-quality evidence. It does not own physics, assembly,
//! solver policy, canonical model meaning, or artifact serialization.

mod affine_geometry;
mod cartesian_mesh;
mod discrete_field;
mod fixed_topology_geometry;
mod mesh;
mod p1_harmonic_geometry;
mod quadrature;
mod reference_topology;
mod remesh_overlap;
mod simplex_quadrature;
mod simplicial_mesh;

pub use affine_geometry::{AffineGeometryLinearization, AffineGeometryMap, AffineMapQuality};
pub use cartesian_mesh::CartesianMesh;
pub use discrete_field::{DiscreteFieldAssociation, DiscreteFieldPayload, DiscreteFieldShape};
pub use fixed_topology_geometry::{
    FixedTopologyCellGeometryAction, FixedTopologyGeometryAction, FixedTopologyGeometryState,
};
pub use mesh::{
    CellId, EntityIncidence, FacetCells1d, FacetId, GeometryMap, LineGeometryMap, LineMesh,
    MeshEntity, MeshGeometry, MeshTopology, OrientationCode, PointGeometry1d, SegmentGeometry1d,
    VertexId,
};
pub use p1_harmonic_geometry::P1HarmonicCoordinateRelation;
pub use quadrature::{QuadraturePoint, QuadratureRule, ReferenceCell, ReferenceCellFamily};
pub use reference_topology::{
    ReferenceEntity, ReferenceIncidence, ReferenceTopology, VertexPermutation,
};
pub use remesh_overlap::{
    OverlapCoordinateChart2d, RetainedFacetSide2d, RevisionCellFragment2d, RevisionFacetFragment2d,
    SimplicialRevisionOverlap2d,
};
pub use simplex_quadrature::{simplex_centroid_rule, simplex_duffy_gauss_legendre};
pub use simplicial_mesh::{MeshQualityGate, MeshQualityReport, SimplicialMesh};
