//! **eqiora-geometry** — geometry identity, geometry-to-mesh correspondence,
//! and kernel-neutral CAD contracts at the L2 boundary.
//!
//! The Semantic Kernel says *which* Domains exist. A geometry revision says
//! which exact geometric entities realize those Domains, and a mesh revision
//! supplies numerical entities. This crate owns geometry revision identity,
//! the pure typed correspondence between those identities, and bounded CAD
//! design, observation, and adapter contracts that contain no kernel objects
//! or entity-enumeration indices.
//!
//! It deliberately does not own a CAD kernel, STEP parsing, concrete B-rep or
//! modeling operations, artifact encoding, physics, transfer operators, or
//! generic topology naming. The bounded authored build atomically projects
//! construction lineage into complete named geometry without exposing private
//! result topology. Concrete CAD responsibilities remain isolated in adapters
//! such as `eqiora-cad-truck`.
//! In particular, [`ParentOutward`] is
//! geometric meaning derived relative to a parent body; it is not a normal
//! sign and is unrelated to
//! [`eqiora_meshing::OrientationCode`], which encodes a local permutation.

mod association;
mod cad;
mod cad_authored_build;
mod cad_authored_cut;
mod cad_authored_face_mesh;
mod cad_authored_graph;
#[allow(
    dead_code,
    reason = "frozen private evidence retains observations outside the public atomic owner path"
)]
mod cad_authored_result_topology;
mod cad_authored_selection;
mod cad_authored_swept_mesh;
mod canonical;
mod cartesian_box_v1;
mod circular_hole;
mod convex_polyhedra;
mod correspondence;
mod identity;
mod planar_adjacent_rectangle_partition_v1;
mod planar_interface;
mod planar_operation;
mod planar_rectangle_v2;
mod region;

pub use association::{BodyAssociationCandidate, RetainedGeometryAssociation, RetentionRejection};
pub use cad::{
    AxisAlignedBox3, CadAdapterIdentityV1, CadBoxDesignV1, CadBoxObservationV1,
    CadBoxRealizationV1, CadKernelAdapter, CadRepairDispositionV1, ConstrainedRectangleV1,
    StepLengthUnitV1, StepSourceDigest,
};
pub use cad_authored_build::GeometryBuildReceipt;
pub use cad_authored_face_mesh::CadAuthoredFaceMesh;
pub use cad_authored_graph::GeometrySolidOperation;
pub use cad_authored_selection::GeometryFaceHandle;
pub use cad_authored_swept_mesh::CadAuthoredSweptMesh;
pub use canonical::{CanonicalGeometryLimits, CanonicalGeometryV1};
pub(crate) use cartesian_box_v1::CanonicalCartesianBoxGeometryV1;
pub(crate) use circular_hole::CircularHoleGeometry;
pub use correspondence::{
    CartesianBodyAssignment, CartesianBoundaryAssignment, GeometryMeshCorrespondence,
};
pub use identity::{
    GeometryEntity, GeometryRevisionReference, GeometryRevisionTopology, ParentOutward,
};
pub(crate) use planar_adjacent_rectangle_partition_v1::CanonicalPlanarAdjacentRectanglePartitionV1;
pub use planar_operation::{
    GeometryGraph, PlanarBoundaryHandle, PlanarOperation, PlanarRegionHandle, PlanarTopologyHandle,
};
pub(crate) use planar_rectangle_v2::CanonicalPlanarRectangleGeometryV2;
pub use region::{
    EDGE_DIMENSION, FACE_DIMENSION, NamedEntitySet, PlanarFace, PlanarRegion, VERTEX_DIMENSION,
};
