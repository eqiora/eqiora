//! Compiler-owned Domain declarations shared by source and hierarchy lowering.

use eqiora_lang::{DomainSyntax, PortSyntax};
use eqiora_schema::kernel::{BoundaryPhysicalConnector, GeometryDigest};

#[derive(Debug, Clone)]
pub(crate) enum LoweringDomainContract {
    Source(DomainSyntax),
    CoordinateInterval(eqiora_schema::kernel::AxisBounds),
    ExternalGeometryRegion {
        geometry: GeometryDigest,
        entity_set: String,
        dimensions: usize,
    },
    ExternalGeometryBoundary {
        entity_set: String,
        parent: String,
    },
    BoundaryPhysical(BoundaryPhysicalConnector),
}

#[derive(Debug, Clone)]
pub(crate) enum LoweringPortContract {
    Source(PortSyntax),
    BoundaryPhysical { connector: String, boundary: String },
}
