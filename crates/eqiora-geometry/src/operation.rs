//! Handle-first construction of exact Cartesian and planar primitives.
//!
//! This stack-only owner is independent of the persisted authored-CAD graph.
//! Operations retain exact construction meaning and non-persisted topology
//! lineage; only a completed named Geometry receives canonical content identity.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;

use crate::{
    CanonicalCartesianBoxGeometryV1, CanonicalGeometryV1,
    CanonicalPlanarAdjacentRectanglePartitionV1, CanonicalPlanarRectangleGeometryV2,
    CircularHoleGeometry, ConstrainedRectangleV1, EDGE_DIMENSION, FACE_DIMENSION,
    GeometryBuildReceipt, GeometryFaceHandle, GeometrySolidOperation, NamedEntitySet,
};

mod polytope;
mod topology;
use topology::{BoundarySource, OperationKind, RegionSource};
pub use topology::{
    GeometryBoundaryHandle, GeometryOperation, GeometryRegionHandle, GeometryTopologyHandle,
};

static NEXT_GRAPH_OWNER: AtomicU64 = AtomicU64::new(1);

fn invalid(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_ARTIFACT, message)
}

/// One non-persisted construction session for all admitted authored operations.
#[derive(Clone, Debug)]
pub struct GeometryGraph {
    owner: u64,
    next_operation: Arc<AtomicU64>,
}

impl Default for GeometryGraph {
    fn default() -> Self {
        Self::new()
    }
}

impl GeometryGraph {
    /// Start one independent handle-ownership session.
    #[must_use]
    pub fn new() -> Self {
        Self {
            owner: NEXT_GRAPH_OWNER.fetch_add(1, Ordering::Relaxed),
            next_operation: Arc::new(AtomicU64::new(1)),
        }
    }

    /// Construct the admitted rectangle-extrusion solid operation.
    pub fn rectangle_extrusion(
        &self,
        x_bounds_m: (f64, f64),
        y_bounds_m: (f64, f64),
        plane_z_m: f64,
        extrusion_depth_m: f64,
        requested_modeling_tolerance_m: f64,
    ) -> Result<GeometrySolidOperation, Diagnostic> {
        let sketch = ConstrainedRectangleV1::new(x_bounds_m, y_bounds_m, plane_z_m)?;
        GeometrySolidOperation::from_rectangle(
            self.owner,
            sketch,
            extrusion_depth_m,
            requested_modeling_tolerance_m,
        )
    }

    /// Decode one accepted solid-operation wire into this graph session.
    pub fn decode_solid(&self, bytes: &[u8]) -> Result<GeometrySolidOperation, Diagnostic> {
        GeometrySolidOperation::decode_for_owner(self.owner, bytes)
    }

    /// Decode and validate one face handle against an owned solid revision.
    pub fn decode_face_handle(
        &self,
        operation: &GeometrySolidOperation,
        bytes: &[u8],
    ) -> Result<GeometryFaceHandle, Diagnostic> {
        self.require_solid_owner(operation)?;
        let handle = GeometryFaceHandle::decode_for_owner(self.owner, bytes)?;
        operation.resolve_face(&handle)?;
        Ok(handle)
    }

    /// Append the admitted circular through-cut to an owned solid operation.
    pub fn circular_through_cut(
        &self,
        target: &GeometrySolidOperation,
        center_m: [f64; 2],
        radius_m: f64,
        requested_boolean_tolerance_m: f64,
    ) -> Result<GeometrySolidOperation, Diagnostic> {
        self.require_solid_owner(target)?;
        target.append_circular_through_cut(center_m, radius_m, requested_boolean_tolerance_m)
    }

    /// Admit the provider observation for one owned solid operation.
    pub fn build_solid(
        &self,
        operation: &GeometrySolidOperation,
    ) -> Result<GeometryBuildReceipt, Diagnostic> {
        self.require_solid_owner(operation)?;
        GeometryBuildReceipt::from_graph(operation)
    }

    /// Publish the admitted planar result of one owned solid operation.
    pub fn build_solid_geometry(
        &self,
        operation: &GeometrySolidOperation,
        named_topology: &BTreeMap<String, Vec<GeometryFaceHandle>>,
    ) -> Result<CanonicalGeometryV1, Diagnostic> {
        self.build_solid(operation)?
            .with_named_topology(named_topology)
    }

    /// Construct one exact axis-aligned planar rectangle.
    pub fn rectangle(
        &self,
        x_bounds_m: [f64; 2],
        y_bounds_m: [f64; 2],
    ) -> Result<GeometryOperation, Diagnostic> {
        validate_interval(x_bounds_m, "x")?;
        validate_interval(y_bounds_m, "y")?;
        Ok(self.operation(OperationKind::Rectangle {
            bounds: [
                normalize_interval(x_bounds_m),
                normalize_interval(y_bounds_m),
            ],
        }))
    }

    /// Construct one exact axis-aligned cuboid with six owned boundary faces.
    pub fn cuboid(
        &self,
        x_bounds_m: [f64; 2],
        y_bounds_m: [f64; 2],
        z_bounds_m: [f64; 2],
    ) -> Result<GeometryOperation, Diagnostic> {
        let mut bounds = [x_bounds_m, y_bounds_m, z_bounds_m];
        CanonicalCartesianBoxGeometryV1::validate_bounds(&mut bounds)?;
        Ok(self.operation(OperationKind::Cuboid { bounds }))
    }

    /// Construct one exact Cartesian interval.
    pub fn interval(&self, bounds_m: [f64; 2]) -> Result<GeometryOperation, Diagnostic> {
        validate_interval(bounds_m, "interval")?;
        Ok(self.operation(OperationKind::Interval {
            bounds: normalize_interval(bounds_m),
        }))
    }

    /// Construct one exact planar circle.
    pub fn circle(
        &self,
        center_m: [f64; 2],
        radius_m: f64,
    ) -> Result<GeometryOperation, Diagnostic> {
        if !center_m.into_iter().all(f64::is_finite) || !radius_m.is_finite() || radius_m <= 0.0 {
            return Err(invalid(
                "circle centre must be finite and radius must be finite and positive",
            ));
        }
        Ok(self.operation(OperationKind::Circle {
            center: center_m.map(normalize_zero),
            radius: radius_m,
        }))
    }

    /// Subtract one exact circle from one exact rectangle.
    pub fn subtract(
        &self,
        rectangle: &GeometryOperation,
        circle: &GeometryOperation,
    ) -> Result<GeometryOperation, Diagnostic> {
        self.require_owner(rectangle)?;
        self.require_owner(circle)?;
        let OperationKind::Rectangle { bounds } = rectangle.kind.clone() else {
            return Err(invalid("subtract target must be a rectangle operation"));
        };
        let OperationKind::Circle { center, radius } = circle.kind.clone() else {
            return Err(invalid("subtract tool must be a circle operation"));
        };
        if center[0] - radius <= bounds[0][0]
            || center[0] + radius >= bounds[0][1]
            || center[1] - radius <= bounds[1][0]
            || center[1] + radius >= bounds[1][1]
        {
            return Err(invalid(
                "subtracted circle must lie strictly inside the rectangle",
            ));
        }
        Ok(self.operation(OperationKind::Subtract {
            bounds,
            center,
            radius,
            rectangle_operation: rectangle.operation,
            circle_operation: circle.operation,
        }))
    }

    /// Join exactly two axis-aligned rectangles across one explicitly named,
    /// complete, oppositely oriented construction edge.
    pub fn partition(
        &self,
        left: &GeometryOperation,
        right: &GeometryOperation,
        interface: [GeometryBoundaryHandle; 2],
    ) -> Result<GeometryOperation, Diagnostic> {
        self.require_owner(left)?;
        self.require_owner(right)?;
        let OperationKind::Rectangle {
            bounds: left_bounds,
        } = left.kind.clone()
        else {
            return Err(invalid("partition inputs must be rectangle operations"));
        };
        let OperationKind::Rectangle {
            bounds: right_bounds,
        } = right.kind.clone()
        else {
            return Err(invalid("partition inputs must be rectangle operations"));
        };
        if interface[0].owner != self.owner || interface[1].owner != self.owner {
            return Err(invalid(
                "partition interface belongs to a foreign construction graph",
            ));
        }
        let expected = [
            GeometryBoundaryHandle {
                owner: self.owner,
                operation: left.operation,
                source: BoundarySource::Rectangle(1),
            },
            GeometryBoundaryHandle {
                owner: self.owner,
                operation: right.operation,
                source: BoundarySource::Rectangle(0),
            },
        ];
        if interface != expected {
            return Err(invalid(
                "partition interface must be the left x-upper and right x-lower handles in parent order",
            ));
        }
        if left_bounds[0][1] != right_bounds[0][0]
            || left_bounds[1] != right_bounds[1]
            || left_bounds[0][0] >= right_bounds[0][1]
        {
            return Err(invalid(
                "partition rectangles must share one complete coincident vertical edge without a gap or overlap",
            ));
        }
        Ok(self.operation(OperationKind::Partition {
            bounds: [[left_bounds[0][0], right_bounds[0][1]], left_bounds[1]],
            interface_x: left_bounds[0][1],
            left_operation: left.operation,
            right_operation: right.operation,
        }))
    }

    /// Publish one completed operation with one atomic semantic-name mapping.
    pub fn build(
        &self,
        operation: &GeometryOperation,
        named_topology: &BTreeMap<String, Vec<GeometryTopologyHandle>>,
    ) -> Result<CanonicalGeometryV1, Diagnostic> {
        self.require_owner(operation)?;
        let mut sets = Vec::with_capacity(named_topology.len());
        for (name, handles) in named_topology {
            let Some(first) = handles.first() else {
                return Err(invalid("named topology groups must not be empty"));
            };
            let dimension = first.dimension();
            let mut members = Vec::with_capacity(handles.len());
            for handle in handles {
                if handle.dimension() != dimension {
                    return Err(invalid("one topology name cannot mix dimensions"));
                }
                members.push(self.project(operation, handle)?);
            }
            sets.push(NamedEntitySet::new(name, dimension, members));
        }
        if matches!(operation.kind, OperationKind::Partition { .. }) {
            let memberships = sets
                .iter()
                .flat_map(|set| {
                    set.members()
                        .iter()
                        .map(move |&member| (set.dimension(), member))
                })
                .collect::<Vec<_>>();
            let unique = memberships.iter().copied().collect::<BTreeSet<_>>();
            let expected = (0..8)
                .map(|edge| (EDGE_DIMENSION, edge))
                .chain((0..2).map(|face| (FACE_DIMENSION, face)))
                .collect::<BTreeSet<_>>();
            if unique != expected || memberships.len() != expected.len() {
                return Err(invalid(
                    "partition naming must cover both regions and every parent-relative interface/exterior boundary exactly once",
                ));
            }
        }
        match operation.kind.clone() {
            OperationKind::Polygon(primitive) | OperationKind::ConvexPolyhedron(primitive) => {
                primitive.with_names(sets)
            }
            OperationKind::Interval { bounds } => {
                CanonicalCartesianBoxGeometryV1::new(vec![bounds], sets)
                    .map(CanonicalGeometryV1::from_cartesian_box_v1)
            }
            OperationKind::Cuboid { bounds } => {
                CanonicalCartesianBoxGeometryV1::new(bounds.to_vec(), sets)
                    .map(CanonicalGeometryV1::from_cartesian_box_v1)
            }
            OperationKind::Rectangle { bounds } => {
                CanonicalPlanarRectangleGeometryV2::new(bounds, sets)
                    .map(CanonicalGeometryV1::from_planar_rectangle_v2)
            }
            OperationKind::Subtract {
                bounds,
                center,
                radius,
                ..
            } => CircularHoleGeometry::new(bounds, center, radius, sets)
                .map(CanonicalGeometryV1::from_circular_hole_geometry),
            OperationKind::Partition {
                bounds,
                interface_x,
                ..
            } => CanonicalPlanarAdjacentRectanglePartitionV1::new(bounds, interface_x, sets)
                .map(CanonicalGeometryV1::from_planar_adjacent_rectangle_partition_v1),
            OperationKind::Circle { .. } => Err(invalid(
                "a circle operation alone does not define an admitted planar region",
            )),
        }
    }

    fn operation(&self, kind: OperationKind) -> GeometryOperation {
        GeometryOperation {
            owner: self.owner,
            operation: self.next_operation.fetch_add(1, Ordering::Relaxed),
            kind,
        }
    }

    fn require_owner(&self, operation: &GeometryOperation) -> Result<(), Diagnostic> {
        if operation.owner != self.owner {
            return Err(invalid("operation belongs to a foreign construction graph"));
        }
        Ok(())
    }

    fn require_solid_owner(&self, operation: &GeometrySolidOperation) -> Result<(), Diagnostic> {
        if operation.authoring_owner() != self.owner {
            return Err(invalid(
                "solid operation belongs to a foreign GeometryGraph",
            ));
        }
        Ok(())
    }

    fn project(
        &self,
        target: &GeometryOperation,
        handle: &GeometryTopologyHandle,
    ) -> Result<usize, Diagnostic> {
        if handle.owner() != self.owner {
            return Err(invalid(
                "topology handle belongs to a foreign construction graph",
            ));
        }
        match (&target.kind, handle) {
            (OperationKind::Polygon(primitive) | OperationKind::ConvexPolyhedron(primitive), _) => {
                polytope::project(target, primitive, handle)
            }

            (OperationKind::Interval { .. }, GeometryTopologyHandle::Region(handle))
                if handle.operation == target.operation
                    && handle.source == RegionSource::Interval =>
            {
                Ok(0)
            }
            (OperationKind::Interval { .. }, GeometryTopologyHandle::Boundary(handle))
                if handle.operation == target.operation =>
            {
                match handle.source {
                    BoundarySource::Interval(member) => Ok(member),
                    _ => Err(invalid(
                        "boundary handle is absent from the interval result",
                    )),
                }
            }
            (OperationKind::Cuboid { .. }, GeometryTopologyHandle::Region(handle))
                if handle.operation == target.operation
                    && handle.source == RegionSource::Cuboid =>
            {
                Ok(0)
            }
            (OperationKind::Cuboid { .. }, GeometryTopologyHandle::Boundary(handle))
                if handle.operation == target.operation =>
            {
                match handle.source {
                    BoundarySource::Cuboid(member) => Ok(member),
                    _ => Err(invalid("boundary handle is absent from the cuboid result")),
                }
            }
            (OperationKind::Rectangle { .. }, GeometryTopologyHandle::Region(handle))
                if handle.operation == target.operation
                    && handle.source == RegionSource::Rectangle =>
            {
                Ok(0)
            }
            (OperationKind::Rectangle { .. }, GeometryTopologyHandle::Boundary(handle))
                if handle.operation == target.operation =>
            {
                match handle.source {
                    BoundarySource::Rectangle(member) => Ok(member),
                    _ => Err(invalid(
                        "boundary handle is absent from the rectangle result",
                    )),
                }
            }
            (OperationKind::Subtract { .. }, GeometryTopologyHandle::Region(handle))
                if handle.operation == target.operation
                    && handle.source == RegionSource::Subtract =>
            {
                Ok(0)
            }
            (
                OperationKind::Subtract {
                    rectangle_operation,
                    circle_operation,
                    ..
                },
                GeometryTopologyHandle::Boundary(handle),
            ) => match handle.source {
                BoundarySource::Subtract(member) if handle.operation == target.operation => {
                    Ok(member)
                }
                BoundarySource::Rectangle(member) if handle.operation == *rectangle_operation => {
                    Ok(member)
                }
                BoundarySource::Circle if handle.operation == *circle_operation => Ok(4),
                _ => Err(invalid(
                    "boundary handle is deleted, stale, or absent from the subtract result",
                )),
            },
            (OperationKind::Subtract { .. }, GeometryTopologyHandle::Region(_)) => Err(invalid(
                "predecessor region handle was deleted by subtraction",
            )),
            (
                OperationKind::Partition {
                    left_operation,
                    right_operation,
                    ..
                },
                GeometryTopologyHandle::Region(handle),
            ) => {
                if handle.operation == *left_operation && handle.source == RegionSource::Rectangle {
                    Ok(0)
                } else if handle.operation == *right_operation
                    && handle.source == RegionSource::Rectangle
                {
                    Ok(1)
                } else {
                    Err(invalid(
                        "region handle is stale or absent from the partition result",
                    ))
                }
            }
            (
                OperationKind::Partition {
                    left_operation,
                    right_operation,
                    ..
                },
                GeometryTopologyHandle::Boundary(handle),
            ) => match (handle.operation, handle.source) {
                (operation, BoundarySource::Rectangle(member)) if operation == *left_operation => {
                    Ok([3, 1, 0, 2][member])
                }
                (operation, BoundarySource::Rectangle(member)) if operation == *right_operation => {
                    Ok([7, 5, 4, 6][member])
                }
                _ => Err(invalid(
                    "boundary handle is stale or absent from the partition result",
                )),
            },
            _ => Err(invalid(
                "topology handle is stale or absent from this result",
            )),
        }
    }
}

fn validate_interval(interval: [f64; 2], axis: &str) -> Result<(), Diagnostic> {
    if !interval[0].is_finite() || !interval[1].is_finite() || interval[0] >= interval[1] {
        return Err(invalid(format!(
            "{axis} bounds must be a finite strict interval"
        )));
    }
    Ok(())
}

fn normalize_interval(interval: [f64; 2]) -> [f64; 2] {
    interval.map(normalize_zero)
}

fn normalize_zero(value: f64) -> f64 {
    if value == 0.0 { 0.0 } else { value }
}
