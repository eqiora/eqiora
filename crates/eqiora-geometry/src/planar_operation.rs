//! Handle-first exact planar construction for the post-#530 geometry path.
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
    ) -> Result<PlanarOperation, Diagnostic> {
        validate_interval(x_bounds_m, "x")?;
        validate_interval(y_bounds_m, "y")?;
        Ok(self.operation(OperationKind::Rectangle {
            bounds: [
                normalize_interval(x_bounds_m),
                normalize_interval(y_bounds_m),
            ],
        }))
    }

    /// Construct one exact Cartesian interval.
    pub fn interval(&self, bounds_m: [f64; 2]) -> Result<PlanarOperation, Diagnostic> {
        validate_interval(bounds_m, "interval")?;
        Ok(self.operation(OperationKind::Interval {
            bounds: normalize_interval(bounds_m),
        }))
    }

    /// Construct one exact planar circle.
    pub fn circle(&self, center_m: [f64; 2], radius_m: f64) -> Result<PlanarOperation, Diagnostic> {
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
        rectangle: &PlanarOperation,
        circle: &PlanarOperation,
    ) -> Result<PlanarOperation, Diagnostic> {
        self.require_owner(rectangle)?;
        self.require_owner(circle)?;
        let OperationKind::Rectangle { bounds } = rectangle.kind else {
            return Err(invalid("subtract target must be a rectangle operation"));
        };
        let OperationKind::Circle { center, radius } = circle.kind else {
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
        left: &PlanarOperation,
        right: &PlanarOperation,
        interface: [PlanarBoundaryHandle; 2],
    ) -> Result<PlanarOperation, Diagnostic> {
        self.require_owner(left)?;
        self.require_owner(right)?;
        let OperationKind::Rectangle {
            bounds: left_bounds,
        } = left.kind
        else {
            return Err(invalid("partition inputs must be rectangle operations"));
        };
        let OperationKind::Rectangle {
            bounds: right_bounds,
        } = right.kind
        else {
            return Err(invalid("partition inputs must be rectangle operations"));
        };
        if interface[0].owner != self.owner || interface[1].owner != self.owner {
            return Err(invalid(
                "partition interface belongs to a foreign construction graph",
            ));
        }
        let expected = [
            PlanarBoundaryHandle {
                owner: self.owner,
                operation: left.operation,
                source: BoundarySource::Rectangle(1),
            },
            PlanarBoundaryHandle {
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
        operation: &PlanarOperation,
        named_topology: &BTreeMap<String, Vec<PlanarTopologyHandle>>,
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
        match operation.kind {
            OperationKind::Interval { bounds } => {
                CanonicalCartesianBoxGeometryV1::new(vec![bounds], sets)
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

    fn operation(&self, kind: OperationKind) -> PlanarOperation {
        PlanarOperation {
            owner: self.owner,
            operation: self.next_operation.fetch_add(1, Ordering::Relaxed),
            kind,
        }
    }

    fn require_owner(&self, operation: &PlanarOperation) -> Result<(), Diagnostic> {
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
        target: &PlanarOperation,
        handle: &PlanarTopologyHandle,
    ) -> Result<usize, Diagnostic> {
        if handle.owner() != self.owner {
            return Err(invalid(
                "topology handle belongs to a foreign construction graph",
            ));
        }
        match (&target.kind, handle) {
            (OperationKind::Interval { .. }, PlanarTopologyHandle::Region(handle))
                if handle.operation == target.operation
                    && handle.source == RegionSource::Interval =>
            {
                Ok(0)
            }
            (OperationKind::Interval { .. }, PlanarTopologyHandle::Boundary(handle))
                if handle.operation == target.operation =>
            {
                match handle.source {
                    BoundarySource::Interval(member) => Ok(member),
                    _ => Err(invalid(
                        "boundary handle is absent from the interval result",
                    )),
                }
            }
            (OperationKind::Rectangle { .. }, PlanarTopologyHandle::Region(handle))
                if handle.operation == target.operation
                    && handle.source == RegionSource::Rectangle =>
            {
                Ok(0)
            }
            (OperationKind::Rectangle { .. }, PlanarTopologyHandle::Boundary(handle))
                if handle.operation == target.operation =>
            {
                match handle.source {
                    BoundarySource::Rectangle(member) => Ok(member),
                    _ => Err(invalid(
                        "boundary handle is absent from the rectangle result",
                    )),
                }
            }
            (OperationKind::Subtract { .. }, PlanarTopologyHandle::Region(handle))
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
                PlanarTopologyHandle::Boundary(handle),
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
            (OperationKind::Subtract { .. }, PlanarTopologyHandle::Region(_)) => Err(invalid(
                "predecessor region handle was deleted by subtraction",
            )),
            (
                OperationKind::Partition {
                    left_operation,
                    right_operation,
                    ..
                },
                PlanarTopologyHandle::Region(handle),
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
                PlanarTopologyHandle::Boundary(handle),
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

/// One immutable result of a primitive or Boolean operation.
#[derive(Clone, Debug, PartialEq)]
pub struct PlanarOperation {
    owner: u64,
    operation: u64,
    kind: OperationKind,
}

impl PlanarOperation {
    /// Direct typed handle for this operation's interior region.
    #[must_use]
    pub const fn region(&self) -> PlanarRegionHandle {
        let source = match self.kind {
            OperationKind::Interval { .. } => RegionSource::Interval,
            OperationKind::Rectangle { .. } => RegionSource::Rectangle,
            OperationKind::Circle { .. } => RegionSource::Circle,
            OperationKind::Subtract { .. } => RegionSource::Subtract,
            OperationKind::Partition { .. } => RegionSource::Partition,
        };
        PlanarRegionHandle {
            owner: self.owner,
            operation: self.operation,
            source,
        }
    }

    /// Direct typed boundary handles in construction order.
    ///
    /// Rectangle and subtract order is x-lower, x-upper, y-lower, y-upper;
    /// subtract then appends the created cut boundary. A circle has one member.
    #[must_use]
    pub fn boundaries(&self) -> Vec<PlanarBoundaryHandle> {
        let sources: &[BoundarySource] = match self.kind {
            OperationKind::Interval { .. } => {
                &[BoundarySource::Interval(0), BoundarySource::Interval(1)]
            }
            OperationKind::Rectangle { .. } => &[
                BoundarySource::Rectangle(0),
                BoundarySource::Rectangle(1),
                BoundarySource::Rectangle(2),
                BoundarySource::Rectangle(3),
            ],
            OperationKind::Circle { .. } => &[BoundarySource::Circle],
            OperationKind::Subtract { .. } => &[
                BoundarySource::Subtract(0),
                BoundarySource::Subtract(1),
                BoundarySource::Subtract(2),
                BoundarySource::Subtract(3),
                BoundarySource::Subtract(4),
            ],
            OperationKind::Partition { .. } => &[],
        };
        sources
            .iter()
            .copied()
            .map(|source| PlanarBoundaryHandle {
                owner: self.owner,
                operation: self.operation,
                source,
            })
            .collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
enum OperationKind {
    Interval {
        bounds: [f64; 2],
    },
    Rectangle {
        bounds: [[f64; 2]; 2],
    },
    Circle {
        center: [f64; 2],
        radius: f64,
    },
    Subtract {
        bounds: [[f64; 2]; 2],
        center: [f64; 2],
        radius: f64,
        rectangle_operation: u64,
        circle_operation: u64,
    },
    Partition {
        bounds: [[f64; 2]; 2],
        interface_x: f64,
        left_operation: u64,
        right_operation: u64,
    },
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RegionSource {
    Interval,
    Rectangle,
    Circle,
    Subtract,
    Partition,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum BoundarySource {
    Interval(usize),
    Rectangle(usize),
    Circle,
    Subtract(usize),
}

/// Direct construction-owned two-dimensional topology handle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlanarRegionHandle {
    owner: u64,
    operation: u64,
    source: RegionSource,
}

impl PlanarRegionHandle {
    /// Topological dimension carried by this construction handle.
    #[must_use]
    pub const fn dimension(self) -> usize {
        match self.source {
            RegionSource::Interval => 1,
            RegionSource::Rectangle
            | RegionSource::Circle
            | RegionSource::Subtract
            | RegionSource::Partition => 2,
        }
    }
}

/// Direct construction-owned one-dimensional topology handle.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PlanarBoundaryHandle {
    owner: u64,
    operation: u64,
    source: BoundarySource,
}

impl PlanarBoundaryHandle {
    /// Topological dimension carried by this construction handle.
    #[must_use]
    pub const fn dimension(self) -> usize {
        match self.source {
            BoundarySource::Interval(_) => 0,
            BoundarySource::Rectangle(_) | BoundarySource::Circle | BoundarySource::Subtract(_) => {
                1
            }
        }
    }
}

/// Dimension-carrying input to atomic semantic naming.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PlanarTopologyHandle {
    /// One exact region.
    Region(PlanarRegionHandle),
    /// One exact boundary.
    Boundary(PlanarBoundaryHandle),
}

impl PlanarTopologyHandle {
    /// Topological dimension of this handle.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        match self {
            Self::Region(handle) => handle.dimension(),
            Self::Boundary(handle) => handle.dimension(),
        }
    }

    const fn owner(&self) -> u64 {
        match self {
            Self::Region(handle) => handle.owner,
            Self::Boundary(handle) => handle.owner,
        }
    }
}

impl From<PlanarRegionHandle> for PlanarTopologyHandle {
    fn from(handle: PlanarRegionHandle) -> Self {
        Self::Region(handle)
    }
}

impl From<PlanarBoundaryHandle> for PlanarTopologyHandle {
    fn from(handle: PlanarBoundaryHandle) -> Self {
        Self::Boundary(handle)
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
