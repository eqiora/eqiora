//! Dimension-carrying construction operations and their owned topology handles.

/// One immutable result of a primitive or Boolean operation.
#[derive(Clone, Debug, PartialEq)]
pub struct GeometryOperation {
    pub(super) owner: u64,
    pub(super) operation: u64,
    pub(super) kind: OperationKind,
}

impl GeometryOperation {
    /// Direct typed handle for this operation's interior region.
    #[must_use]
    pub const fn region(&self) -> GeometryRegionHandle {
        let source = match self.kind {
            OperationKind::Interval { .. } => RegionSource::Interval,
            OperationKind::Cuboid { .. } => RegionSource::Cuboid,
            OperationKind::Rectangle { .. } => RegionSource::Rectangle,
            OperationKind::Circle { .. } => RegionSource::Circle,
            OperationKind::Subtract { .. } => RegionSource::Subtract,
            OperationKind::Partition { .. } => RegionSource::Partition,
        };
        GeometryRegionHandle {
            owner: self.owner,
            operation: self.operation,
            source,
        }
    }

    /// Direct typed boundary handles in construction order.
    ///
    /// Rectangle and subtract order is x-lower, x-upper, y-lower, y-upper;
    /// A cuboid appends z-lower and z-upper to that axis-side order;
    /// subtract then appends the created cut boundary. A circle has one member.
    #[must_use]
    pub fn boundaries(&self) -> Vec<GeometryBoundaryHandle> {
        let sources: &[BoundarySource] = match self.kind {
            OperationKind::Interval { .. } => {
                &[BoundarySource::Interval(0), BoundarySource::Interval(1)]
            }
            OperationKind::Cuboid { .. } => &[
                BoundarySource::Cuboid(0),
                BoundarySource::Cuboid(1),
                BoundarySource::Cuboid(2),
                BoundarySource::Cuboid(3),
                BoundarySource::Cuboid(4),
                BoundarySource::Cuboid(5),
            ],
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
            .map(|source| GeometryBoundaryHandle {
                owner: self.owner,
                operation: self.operation,
                source,
            })
            .collect()
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub(super) enum OperationKind {
    Cuboid {
        bounds: [[f64; 2]; 3],
    },
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
pub(super) enum RegionSource {
    Cuboid,
    Interval,
    Rectangle,
    Circle,
    Subtract,
    Partition,
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(super) enum BoundarySource {
    Cuboid(usize),
    Interval(usize),
    Rectangle(usize),
    Circle,
    Subtract(usize),
}

/// Direct construction-owned interior handle carrying its dimension.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GeometryRegionHandle {
    pub(super) owner: u64,
    pub(super) operation: u64,
    pub(super) source: RegionSource,
}

impl GeometryRegionHandle {
    /// Topological dimension carried by this construction handle.
    #[must_use]
    pub const fn dimension(self) -> usize {
        match self.source {
            RegionSource::Interval => 1,
            RegionSource::Cuboid => 3,
            RegionSource::Rectangle
            | RegionSource::Circle
            | RegionSource::Subtract
            | RegionSource::Partition => 2,
        }
    }
}

/// Direct construction-owned boundary handle carrying its dimension.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct GeometryBoundaryHandle {
    pub(super) owner: u64,
    pub(super) operation: u64,
    pub(super) source: BoundarySource,
}

impl GeometryBoundaryHandle {
    /// Topological dimension carried by this construction handle.
    #[must_use]
    pub const fn dimension(self) -> usize {
        match self.source {
            BoundarySource::Interval(_) => 0,
            BoundarySource::Cuboid(_) => 2,
            BoundarySource::Rectangle(_) | BoundarySource::Circle | BoundarySource::Subtract(_) => {
                1
            }
        }
    }
}

/// Dimension-carrying input to atomic semantic naming.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GeometryTopologyHandle {
    /// One exact region.
    Region(GeometryRegionHandle),
    /// One exact boundary.
    Boundary(GeometryBoundaryHandle),
}

impl GeometryTopologyHandle {
    /// Topological dimension of this handle.
    #[must_use]
    pub const fn dimension(&self) -> usize {
        match self {
            Self::Region(handle) => handle.dimension(),
            Self::Boundary(handle) => handle.dimension(),
        }
    }

    pub(super) const fn owner(&self) -> u64 {
        match self {
            Self::Region(handle) => handle.owner,
            Self::Boundary(handle) => handle.owner,
        }
    }
}

impl From<GeometryRegionHandle> for GeometryTopologyHandle {
    fn from(handle: GeometryRegionHandle) -> Self {
        Self::Region(handle)
    }
}

impl From<GeometryBoundaryHandle> for GeometryTopologyHandle {
    fn from(handle: GeometryBoundaryHandle) -> Self {
        Self::Boundary(handle)
    }
}
