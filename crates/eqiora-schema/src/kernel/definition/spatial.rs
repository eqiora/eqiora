//! What a spatial Domain is: its extent, its boundaries, and how it names one.
//!
//! Continuous supports are Model meaning: dimensioned factors and ordered products,
//! physical boxes, or exact authored Geometry selections. Their meaning is settled
//! here rather than by whichever mesh happens to realize them.

use eqiora_core::diagnostic::codes;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, DimExponents, DynQuantity, Id};

use super::*;

/// One finite increasing coordinate interval with one exact physical dimension.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AxisBounds {
    lower: DynQuantity,
    upper: DynQuantity,
}

impl AxisBounds {
    /// Construct finite, increasing bounds with equal physical dimensions.
    ///
    /// # Errors
    /// Returns `EQ0302` when dimensions differ, either value is non-finite, or
    /// does not form an increasing interval.
    pub fn new(lower: DynQuantity, upper: DynQuantity) -> Result<Self, Diagnostic> {
        if lower.dim() != upper.dim() {
            return Err(Diagnostic::error(
                codes::INVALID_KERNEL_DEFINITION,
                "coordinate interval bounds must have equal physical dimensions",
            ));
        }
        if !lower.value().is_finite()
            || !upper.value().is_finite()
            || upper.value() <= lower.value()
        {
            return Err(Diagnostic::error(
                codes::INVALID_KERNEL_DEFINITION,
                "coordinate interval bounds must be finite and strictly increasing",
            ));
        }
        Ok(Self { lower, upper })
    }

    /// Lower coordinate in coherent SI units.
    #[must_use]
    pub const fn lower(self) -> DynQuantity {
        self.lower
    }

    /// Upper coordinate in coherent SI units.
    #[must_use]
    pub const fn upper(self) -> DynQuantity {
        self.upper
    }
}

/// One unevaluated Cartesian coordinate endpoint.
///
/// A fixed endpoint carries a coherent-SI length. A Parameter endpoint names
/// one exact root Model Parameter whose revision-local value is resolved by
/// the Semantic Kernel.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CartesianCoordinateSource {
    /// One fixed finite coherent-SI length.
    Fixed(DynQuantity),
    /// One exact root Model Parameter identity.
    Parameter(Id<kinds::Parameter>),
}

impl CartesianCoordinateSource {
    /// Construct one locally valid fixed coordinate.
    ///
    /// Increasing-axis validation belongs to whole-Model resolution because
    /// the other endpoint may be Parameter-driven.
    ///
    /// # Errors
    /// Returns `EQ0302` when `value` is not a finite length.
    pub fn fixed(value: DynQuantity) -> Result<Self, Diagnostic> {
        let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension");
        if value.dim() != length || !value.value().is_finite() {
            return Err(Diagnostic::error(
                codes::INVALID_KERNEL_DEFINITION,
                "fixed Cartesian coordinate must be a finite physical length",
            ));
        }
        Ok(Self::Fixed(value))
    }

    /// Construct one Parameter-backed coordinate.
    #[must_use]
    pub const fn parameter(parameter: Id<kinds::Parameter>) -> Self {
        Self::Parameter(parameter)
    }

    fn validate(self) -> Result<(), Diagnostic> {
        match self {
            Self::Fixed(value) => Self::fixed(value).map(|_| ()),
            Self::Parameter(_) => Ok(()),
        }
    }
}

/// Unevaluated lower and upper sources for one Cartesian axis.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CartesianAxisDefinition {
    lower: CartesianCoordinateSource,
    upper: CartesianCoordinateSource,
}

impl CartesianAxisDefinition {
    /// Construct one axis recipe from two locally valid coordinate sources.
    #[must_use]
    pub const fn new(lower: CartesianCoordinateSource, upper: CartesianCoordinateSource) -> Self {
        Self { lower, upper }
    }

    /// Convert a fixed interval into its source recipe; Domain admission checks length units.
    #[must_use]
    pub const fn fixed(bounds: AxisBounds) -> Self {
        Self {
            lower: CartesianCoordinateSource::Fixed(bounds.lower()),
            upper: CartesianCoordinateSource::Fixed(bounds.upper()),
        }
    }

    /// Lower endpoint source.
    #[must_use]
    pub const fn lower(self) -> CartesianCoordinateSource {
        self.lower
    }

    /// Upper endpoint source.
    #[must_use]
    pub const fn upper(self) -> CartesianCoordinateSource {
        self.upper
    }
}

/// Outward side of one Cartesian coordinate axis.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BoundarySide {
    /// Side at the lower coordinate bound.
    Lower,
    /// Side at the upper coordinate bound.
    Upper,
}

/// Content identity of one authored geometry, as raw digest bytes.
///
/// The Kernel names a geometry rather than describing one. Carrying the digest
/// instead of the shape keeps a Model's meaning exact while leaving the
/// geometry artifact outside the Kernel, and it is why two Realizations that
/// differ only in mesh are still one Model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct GeometryDigest([u8; 32]);

impl GeometryDigest {
    /// One geometry identity from its exact digest bytes.
    #[must_use]
    pub const fn new(bytes: [u8; 32]) -> Self {
        Self(bytes)
    }

    /// Exact digest bytes.
    #[must_use]
    pub const fn bytes(self) -> [u8; 32] {
        self.0
    }
}

/// Canonical continuous-domain shape, independent of any mesh.
#[derive(Debug, Clone, PartialEq)]
#[non_exhaustive]
pub enum DomainKind {
    /// One abstract bounded coordinate factor with increasing-coordinate measure.
    CoordinateInterval { bounds: AxisBounds },
    /// Ordered product of exact coordinate-factor supports; no ambient space is implied.
    CoordinateProduct { factors: Vec<Id<kinds::Domain>> },
    /// Identity-only domain retained for non-spatial and schema-defined uses.
    Abstract,
    /// Runtime-dimensional Cartesian box in physical space.
    CartesianBox {
        /// Coordinate recipes in physical axis order.
        coordinates: Vec<CartesianAxisDefinition>,
    },
    /// One oriented side of a parent Cartesian box. The parent is supplied by
    /// exactly one `BoundaryOf` graph edge.
    CartesianBoundary { axis: usize, side: BoundarySide },
    /// One region of an authored geometry, named by content digest and by an
    /// entity set within it. The shape is the geometry's; this Domain selects
    /// part of it and gives that part meaning.
    GeometryRegion {
        geometry: GeometryDigest,
        entity_set: String,
    },
    /// One boundary of a parent geometry region, named by an entity set of the
    /// same geometry. The parent is supplied by exactly one `BoundaryOf` graph
    /// edge, so the digest is never repeated here: a boundary free to name a
    /// different geometry from its parent would be a boundary of nothing.
    GeometryBoundary { entity_set: String },
    /// One nominal scalar conserving domain. The Domain ID is part of the
    /// physical type; dimensions alone never make two domains compatible.
    ScalarPhysical {
        across_type: eqiora_core::ValueType,
        through_type: eqiora_core::ValueType,
    },
    /// One nominal field-valued boundary connector. The Domain ID plus closed
    /// trace/flux role is the exact quantity identity.
    BoundaryPhysical {
        connector: BoundaryPhysicalConnector,
    },
}

/// Domain definition. Continuous geometry is model meaning; meshes and
/// geometry maps remain realization concerns.
#[derive(Debug, Clone, PartialEq)]
pub struct DomainDef {
    id: Id<kinds::Domain>,
    kind: DomainKind,
}

impl DomainDef {
    /// Bind an abstract coordinate interval without fabricating physical Geometry.
    #[must_use]
    pub const fn coordinate_interval(id: Id<kinds::Domain>, bounds: AxisBounds) -> Self {
        Self {
            id,
            kind: DomainKind::CoordinateInterval { bounds },
        }
    }

    /// Compose an ordered nonempty product of exact coordinate supports.
    /// Whole-Model validation resolves factors, rejects cycles and repeated factors,
    /// and derives units and intrinsic dimension independently of component frames.
    pub fn coordinate_product(
        id: Id<kinds::Domain>,
        factors: Vec<Id<kinds::Domain>>,
    ) -> Result<Self, Diagnostic> {
        if factors.is_empty() {
            return Err(Diagnostic::error(
                codes::INVALID_KERNEL_DEFINITION,
                "a coordinate product requires at least one factor",
            )
            .with_graph_path(kernel_path(id.erase())));
        }
        Ok(Self {
            id,
            kind: DomainKind::CoordinateProduct { factors },
        })
    }

    /// Construct an abstract domain for schema-defined or non-spatial use.
    #[must_use]
    pub const fn new(id: Id<kinds::Domain>) -> Self {
        Self {
            id,
            kind: DomainKind::Abstract,
        }
    }

    /// Construct a non-empty Cartesian box of arbitrary runtime dimension.
    ///
    /// # Errors
    /// Returns `EQ0302` for an empty axis set or a locally invalid fixed
    /// coordinate.
    pub fn cartesian_box(
        id: Id<kinds::Domain>,
        bounds: Vec<AxisBounds>,
    ) -> Result<Self, Diagnostic> {
        Self::cartesian_box_from_sources(
            id,
            bounds
                .into_iter()
                .map(CartesianAxisDefinition::fixed)
                .collect(),
        )
    }

    /// Construct a non-empty Cartesian box from unevaluated coordinate
    /// sources.
    ///
    /// Whole-Model validation resolves Parameter values and proves that every
    /// resulting interval is finite and strictly increasing.
    ///
    /// # Errors
    /// Returns `EQ0302` for an empty axis set.
    pub fn cartesian_box_from_sources(
        id: Id<kinds::Domain>,
        coordinates: Vec<CartesianAxisDefinition>,
    ) -> Result<Self, Diagnostic> {
        if coordinates.is_empty() {
            return Err(Diagnostic::error(
                codes::INVALID_KERNEL_DEFINITION,
                "Cartesian Domain requires at least one coordinate axis",
            )
            .with_graph_path(kernel_path(id.erase())));
        }
        for axis in &coordinates {
            axis.lower()
                .validate()
                .map_err(|diagnostic| diagnostic.with_graph_path(kernel_path(id.erase())))?;
            axis.upper()
                .validate()
                .map_err(|diagnostic| diagnostic.with_graph_path(kernel_path(id.erase())))?;
        }
        Ok(Self {
            id,
            kind: DomainKind::CartesianBox { coordinates },
        })
    }

    /// Construct one oriented boundary selector. Whole-model validation
    /// checks the axis against its unique `BoundaryOf` parent.
    #[must_use]
    pub const fn cartesian_boundary(
        id: Id<kinds::Domain>,
        axis: usize,
        side: BoundarySide,
    ) -> Self {
        Self {
            id,
            kind: DomainKind::CartesianBoundary { axis, side },
        }
    }

    /// Construct one region selected from an authored geometry.
    ///
    /// # Errors
    /// Returns `EQ0302` for an unnamed entity set.
    pub fn geometry_region(
        id: Id<kinds::Domain>,
        geometry: GeometryDigest,
        entity_set: impl Into<String>,
    ) -> Result<Self, Diagnostic> {
        Ok(Self {
            id,
            kind: DomainKind::GeometryRegion {
                geometry,
                entity_set: named_entity_set(id, entity_set)?,
            },
        })
    }

    /// Construct one boundary selected from its parent region's geometry.
    ///
    /// Whole-model validation checks that exactly one `BoundaryOf` parent is a
    /// geometry region. Artifact admission separately checks that both names
    /// select compatible entity sets in the referenced geometry.
    ///
    /// # Errors
    /// Returns `EQ0302` for an unnamed entity set.
    pub fn geometry_boundary(
        id: Id<kinds::Domain>,
        entity_set: impl Into<String>,
    ) -> Result<Self, Diagnostic> {
        Ok(Self {
            id,
            kind: DomainKind::GeometryBoundary {
                entity_set: named_entity_set(id, entity_set)?,
            },
        })
    }

    /// Construct a nominal scalar conserving domain.
    ///
    /// # Errors
    /// Rejects shaped across or through quantities.
    pub fn scalar_physical(
        id: Id<kinds::Domain>,
        across_type: eqiora_core::ValueType,
        through_type: eqiora_core::ValueType,
    ) -> Result<Self, Diagnostic> {
        if !across_type.shape().is_scalar() || !through_type.shape().is_scalar() {
            return Err(Diagnostic::error(
                codes::INVALID_KERNEL_DEFINITION,
                "scalar physical quantities require scalar mathematical types",
            ));
        }
        Ok(Self {
            id,
            kind: DomainKind::ScalarPhysical {
                across_type,
                through_type,
            },
        })
    }

    /// Construct a nominal field-valued boundary connector Domain.
    #[must_use]
    pub const fn boundary_physical(
        id: Id<kinds::Domain>,
        connector: BoundaryPhysicalConnector,
    ) -> Self {
        Self {
            id,
            kind: DomainKind::BoundaryPhysical { connector },
        }
    }

    /// Typed node ID.
    #[must_use]
    pub const fn id(&self) -> Id<kinds::Domain> {
        self.id
    }

    /// Canonical domain kind.
    #[must_use]
    pub const fn kind(&self) -> &DomainKind {
        &self.kind
    }
}
