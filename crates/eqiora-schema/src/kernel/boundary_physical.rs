use eqiora_core::{ScalarDomain, ValueShape, ValueType};

use super::{AxisBounds, BoundarySide};
use eqiora_core::ValueFrame;

/// Closed dual pairing for one field-valued physical connector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum BoundaryPairing {
    /// Pointwise componentwise Euclidean pairing on the boundary.
    EuclideanBoundaryDuality,
}

/// Closed, mesh-independent contract of one field-valued physical connector.
///
/// The exact nominal identity is supplied by the owning Domain node.
/// Euclidean boundary duality pairs real quantities with equal component
/// spaces. Support and outward orientation belong to Ports,
/// not to this reusable nominal contract.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BoundaryPhysicalConnector {
    trace_type: ValueType,
    flux_type: ValueType,
    pairing: BoundaryPairing,
}

impl BoundaryPhysicalConnector {
    /// Construct one exact trace/flux dual pair.
    ///
    /// # Errors
    /// Rejects non-real Euclidean quantities or unequal component roles, shape or frame.
    pub fn new(
        trace_type: ValueType,
        flux_type: ValueType,
        pairing: BoundaryPairing,
    ) -> Result<Self, BoundaryPhysicalViolation> {
        if trace_type.scalar_domain() != ScalarDomain::Real
            || flux_type.scalar_domain() != ScalarDomain::Real
        {
            return Err(BoundaryPhysicalViolation::EuclideanRequiresReal);
        }
        if trace_type
            .clone()
            .with_dimension(flux_type.dimension())
            .as_ref()
            != Ok(&flux_type)
        {
            return Err(BoundaryPhysicalViolation::ComponentTypeMismatch);
        }
        Ok(Self {
            trace_type,
            flux_type,
            pairing,
        })
    }

    /// Complete mathematical type of the trace quantity.
    #[must_use]
    pub const fn trace_type(&self) -> &ValueType {
        &self.trace_type
    }

    /// Complete mathematical type of the outward-flux quantity.
    #[must_use]
    pub const fn flux_type(&self) -> &ValueType {
        &self.flux_type
    }

    /// Exact common component shape.
    #[must_use]
    pub const fn shape(&self) -> &ValueShape {
        self.trace_type.shape()
    }

    /// Coordinate-frame meaning of components.
    #[must_use]
    pub const fn frame(&self) -> ValueFrame {
        self.trace_type.frame()
    }

    /// Closed boundary dual pairing.
    #[must_use]
    pub const fn pairing(&self) -> BoundaryPairing {
        self.pairing
    }
}

/// Pure construction/type failure for a boundary-physical contract.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryPhysicalViolation {
    /// Euclidean boundary duality requires real quantities.
    EuclideanRequiresReal,
    /// Trace and flux have different component roles, shape or frame.
    ComponentTypeMismatch,
}

/// Mesh-independent embedding of one axis-aligned Cartesian boundary.
///
/// This is derived validation data, never a second canonical payload. Exact
/// equality means the boundaries denote the same point set in the model-global
/// Cartesian frame. IEEE `-0.0` and `0.0` compare equal by numeric equality.
#[derive(Debug, Clone)]
pub struct CartesianBoundaryEmbedding {
    ambient_dimension: usize,
    normal_axis: usize,
    side: BoundarySide,
    coordinate: f64,
    tangential_intervals: Box<[(f64, f64)]>,
}

impl PartialEq for CartesianBoundaryEmbedding {
    fn eq(&self, other: &Self) -> bool {
        self.ambient_dimension == other.ambient_dimension
            && self.normal_axis == other.normal_axis
            && self.coordinate == other.coordinate
            && self.tangential_intervals == other.tangential_intervals
    }
}

impl CartesianBoundaryEmbedding {
    /// Derive one boundary embedding from its exact Cartesian parent.
    #[must_use]
    pub fn derive(
        parent_bounds: &[AxisBounds],
        normal_axis: usize,
        side: BoundarySide,
    ) -> Option<Self> {
        let normal = parent_bounds.get(normal_axis)?;
        let coordinate = match side {
            BoundarySide::Lower => normal.lower().value(),
            BoundarySide::Upper => normal.upper().value(),
        };
        let tangential_intervals = parent_bounds
            .iter()
            .enumerate()
            .filter(|(axis, _)| *axis != normal_axis)
            .map(|(_, bounds)| (bounds.lower().value(), bounds.upper().value()))
            .collect();
        Some(Self {
            ambient_dimension: parent_bounds.len(),
            normal_axis,
            side,
            coordinate,
            tangential_intervals,
        })
    }

    /// Parent Cartesian dimension.
    #[must_use]
    pub const fn ambient_dimension(&self) -> usize {
        self.ambient_dimension
    }

    /// Axis normal to the boundary hyperplane.
    #[must_use]
    pub const fn normal_axis(&self) -> usize {
        self.normal_axis
    }

    /// Parent-outward side from which the coordinate was derived.
    #[must_use]
    pub const fn side(&self) -> BoundarySide {
        self.side
    }

    /// Fixed coherent-SI coordinate of the boundary hyperplane.
    #[must_use]
    pub const fn coordinate(&self) -> f64 {
        self.coordinate
    }

    /// Parent intervals along every non-normal axis, in axis order.
    #[must_use]
    pub const fn tangential_intervals(&self) -> &[(f64, f64)] {
        &self.tangential_intervals
    }
}

/// Resolved member of one boundary-physical conserving set.
#[derive(Debug, Clone, PartialEq)]
pub struct BoundaryPhysicalPortContract<I> {
    /// Exact nominal Connector identity.
    pub connector: I,
    /// Exact boundary Domain identity.
    pub boundary: I,
    /// Exact parent volume Domain identity.
    pub parent: I,
    /// Derived mesh-independent Cartesian embedding.
    pub embedding: CartesianBoundaryEmbedding,
}

/// Pure incompatibility in one field-valued conserving set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BoundaryPhysicalConnectionViolation {
    /// A conserving set must contain at least two Ports.
    TooFewPorts { found: usize },
    /// Members do not name one exact specialized Connector.
    ConnectorMismatch,
    /// Member boundaries do not denote one coincident Cartesian point set.
    NoncoincidentBoundaries,
}

/// Derived geometry of one validated spatial-periodic Cartesian pair.
///
/// The value is a projection of the two Port supports and their exact parent
/// Domain. It is never a second canonical payload. Lower-to-upper translation
/// is zero on every tangential axis and [`Self::period`] on
/// [`Self::normal_axis`].
#[derive(Debug, Clone, PartialEq)]
pub struct CartesianPeriodicBoundaryIdentification {
    ambient_dimension: usize,
    normal_axis: usize,
    lower_coordinate: f64,
    upper_coordinate: f64,
    tangential_intervals: Box<[(f64, f64)]>,
}

impl CartesianPeriodicBoundaryIdentification {
    /// Parent Cartesian dimension.
    #[must_use]
    pub const fn ambient_dimension(&self) -> usize {
        self.ambient_dimension
    }

    /// Axis normal to both identified boundaries.
    #[must_use]
    pub const fn normal_axis(&self) -> usize {
        self.normal_axis
    }

    /// Lower boundary coordinate in coherent SI units.
    #[must_use]
    pub const fn lower_coordinate(&self) -> f64 {
        self.lower_coordinate
    }

    /// Upper boundary coordinate in coherent SI units.
    #[must_use]
    pub const fn upper_coordinate(&self) -> f64 {
        self.upper_coordinate
    }

    /// Positive lower-to-upper translation along the normal axis.
    #[must_use]
    pub fn period(&self) -> f64 {
        self.upper_coordinate - self.lower_coordinate
    }

    /// Common intervals on every tangential axis, in axis order.
    #[must_use]
    pub const fn tangential_intervals(&self) -> &[(f64, f64)] {
        &self.tangential_intervals
    }
}

/// Pure incompatibility in one spatial-periodic boundary pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SpatialPeriodicBoundaryViolation {
    /// The first profile is an exact pair, never an N-ary connection set.
    WrongPortCount { found: usize },
    /// Members do not name one exact specialized Connector.
    ConnectorMismatch,
    /// Members do not belong to one exact parent Domain.
    ParentMismatch,
    /// Members are not parallel sides of the same Cartesian axis.
    NormalAxisMismatch,
    /// Members do not contain exactly one lower and one upper side.
    SidesNotOpposite,
    /// Tangential support or lower-to-upper translation is inconsistent.
    GeometryMismatch,
}

/// Validate the identity and geometry shared by one field-valued set.
pub fn validate_boundary_physical_connection<I: Eq>(
    ports: &[BoundaryPhysicalPortContract<I>],
) -> Result<(), BoundaryPhysicalConnectionViolation> {
    if ports.len() < 2 {
        return Err(BoundaryPhysicalConnectionViolation::TooFewPorts { found: ports.len() });
    }
    let anchor = &ports[0];
    if ports
        .iter()
        .skip(1)
        .any(|port| port.connector != anchor.connector)
    {
        return Err(BoundaryPhysicalConnectionViolation::ConnectorMismatch);
    }
    if ports
        .iter()
        .skip(1)
        .any(|port| port.embedding != anchor.embedding)
    {
        return Err(BoundaryPhysicalConnectionViolation::NoncoincidentBoundaries);
    }
    Ok(())
}

/// Validate and derive one exact Cartesian translation identification.
///
/// Unlike an ordinary conserving connection, the two supports are expected
/// to be noncoincident. The exact parent and opposite-side contract makes the
/// translation unique, so no tolerance or duplicate vector enters canonical
/// meaning.
pub fn validate_spatial_periodic_boundary_connection<I: Eq>(
    ports: &[BoundaryPhysicalPortContract<I>],
) -> Result<CartesianPeriodicBoundaryIdentification, SpatialPeriodicBoundaryViolation> {
    if ports.len() != 2 {
        return Err(SpatialPeriodicBoundaryViolation::WrongPortCount { found: ports.len() });
    }
    let lower = ports
        .iter()
        .find(|port| port.embedding.side() == BoundarySide::Lower)
        .ok_or(SpatialPeriodicBoundaryViolation::SidesNotOpposite)?;
    let upper = ports
        .iter()
        .find(|port| port.embedding.side() == BoundarySide::Upper)
        .ok_or(SpatialPeriodicBoundaryViolation::SidesNotOpposite)?;
    if lower.connector != upper.connector {
        return Err(SpatialPeriodicBoundaryViolation::ConnectorMismatch);
    }
    if lower.parent != upper.parent {
        return Err(SpatialPeriodicBoundaryViolation::ParentMismatch);
    }
    if lower.embedding.normal_axis() != upper.embedding.normal_axis()
        || lower.embedding.ambient_dimension() != upper.embedding.ambient_dimension()
    {
        return Err(SpatialPeriodicBoundaryViolation::NormalAxisMismatch);
    }
    let lower_coordinate = lower.embedding.coordinate();
    let upper_coordinate = upper.embedding.coordinate();
    let period = upper_coordinate - lower_coordinate;
    if lower.boundary == upper.boundary
        || lower.embedding.tangential_intervals() != upper.embedding.tangential_intervals()
        || !lower_coordinate.is_finite()
        || !upper_coordinate.is_finite()
        || !period.is_finite()
        || period <= 0.0
    {
        return Err(SpatialPeriodicBoundaryViolation::GeometryMismatch);
    }
    Ok(CartesianPeriodicBoundaryIdentification {
        ambient_dimension: lower.embedding.ambient_dimension(),
        normal_axis: lower.embedding.normal_axis(),
        lower_coordinate,
        upper_coordinate,
        tangential_intervals: lower.embedding.tangential_intervals().into(),
    })
}

#[cfg(test)]
mod tests {
    use eqiora_core::{DimExponents, DynQuantity, ScalarDomain, ValueShape, ValueType};

    use super::{
        BoundaryPairing, BoundaryPhysicalConnectionViolation, BoundaryPhysicalConnector,
        BoundaryPhysicalPortContract, BoundaryPhysicalViolation, CartesianBoundaryEmbedding,
        SpatialPeriodicBoundaryViolation, validate_boundary_physical_connection,
        validate_spatial_periodic_boundary_connection,
    };
    use crate::kernel::{AxisBounds, BoundarySide};
    use eqiora_core::ValueFrame;

    fn length(value: f64) -> DynQuantity {
        DynQuantity::new(
            value,
            DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension"),
        )
    }

    fn vertical_boundary(
        lower_x: f64,
        upper_x: f64,
        side: BoundarySide,
    ) -> CartesianBoundaryEmbedding {
        CartesianBoundaryEmbedding::derive(
            &[
                AxisBounds::new(length(lower_x), length(upper_x)).unwrap(),
                AxisBounds::new(length(0.0), length(1.0)).unwrap(),
            ],
            0,
            side,
        )
        .unwrap()
    }

    #[test]
    fn connector_keeps_exact_shape_frame_and_dual_dimensions() {
        let velocity =
            DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).expect("bounded dimension");
        let traction =
            DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0]).expect("bounded dimension");
        let trace_type = ValueType::shaped(
            ScalarDomain::Real,
            velocity,
            ValueShape::new([2]).unwrap(),
            ValueFrame::SpatialCartesian,
        )
        .unwrap();
        let connector = BoundaryPhysicalConnector::new(
            trace_type.clone(),
            trace_type
                .with_dimension(traction)
                .expect("checked numeric dimension"),
            BoundaryPairing::EuclideanBoundaryDuality,
        )
        .unwrap();

        assert_eq!(connector.trace_type().dimension(), velocity);
        assert_eq!(connector.flux_type().dimension(), traction);
        assert_eq!(connector.shape().extents()[0].get(), 2);
        assert_eq!(connector.frame(), ValueFrame::SpatialCartesian);
    }

    #[test]
    fn euclidean_pairing_requires_real_equal_component_types() {
        let scalar = ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
            .expect("checked scalar type");
        let complex = ValueType::scalar(ScalarDomain::Complex, DimExponents::DIMENSIONLESS)
            .expect("checked scalar type");
        for (trace, flux) in [
            (scalar.clone(), complex.clone()),
            (complex.clone(), scalar.clone()),
            (complex.clone(), complex),
        ] {
            assert_eq!(
                BoundaryPhysicalConnector::new(
                    trace,
                    flux,
                    BoundaryPairing::EuclideanBoundaryDuality
                ),
                Err(BoundaryPhysicalViolation::EuclideanRequiresReal)
            );
        }
        let vector = ValueType::shaped(
            ScalarDomain::Real,
            DimExponents::DIMENSIONLESS,
            ValueShape::new([2]).unwrap(),
            ValueFrame::SpatialCartesian,
        )
        .unwrap();
        let array = scalar.clone().array(2).unwrap();
        for (trace, flux) in [
            (scalar, array.clone()),
            (vector.clone(), array),
            (vector.clone(), vector.clone().array(1).unwrap()),
        ] {
            assert_eq!(
                BoundaryPhysicalConnector::new(
                    trace,
                    flux,
                    BoundaryPairing::EuclideanBoundaryDuality
                ),
                Err(BoundaryPhysicalViolation::ComponentTypeMismatch)
            );
        }
        let nested = vector.array(3).unwrap();
        let flux = nested
            .clone()
            .with_dimension(DimExponents::from_integers([1, 0, 0, 0, 0, 0, 0]).unwrap())
            .expect("checked numeric dimension");
        let connector = BoundaryPhysicalConnector::new(
            nested.clone(),
            flux.clone(),
            BoundaryPairing::EuclideanBoundaryDuality,
        )
        .unwrap();
        assert_eq!(connector.trace_type(), &nested);
        assert_eq!(connector.flux_type(), &flux);
    }

    #[test]
    fn connection_admission_is_exactly_nominal_and_geometric() {
        let left = BoundaryPhysicalPortContract {
            connector: "mechanical",
            boundary: "left-wall",
            parent: "left-body",
            embedding: vertical_boundary(0.0, 1.0, BoundarySide::Upper),
        };
        let right = BoundaryPhysicalPortContract {
            connector: "mechanical",
            boundary: "right-wall",
            parent: "right-body",
            embedding: vertical_boundary(1.0, 2.0, BoundarySide::Lower),
        };
        assert_eq!(
            validate_boundary_physical_connection(&[left.clone(), right.clone()]),
            Ok(())
        );
        assert_eq!(
            validate_boundary_physical_connection(std::slice::from_ref(&left)),
            Err(BoundaryPhysicalConnectionViolation::TooFewPorts { found: 1 })
        );
        assert_eq!(
            validate_boundary_physical_connection(&[
                left.clone(),
                BoundaryPhysicalPortContract {
                    connector: "other-mechanical",
                    ..right.clone()
                },
            ]),
            Err(BoundaryPhysicalConnectionViolation::ConnectorMismatch)
        );
        assert_eq!(
            validate_boundary_physical_connection(&[
                left,
                BoundaryPhysicalPortContract {
                    embedding: vertical_boundary(1.25, 2.0, BoundarySide::Lower),
                    ..right
                },
            ]),
            Err(BoundaryPhysicalConnectionViolation::NoncoincidentBoundaries)
        );
    }

    #[test]
    fn spatial_periodic_pair_derives_one_exact_translation() {
        let lower = BoundaryPhysicalPortContract {
            connector: "transport",
            boundary: "x-lower",
            parent: "body",
            embedding: vertical_boundary(-2.0, 3.0, BoundarySide::Lower),
        };
        let upper = BoundaryPhysicalPortContract {
            connector: "transport",
            boundary: "x-upper",
            parent: "body",
            embedding: vertical_boundary(-2.0, 3.0, BoundarySide::Upper),
        };

        let identification =
            validate_spatial_periodic_boundary_connection(&[upper.clone(), lower.clone()]).unwrap();
        assert_eq!(identification.ambient_dimension(), 2);
        assert_eq!(identification.normal_axis(), 0);
        assert_eq!(identification.lower_coordinate(), -2.0);
        assert_eq!(identification.upper_coordinate(), 3.0);
        assert_eq!(identification.period(), 5.0);
        assert_eq!(identification.tangential_intervals(), &[(0.0, 1.0)]);

        assert_eq!(
            validate_spatial_periodic_boundary_connection(std::slice::from_ref(&lower)),
            Err(SpatialPeriodicBoundaryViolation::WrongPortCount { found: 1 })
        );
        assert_eq!(
            validate_spatial_periodic_boundary_connection(&[
                lower.clone(),
                BoundaryPhysicalPortContract {
                    connector: "other",
                    ..upper.clone()
                },
            ]),
            Err(SpatialPeriodicBoundaryViolation::ConnectorMismatch)
        );
        assert_eq!(
            validate_spatial_periodic_boundary_connection(&[
                lower.clone(),
                BoundaryPhysicalPortContract {
                    parent: "other-body",
                    ..upper.clone()
                },
            ]),
            Err(SpatialPeriodicBoundaryViolation::ParentMismatch)
        );
        assert_eq!(
            validate_spatial_periodic_boundary_connection(&[
                lower.clone(),
                BoundaryPhysicalPortContract {
                    embedding: CartesianBoundaryEmbedding::derive(
                        &[
                            AxisBounds::new(length(-2.0), length(3.0)).unwrap(),
                            AxisBounds::new(length(0.0), length(1.0)).unwrap(),
                        ],
                        0,
                        BoundarySide::Lower,
                    )
                    .unwrap(),
                    ..upper.clone()
                },
            ]),
            Err(SpatialPeriodicBoundaryViolation::SidesNotOpposite)
        );
        assert_eq!(
            validate_spatial_periodic_boundary_connection(&[
                lower,
                BoundaryPhysicalPortContract {
                    embedding: CartesianBoundaryEmbedding::derive(
                        &[
                            AxisBounds::new(length(-2.0), length(3.0)).unwrap(),
                            AxisBounds::new(length(0.0), length(1.0)).unwrap(),
                        ],
                        1,
                        BoundarySide::Upper,
                    )
                    .unwrap(),
                    ..upper
                },
            ]),
            Err(SpatialPeriodicBoundaryViolation::NormalAxisMismatch)
        );

        let overflow_lower = BoundaryPhysicalPortContract {
            connector: "transport",
            boundary: "x-lower",
            parent: "body",
            embedding: vertical_boundary(-f64::MAX, f64::MAX, BoundarySide::Lower),
        };
        let overflow_upper = BoundaryPhysicalPortContract {
            connector: "transport",
            boundary: "x-upper",
            parent: "body",
            embedding: vertical_boundary(-f64::MAX, f64::MAX, BoundarySide::Upper),
        };
        assert_eq!(
            validate_spatial_periodic_boundary_connection(&[overflow_lower, overflow_upper]),
            Err(SpatialPeriodicBoundaryViolation::GeometryMismatch)
        );
    }
}
