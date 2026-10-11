mod finite;
pub use finite::FiniteBasis;

use crate::entity::kinds;
use crate::{DimExponents, Id, ScalarDomain, ValueShape};

/// Coordinate-frame meaning of mathematical value components.
///
/// Spatial components use the model-global Cartesian frame. Channel-array
/// axes do not introduce a frame or change their element's frame.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ValueFrame {
    /// Components are unchanged by a Cartesian spatial frame change.
    Invariant,
    /// Components are expressed in the model-global Cartesian spatial frame.
    SpatialCartesian,
}

/// Checked mathematical value type. Support and activation belong to its use site.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ValueType {
    scalar_domain: ScalarDomain,
    dimension: DimExponents,
    shape: ValueShape,
    frame: ValueFrame,
    array_rank: usize,
    meaning: Meaning,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum Meaning {
    Ordinary,
    Enum {
        definition: Id<kinds::Enum>,
        members: u32,
    },
    Coordinates(Box<FiniteBasis>),
    Counts(Box<FiniteBasis>),
    LinearMap(Box<(FiniteBasis, FiniteBasis)>),
    Index {
        set: Id<kinds::IndexSet>,
        extent: u32,
    },
}

impl ValueType {
    /// Shared finite enum declaration bound; executable case DAGs have separate limits.
    pub const MAX_ENUM_MEMBERS: u32 = 65_536;

    /// One closed nominal enum declaration; members are not numeric ordinals.
    pub fn enumeration(
        definition: Id<kinds::Enum>,
        member_count: u32,
    ) -> Result<Self, InvalidValueType> {
        if member_count == 0 || member_count > Self::MAX_ENUM_MEMBERS {
            return Err(InvalidValueType::EnumType);
        }
        Ok(Self {
            scalar_domain: ScalarDomain::Enum,
            dimension: DimExponents::DIMENSIONLESS,
            shape: ValueShape::scalar(),
            frame: ValueFrame::Invariant,
            array_rank: 0,
            meaning: Meaning::Enum {
                definition,
                members: member_count,
            },
        })
    }

    /// Exact enum declaration identity.
    pub const fn enum_definition(&self) -> Option<Id<kinds::Enum>> {
        match self.meaning {
            Meaning::Enum { definition, .. } => Some(definition),
            _ => None,
        }
    }
    /// Exact declared member count, cross-checked by semantic admission.
    pub const fn enum_member_count(&self) -> Option<u32> {
        match self.meaning {
            Meaning::Enum { members, .. } => Some(members),
            _ => None,
        }
    }

    /// Dimensionless invariant logical scalar, distinct from every numeric domain.
    #[must_use]
    pub fn boolean() -> Self {
        Self::scalar(ScalarDomain::Boolean, DimExponents::DIMENSIONLESS).expect("canonical Boolean")
    }

    /// Bounded ordinal tied to one exact IndexSet declaration.
    pub fn index(set: Id<kinds::IndexSet>, extent: u32) -> Result<Self, InvalidValueType> {
        if extent == 0 {
            return Err(InvalidValueType::ArrayExtent);
        }
        let mut value = Self::scalar(ScalarDomain::Integer, DimExponents::DIMENSIONLESS)?;
        value.meaning = Meaning::Index { set, extent };
        Ok(value)
    }
    /// Exact nominal IndexSet identity, absent for ordinary integers.
    #[must_use]
    pub const fn index_set(&self) -> Option<Id<kinds::IndexSet>> {
        match self.meaning {
            Meaning::Index { set, .. } => Some(set),
            _ => None,
        }
    }
    /// Declared exclusive index bound, validated against the semantic declaration.
    #[must_use]
    pub const fn index_extent(&self) -> Option<u32> {
        match self.meaning {
            Meaning::Index { extent, .. } => Some(extent),
            _ => None,
        }
    }

    /// Promote scalar components to the smallest common domain without changing their roles.
    #[must_use]
    pub fn with_common_scalar_domain(mut self, other: &Self) -> Option<Self> {
        if self.meaning != other.meaning {
            return None;
        }
        self.scalar_domain = self.scalar_domain.common(other.scalar_domain)?;
        Some(self)
    }

    /// Wrap this complete element type in one ordered channel-array axis.
    ///
    /// # Errors
    /// Rejects a zero extent or an unrepresentable component count.
    pub fn array(self, extent: u32) -> Result<Self, InvalidValueType> {
        if self.scalar_domain == ScalarDomain::Enum {
            return Err(InvalidValueType::EnumType);
        }
        if self.scalar_domain == ScalarDomain::Boolean {
            return Err(InvalidValueType::BooleanType);
        }
        if self.finite_bases().next().is_some() || self.index_set().is_some() {
            return Err(InvalidValueType::FiniteSpaceShape);
        }
        let shape = ValueShape::new(
            [extent]
                .into_iter()
                .chain(self.shape.extents().iter().map(|n| n.get())),
        )
        .map_err(|_| InvalidValueType::ArrayExtent)?;
        if shape.component_count().is_none() {
            return Err(InvalidValueType::ComponentCountOverflow);
        }
        Ok(Self {
            shape,
            array_rank: self.array_rank + 1,
            ..self
        })
    }

    /// Number of outer channel-array axes, distinct from inner spatial axes.
    #[must_use]
    pub const fn array_rank(&self) -> usize {
        self.array_rank
    }

    /// Preserve the scalar domain and component meaning with a checked derived dimension.
    pub fn with_dimension(mut self, dimension: DimExponents) -> Result<Self, InvalidValueType> {
        if self.scalar_domain == ScalarDomain::Enum && dimension != DimExponents::DIMENSIONLESS {
            return Err(InvalidValueType::EnumType);
        }
        if self.scalar_domain == ScalarDomain::Boolean && dimension != DimExponents::DIMENSIONLESS {
            return Err(InvalidValueType::BooleanType);
        }
        self.dimension = dimension;
        Ok(self)
    }
    /// Construct a checked invariant scalar type. Nominal enums require `enumeration`.
    pub fn scalar(
        scalar_domain: ScalarDomain,
        dimension: DimExponents,
    ) -> Result<Self, InvalidValueType> {
        Self::shaped(
            scalar_domain,
            dimension,
            ValueShape::scalar(),
            ValueFrame::Invariant,
        )
    }

    /// Construct exact channel axes (invariant) or spatial axes (Cartesian).
    /// Wrap spatial elements with [`Self::array`] for arrays of vectors or tensors.
    ///
    /// # Errors
    /// Rejects an unrepresentable component count or a frame-bearing scalar.
    pub fn shaped(
        scalar_domain: ScalarDomain,
        dimension: DimExponents,
        shape: ValueShape,
        frame: ValueFrame,
    ) -> Result<Self, InvalidValueType> {
        if scalar_domain == ScalarDomain::Enum {
            return Err(InvalidValueType::EnumType);
        }
        if scalar_domain == ScalarDomain::Boolean
            && (dimension != DimExponents::DIMENSIONLESS
                || !shape.is_scalar()
                || frame != ValueFrame::Invariant)
        {
            return Err(InvalidValueType::BooleanType);
        }
        if shape.component_count().is_none() {
            return Err(InvalidValueType::ComponentCountOverflow);
        }
        if shape.is_scalar() && frame != ValueFrame::Invariant {
            return Err(InvalidValueType::ScalarFrame);
        }
        Ok(Self {
            scalar_domain,
            dimension,
            meaning: Meaning::Ordinary,
            array_rank: if frame == ValueFrame::Invariant {
                shape.rank()
            } else {
                0
            },
            shape,
            frame,
        })
    }

    /// Mathematical domain of each scalar component.
    #[must_use]
    pub const fn scalar_domain(&self) -> ScalarDomain {
        self.scalar_domain
    }

    /// Exact physical dimension of each component.
    #[must_use]
    pub const fn dimension(&self) -> DimExponents {
        self.dimension
    }

    /// Ordered mathematical component extents.
    #[must_use]
    pub const fn shape(&self) -> &ValueShape {
        &self.shape
    }

    /// Coordinate-frame meaning of the components.
    #[must_use]
    pub const fn frame(&self) -> ValueFrame {
        self.frame
    }
}

/// Invalid mathematical shape/frame combination.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InvalidValueType {
    /// Boolean values require dimensionless invariant scalars.
    BooleanType,
    /// Enums require an exact nonempty declaration and dimensionless invariant scalar shape.
    EnumType,
    /// Finite coordinates or maps require a numeric domain and explicit basis axes.
    FiniteSpaceType,
    /// Finite basis coordinates cannot acquire implicit channel axes.
    FiniteSpaceShape,
    /// An array axis must contain at least one element.
    ArrayExtent,
    /// Component count cannot be represented on this target.
    ComponentCountOverflow,
    /// A derived physical dimension exceeds the exact exponent representation.
    DimensionOverflow,
    /// A scalar cannot carry component-frame axes.
    ScalarFrame,
}

impl core::fmt::Display for InvalidValueType {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter.write_str(match self {
            Self::EnumType => {
                "enum values require a closed nominal dimensionless invariant scalar type"
            }
            Self::BooleanType => "Boolean values require dimensionless invariant scalar types",
            Self::FiniteSpaceType => "finite coordinates and maps require admitted scalar domains and explicit basis axes",
            Self::FiniteSpaceShape => "finite basis coordinates are not channel arrays",
            Self::ArrayExtent => "array extent must be positive",
            Self::ComponentCountOverflow => "mathematical component count is not representable",
            Self::DimensionOverflow => "derived physical dimension is not representable",
            Self::ScalarFrame => "a scalar must have an invariant component frame",
        })
    }
}

impl std::error::Error for InvalidValueType {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn boolean_type_has_no_numeric_domain_or_shaped_embedding() {
        let boolean = ValueType::boolean();
        assert_eq!(boolean.scalar_domain(), ScalarDomain::Boolean);
        assert_eq!(boolean.dimension(), DimExponents::DIMENSIONLESS);
        assert!(boolean.shape().is_scalar());
        assert_eq!(boolean.frame(), ValueFrame::Invariant);
        assert_eq!(boolean.array_rank(), 0);
        assert!(boolean.finite_space().is_none());
        assert!(boolean.index_set().is_none());
        assert!(!boolean.is_count());
        assert_eq!(
            ScalarDomain::Boolean.common(ScalarDomain::Boolean),
            Some(ScalarDomain::Boolean)
        );
        for domain in [
            ScalarDomain::Integer,
            ScalarDomain::Real,
            ScalarDomain::Complex,
        ] {
            assert_eq!(ScalarDomain::Boolean.common(domain), None);
            assert_eq!(domain.common(ScalarDomain::Boolean), None);
            assert!(
                boolean
                    .clone()
                    .with_common_scalar_domain(
                        &ValueType::scalar(domain, DimExponents::DIMENSIONLESS)
                            .expect("checked scalar type")
                    )
                    .is_none()
            );
        }
        assert_eq!(boolean.clone().array(1), Err(InvalidValueType::BooleanType));
        for (dimension, shape, frame) in [
            (
                DimExponents::DIMENSIONLESS,
                ValueShape::new([1]).unwrap(),
                ValueFrame::Invariant,
            ),
            (
                DimExponents::DIMENSIONLESS,
                ValueShape::scalar(),
                ValueFrame::SpatialCartesian,
            ),
            (
                DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap(),
                ValueShape::scalar(),
                ValueFrame::Invariant,
            ),
        ] {
            assert_eq!(
                ValueType::shaped(ScalarDomain::Boolean, dimension, shape, frame),
                Err(InvalidValueType::BooleanType)
            );
        }
        assert_eq!(
            ValueType::shaped(
                ScalarDomain::Boolean,
                DimExponents::DIMENSIONLESS,
                ValueShape::scalar(),
                ValueFrame::Invariant
            )
            .unwrap(),
            boolean
        );
    }

    #[test]
    fn scalar_domain_dimension_and_frame_are_independent_type_identity() {
        let real = ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
            .expect("checked scalar type");
        let complex = ValueType::scalar(ScalarDomain::Complex, real.dimension())
            .expect("checked scalar type");
        assert_ne!(real, complex);
        assert_eq!(
            real.clone().with_common_scalar_domain(&complex),
            Some(complex.clone())
        );
        assert_eq!(
            complex.clone().with_common_scalar_domain(&real),
            Some(complex)
        );
        let channels = ValueType::shaped(
            ScalarDomain::Complex,
            real.dimension(),
            ValueShape::new([3]).unwrap(),
            ValueFrame::Invariant,
        )
        .unwrap();
        let vector = ValueType::shaped(
            ScalarDomain::Complex,
            real.dimension(),
            ValueShape::new([3]).unwrap(),
            ValueFrame::SpatialCartesian,
        )
        .unwrap();
        assert_ne!(channels, vector);
        assert_eq!(channels.shape(), vector.shape());
        let dimensioned = ValueType::scalar(
            ScalarDomain::Real,
            DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap(),
        )
        .expect("checked scalar type");
        assert_ne!(real, dimensioned);
    }

    #[test]
    fn malformed_shape_frame_types_are_not_constructible() {
        assert_eq!(
            ValueType::shaped(
                ScalarDomain::Real,
                DimExponents::DIMENSIONLESS,
                ValueShape::scalar(),
                ValueFrame::SpatialCartesian,
            ),
            Err(InvalidValueType::ScalarFrame)
        );
        assert_eq!(
            ValueType::shaped(
                ScalarDomain::Complex,
                DimExponents::DIMENSIONLESS,
                ValueShape::new([u32::MAX; 3]).unwrap(),
                ValueFrame::Invariant,
            ),
            Err(InvalidValueType::ComponentCountOverflow)
        );
    }
}

#[cfg(test)]
mod enum_bound_tests {
    use super::*;
    #[test]
    fn enum_cardinality_uses_the_shared_finite_declaration_bound() {
        let id = Id::new();
        let largest = ValueType::enumeration(id, ValueType::MAX_ENUM_MEMBERS).unwrap();
        assert_eq!(largest.enum_member_count(), Some(65_536));
        assert!(ValueType::enumeration(id, 65_537).is_err());
        assert!(ValueType::enumeration(id, u32::MAX).is_err());
    }
}
