use super::{InvalidValueType, Meaning, ValueFrame, ValueType};
use crate::{DimExponents, Id, ScalarDomain, ValueShape, entity::kinds};

/// An exact declared component basis or its coordinate dual.
/// Extent is cross-checked against the selected FiniteSpace by semantic admission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct FiniteBasis {
    space: Id<kinds::FiniteSpace>,
    extent: u32,
    dual: bool,
    second: Option<(Id<kinds::FiniteSpace>, u32)>,
}

impl FiniteBasis {
    /// Refer to the primal ordered basis of one nonempty declaration.
    pub fn new(space: Id<kinds::FiniteSpace>, extent: u32) -> Result<Self, InvalidValueType> {
        if extent == 0 {
            return Err(InvalidValueType::ArrayExtent);
        }
        Ok(Self {
            space,
            extent,
            dual: false,
            second: None,
        })
    }
    /// Exact atomic declaration identity; a product is defined by its ordered factors.
    pub const fn space(self) -> Option<Id<kinds::FiniteSpace>> {
        if self.second.is_none() {
            Some(self.space)
        } else {
            None
        }
    }
    /// Ordered tensor product of two atomic bases with the same dual role.
    /// Nested products and mixed primal/dual factors are outside the bounded profile.
    pub fn product(left: Self, right: Self) -> Result<Self, InvalidValueType> {
        if left.second.is_some() || right.second.is_some() || left.dual != right.dual {
            return Err(InvalidValueType::FiniteSpaceType);
        }
        left.extent
            .checked_mul(right.extent)
            .ok_or(InvalidValueType::ComponentCountOverflow)?;
        Ok(Self {
            second: Some((right.space, right.extent)),
            ..left
        })
    }
    /// The ordered atomic factors of a two-factor product, excluding aliases.
    pub const fn factors(self) -> Option<[Self; 2]> {
        match self.second {
            Some((space, extent)) => Some([
                Self {
                    second: None,
                    ..self
                },
                Self {
                    space,
                    extent,
                    dual: self.dual,
                    second: None,
                },
            ]),
            None => None,
        }
    }
    /// Atomic declarations in semantic factor order, including their dual roles.
    pub fn atoms(self) -> impl Iterator<Item = Self> {
        match self.factors() {
            Some([left, right]) => [Some(left), Some(right)],
            None => [Some(self), None],
        }
        .into_iter()
        .flatten()
    }
    /// Reverse the two factors explicitly; an atomic basis has no factor permutation.
    pub fn swapped(self) -> Result<Self, InvalidValueType> {
        let [left, right] = self.factors().ok_or(InvalidValueType::FiniteSpaceType)?;
        Self::product(right, left)
    }
    /// Declared number of basis elements.
    pub const fn extent(self) -> u32 {
        match self.second {
            Some((_, extent)) => self.extent * extent,
            None => self.extent,
        }
    }
    /// Whether coordinates refer to the algebraic dual basis.
    pub const fn is_dual(self) -> bool {
        self.dual
    }
    /// Dualize the basis identity without conjugating any value or applying a metric.
    pub const fn dual(self) -> Self {
        Self {
            dual: !self.dual,
            ..self
        }
    }
}

impl ValueType {
    /// Numeric coordinates in one exact primal or dual basis.
    /// Integer coordinates describe the primal lattice; duals require real or complex scalars.
    pub fn coordinates(
        basis: FiniteBasis,
        domain: ScalarDomain,
        dimension: DimExponents,
    ) -> Result<Self, InvalidValueType> {
        if !matches!(
            domain,
            ScalarDomain::Integer | ScalarDomain::Real | ScalarDomain::Complex
        ) || (domain == ScalarDomain::Integer
            && (basis.is_dual()
                || basis.factors().is_some()
                || dimension != DimExponents::DIMENSIONLESS))
        {
            return Err(InvalidValueType::FiniteSpaceType);
        }
        Ok(Self {
            scalar_domain: domain,
            dimension,
            shape: ValueShape::new([basis.extent()]).map_err(|_| InvalidValueType::ArrayExtent)?,
            frame: ValueFrame::Invariant,
            array_rank: 0,
            meaning: Meaning::Coordinates(Box::new(basis)),
        })
    }

    /// Nonnegative exact counts in one primal basis, distinct from signed coordinates.
    pub fn counts(space: Id<kinds::FiniteSpace>, extent: u32) -> Result<Self, InvalidValueType> {
        let basis = FiniteBasis::new(space, extent)?;
        let mut value =
            Self::coordinates(basis, ScalarDomain::Integer, DimExponents::DIMENSIONLESS)?;
        value.meaning = Meaning::Counts(Box::new(basis));
        Ok(value)
    }

    /// A real or complex linear map with ordered input and output basis identities.
    /// Storage axes are output rows followed by input columns; physical dimension is the map's.
    pub fn linear_map(
        source: FiniteBasis,
        target: FiniteBasis,
        domain: ScalarDomain,
        dimension: DimExponents,
    ) -> Result<Self, InvalidValueType> {
        if !matches!(domain, ScalarDomain::Real | ScalarDomain::Complex) {
            return Err(InvalidValueType::FiniteSpaceType);
        }
        let shape = ValueShape::new([target.extent(), source.extent()])
            .map_err(|_| InvalidValueType::ArrayExtent)?;
        if shape.component_count().is_none() {
            return Err(InvalidValueType::ComponentCountOverflow);
        }
        Ok(Self {
            scalar_domain: domain,
            dimension,
            shape,
            frame: ValueFrame::Invariant,
            array_rank: 0,
            meaning: Meaning::LinearMap(Box::new((source, target))),
        })
    }

    /// Exact basis of coordinates, excluding counts and linear maps.
    pub const fn coordinate_basis(&self) -> Option<FiniteBasis> {
        match &self.meaning {
            Meaning::Coordinates(basis) => Some(**basis),
            _ => None,
        }
    }
    /// Ordered input and output bases of a linear map.
    pub const fn map_bases(&self) -> Option<(FiniteBasis, FiniteBasis)> {
        match &self.meaning {
            Meaning::LinearMap(bases) => Some(**bases),
            _ => None,
        }
    }
    /// Every nominal finite basis referenced by this type, input before output for maps.
    pub fn finite_bases(&self) -> impl Iterator<Item = FiniteBasis> {
        match &self.meaning {
            Meaning::Coordinates(basis) | Meaning::Counts(basis) => [Some(**basis), None],
            Meaning::LinearMap(bases) => [Some(bases.0), Some(bases.1)],
            _ => [None, None],
        }
        .into_iter()
        .flatten()
    }
    /// Single finite declaration of a coordinate or count; maps have two explicit bases.
    pub const fn finite_space(&self) -> Option<Id<kinds::FiniteSpace>> {
        match &self.meaning {
            Meaning::Coordinates(basis) | Meaning::Counts(basis) => basis.space(),
            _ => None,
        }
    }
    /// Whether this value carries the nonnegative count contract.
    pub const fn is_count(&self) -> bool {
        matches!(self.meaning, Meaning::Counts(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ValueLiteral;

    #[test]
    fn products_retain_factor_identity_order_and_duality_without_alias_identity() {
        let left = FiniteBasis::new(Id::new(), 2).unwrap();
        let right = FiniteBasis::new(Id::new(), 3).unwrap();
        let product = FiniteBasis::product(left, right).unwrap();
        assert_eq!(product.extent(), 6);
        assert_eq!(product.space(), None);
        assert_eq!(product.factors(), Some([left, right]));
        assert_eq!(product.atoms().collect::<Vec<_>>(), [left, right]);
        assert_eq!(product.dual().factors(), Some([left.dual(), right.dual()]));
        assert_eq!(
            FiniteBasis::product(left.dual(), right.dual()).unwrap(),
            product.dual()
        );
        assert_eq!(product.swapped().unwrap().swapped().unwrap(), product);
        assert_ne!(product, product.swapped().unwrap());
        assert_ne!(product, FiniteBasis::new(Id::new(), 6).unwrap());
        assert!(FiniteBasis::product(product, left).is_err());
        assert!(FiniteBasis::product(left.dual(), right).is_err());
        assert!(
            FiniteBasis::product(FiniteBasis::new(Id::new(), u32::MAX).unwrap(), left).is_err()
        );
        assert!(
            ValueType::coordinates(product, ScalarDomain::Integer, DimExponents::DIMENSIONLESS)
                .is_err()
        );
        let coordinate =
            ValueType::coordinates(product, ScalarDomain::Complex, DimExponents::DIMENSIONLESS)
                .unwrap();
        assert_eq!(coordinate.coordinate_basis(), Some(product));
        assert!(coordinate.finite_space().is_none());
        assert!(coordinate.array(1).is_err());
    }

    #[test]
    fn nominal_identity_variance_and_channel_axes_are_not_interchangeable() {
        let spin = FiniteBasis::new(Id::new(), 2).unwrap();
        let control = FiniteBasis::new(Id::new(), 2).unwrap();
        let coordinate = |basis| {
            ValueType::coordinates(basis, ScalarDomain::Complex, DimExponents::DIMENSIONLESS)
                .unwrap()
        };
        let ket = coordinate(spin);
        let covector = coordinate(spin.dual());
        assert_eq!(spin.dual().dual(), spin);
        assert_ne!(ket, covector);
        assert_ne!(ket, coordinate(control));
        assert!(ket.clone().with_common_scalar_domain(&covector).is_none());
        assert!(
            ket.clone()
                .with_common_scalar_domain(&coordinate(control))
                .is_none()
        );
        let array = ValueType::scalar(ScalarDomain::Complex, DimExponents::DIMENSIONLESS)
            .unwrap()
            .array(2)
            .unwrap();
        let spatial = ValueType::shaped(
            ScalarDomain::Complex,
            DimExponents::DIMENSIONLESS,
            ket.shape().clone(),
            ValueFrame::SpatialCartesian,
        )
        .unwrap();
        for foreign in [array, spatial] {
            assert_eq!(foreign.shape(), ket.shape());
            assert!(ket.clone().with_common_scalar_domain(&foreign).is_none());
        }
        assert!(ket.array(2).is_err());
    }

    #[test]
    fn maps_retain_both_endpoints_and_row_column_order() {
        let input = FiniteBasis::new(Id::new(), 2).unwrap();
        let output = FiniteBasis::new(Id::new(), 3).unwrap();
        let dimension = DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).unwrap();
        let map = ValueType::linear_map(input, output, ScalarDomain::Complex, dimension).unwrap();
        assert_eq!(
            map.shape()
                .extents()
                .iter()
                .map(|n| n.get())
                .collect::<Vec<_>>(),
            [3, 2]
        );
        assert_eq!(map.map_bases(), Some((input, output)));
        assert_eq!(map.finite_bases().collect::<Vec<_>>(), [input, output]);
        assert_eq!(map.dimension(), dimension);
        let transpose = ValueType::linear_map(
            output.dual(),
            input.dual(),
            ScalarDomain::Complex,
            dimension,
        )
        .unwrap();
        let adjoint =
            ValueType::linear_map(output, input, ScalarDomain::Complex, dimension).unwrap();
        assert_ne!(transpose, adjoint);
        assert_eq!(transpose.shape(), adjoint.shape());
        assert!(map.array(2).is_err());
    }

    #[test]
    fn numeric_literals_preserve_finite_coordinate_meaning() {
        let basis = FiniteBasis::new(Id::new(), 2).unwrap();
        let seconds = DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).unwrap();
        let value_type = ValueType::coordinates(basis, ScalarDomain::Complex, seconds).unwrap();
        let value = ValueLiteral::new(value_type.clone(), [(1.0, 2.0), (-3.0, 4.0)]).unwrap();
        assert_eq!(value.value_type(), &value_type);
        assert_eq!(value.component_count(), 2);
        assert!(ValueLiteral::new(value_type, [(1.0, 0.0)]).is_err());
        for domain in [ScalarDomain::Boolean, ScalarDomain::Enum] {
            assert!(ValueType::coordinates(basis, domain, DimExponents::DIMENSIONLESS).is_err());
            assert!(
                ValueType::linear_map(basis, basis, domain, DimExponents::DIMENSIONLESS).is_err()
            );
        }
        assert!(
            ValueType::coordinates(
                basis.dual(),
                ScalarDomain::Integer,
                DimExponents::DIMENSIONLESS
            )
            .is_err()
        );
        assert!(ValueType::coordinates(basis, ScalarDomain::Integer, seconds).is_err());
        assert!(
            ValueType::linear_map(
                basis,
                basis,
                ScalarDomain::Integer,
                DimExponents::DIMENSIONLESS
            )
            .is_err()
        );
        assert!(FiniteBasis::new(Id::new(), 0).is_err());
    }
}
