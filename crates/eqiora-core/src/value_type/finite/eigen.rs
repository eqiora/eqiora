use super::{InvalidValueType, ScalarDomain, ValueType};

impl ValueType {
    /// Derive `(eigenvalue, mode)` types for a Hermitian pencil `A u = lambda B u`.
    ///
    /// Both maps must be endomorphisms of the same exact declared orthonormal
    /// basis. Eigenvalues are real with dimension `A/B`; modes have dimension
    /// `B^(-1/2)` for the dimensionless normalization `uᴴ B u = 1`.
    /// Either complex map requires complex modes. A standard eigenproblem uses
    /// an explicitly dimensionless identity metric in the same basis.
    ///
    /// This checks types only. Execution must separately establish Hermitian
    /// coefficients and a positive-definite metric on the admitted space;
    /// neither property follows from this type derivation.
    pub fn hermitian_eigenpair_types(
        &self,
        metric: &Self,
    ) -> Result<(Self, Self), InvalidValueType> {
        let (source, target) = self.map_bases().ok_or(InvalidValueType::FiniteSpaceType)?;
        if source != target || metric.map_bases() != Some((source, target)) {
            return Err(InvalidValueType::FiniteSpaceType);
        }
        let eigenvalue_dimension = self
            .dimension()
            .div(metric.dimension())
            .ok_or(InvalidValueType::DimensionOverflow)?;
        let mode_dimension = metric
            .dimension()
            .pow(-1, 2)
            .ok_or(InvalidValueType::DimensionOverflow)?;
        let mode_domain = if self.scalar_domain() == ScalarDomain::Complex
            || metric.scalar_domain() == ScalarDomain::Complex
        {
            ScalarDomain::Complex
        } else {
            ScalarDomain::Real
        };
        Ok((
            Self::scalar(ScalarDomain::Real, eigenvalue_dimension)?,
            Self::coordinates(source, mode_domain, mode_dimension)?,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{DimExponents, FiniteBasis, Id};

    #[test]
    fn hermitian_eigenpair_types_preserve_mass_normalization_and_energy_units() {
        // kg/s² divided by kg gives s^-2; mass-normalized modes carry kg^-1/2.
        // Use six components: physical-space dimension is not the mode extent.
        let basis = FiniteBasis::new(Id::new(), 6).unwrap();
        let stiffness = DimExponents::from_integers([1, 0, -2, 0, 0, 0, 0]).unwrap();
        let mass = DimExponents::from_integers([1, 0, 0, 0, 0, 0, 0]).unwrap();
        let a = ValueType::linear_map(basis, basis, ScalarDomain::Real, stiffness).unwrap();
        let b = ValueType::linear_map(basis, basis, ScalarDomain::Real, mass).unwrap();
        let (value, mode) = a.hermitian_eigenpair_types(&b).unwrap();
        assert_eq!(value.scalar_domain(), ScalarDomain::Real);
        assert!(value.shape().is_scalar());
        assert_eq!(
            value.dimension().exponents(),
            [(0, 1), (0, 1), (-2, 1), (0, 1), (0, 1), (0, 1), (0, 1)]
        );
        assert_eq!(mode.coordinate_basis(), Some(basis));
        assert_eq!(mode.scalar_domain(), ScalarDomain::Real);
        assert_eq!(
            mode.dimension().exponents(),
            [(-1, 2), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1), (0, 1)]
        );

        let energy = DimExponents::from_integers([1, 2, -2, 0, 0, 0, 0]).unwrap();
        let h = ValueType::linear_map(basis, basis, ScalarDomain::Complex, energy).unwrap();
        let identity = ValueType::linear_map(
            basis,
            basis,
            ScalarDomain::Real,
            DimExponents::DIMENSIONLESS,
        )
        .unwrap();
        let (value, mode) = h.hermitian_eigenpair_types(&identity).unwrap();
        assert_eq!(
            value,
            ValueType::scalar(ScalarDomain::Real, energy).unwrap()
        );
        assert_eq!(
            mode,
            ValueType::coordinates(basis, ScalarDomain::Complex, DimExponents::DIMENSIONLESS)
                .unwrap()
        );
        // Complex metrics also promote modes when A is real.
        assert_eq!(
            identity
                .hermitian_eigenpair_types(&h)
                .unwrap()
                .1
                .scalar_domain(),
            ScalarDomain::Complex
        );
    }

    #[test]
    fn hermitian_eigenpair_types_reject_equal_extent_foreign_or_dual_bases() {
        let basis = FiniteBasis::new(Id::new(), 6).unwrap();
        let foreign = FiniteBasis::new(Id::new(), 6).unwrap();
        let map = |source, target| {
            ValueType::linear_map(
                source,
                target,
                ScalarDomain::Real,
                DimExponents::DIMENSIONLESS,
            )
            .unwrap()
        };
        let a = map(basis, basis);
        for b in [
            map(foreign, foreign),
            map(basis.dual(), basis.dual()),
            map(basis, foreign),
        ] {
            assert_eq!(
                a.hermitian_eigenpair_types(&b),
                Err(InvalidValueType::FiniteSpaceType)
            );
        }
        assert_eq!(
            map(basis, foreign).hermitian_eigenpair_types(&map(basis, foreign)),
            Err(InvalidValueType::FiniteSpaceType)
        );
        let scalar = ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS).unwrap();
        assert_eq!(
            a.hermitian_eigenpair_types(&scalar),
            Err(InvalidValueType::FiniteSpaceType)
        );
        assert_eq!(
            scalar.hermitian_eigenpair_types(&a),
            Err(InvalidValueType::FiniteSpaceType)
        );
    }

    #[test]
    fn hermitian_eigenpair_types_reject_unrepresentable_derived_dimensions() {
        let basis = FiniteBasis::new(Id::new(), 1).unwrap();
        let map =
            |dimension| ValueType::linear_map(basis, basis, ScalarDomain::Real, dimension).unwrap();
        let huge = map(DimExponents::from_integers([i32::MAX, 0, 0, 0, 0, 0, 0]).unwrap());
        let inverse_mass = map(DimExponents::from_integers([-1, 0, 0, 0, 0, 0, 0]).unwrap());
        assert_eq!(
            huge.hermitian_eigenpair_types(&inverse_mass),
            Err(InvalidValueType::DimensionOverflow)
        );
        let fractional = map(DimExponents::from_rationals([
            (1, i32::MAX),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
            (0, 1),
        ])
        .unwrap());
        // A/B is dimensionless, but B^-1/2 has an unrepresentable denominator.
        assert_eq!(
            fractional.hermitian_eigenpair_types(&fractional),
            Err(InvalidValueType::DimensionOverflow)
        );
    }
}
