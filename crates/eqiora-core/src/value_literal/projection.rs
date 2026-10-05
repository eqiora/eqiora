//! Mathematical component projections; no phasor, power or probability convention.
use super::{InvalidValueLiteral, ValueLiteral};
use crate::{DimExponents, DynQuantity};

/// A mathematical projection of one declared real or complex component.
/// Shape, basis, support and physical interpretation remain with the source value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ComplexProjection {
    /// Real part, with the source dimension.
    Real,
    /// Imaginary part, with the source dimension.
    Imaginary,
    /// Euclidean component magnitude, with the source dimension.
    Magnitude,
    /// Squared component magnitude, with the squared source dimension.
    SquaredMagnitude,
    /// Principal argument in radians, undefined exactly at zero; no unwrapping.
    Phase,
}

impl ValueLiteral {
    /// Real part of one declared row-major component, with the source dimension.
    /// # Errors
    /// Rejects nonnumeric domains and indexes outside the exact shape.
    pub fn component_real(&self, index: usize) -> Result<DynQuantity, InvalidValueLiteral> {
        self.defined_component(index, ComplexProjection::Real)
    }

    /// Imaginary part of one declared row-major component, with the source dimension.
    /// # Errors
    /// Rejects nonnumeric domains and indexes outside the exact shape.
    pub fn component_imaginary(&self, index: usize) -> Result<DynQuantity, InvalidValueLiteral> {
        self.defined_component(index, ComplexProjection::Imaginary)
    }

    /// Euclidean magnitude of one declared component, with the source dimension.
    /// No peak/RMS, power or probability interpretation is inferred.
    /// # Errors
    /// Rejects nonnumeric domains, out-of-shape indexes and numerical overflow.
    pub fn component_magnitude(&self, index: usize) -> Result<DynQuantity, InvalidValueLiteral> {
        self.defined_component(index, ComplexProjection::Magnitude)
    }

    /// Squared magnitude of one declared component, with the squared source dimension.
    /// This does not sum components or infer a physical normalization.
    /// # Errors
    /// Rejects nonnumeric domains, out-of-shape indexes, derived-dimension overflow
    /// and numerical overflow.
    pub fn component_squared_magnitude(
        &self,
        index: usize,
    ) -> Result<DynQuantity, InvalidValueLiteral> {
        self.defined_component(index, ComplexProjection::SquaredMagnitude)
    }

    /// Principal argument in radians (dimensionless), undefined exactly at zero.
    /// No threshold, phase unwrapping or phasor convention is implicit. The source
    /// value retains its exact shape, support and nominal component basis.
    /// # Errors
    /// Rejects nonnumeric domains and indexes outside the exact shape.
    pub fn component_phase(
        &self,
        index: usize,
    ) -> Result<Option<DynQuantity>, InvalidValueLiteral> {
        self.project_component(index, ComplexProjection::Phase)
    }

    fn defined_component(
        &self,
        index: usize,
        projection: ComplexProjection,
    ) -> Result<DynQuantity, InvalidValueLiteral> {
        self.project_component(index, projection)
            .map(|value| value.expect("non-phase component is defined"))
    }

    /// Project one row-major component without dropping its physical dimension.
    ///
    /// Only phase at exact zero returns `None`. No threshold, peak/RMS conversion,
    /// power factor, normalization or probability interpretation is implicit.
    /// The returned quantity does not replace the source's shape or nominal basis.
    /// # Errors
    /// Rejects nonnumeric domains, an out-of-range component, unrepresentable
    /// derived dimensions, and overflow of the requested projection.
    fn project_component(
        &self,
        index: usize,
        projection: ComplexProjection,
    ) -> Result<Option<DynQuantity>, InvalidValueLiteral> {
        let values = self.components().ok_or(InvalidValueLiteral::ScalarDomain)?;
        if index >= values.len() {
            return Err(InvalidValueLiteral::ComponentCount);
        }
        let (real, imaginary) = self.component(index).expect("checked component");
        let dimension = self.value_type().dimension();
        let (value, dimension) = match projection {
            ComplexProjection::Real => (real, dimension),
            ComplexProjection::Imaginary => (imaginary, dimension),
            ComplexProjection::Magnitude => (real.hypot(imaginary), dimension),
            ComplexProjection::SquaredMagnitude => (
                real.mul_add(real, imaginary * imaginary),
                dimension
                    .pow(2, 1)
                    .ok_or(InvalidValueLiteral::ScalarDomain)?,
            ),
            ComplexProjection::Phase => {
                if real == 0.0 && imaginary == 0.0 {
                    return Ok(None);
                }
                (imaginary.atan2(real), DimExponents::DIMENSIONLESS)
            }
        };
        if !value.is_finite() {
            return Err(InvalidValueLiteral::NonFinite);
        }
        Ok(Some(DynQuantity::new(value, dimension)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ScalarDomain, ValueType};

    #[test]
    fn independent_component_geometry_and_units() {
        let meters = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
        let value = ValueLiteral::new(
            ValueType::scalar(ScalarDomain::Complex, meters)
                .unwrap()
                .array(3)
                .unwrap(),
            [(3.0, 4.0), (-1.0, 1.0), (0.0, 0.0)],
        )
        .unwrap();
        for (projection, expected, dimension) in [
            (ComplexProjection::Real, 3.0, meters),
            (ComplexProjection::Imaginary, 4.0, meters),
            (ComplexProjection::Magnitude, 5.0, meters),
            (
                ComplexProjection::SquaredMagnitude,
                25.0,
                meters.pow(2, 1).unwrap(),
            ),
        ] {
            let actual = value.project_component(0, projection).unwrap().unwrap();
            assert_eq!(actual.value(), expected);
            assert_eq!(actual.dim(), dimension);
        }
        let phase = value
            .project_component(1, ComplexProjection::Phase)
            .unwrap()
            .unwrap();
        assert!((phase.value() - 3.0 * std::f64::consts::FRAC_PI_4).abs() < 1e-15);
        assert_eq!(phase.dim(), DimExponents::DIMENSIONLESS);
        assert_eq!(
            value
                .project_component(2, ComplexProjection::Phase)
                .unwrap(),
            None
        );
        assert_eq!(
            value.project_component(3, ComplexProjection::Real),
            Err(InvalidValueLiteral::ComponentCount)
        );
        assert_eq!(
            ValueLiteral::boolean(true).project_component(0, ComplexProjection::Real),
            Err(InvalidValueLiteral::ScalarDomain)
        );
    }

    #[test]
    fn requested_projection_owns_its_numerical_range() {
        let value = ValueLiteral::new(
            ValueType::scalar(ScalarDomain::Complex, DimExponents::DIMENSIONLESS).unwrap(),
            [(3e200, 4e200)],
        )
        .unwrap();
        let magnitude = value
            .project_component(0, ComplexProjection::Magnitude)
            .unwrap()
            .unwrap();
        assert!((magnitude.value() / 1e200 - 5.0).abs() < 1e-14);
        assert_eq!(
            value.project_component(0, ComplexProjection::SquaredMagnitude),
            Err(InvalidValueLiteral::NonFinite)
        );
        let tiny =
            ValueLiteral::new(value.value_type().clone(), [(0.0, f64::MIN_POSITIVE)]).unwrap();
        assert_eq!(
            tiny.project_component(0, ComplexProjection::Phase)
                .unwrap()
                .unwrap()
                .value(),
            std::f64::consts::FRAC_PI_2
        );
    }
}
