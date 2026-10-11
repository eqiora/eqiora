//! One measure typing rule for full and factor-selected integrals.
use super::*;

/// Integration measure on one exact continuous support.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ObservableMeasure {
    /// Cartesian product measure: physical volume or the product of abstract factor units.
    Volume,
    /// Codimension-one surface measure, including an exact physical interface.
    /// Orientation belongs to the support's normal; an interface is counted once.
    Boundary,
    /// Spherical-symmetry volume measure `4*pi*r^2 dr` on one exact radial
    /// coordinate factor. Its coordinate has length units; its measure has
    /// volume units. Domain admission separately requires bounds `[0, R]`.
    SphericalVolume,
}

impl ObservableMeasure {
    /// Infer the value type and exact remaining support after integration.
    ///
    /// Source checking and Semantic Model admission use this same rule.
    /// The measure removes only its exact factors, preserving all remaining factor order.
    /// A full integral has no output support; a partial integral must name that support.
    /// # Errors
    /// Rejects a wrong measure, foreign support or unrepresentable dimension.
    pub fn output_type<I: Clone + PartialEq>(
        self,
        root: &ExpressionType<I>,
        input: &SpatialSupport<I>,
        support: &SpatialSupport<I>,
        output: Option<&SpatialSupport<I>>,
    ) -> Result<ExpressionType<I>, Diagnostic> {
        if !matches!(
            root.value_type.scalar_domain(),
            eqiora_core::ScalarDomain::Real
                | eqiora_core::ScalarDomain::Complex
                | eqiora_core::ScalarDomain::Integer
        ) {
            return Err(invalid("Observable integral requires a numeric integrand"));
        }
        let measure_dimension = match (self, support) {
            (ObservableMeasure::Volume, SpatialSupport::Coordinates { factors, .. }) => factors
                .iter()
                .try_fold(
                    DimExponents::DIMENSIONLESS,
                    |product, (_, dimension, axes)| {
                        product.mul(dimension.pow(i32::try_from(*axes).ok()?, 1)?)
                    },
                )
                .ok_or_else(|| {
                    invalid("coordinate product measure dimension exceeds its exact representation")
                })?,
            (ObservableMeasure::SphericalVolume, SpatialSupport::Coordinates { factors, .. })
                if matches!(factors.as_slice(), [(_, dimension, 1)] if *dimension == length_measure(1)?) =>
            {
                length_measure(3)?
            }
            (ObservableMeasure::Volume, SpatialSupport::Volume { dimensions, .. }) => {
                length_measure(*dimensions)?
            }
            (
                ObservableMeasure::Boundary,
                SpatialSupport::Boundary { dimensions, .. }
                | SpatialSupport::PhysicalInterface { dimensions, .. },
            ) => length_measure(
                dimensions
                    .checked_sub(1)
                    .ok_or_else(|| invalid("Observable boundary has no ambient dimension"))?,
            )?,
            _ => return Err(invalid("Observable measure does not match its Domain kind")),
        };
        if root.support.as_ref().is_some_and(|actual| actual != input) {
            return Err(invalid(
                "integrand does not have the exact input support; boundary traces must be explicit",
            ));
        }
        check_projection(input, support, output)?;
        let dimension = root.dimension().mul(measure_dimension).ok_or_else(|| {
            invalid("Observable integral dimension exceeds its exact representation")
        })?;
        let value_type = root
            .value_type
            .clone()
            .with_dimension(dimension)
            .map_err(|_| invalid("Observable integral requires a numeric result type"))?;
        Ok(ExpressionType::new(value_type, output.cloned()))
    }
}

fn factors<I: Clone + PartialEq>(
    support: &SpatialSupport<I>,
) -> Result<Vec<(I, DimExponents, usize)>, Diagnostic> {
    let factors = match support {
        SpatialSupport::Coordinates { factors, .. } => factors.clone(),
        SpatialSupport::Volume { domain, dimensions } => {
            vec![(domain.clone(), length_measure(1)?, *dimensions)]
        }
        _ => {
            return Err(invalid(
                "partial integration requires Cartesian coordinate factors",
            ));
        }
    };
    if factors.is_empty()
        || factors.iter().enumerate().any(|(index, (id, _, axes))| {
            *axes == 0
                || factors[..index]
                    .iter()
                    .any(|(previous, _, _)| previous == id)
        })
    {
        return Err(invalid(
            "integration support requires nonempty unique coordinate factors",
        ));
    }
    Ok(factors)
}

fn check_projection<I: Clone + PartialEq>(
    input: &SpatialSupport<I>,
    measure: &SpatialSupport<I>,
    output: Option<&SpatialSupport<I>>,
) -> Result<(), Diagnostic> {
    if input == measure {
        return if output.is_none() {
            Ok(())
        } else {
            Err(invalid("a full integral cannot retain an output support"))
        };
    }
    let input = factors(input)?;
    let selected = factors(measure)?;
    if selected.iter().any(|factor| !input.contains(factor)) {
        return Err(invalid(
            "integration measure names a foreign or differently typed coordinate factor",
        ));
    }
    let remaining = input
        .into_iter()
        .filter(|factor| !selected.contains(factor))
        .collect::<Vec<_>>();
    match output {
        None if remaining.is_empty() => Ok(()),
        Some(output) if !remaining.is_empty() && factors(output)? == remaining => Ok(()),
        _ => Err(invalid(
            "integral output support differs from its exact ordered remaining factors",
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(length: i32, time: i32) -> DimExponents {
        DimExponents::from_integers([0, length, time, 0, 0, 0, 0]).unwrap()
    }

    #[test]
    fn interface_measure_has_one_surface_dimension_and_exact_ordered_support() {
        for (dimensions, expected) in [(1, unit(0, 0)), (2, unit(1, 0)), (3, unit(2, 0))] {
            let support = SpatialSupport::PhysicalInterface {
                domain: 1,
                boundaries: Box::new([2, 3]),
                parents: Box::new([4, 5]),
                dimensions,
            };
            let value = ExpressionType::scalar(unit(0, 0), Some(support.clone()));
            let result = ObservableMeasure::Boundary
                .output_type(&value, &support, &support, None)
                .unwrap();
            assert_eq!(result.dimension(), expected);
            assert_eq!(result.support, None);
            assert!(
                ObservableMeasure::Volume
                    .output_type(&value, &support, &support, None)
                    .is_err()
            );
            let reversed = SpatialSupport::PhysicalInterface {
                domain: 1,
                boundaries: Box::new([3, 2]),
                parents: Box::new([5, 4]),
                dimensions,
            };
            assert!(
                ObservableMeasure::Boundary
                    .output_type(&value, &support, &reversed, None)
                    .is_err()
            );
        }
    }
    fn coordinates(
        domain: &'static str,
        factors: Vec<(&'static str, DimExponents, usize)>,
    ) -> SpatialSupport<&'static str> {
        SpatialSupport::Coordinates { domain, factors }
    }

    #[test]
    fn spherical_measure_has_volume_units_and_removes_only_the_exact_radial_factor() {
        let r = ("radius", unit(1, 0), 1);
        let x = ("position", unit(1, 0), 1);
        let radial = coordinates("radius", vec![r]);
        let position = coordinates("position", vec![x]);
        let phase = coordinates("position_radius", vec![x, r]);
        let concentration = ExpressionType::scalar(unit(-3, 0), Some(phase.clone()));
        let total = ObservableMeasure::SphericalVolume
            .output_type(&concentration, &phase, &radial, Some(&position))
            .unwrap();
        assert_eq!(total.dimension(), DimExponents::DIMENSIONLESS);
        assert_eq!(total.support, Some(position.clone()));
        let cartesian = ObservableMeasure::Volume
            .output_type(&concentration, &phase, &radial, Some(&position))
            .unwrap();
        assert_eq!(cartesian.dimension(), unit(-2, 0));
        for wrong in [
            coordinates("speed", vec![("speed", unit(1, -1), 1)]),
            coordinates("two_axes", vec![("two_axes", unit(1, 0), 2)]),
            coordinates("two_factors", vec![x, r]),
            SpatialSupport::Volume {
                domain: "box",
                dimensions: 1,
            },
        ] {
            let density = ExpressionType::scalar(unit(-3, 0), Some(wrong.clone()));
            assert!(
                ObservableMeasure::SphericalVolume
                    .output_type(&density, &wrong, &wrong, None)
                    .is_err()
            );
        }
        let foreign = coordinates("foreign", vec![("foreign", unit(1, 0), 1)]);
        assert!(
            ObservableMeasure::SphericalVolume
                .output_type(&concentration, &phase, &foreign, Some(&position))
                .is_err()
        );
        assert!(
            ObservableMeasure::SphericalVolume
                .output_type(&concentration, &phase, &radial, None)
                .is_err()
        );
    }

    #[test]
    fn position_velocity_reductions_preserve_only_the_unintegrated_factor_and_units() {
        let x = ("position", unit(1, 0), 1);
        let v = ("velocity", unit(1, -1), 1);
        let phase = coordinates("phase", vec![x, v]);
        let position = coordinates("position", vec![x]);
        let velocity = coordinates("velocity", vec![v]);
        let density = ExpressionType::scalar(unit(-2, 1), Some(phase.clone()));
        // f has units s/m²: velocity integration gives number density 1/m,
        // while position integration gives a velocity density s/m.
        for (measure, output, expected) in [
            (&velocity, &position, unit(-1, 0)),
            (&position, &velocity, unit(-1, 1)),
        ] {
            let result = ObservableMeasure::Volume
                .output_type(&density, &phase, measure, Some(output))
                .unwrap();
            assert_eq!(result.dimension(), expected);
            assert_eq!(result.support.as_ref(), Some(output));
        }
        let count = ObservableMeasure::Volume
            .output_type(&density, &phase, &phase, None)
            .unwrap();
        assert_eq!(count.dimension(), DimExponents::DIMENSIONLESS);
        assert!(count.support.is_none());
        assert!(
            ObservableMeasure::Volume
                .output_type(&density, &phase, &velocity, None)
                .is_err()
        );
        assert!(
            ObservableMeasure::Volume
                .output_type(&density, &phase, &phase, Some(&position))
                .is_err()
        );
        let foreign = coordinates("foreign", vec![("foreign", unit(1, -1), 1)]);
        assert!(
            ObservableMeasure::Volume
                .output_type(&density, &phase, &foreign, Some(&position))
                .is_err()
        );
        let stale = ExpressionType::scalar(unit(-2, 1), Some(foreign));
        assert!(
            ObservableMeasure::Volume
                .output_type(&stale, &phase, &velocity, Some(&position))
                .is_err()
        );
    }

    #[test]
    fn partial_measure_preserves_remaining_factor_order_and_physical_axis_multiplicity() {
        let x = ("position", unit(1, 0), 2);
        let v = ("velocity", unit(1, -1), 1);
        let t = ("clock_coordinate", unit(0, 1), 1);
        let phase = coordinates("phase", vec![x, v, t]);
        let velocity = coordinates("velocity", vec![v]);
        let remaining = coordinates("position_time", vec![x, t]);
        let scalar = ExpressionType::scalar(DimExponents::DIMENSIONLESS, Some(phase.clone()));
        let result = ObservableMeasure::Volume
            .output_type(&scalar, &phase, &velocity, Some(&remaining))
            .unwrap();
        assert_eq!(result.dimension(), unit(1, -1)); // no implicit normalization
        assert_eq!(result.support, Some(remaining));
        let reversed = coordinates("position_time", vec![t, x]);
        assert!(
            ObservableMeasure::Volume
                .output_type(&scalar, &phase, &velocity, Some(&reversed))
                .is_err()
        );
        let wrong_axes = coordinates("position_time", vec![(x.0, x.1, 1), t]);
        assert!(
            ObservableMeasure::Volume
                .output_type(&scalar, &phase, &velocity, Some(&wrong_axes))
                .is_err()
        );
        let physical = SpatialSupport::Volume {
            domain: "position",
            dimensions: 2,
        };
        let velocity_time = coordinates("velocity_time", vec![v, t]);
        let result = ObservableMeasure::Volume
            .output_type(&scalar, &phase, &physical, Some(&velocity_time))
            .unwrap();
        assert_eq!(result.dimension(), unit(2, 0));
        assert_eq!(result.support, Some(velocity_time));
    }
}
