use super::*;

impl ScalarEllipticCartesianModel {
    /// Evaluate one exact side condition and its coordinate/Parameter JVP.
    ///
    /// # Errors
    /// Returns a lowering diagnostic for an unknown side, mismatched tangent
    /// shape, or non-finite expression evaluation.
    pub fn boundary_jvp(
        &self,
        axis: usize,
        side: BoundarySide,
        coordinates: &[f64],
        coordinate_tangent: &[f64],
        parameter_tangent: &[f64],
    ) -> Result<(&ScalarEllipticCartesianBoundary, f64, f64), Diagnostic> {
        let boundary = self.boundary(axis, side).ok_or_else(|| {
            lowering_error(
                self.domain(),
                "Cartesian boundary side is not present in the Model",
            )
        })?;
        let (value, tangent) = self.evaluate_expression_jvp(
            boundary.value(),
            coordinates,
            coordinate_tangent,
            parameter_tangent,
        )?;
        Ok((boundary, value, tangent))
    }

    /// Evaluate the constitutive coefficient and one canonical-Parameter JVP.
    ///
    /// # Errors
    /// Preserves lowered-expression shape and finite-value diagnostics.
    pub fn coefficient_jvp(
        &self,
        coordinates: &[f64],
        coordinate_tangent: &[f64],
        parameter_tangent: &[f64],
    ) -> Result<(f64, f64), Diagnostic> {
        self.evaluate_expression_jvp(
            &self.coefficient,
            coordinates,
            coordinate_tangent,
            parameter_tangent,
        )
    }
}

pub(crate) fn validate_positive_affine_coefficient(
    coefficient: &ScalarSpatialExpression<f64>,
    bounds: &[[f64; 2]],
    owner: RawId,
) -> Result<(), Diagnostic> {
    if coefficient.coordinate_dimension() != bounds.len() || coefficient.affine_gradient().is_none()
    {
        return Err(lowering_error(
            owner,
            "scalar elliptic coefficient must be an affine expression of coordinates and bound Parameters",
        ));
    }
    let corner_count = 1_usize
        .checked_shl(u32::try_from(bounds.len()).unwrap_or(u32::MAX))
        .ok_or_else(|| lowering_error(owner, "coefficient positivity corner count overflows"))?;
    for corner in 0..corner_count {
        let coordinates = bounds
            .iter()
            .enumerate()
            .map(|(axis, axis_bounds)| axis_bounds[(corner >> axis) & 1])
            .collect::<Vec<_>>();
        let value = coefficient.evaluate(&coordinates)?;
        if !value.is_finite() || value <= 0.0 {
            return Err(lowering_error(
                owner,
                "scalar elliptic coefficient must be finite and strictly positive at every Domain corner",
            ));
        }
    }
    Ok(())
}
