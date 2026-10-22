//! One rate witness for snapshot compatibility, checked in the original equations.
use super::*;
use crate::form_compiler::region::coordinates::Projection;
use crate::region_assembly::mapping::assembly::MappedRegionAssembly;

impl<S: Coefficient + Send + Sync, P: Projection<S, Scalar = f64> + Send + Sync>
    MappedRegionAssembly<S, P>
{
    pub(super) fn consistent_rates(
        &self,
        coordinates: &RegionRealCoordinates<'_, S>,
        point: (&[f64], &[f64]),
        proof: &ConstantDerivativeMatrixProof,
        rhs: &[f64],
        tangent_rows: &BTreeSet<usize>,
        solver: SolverPlan,
    ) -> Result<Vec<f64>, Diagnostic> {
        let (values, fixed_rates) = point;
        let scale = rhs.iter().fold(0_f64, |norm, value| norm.hypot(*value));
        let exact_rhs = rhs
            .iter()
            .map(|value| {
                num_rational::BigRational::from_float(*value)
                    .ok_or_else(|| invalid("initial rate compatibility requires finite residuals"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let projection = proof.compatibility_projection(&[], &exact_rhs)?;
        let norm = projection
            .residual()?
            .iter()
            .fold(0_f64, |norm, value| norm.hypot(*value));
        let target = solver.residual_target(scale)?;
        if !norm.is_finite() || !scale.is_finite() || norm > target {
            return Err(invalid(&format!(
                "spatial State initial rate compatibility residual {norm:e} exceeds target {target:e}"
            )));
        }
        // Retain analytic fixed-trace rates when lifting the free rate witness.
        // Round only after exact compatibility, then evaluate original R(u,v).
        let reduced_rates = projection.coefficients()?;
        let rates = coordinates.lift_rates(&reduced_rates, fixed_rates)?;
        let mut residual = self.work.residual(
            &self.plan,
            self.plan.target_id(0).expect("reduced target"),
            RegionPoint::Independent {
                state: values,
                rate_target: self.plan.target_id(1).expect("full target"),
                rates: &rates,
                pullback: FieldPullback::Uniform {
                    current: None,
                    rate: None,
                },
            },
        )?;
        for &row in tangent_rows {
            residual[row] = proof
                .row(row)
                .expect("rate row")
                .iter()
                .zip(&reduced_rates)
                .map(|(a, v)| a * v)
                .sum::<f64>()
                - rhs[row];
        }
        let norm = residual
            .iter()
            .fold(0_f64, |norm, value| norm.hypot(*value));
        if !norm.is_finite() || norm > target {
            return Err(invalid(&format!(
                "spatial State rounded initial rate witness residual {norm:e} exceeds target {target:e}"
            )));
        }
        Ok(rates)
    }
}

impl<S: Coefficient + Send + Sync> RegionDofMap<S>
where
    Self: RegionAlgebraicCoordinates<f64, S>,
{
    /// Lift only the analytic derivative of prescribed traces. A rate witness
    /// for a regular mass equation does not demand an explicit Time derivative
    /// of volume coefficients, forcing, or natural boundary data.
    pub(crate) fn prescribed_rates(&self, partial_mapping: &Self) -> Result<Vec<f64>, Diagnostic> {
        if !partial_mapping.has_same_coordinate_layout(self) {
            return Err(invalid(
                "Time partial changed the exact Field constraint layout",
            ));
        }
        let fixed = partial_mapping.lift(&vec![S::zero(); self.free_count()], false)?;
        RegionRealCoordinates::new(self)?.encode_full(&fixed)
    }

    pub(super) fn initial_rate_data<'mesh>(
        &self,
        mesh: &'mesh impl MeshGeometry<Map<'mesh> = AffineGeometryMap>,
        input: &RegionSolveInput<S>,
        values: &[f64],
        (partial_mapping, mut partial): (Self, RegionSolveInput<S>),
    ) -> Result<(Vec<f64>, Vec<f64>), Diagnostic> {
        let fixed = self.prescribed_rates(&partial_mapping)?;
        partial.retain_primal_integration(input)?;
        partial.previous = input.previous.clone();
        let coordinates = RegionRealCoordinates::new(self)?;
        let prepared = self.prepare_numeric_assembly(mesh, &partial)?;
        let residual = prepared.work.residual(
            &prepared.plan,
            prepared.plan.target_id(0).expect("reduced target"),
            RegionPoint::Independent {
                state: values,
                rate_target: prepared.plan.target_id(1).expect("full target"),
                rates: &vec![0.; coordinates.full_count()],
                pullback: FieldPullback::Uniform {
                    current: None,
                    rate: None,
                },
            },
        )?;
        Ok((fixed, residual))
    }
}
