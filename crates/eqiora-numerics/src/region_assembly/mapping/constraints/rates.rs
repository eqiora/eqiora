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
        values: &[f64],
        proof: &ConstantDerivativeMatrixProof,
        residual: &[f64],
        tangent_rows: &BTreeSet<usize>,
        solver: SolverPlan,
    ) -> Result<Vec<f64>, Diagnostic> {
        let rhs = residual
            .iter()
            .enumerate()
            .map(|(row, value)| {
                if tangent_rows.contains(&row) {
                    0.
                } else {
                    -*value
                }
            })
            .collect::<Vec<_>>();
        let scale = rhs.iter().fold(0_f64, |norm, value| norm.hypot(*value));
        let rhs = rhs
            .iter()
            .map(|value| {
                num_rational::BigRational::from_float(*value)
                    .ok_or_else(|| invalid("initial rate compatibility requires finite residuals"))
            })
            .collect::<Result<Vec<_>, _>>()?;
        let projection = proof.compatibility_projection(&[], &rhs)?;
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
        // Fixed traces have zero rates in this autonomous snapshot profile.
        // Round only after exact compatibility, then evaluate original R(u,v).
        let reduced_rates = projection.coefficients()?;
        let rates = (0..coordinates.full_count())
            .map(|global| {
                coordinates
                    .free_coordinate(global)
                    .map_or(0., |free| reduced_rates[free])
            })
            .collect::<Vec<_>>();
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
                .sum();
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
