//! Recover algebraic State and differential rates from the original equations.
use super::*;
use crate::nonlinear_execution::{
    NewtonEvidence, NewtonOptions, NewtonTrial, solve_damped_newton, stage_merit,
};

impl<S: Coefficient + Send + Sync> SpatialImplicitSystem<'_, S>
where
    RegionDofMap<S>: RegionAlgebraicCoordinates<f64, S>,
{
    pub(in crate::numerical_admission) fn differential_coordinates(
        &self,
        full: &[f64],
    ) -> Vec<f64> {
        full.iter()
            .enumerate()
            .filter_map(|(i, value)| (!self.algebraic.contains(&i)).then_some(*value))
            .collect()
    }

    pub(super) fn lift_differential(
        &self,
        differential: &[f64],
        reference: &[f64],
    ) -> Result<Vec<f64>, Diagnostic> {
        self.validate(reference)?;
        if differential.len() != self.dimension - self.algebraic.len()
            || differential.iter().any(|value| !value.is_finite())
        {
            return Err(invalid(
                "history has a foreign differential coordinate inventory",
            ));
        }
        let mut full = reference.to_vec();
        let mut values = differential.iter();
        for (i, value) in full.iter_mut().enumerate() {
            if !self.algebraic.contains(&i) {
                *value = *values.next().expect("checked inventory");
            }
        }
        Ok(full)
    }

    pub(super) fn interpolated_recovery_seed(
        &self,
        step: &eqiora_time::TimeHistoryStep,
        time: f64,
    ) -> Result<(Vec<f64>, Vec<f64>), Diagnostic> {
        let fixed = step.sample(time)?;
        let rate = step.polynomial_derivative(time)?;
        self.validate(&fixed)?;
        self.validate(&rate)?;
        let h = step.end_time() - step.start_time();
        let seed = fixed
            .iter()
            .enumerate()
            .map(|(i, value)| {
                if self.algebraic.contains(&i) {
                    *value
                } else {
                    h * rate[i]
                }
            })
            .collect::<Vec<_>>();
        self.validate(&seed)?;
        Ok((fixed, seed))
    }

    fn stage_recovery_seed(
        &self,
        initial: &[f64],
        unknown: &[f64],
        h: f64,
    ) -> Result<(Vec<f64>, Vec<f64>), Diagnostic> {
        self.validate(initial)?;
        self.validate(unknown)?;
        let (stage, _) = self.stage_coordinates(initial, unknown, h);
        let mut seed = stage.clone();
        for i in 0..self.dimension {
            if !self.algebraic.contains(&i) {
                seed[i] = unknown[i] - initial[i];
            }
        }
        Ok((stage, seed))
    }

    pub(super) fn endpoint_recovery_seed(
        &self,
        initial: &[f64],
        unknown: &[f64],
        h: f64,
    ) -> Result<(Vec<f64>, Vec<f64>), Diagnostic> {
        let (mut fixed, seed) = self.stage_recovery_seed(initial, unknown, h)?;
        for i in 0..self.dimension {
            if !self.algebraic.contains(&i) {
                fixed[i] = unknown[i];
            }
        }
        Ok((fixed, seed))
    }

    pub(super) fn require_stage_regularity(
        &self,
        time: f64,
        h: f64,
        initial: &[f64],
        unknown: &[f64],
    ) -> Result<(), Diagnostic> {
        if !self.algebraic.is_empty() {
            let (stage, seed) = self.stage_recovery_seed(initial, unknown, h)?;
            // A regular time-discrete stage matrix can hide a singular DAE chart.
            // Check the actual collocation point, independently of observations.
            self.recovery_point(time + 0.5 * h, h, &stage, &seed)?;
        }
        Ok(())
    }

    /// Hold differential State fixed. Unknowns are algebraic State and h*v_D;
    /// inactive algebraic rates are zero, not claimed physical tangent values.
    pub(in crate::numerical_admission) fn recover_consistent(
        &self,
        time: f64,
        h: f64,
        fixed: &[f64],
        seed: &[f64],
        backend: &dyn LinearSolverBackend,
    ) -> Result<(Vec<f64>, NewtonEvidence), Diagnostic> {
        self.validate(fixed)?;
        self.validate(seed)?;
        let temporal = self
            .plan
            .admission
            .temporal
            .as_ref()
            .and_then(CommonTemporalPolicy::as_ode)
            .ok_or_else(|| invalid("constraint recovery requires continuous time controls"))?;
        let nonlinear = self.plan.admission.nonlinear;
        let absolute = self.correction_tolerances(temporal)?;
        let structure = self.equations.algebraic_structure(None)?;
        let checked = self
            .plan
            .admission
            .linear
            .checked_backend(backend, Some(&structure))?;
        let linear = LinearSolveRequest::new(&checked, self.plan.linear());
        let (accepted, evidence) = solve_damped_newton(
            seed.to_vec(),
            NewtonOptions::time_stage(
                nonlinear,
                self.plan.linear(),
                temporal.relative_tolerance(),
                &absolute,
            ),
            |unknown| self.recovery_point(time, h, fixed, unknown),
            |point| {
                stage_merit(
                    &point.assembly.residuals[0],
                    &point.row_scales,
                    nonlinear.is_some(),
                )
            },
            |_, point| {
                point
                    .assembly
                    .solve_correction(0, linear, self.plan.admission.linear.workers)
            },
            |unknown, current| {
                let (state, rate) = self.recovery_coordinates(h, fixed, unknown)?;
                let mut residual = vec![0.; self.dimension];
                self.residual(time, &state, &rate, &mut residual)?;
                Ok(Some(NewtonTrial {
                    residual_norm: stage_merit(
                        &residual,
                        &current.row_scales,
                        nonlinear.is_some(),
                    )?,
                    linearization: None,
                }))
            },
        )?;
        Ok((accepted.unknown, evidence))
    }

    pub(super) fn recovery_point(
        &self,
        time: f64,
        h: f64,
        fixed: &[f64],
        unknown: &[f64],
    ) -> Result<ImplicitPoint, Diagnostic> {
        let (state, rate) = self.recovery_coordinates(h, fixed, unknown)?;
        let weights = (0..self.dimension)
            .map(|i| {
                if self.algebraic.contains(&i) {
                    [Some(1.), None]
                } else {
                    [None, Some(1. / h)]
                }
            })
            .collect::<Vec<_>>();
        let assembly = self.linearize(time, &state, &rate, FieldPullback::Coordinates(&weights))?;
        // A zero residual alone does not prove a locally unique index-one chart.
        // Reuse the existing exact rank owner, including for mixed equation rows.
        let coordinates = (0..self.dimension).collect::<Vec<_>>();
        let proof = eqiora_time::ConstantDerivativeMatrixProof::new(
            self.dimension,
            crate::region_assembly::mapping::square_projection(
                assembly.systems[0].matrix(),
                &coordinates,
            )?,
        )?;
        if proof.exact_rank() != self.dimension {
            return Err(invalid(
                "constraint recovery requires a locally regular differential-rate/algebraic-State Jacobian",
            ));
        }
        ImplicitPoint::new(unknown, assembly)
    }

    pub(in crate::numerical_admission) fn recovery_coordinates(
        &self,
        h: f64,
        fixed: &[f64],
        unknown: &[f64],
    ) -> Result<(Vec<f64>, Vec<f64>), Diagnostic> {
        self.validate(fixed)?;
        self.validate(unknown)?;
        if !h.is_finite() || h <= 0. || !(1. / h).is_finite() {
            return Err(invalid(
                "constraint recovery requires a finite positive rate scale",
            ));
        }
        let mut state = fixed.to_vec();
        let mut rate = vec![0.; self.dimension];
        for i in 0..self.dimension {
            if self.algebraic.contains(&i) {
                state[i] = unknown[i];
            } else {
                rate[i] = unknown[i] / h;
            }
        }
        self.validate(&rate)?;
        Ok((state, rate))
    }
}
