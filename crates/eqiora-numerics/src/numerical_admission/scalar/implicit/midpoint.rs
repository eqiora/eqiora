//! Residual-native midpoint stages use the shared Newton and canonical solver owners.
use super::*;
use crate::nonlinear_execution::{
    NewtonEvidence, NewtonOptions, NewtonTrial, solve_damped_newton, stage_merit,
};

impl<S: Coefficient + Send + Sync> SpatialImplicitSystem<'_, S>
where
    RegionDofMap<S>: RegionAlgebraicCoordinates<f64, S>,
{
    /// An immutable stage attempt. The caller owns clocks, observations and State publication.
    pub(in crate::numerical_admission) fn midpoint_step(
        &self,
        time_s: f64,
        step_s: f64,
        initial: &[f64],
        backend: &dyn LinearSolverBackend,
    ) -> Result<(Vec<f64>, NewtonEvidence), Diagnostic> {
        self.validate(initial)?;
        let midpoint_time = time_s + 0.5 * step_s;
        if !time_s.is_finite()
            || !step_s.is_finite()
            || step_s <= 0.
            || !midpoint_time.is_finite()
            || midpoint_time <= time_s
            || !(time_s + step_s).is_finite()
            || !(1. / step_s).is_finite()
        {
            return Err(invalid(
                "midpoint requires a finite representable forward step",
            ));
        }
        let temporal = self
            .plan
            .admission
            .temporal
            .as_ref()
            .and_then(CommonTemporalPolicy::as_ode)
            .filter(|policy| policy.method() == eqiora_time::TimeMethod::ImplicitMidpoint)
            .ok_or_else(|| invalid("midpoint execution requires the admitted midpoint policy"))?;
        let nonlinear = self.plan.admission.nonlinear;
        let absolute = self.midpoint_tolerances(temporal)?;
        let structure = self.equations.algebraic_structure(None)?;
        let checked = self
            .plan
            .admission
            .linear
            .checked_backend(backend, Some(&structure))?;
        let linear = LinearSolveRequest::new(&checked, self.plan.linear());
        let (accepted, evidence) = solve_damped_newton(
            self.midpoint_seed(initial),
            NewtonOptions::time_stage(
                nonlinear,
                self.plan.linear(),
                temporal.relative_tolerance(),
                &absolute,
            ),
            |endpoint| self.midpoint_point(time_s, step_s, initial, endpoint),
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
            |endpoint, current| {
                let (state, rate) = self.stage_coordinates(initial, endpoint, step_s);
                let mut residual = vec![0.; self.dimension];
                self.residual(midpoint_time, &state, &rate, &mut residual)?;
                Ok(Some(NewtonTrial {
                    // Freeze this iteration's row scaling throughout its line search.
                    residual_norm: stage_merit(
                        &residual,
                        &current.row_scales,
                        nonlinear.is_some(),
                    )?,
                    linearization: None,
                }))
            },
        )?;
        self.require_stage_regularity(time_s, step_s, initial, &accepted.unknown)?;
        Ok((accepted.unknown, evidence))
    }

    pub(super) fn midpoint_point(
        &self,
        time: f64,
        h: f64,
        initial: &[f64],
        endpoint: &[f64],
    ) -> Result<ImplicitPoint, Diagnostic> {
        self.validate(initial)?;
        self.validate(endpoint)?;
        let (state, rate) = self.stage_coordinates(initial, endpoint, h);
        let weights = (0..self.dimension)
            .map(|i| [Some(0.5), (!self.algebraic.contains(&i)).then_some(1. / h)])
            .collect::<Vec<_>>();
        let assembly = self.linearize(
            time + 0.5 * h,
            &state,
            &rate,
            FieldPullback::Coordinates(&weights),
        )?;
        ImplicitPoint::new(endpoint, assembly)
    }

    /// Differential endpoint and twice the algebraic midpoint value. Scaling
    /// both the latter unknown and its absolute tolerance preserves State units.
    pub(super) fn midpoint_seed(&self, initial: &[f64]) -> Vec<f64> {
        initial
            .iter()
            .enumerate()
            .map(|(i, value)| {
                if self.algebraic.contains(&i) {
                    2. * value
                } else {
                    *value
                }
            })
            .collect()
    }

    pub(super) fn midpoint_tolerances(
        &self,
        policy: &CommonOdePolicy,
    ) -> Result<Vec<f64>, Diagnostic> {
        let mut absolute = self.correction_tolerances(policy)?;
        for &i in &self.algebraic {
            absolute[i] *= 2.;
        }
        if absolute.iter().any(|value| !value.is_finite()) {
            return Err(invalid("midpoint algebraic tolerance scaling overflowed"));
        }
        Ok(absolute)
    }

    pub(super) fn stage_coordinates(
        &self,
        initial: &[f64],
        unknown: &[f64],
        h: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let mut state = Vec::with_capacity(self.dimension);
        let mut rate = Vec::with_capacity(self.dimension);
        for (i, (&a, &b)) in initial.iter().zip(unknown).enumerate() {
            if self.algebraic.contains(&i) {
                state.push(0.5 * b);
                rate.push(0.);
            } else {
                state.push(0.5 * a + 0.5 * b);
                rate.push((b - a) / h);
            }
        }
        (state, rate)
    }

    pub(super) fn correction_tolerances(
        &self,
        policy: &CommonOdePolicy,
    ) -> Result<Vec<f64>, Diagnostic> {
        let coordinates = RegionRealCoordinates::new(&self.mapping)?;
        let mut absolute = vec![f64::INFINITY; self.dimension];
        for key in self.mapping.keys() {
            let field_scale = self.mapping.field_scale(key.field)?;
            for (part, global) in coordinates.global_coordinates(key).enumerate() {
                if let Some(free) = coordinates.free_coordinate(global) {
                    let tolerance = policy
                        .absolute_tolerances()
                        .iter()
                        .find(|(coordinate, _)| {
                            coordinate.field().erase() == key.field
                                && coordinate.derivative_order() == 0
                                && coordinate.component() == key.component
                                && coordinate.is_imaginary() == (part == 1)
                        })
                        .ok_or_else(|| {
                            invalid("midpoint tolerance omitted an exact Field coordinate")
                        })?
                        .1
                        / field_scale;
                    // A trace quotient must satisfy every physical Field's tolerance.
                    absolute[free] = absolute[free].min(tolerance);
                }
            }
        }
        if absolute
            .iter()
            .any(|value| !value.is_finite() || *value <= 0.)
        {
            return Err(invalid(
                "midpoint tolerances must cover every finite free coordinate",
            ));
        }
        Ok(absolute)
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn nonzero_scaled_residual_cannot_underflow_into_exact_zero_evidence() {
        assert!(super::stage_merit(&[f64::from_bits(1)], &[1e308], true).is_err());
    }
}
