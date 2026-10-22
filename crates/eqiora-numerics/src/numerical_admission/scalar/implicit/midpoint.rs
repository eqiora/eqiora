//! Residual-native midpoint stages use the shared Newton and canonical solver owners.
use super::*;
use crate::nonlinear_execution::{
    NewtonEvidence, NewtonOptions, NewtonStopping, NewtonTrial, solve_damped_newton,
};
use crate::region_assembly::nonlinear::RegionAssemblyLinearization;

struct MidpointPoint {
    endpoint: Vec<f64>,
    assembly: RegionAssemblyLinearization,
    row_scales: Vec<f64>,
}

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
        let nonlinear = self.plan.admission.nonlinear.ok_or_else(|| {
            invalid("implicit spatial stages require declared Newton iteration bounds")
        })?;
        let absolute = self.correction_tolerances(temporal)?;
        let structure = self.equations.algebraic_structure(None)?;
        let checked = self
            .plan
            .admission
            .linear
            .checked_backend(backend, Some(&structure))?;
        let linear = LinearSolveRequest::new(&checked, self.plan.linear());
        let stage = |endpoint: &[f64]| {
            let state = initial
                .iter()
                .zip(endpoint)
                .map(|(old, next)| 0.5 * old + 0.5 * next)
                .collect::<Vec<_>>();
            let rate = initial
                .iter()
                .zip(endpoint)
                .map(|(old, next)| (next - old) / step_s)
                .collect::<Vec<_>>();
            (state, rate)
        };
        let (accepted, evidence) = solve_damped_newton(
            initial.to_vec(),
            NewtonOptions {
                maximum_iterations: nonlinear.maximum_iterations(),
                maximum_line_search_steps: nonlinear.maximum_line_search_steps(),
                stopping: NewtonStopping::Correction {
                    relative: temporal.relative_tolerance(),
                    absolute: &absolute,
                },
            },
            |endpoint| {
                let (state, rate) = stage(endpoint);
                let assembly = self.linearize(
                    midpoint_time,
                    &state,
                    &rate,
                    FieldPullback::Uniform {
                        current: Some(0.5),
                        rate: Some(1. / step_s),
                    },
                )?;
                let matrix = assembly.systems[0].matrix();
                let row_scales = matrix
                    .row_offsets()
                    .windows(2)
                    .map(|offsets| {
                        matrix.values()[offsets[0]..offsets[1]]
                            .iter()
                            .fold(0_f64, |scale, value| scale.max(value.abs()))
                    })
                    .collect::<Vec<_>>();
                if row_scales
                    .iter()
                    .any(|scale| !scale.is_finite() || *scale <= 0.)
                {
                    return Err(invalid(
                        "midpoint Jacobian requires finite nonzero equation rows",
                    ));
                }
                Ok(MidpointPoint {
                    endpoint: endpoint.to_vec(),
                    assembly,
                    row_scales,
                })
            },
            |point| merit(&point.assembly.residuals[0], &point.row_scales),
            |_, point| {
                point
                    .assembly
                    .solve_correction(0, linear, self.plan.admission.linear.workers)
            },
            |endpoint, current| {
                let (state, rate) = stage(endpoint);
                let mut residual = vec![0.; self.dimension];
                self.residual(midpoint_time, &state, &rate, &mut residual)?;
                Ok(Some(NewtonTrial {
                    // Freeze this iteration's row scaling throughout its line search.
                    residual_norm: merit(&residual, &current.row_scales)?,
                    linearization: None,
                }))
            },
        )?;
        Ok((accepted.endpoint, evidence))
    }

    fn correction_tolerances(&self, policy: &CommonOdePolicy) -> Result<Vec<f64>, Diagnostic> {
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

fn merit(residual: &[f64], scales: &[f64]) -> Result<f64, Diagnostic> {
    if residual.len() != scales.len() {
        return Err(invalid("midpoint residual and equation scaling differ"));
    }
    let norm = residual
        .iter()
        .zip(scales)
        .fold(0_f64, |norm, (value, scale)| norm.hypot(value / scale));
    if norm.is_finite() {
        Ok(norm)
    } else {
        Err(invalid("midpoint scaled residual is non-finite"))
    }
}
