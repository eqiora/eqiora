//! Reaccept physical step coefficients against the ordinary assembled equations.
use super::*;
use eqiora_assembly::REFERENCE_ASSEMBLY_BACKEND;
use eqiora_meshing::{AffineGeometryMap, MeshGeometry};

impl<S: Coefficient + Send + Sync> RegionDofMap<S> {
    fn validated_step_point(
        &self,
        input: &RegionSolveInput<S>,
        fields: &BTreeMap<RawId, RecoveredRegionField<S>>,
    ) -> Result<Vec<S>, Diagnostic> {
        self.validate_physical(fields)?;
        let previous = input
            .previous
            .as_ref()
            .ok_or_else(|| invalid("step replay requires exact previous physical Fields"))?;
        if let Some(step) = super::solve::kinematic_step(&input.forms)? {
            self.validate_step_history(previous, &step)?;
            self.validate_step_history(fields, &step)?;
            self.validate_state_constraints(previous, &step, &input.prescribed_states)?;
            for state in step.eliminated_states() {
                let pair = state.pair();
                for (key, old) in &previous[&pair.state().erase()].coefficients {
                    let rate_key = FieldDof {
                        field: pair.rate().erase(),
                        ..*key
                    };
                    let new = fields[&pair.state().erase()].coefficients[key];
                    let increment = fields[&pair.rate().erase()].coefficients[&rate_key]
                        * step.duration().value();
                    // Same floating-operation bound used by physical volume sampling.
                    let roundoff = (8. * f64::EPSILON * old.abs())
                        + (8. * f64::EPSILON * new.abs())
                        + (8. * f64::EPSILON * increment.abs());
                    if !increment.is_finite() || (new - *old - increment).abs() > roundoff {
                        return Err(invalid("accepted step violates backward-Euler kinematics"));
                    }
                    if input
                        .prescribed_states
                        .get(key)
                        .is_some_and(|expected| *expected != new)
                    {
                        return Err(invalid(
                            "accepted step contradicts its prescribed physical state",
                        ));
                    }
                }
            }
        }
        self.validated_algebraic_point(fields)
    }
}

impl<S: Coefficient + Send + Sync> RegionDofMap<S>
where
    Self: RegionAlgebraicCoordinates<f64, S>,
{
    pub(crate) fn validate_step_coefficients<'mesh>(
        &self,
        mesh: &'mesh impl MeshGeometry<Map<'mesh> = AffineGeometryMap>,
        input: &RegionSolveInput<S>,
        fields: &BTreeMap<RawId, RecoveredRegionField<S>>,
        solver: eqiora_solver::SolverPlan,
        workers: std::num::NonZeroUsize,
    ) -> Result<(f64, f64, usize, usize), Diagnostic> {
        self.validate_solve_input(mesh, input)?;
        let reduced = self.encode_coordinates(&self.validated_step_point(input, fields)?)?;
        let (core, assembly, _) = self.finalize_linear::<f64>(mesh, input, workers, solver)?;
        let (norm, target) = core.validate_coefficients(&reduced)?;
        Ok((
            norm,
            target,
            assembly.packet_count(),
            assembly.target_count(),
        ))
    }
}

impl<S: Coefficient + Send + Sync> RegionDofMap<S>
where
    Self: RegionAlgebraicCoordinates<f64, S>,
{
    pub(crate) fn validate_nonlinear_coefficients<'mesh>(
        &self,
        mesh: &'mesh impl MeshGeometry<Map<'mesh> = AffineGeometryMap>,
        input: &RegionSolveInput<S>,
        fields: &BTreeMap<RawId, RecoveredRegionField<S>>,
        initial: &BTreeMap<RawId, RecoveredRegionField<S>>,
        nonlinear: eqiora_realization::NonlinearSolvePlan,
        iterations: usize,
    ) -> Result<(f64, f64, usize, usize, Option<f64>), Diagnostic> {
        let accepted =
            self.accept_nonlinear_coefficients(mesh, input, fields, initial, nonlinear)?;
        if iterations > nonlinear.maximum_iterations().get() {
            return Err(invalid("Newton iteration history exceeds its Plan limit"));
        }
        accepted
            .initial
            .validate_updates(accepted.residual_target, iterations)?;
        Ok((
            accepted.residual_norm,
            accepted.residual_target,
            accepted.assembly.report.packet_count(),
            accepted.assembly.report.target_count(),
            Some(accepted.initial.residual_norm),
        ))
    }

    /// Mathematical acceptance shared by Result replay and implicit derivatives.
    /// This derives no producer history and executes no solve.
    pub(crate) fn accept_nonlinear_coefficients<'mesh>(
        &self,
        mesh: &'mesh impl MeshGeometry<Map<'mesh> = AffineGeometryMap>,
        input: &RegionSolveInput<S>,
        fields: &BTreeMap<RawId, RecoveredRegionField<S>>,
        initial: &BTreeMap<RawId, RecoveredRegionField<S>>,
        nonlinear: eqiora_realization::NonlinearSolvePlan,
    ) -> Result<AcceptedNonlinearPoint, Diagnostic> {
        self.validate_solve_input(mesh, input)?;
        let reduced = if input.previous.is_some() {
            self.encode_coordinates(&self.validated_step_point(input, fields)?)?
        } else {
            self.validated_coordinates(fields)?
        };
        let seed = self.nonlinear_predictor(input, initial)?;
        let prepared = self.prepare_numeric_assembly(mesh, input)?;
        let target = prepared.plan.target_id(0).expect("reduced target");
        let norm = |point: &[f64]| -> Result<f64, Diagnostic> {
            super::super::newton::residual_norm(&prepared.work.residual(
                &prepared.plan,
                target,
                point,
            )?)
        };
        let initial = norm(&seed)?;
        let residual_target = nonlinear
            .absolute_tolerance()
            .max(nonlinear.relative_tolerance() * initial);
        let residual = norm(&reduced)?;
        if residual > residual_target {
            return Err(invalid(
                "supplied Region coefficients violate the original nonlinear residual",
            ));
        }
        let action = prepared.work.linearize(
            &prepared.plan,
            target,
            &reduced,
            &REFERENCE_ASSEMBLY_BACKEND,
        )?;
        let multipliers = input
            .forms
            .iter()
            .flat_map(|(form, _)| form.constraint_multipliers())
            .collect();
        self.require_numeric_multiplier_closure(action.systems[0].matrix(), &multipliers)?;
        // Both ordinary residual evaluation and the retained differentiated action
        // must satisfy the original target; neither is replaced by J*x-b.
        let assembled_norm = super::super::newton::residual_norm(&action.residuals[0])?;
        if assembled_norm > residual_target {
            return Err(invalid(
                "supplied Region linearization violates the original nonlinear residual",
            ));
        }
        let at_seed = seed == reduced;
        Ok(AcceptedNonlinearPoint {
            values: reduced,
            residual_norm: residual,
            residual_target,
            initial: crate::nonlinear_execution::NewtonInitialPoint {
                residual_norm: initial,
                at_seed,
            },
            assembly: action,
        })
    }
}

pub(crate) struct AcceptedNonlinearPoint {
    pub(crate) values: Vec<f64>,
    pub(crate) assembly: crate::region_assembly::nonlinear::RegionAssemblyLinearization,
    pub(crate) residual_norm: f64,
    pub(crate) residual_target: f64,
    pub(crate) initial: crate::nonlinear_execution::NewtonInitialPoint,
}
