//! One private provider session for the accepted actions of one scalar Run.
use super::*;

pub(super) struct PreparedScalarExecution<'a> {
    plan: &'a CommonSpatialPlan,
    backend: &'a dyn LinearSolverBackend,
    linear: Option<Box<dyn eqiora_solver::PreparedLinearSolver>>,
}

impl<'a> PreparedScalarExecution<'a> {
    pub(super) fn new(
        plan: &'a CommonSpatialPlan,
        state: &CommonState,
        backend: &'a dyn LinearSolverBackend,
    ) -> Result<Self, Diagnostic> {
        plan.reauthenticate_portable_realization()?;
        plan.admission.revalidate()?;
        if state.state_space_identity() != plan.identity() {
            return Err(invalid(
                "State belongs to a different exact common state space",
            ));
        }
        let linear = {
            let structure = match plan.admission.recognized_model() {
                RecognizedNativeModel::Spatial(equations) => equations.algebraic_structure(None)?,
                RecognizedNativeModel::ComplexSpatial(equations) => {
                    equations.algebraic_structure(None)?
                }
                _ => return Err(invalid("missing spatial inventory")),
            };
            plan.admission
                .linear
                .checked_backend(backend, Some(&structure))?
                .prepare_linear(plan.admission.linear.solver)?
        };
        Ok(Self {
            plan,
            backend,
            linear,
        })
    }

    pub(super) fn advance(
        &mut self,
        state: &CommonState,
    ) -> Result<CommonAcceptedSpatialStep, Diagnostic> {
        if let Some(linear) = &mut self.linear {
            linear.begin_action()?;
        }
        let result = self.plan.advance_scalar_prepared(
            state,
            self.backend,
            self.linear
                .as_mut()
                .map(|prepared| prepared.as_mut() as &mut dyn eqiora_solver::PreparedLinearSolver),
        );
        match result {
            Ok(accepted) => {
                if let Some(linear) = &mut self.linear {
                    if let Err(error) = linear.commit_action() {
                        let _ = linear.discard_action();
                        return Err(error);
                    }
                }
                Ok(accepted)
            }
            Err(error) => {
                if let Some(linear) = &mut self.linear {
                    let _ = linear.discard_action();
                }
                Err(error)
            }
        }
    }
}
