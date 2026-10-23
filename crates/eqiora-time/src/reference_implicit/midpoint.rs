//! Bounded reference stage solve; production midpoint owns no implicit solver choice.
use super::*;
use crate::midpoint::midpoint;
use crate::{MassMatrixRank, TimeProblem};

impl ReferenceImplicitTimeBackend {
    /// Bounded midpoint reference oracle for time-contract tests.
    ///
    /// Uses the reference Newton iteration budgets and dense correction solve.
    /// Production Runs supply their admitted stage solver to the shared midpoint
    /// clock/history owner instead. This oracle accepts only provided ODE or
    /// full-rank mass initial data; it does not admit a production solve policy.
    ///
    /// # Errors
    /// Rejects unsupported profiles, nonfinite callbacks and failed reference solves.
    pub fn solve_midpoint(
        &self,
        problem: &TimeProblem<'_>,
        plan: &TimePlan,
    ) -> Result<TimeSolution, Diagnostic> {
        plan.validate_for(problem)?;
        if plan.method() != TimeMethod::ImplicitMidpoint {
            return Err(invalid_plan(
                "implicit-midpoint backend requires the ImplicitMidpoint method",
            ));
        }
        if !matches!(
            problem.equation_class(),
            TimeEquationClass::ExplicitOde
                | TimeEquationClass::MassMatrix {
                    rank: MassMatrixRank::Full
                }
        ) || problem.initial_condition() != InitialConditionPolicy::Provided
        {
            return Err(invalid_plan(
                "implicit midpoint requires an ODE with a provided consistent state",
            ));
        }
        crate::ImplicitMidpointTimeBackend::new().solve_with_stages(
            problem.initial_state(),
            plan,
            problem.equation_class(),
            |time, next_time, state| {
                let n = problem.dimension();
                let step = next_time - time;
                let middle_time = time + step / 2.;
                let mut candidate = state.to_vec();
                crate::reference_implicit::newton(
                    &mut candidate,
                    plan.absolute_tolerances(),
                    plan.relative_tolerance(),
                    |next, residual| {
                        let middle = midpoint(state, next);
                        let difference = next
                            .iter()
                            .zip(state)
                            .map(|(b, a)| b - a)
                            .collect::<Vec<_>>();
                        mass(problem, middle_time, &difference, residual)?;
                        let mut rhs = vec![0.; n];
                        problem.system().rhs(middle_time, &middle, &mut rhs)?;
                        for (value, rhs) in residual.iter_mut().zip(rhs) {
                            *value -= step * rhs;
                        }
                        Ok(())
                    },
                    |next, direction, output| {
                        mass(problem, middle_time, direction, output)?;
                        let mut rhs = vec![0.; n];
                        problem.system().rhs_jvp(
                            middle_time,
                            &midpoint(state, next),
                            direction,
                            &mut rhs,
                        )?;
                        for (value, rhs) in output.iter_mut().zip(rhs) {
                            *value -= step / 2. * rhs;
                        }
                        Ok(())
                    },
                )?;
                Ok(candidate)
            },
        )
    }
}
fn mass(
    problem: &TimeProblem<'_>,
    time: f64,
    direction: &[f64],
    output: &mut [f64],
) -> Result<(), Diagnostic> {
    if problem.equation_class() == TimeEquationClass::ExplicitOde {
        output.copy_from_slice(direction);
        Ok(())
    } else {
        problem.system().mass_action(time, direction, output)
    }
}
