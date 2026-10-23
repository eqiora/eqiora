//! Transfer the common accepted initial pair into either native method state.
use super::*;
use diffsol::{OdeEquations, OdeSolverProblem};

pub(super) fn state<E, S>(
    ode: &OdeSolverProblem<E>,
    problem: &TimeProblem<'_>,
    sensitivity: Option<&ForwardSensitivityProblem<'_>>,
    method: TimeMethod,
    failures: &CallbackFailures,
) -> Result<S, Diagnostic>
where
    E: OdeEquations<T = f64, V = NalgebraVec<f64>, C = diffsol::NalgebraContext>,
    S: OdeSolverState<NalgebraVec<f64>>,
{
    let mut state = S::new_without_initialise(ode)
        .map_err(|error| map_failure(failures, "construct accepted Diffsol initial state", error))?
        .into_common();
    failures.check()?;
    copy_initial_state(problem.initial_derivative(), &mut state.dy);
    if let Some(sensitivity) = sensitivity {
        let mut direction = vec![0.0; sensitivity.parameter_dimension()];
        for parameter in 0..sensitivity.parameter_dimension() {
            direction[parameter] = 1.0;
            let tangent = sensitivity.initialization_parameter_jvp(ode.t0, &direction)?;
            let mut values = state.y.clone();
            let mut rates = state.dy.clone();
            copy_initial_state(tangent.state(), &mut values);
            copy_initial_state(tangent.derivative(), &mut rates);
            state.s.push(values);
            state.ds.push(rates);
            direction[parameter] = 0.0;
        }
    }
    let mut state = S::new_from_common(state);
    // The native startup estimate evaluates f as a physical rate, which is
    // valid only after explicit-ODE normalization. Preserve that controller
    // choice for ODEs; mass systems retain the requested initial step.
    if problem.equation_class() == TimeEquationClass::ExplicitOde {
        let order = match method {
            TimeMethod::Bdf => 1,
            TimeMethod::Tsitouras45 => {
                diffsol::Tableau::<NalgebraMat<f64>>::tsit45(Default::default()).order()
            }
            _ => unreachable!("native method admission precedes initialization"),
        };
        state
            .as_mut()
            .set_step_size(ode.h0, &ode.atol, ode.rtol, &ode.eqn, order);
        failures.check()?;
    }
    Ok(state)
}
