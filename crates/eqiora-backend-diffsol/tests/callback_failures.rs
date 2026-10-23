#![cfg(feature = "diffsol-runtime")]

use eqiora_backend_diffsol::DiffsolTimeBackend;
use eqiora_core::{Diagnostic, diagnostic::codes};
use eqiora_time::{
    ImplicitDaeInitialization, TimeEquationClass, TimeMethod, TimePlan, TimeProblem, TimeSystem,
};

struct FailingDecay {
    fail_after: f64,
}

fn injected_failure() -> Diagnostic {
    Diagnostic::error(codes::INVALID_REALIZATION, "original model RHS failure")
}

impl TimeSystem for FailingDecay {
    fn dimension(&self) -> usize {
        1
    }

    fn rhs(&self, time: f64, state: &[f64], output: &mut [f64]) -> Result<(), Diagnostic> {
        if time >= self.fail_after {
            return Err(injected_failure());
        }
        output[0] = -state[0];
        Ok(())
    }

    fn rhs_jvp(
        &self,
        _time: f64,
        _state: &[f64],
        direction: &[f64],
        output: &mut [f64],
    ) -> Result<(), Diagnostic> {
        output[0] = -direction[0];
        Ok(())
    }
}

#[test]
fn callback_failure_precedes_native_history_acceptance() {
    for method in [TimeMethod::Tsitouras45, TimeMethod::Bdf] {
        // The first case fails in initialization; the second after valid steps.
        for fail_after in [0.0, 0.25] {
            let system = FailingDecay { fail_after };
            let problem = TimeProblem::new(
                &system,
                TimeEquationClass::ExplicitOde,
                ImplicitDaeInitialization::accepted(vec![1.0], vec![-1.0]).unwrap(),
            )
            .unwrap();
            let plan = TimePlan::new(method, 0.0, 1e-3, 1e-8, vec![1e-10], vec![1.0]).unwrap();
            let error = DiffsolTimeBackend::new()
                .solve(&problem, &plan)
                .unwrap_err();
            assert_eq!(error, injected_failure(), "{method:?}, {fail_after}");
        }
    }
}
