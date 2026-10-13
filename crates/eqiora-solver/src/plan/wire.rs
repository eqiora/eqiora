//! The one closed representation of linear-solve controls.

use super::*;

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct WireSolverPlan {
    algorithm: LinearSolver,
    preconditioner: PreconditionerPolicy,
    reduction: ReductionPolicy,
    relative_tolerance: f64,
    absolute_tolerance: f64,
    maximum_iterations: NonZeroUsize,
}

impl From<SolverPlan> for WireSolverPlan {
    fn from(value: SolverPlan) -> Self {
        let canonical_zero = |value: f64| if value == 0.0 { 0.0 } else { value };
        Self {
            algorithm: value.algorithm,
            preconditioner: value.preconditioner,
            reduction: value.reduction,
            relative_tolerance: canonical_zero(value.relative_tolerance),
            absolute_tolerance: canonical_zero(value.absolute_tolerance),
            maximum_iterations: value.maximum_iterations,
        }
    }
}

impl TryFrom<WireSolverPlan> for SolverPlan {
    type Error = Diagnostic;

    fn try_from(value: WireSolverPlan) -> Result<Self, Self::Error> {
        Self::new(
            value.algorithm,
            value.relative_tolerance,
            value.absolute_tolerance,
            value.maximum_iterations,
        )
        .map(|plan| {
            plan.with_preconditioner(value.preconditioner)
                .with_reduction(value.reduction)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WIRE: &str = r#"{"algorithm":"minimum-residual","preconditioner":"jacobi","reduction":"fast","relative_tolerance":0.125,"absolute_tolerance":0.0,"maximum_iterations":7}"#;

    #[test]
    fn current_solver_controls_have_one_closed_wire() {
        for (algorithm, tag) in [
            (LinearSolver::ConjugateGradient, "conjugate-gradient"),
            (LinearSolver::MinimumResidual, "minimum-residual"),
            (
                LinearSolver::BiConjugateGradientStabilized,
                "bi-conjugate-gradient-stabilized",
            ),
            (LinearSolver::SparseLu, "sparse-lu"),
        ] {
            let plan = SolverPlan::new(algorithm, 0.125, -0.0, NonZeroUsize::new(7).unwrap())
                .unwrap()
                .with_preconditioner(PreconditionerPolicy::Jacobi)
                .with_reduction(ReductionPolicy::Fast);
            let expected = WIRE.replace("minimum-residual", tag);
            assert_eq!(serde_json::to_string(&plan).unwrap(), expected);
            assert_eq!(serde_json::from_str::<SolverPlan>(&expected).unwrap(), plan);
        }
        let defaults = SolverPlan::new(
            LinearSolver::MinimumResidual,
            0.125,
            0.0,
            NonZeroUsize::new(7).unwrap(),
        )
        .unwrap();
        assert_eq!(
            serde_json::to_string(&defaults).unwrap(),
            WIRE.replace("jacobi", "identity")
                .replace("fast", "reproducible")
        );
    }

    #[test]
    fn wire_cannot_bypass_tolerance_count_or_vocabulary_admission() {
        for (before, after) in [
            ("0.125", "-0.125"),
            ("0.125", "0.0"),
            ("0.125", "1e400"),
            ("0.125", "null"),
            ("0.0", "-1.0"),
            ("7", "0"),
            ("7", "-1"),
            ("7", "1.5"),
            ("7", "18446744073709551616"),
            ("minimum-residual", "unknown"),
            ("jacobi", "unknown"),
            ("fast", "unknown"),
            (
                "\"maximum_iterations\":7",
                "\"maximum_iterations\":7,\"extra\":1",
            ),
            (
                "\"maximum_iterations\":7",
                "\"maximum_iterations\":7,\"maximum_iterations\":8",
            ),
            (",\"maximum_iterations\":7", ""),
        ] {
            let changed = WIRE.replace(before, after);
            assert_ne!(changed, WIRE);
            assert!(
                serde_json::from_str::<SolverPlan>(&changed).is_err(),
                "{changed}"
            );
        }
    }
}
