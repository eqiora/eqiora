//! Scalar-domain conversion and actual primal acceptance, independent of derivative provenance.
use super::*;
use eqiora_solver::CanonicalCsrSystemView;
use num_complex::Complex64;
use std::sync::Arc;

pub(super) trait DifferentiationScalar: Coefficient + ResidualScalar + Send {
    const WIDTH: usize;
    fn from_coordinates(values: &[f64]) -> Result<Vec<Self>, Diagnostic>;
    fn real_system(
        core: &FinalizedLinearCore<Self>,
    ) -> Result<Arc<CanonicalCsrSystemView<f64>>, Diagnostic>;
    fn solve(
        plan: &CommonSpatialPlan,
        equations: &ExecutableSpatialEquations<Self>,
        core: &FinalizedLinearCore<Self>,
    ) -> Result<(Vec<Self>, Primal), Diagnostic>;
}

pub(super) fn coordinates<S: DifferentiationScalar>(
    values: impl IntoIterator<Item = S>,
) -> impl Iterator<Item = f64> {
    values
        .into_iter()
        .flat_map(|value| [value.re(), value.im()].into_iter().take(S::WIDTH))
}

impl DifferentiationScalar for f64 {
    const WIDTH: usize = 1;
    fn from_coordinates(values: &[f64]) -> Result<Vec<Self>, Diagnostic> {
        Ok(values.to_vec())
    }
    fn real_system(
        core: &FinalizedLinearCore<Self>,
    ) -> Result<Arc<CanonicalCsrSystemView<f64>>, Diagnostic> {
        Ok(core.shared_canonical_system())
    }
    fn solve(
        plan: &CommonSpatialPlan,
        _: &ExecutableSpatialEquations<Self>,
        core: &FinalizedLinearCore<Self>,
    ) -> Result<(Vec<Self>, Primal), Diagnostic> {
        let linear = &plan.admission.linear;
        let executor = HostExecutorDescriptor::new(
            linear.provider,
            linear.execution,
            linear.workers,
            linear.capabilities.clone(),
        );
        let binding = DeploymentBinding::bind_host(&plan.portable, executor)?;
        let admitted = AdmittedExecution::admit_host_linear(
            &plan.portable,
            core.canonical_csr_system_view(),
            binding,
        )?;
        let produced =
            REFERENCE_LINEAR_SOLVER.solve(&core.linear_problem()?, core.solver_plan())?;
        let (solution, receipt) = admitted.accept(produced)?.into_parts();
        core.validate_solution(&solution)?;
        Ok((solution.into_parts().0, Primal::Linear(Box::new(receipt))))
    }
}

impl DifferentiationScalar for Complex64 {
    const WIDTH: usize = 2;
    fn from_coordinates(values: &[f64]) -> Result<Vec<Self>, Diagnostic> {
        let (pairs, remainder) = values.as_chunks::<2>();
        if !remainder.is_empty() {
            return Err(invalid(
                "complex state requires complete real/imaginary coordinates",
            ));
        }
        Ok(pairs
            .iter()
            .map(|pair| Self::new(pair[0], pair[1]))
            .collect())
    }
    fn real_system(
        core: &FinalizedLinearCore<Self>,
    ) -> Result<Arc<CanonicalCsrSystemView<f64>>, Diagnostic> {
        let source = core.canonical_csr_system_view();
        source
            .real_coordinate_system(
                eqiora_solver::LinearOperatorOrientation::Normal,
                source.right_hand_side(),
            )
            .map(Arc::new)
    }
    fn solve(
        plan: &CommonSpatialPlan,
        equations: &ExecutableSpatialEquations<Self>,
        core: &FinalizedLinearCore<Self>,
    ) -> Result<(Vec<Self>, Primal), Diagnostic> {
        let structure = equations.algebraic_structure(None)?;
        let checked = plan
            .admission
            .linear
            .checked_complex_backend(&REFERENCE_LINEAR_SOLVER, Some(&structure))?;
        let solution =
            LinearSolveRequest::new(&checked, core.solver_plan()).solve(&core.linear_problem()?)?;
        core.validate_solution(&solution)?;
        let (values, report) = solution.into_parts();
        // Retain the genuine complex solve report. No real-system execution receipt
        // was produced by this solve or by the subsequent coordinate conversion.
        Ok((values, Primal::Affine(report)))
    }
}
