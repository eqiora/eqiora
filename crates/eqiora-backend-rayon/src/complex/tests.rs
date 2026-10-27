use super::*;

#[test]
fn typed_reductions_conjugate_first_and_share_the_exact_real_execution() {
    for count in [511, 512, 513, 1025] {
        let left = vec![Complex64::new(1., 1.); count];
        let right = vec![Complex64::new(2., -1.); count];
        let action = FixedOrderInnerProduct::new(&left, &right).unwrap();
        // conj(1+i)*(2-i)=1-3i, including both sides of partial boundaries.
        let expected = Complex64::new(count as f64, -3. * count as f64);
        for workers in [NonZeroUsize::MIN, NonZeroUsize::new(4).unwrap()] {
            let pool = CpuThreadPool::new(workers).unwrap();
            let execution = RayonLinearExecution { pool: &pool };
            assert_eq!(execution.inner_product(action).unwrap(), expected);
            assert_eq!(
                execution.inner_product(action).unwrap(),
                SERIAL_LINEAR_EXECUTION.inner_product(action).unwrap()
            );
            let real =
                <RayonLinearExecution<'_> as ReplicatedLinearExecution<Complex64>>::real_execution(
                    &execution,
                )
                .unwrap();
            assert_eq!(real.provider(), execution.provider());
            assert_eq!(real.report(), execution.report());
            let packed = left.iter().flat_map(|x| [x.re, x.im]).collect::<Vec<_>>();
            let complex_norm = FixedOrderInnerProduct::norm(&execution, &left).unwrap();
            assert_eq!(
                complex_norm,
                FixedOrderInnerProduct::norm(real, &packed).unwrap()
            );
            assert_eq!(
                complex_norm,
                FixedOrderInnerProduct::norm(&SERIAL_LINEAR_EXECUTION, &left).unwrap()
            );
        }
    }
}

#[derive(Debug)]
struct StaleSerial;
impl<S> ReplicatedLinearExecution<S> for StaleSerial {
    fn provider(&self) -> ExecutionProvider {
        ExecutionProvider::new(SERIAL_LINEAR_EXECUTION.provider().id(), "stale", &[])
    }
    fn report(&self) -> ExecutionReport {
        SERIAL_LINEAR_EXECUTION.report()
    }
    fn require_reduction(&self, _: ReductionPolicy) -> Result<(), Diagnostic> {
        panic!("entry must reject before work")
    }
    fn apply(
        &self,
        _: &dyn LinearOperator<Scalar = S>,
        _: &[S],
        _: &mut [S],
    ) -> Result<(), Diagnostic> {
        panic!("entry must reject before work")
    }
    fn inner_product(&self, _: FixedOrderInnerProduct<'_, S>) -> Result<S, Diagnostic> {
        panic!("entry must reject before work")
    }
}

#[derive(Debug)]
struct Identity<S>(std::marker::PhantomData<S>);
impl<S: Copy + fmt::Debug + Sync> LinearOperator for Identity<S> {
    type Scalar = S;
    fn rows(&self) -> usize {
        1
    }
    fn columns(&self) -> usize {
        1
    }
    fn apply(&self, input: &[S], output: &mut [S]) -> Result<(), Diagnostic> {
        output.copy_from_slice(input);
        Ok(())
    }
}

#[test]
fn typed_solver_entries_reject_same_id_stale_execution_before_work() {
    use eqiora_solver::{LinearOperatorProperties, LinearSolver, REFERENCE_LINEAR_SOLVER};
    let workers = NonZeroUsize::new(2).unwrap();
    let pool = CpuThreadPool::new(workers).unwrap();
    let backend = pool
        .solver(
            Target::HostCpu { threads: workers },
            &REFERENCE_LINEAR_SOLVER,
        )
        .unwrap();
    let plan = SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        0.,
        1e-12,
        NonZeroUsize::MIN,
    )
    .unwrap();
    let real_operator = Identity::<f64>(std::marker::PhantomData);
    let real =
        LinearProblem::new(&real_operator, &[1.], LinearOperatorProperties::General).unwrap();
    let complex_operator = Identity::<Complex64>(std::marker::PhantomData);
    let rhs = [Complex64::new(1., 0.)];
    let complex =
        LinearProblem::new(&complex_operator, &rhs, LinearOperatorProperties::General).unwrap();
    for error in [
        backend
            .solve_with_execution(&real, plan, &StaleSerial)
            .unwrap_err(),
        backend
            .solve_with_execution(&complex, plan, &StaleSerial)
            .unwrap_err(),
    ] {
        assert!(
            error
                .message()
                .contains("nested inside another replicated execution")
        );
    }
}
