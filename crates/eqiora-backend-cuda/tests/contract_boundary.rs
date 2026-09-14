use eqiora_backend_cuda::CUDA_RUNTIME_ID;

#[test]
fn runtime_identity_exists_without_linking_or_loading_cuda() {
    assert_eq!(CUDA_RUNTIME_ID.as_str(), "eqiora.cuda.cudarc");
}

#[cfg(feature = "cuda-runtime")]
mod cuda_runtime {
    use eqiora_core::ScalarType;
    use std::num::NonZeroUsize;

    use eqiora_assembly::{AssemblyMap, CooAssembler, DofId, LocalContribution, LocalUnknown};
    use eqiora_backend_cuda::{
        CUDA_ADAPTER_VERSION, CUDA_BINDING_TOOLKIT, CUDARC_VERSION, CudaLinearSolveResult,
        CudaLinearSolver, CudaResidentCsrActionSession, CudaRuntime, verify_csr_action,
        verify_csr_action_against,
    };
    use eqiora_core::diagnostic::codes;
    use eqiora_device::{
        DeviceCapability, DeviceDescriptor, DeviceId, MemoryRegion, QueueSlot, RuntimeId,
        SparseActionPolicy, SparseActionTolerance,
    };
    use eqiora_solver::{
        CanonicalCsrSystemView, ExecutionReport, LinearOperatorProperties, LinearSolver,
        PreconditionerPolicy, ReductionPolicy, SolverPlan,
    };

    fn matrix() -> eqiora_assembly::CsrMatrix {
        system(vec![4.0, -1.0, -1.0, 3.0], vec![0.0, 0.0])
            .matrix()
            .clone()
    }

    fn system(matrix: Vec<f64>, rhs: Vec<f64>) -> eqiora_assembly::LinearSystem {
        let local = LocalContribution::new(2, 2, matrix, rhs).unwrap();
        let map = AssemblyMap::new(
            vec![Some(DofId::new(0)), Some(DofId::new(1))],
            vec![
                LocalUnknown::Free(DofId::new(0)),
                LocalUnknown::Free(DofId::new(1)),
            ],
        )
        .unwrap();
        let mut assembler = CooAssembler::new(2).unwrap();
        assembler.scatter(&map, &local).unwrap();
        assembler.finish().unwrap()
    }

    fn canonical_system(
        matrix: Vec<f64>,
        rhs: Vec<f64>,
        properties: LinearOperatorProperties,
    ) -> CanonicalCsrSystemView {
        CanonicalCsrSystemView::new(&system(matrix, rhs), properties).unwrap()
    }

    fn plan(algorithm: LinearSolver) -> SolverPlan {
        SolverPlan::new(algorithm, 1.0e-12, 1.0e-14, NonZeroUsize::new(32).unwrap()).unwrap()
    }

    #[test]
    fn invalid_input_is_rejected_before_runtime_loading() {
        let error = verify_csr_action(
            &matrix(),
            &[1.0],
            0,
            SparseActionPolicy::Deterministic,
            SparseActionTolerance::new(1.0e-12, 1.0e-12).unwrap(),
        )
        .unwrap_err();
        assert!(error.message().contains("input has 1 values"));
    }

    #[test]
    fn invalid_explicit_reference_is_rejected_before_runtime_loading() {
        let tolerance = SparseActionTolerance::new(1.0e-12, 1.0e-12).unwrap();
        let wrong_shape = verify_csr_action_against(
            &matrix(),
            &[1.0, 2.0],
            &[2.0],
            0,
            SparseActionPolicy::Deterministic,
            tolerance,
        )
        .unwrap_err();
        assert!(wrong_shape.message().contains("reference has 1 values"));

        let non_finite = verify_csr_action_against(
            &matrix(),
            &[1.0, 2.0],
            &[2.0, f64::NAN],
            0,
            SparseActionPolicy::Deterministic,
            tolerance,
        )
        .unwrap_err();
        assert!(non_finite.message().contains("reference requires finite"));
    }

    #[test]
    fn unsupported_reduction_is_rejected_before_runtime_loading() {
        let problem = canonical_system(
            vec![4.0, -1.0, -1.0, 3.0],
            vec![2.0, 5.0],
            LinearOperatorProperties::SymmetricPositiveDefinite,
        );
        let error = CudaLinearSolver::new(0)
            .solve(&problem, None, plan(LinearSolver::ConjugateGradient))
            .unwrap_err();
        assert_eq!(error.code(), codes::INVALID_REALIZATION);
        assert!(error.message().contains("ConjugateGradient"));
        assert!(error.message().contains("Reproducible"));
        assert!(error.message().contains("exact"));
    }

    #[test]
    fn minres_symmetric_indefinite_fast_identity_is_an_exact_capability() {
        let supported = plan(LinearSolver::MinimumResidual)
            .with_preconditioner(PreconditionerPolicy::Identity)
            .with_reduction(ReductionPolicy::Fast);
        CudaLinearSolver::capabilities()
            .require_problem(
                supported,
                ScalarType::F64,
                LinearOperatorProperties::SymmetricIndefinite,
            )
            .unwrap();
    }

    #[test]
    fn independent_cuda_axes_do_not_create_unverified_solver_tuples() {
        for (plan, properties, missing_policy) in [
            (
                plan(LinearSolver::ConjugateGradient)
                    .with_preconditioner(PreconditionerPolicy::Identity)
                    .with_reduction(ReductionPolicy::Fast),
                LinearOperatorProperties::SymmetricPositiveDefinite,
                "Identity",
            ),
            (
                plan(LinearSolver::BiConjugateGradientStabilized)
                    .with_preconditioner(PreconditionerPolicy::Jacobi)
                    .with_reduction(ReductionPolicy::Fast),
                LinearOperatorProperties::General,
                "Jacobi",
            ),
            (
                plan(LinearSolver::MinimumResidual)
                    .with_preconditioner(PreconditionerPolicy::Jacobi)
                    .with_reduction(ReductionPolicy::Fast),
                LinearOperatorProperties::SymmetricIndefinite,
                "Jacobi",
            ),
            (
                plan(LinearSolver::MinimumResidual)
                    .with_preconditioner(PreconditionerPolicy::Identity)
                    .with_reduction(ReductionPolicy::Reproducible),
                LinearOperatorProperties::SymmetricIndefinite,
                "Reproducible",
            ),
            (
                plan(LinearSolver::MinimumResidual)
                    .with_preconditioner(PreconditionerPolicy::Identity)
                    .with_reduction(ReductionPolicy::Fast),
                LinearOperatorProperties::SymmetricPositiveDefinite,
                "SymmetricPositiveDefinite",
            ),
        ] {
            let error = CudaLinearSolver::capabilities()
                .require_problem(plan, ScalarType::F64, properties)
                .unwrap_err();
            assert_eq!(error.code(), codes::INVALID_REALIZATION);
            assert!(error.message().contains(missing_policy));
            assert!(error.message().contains("exact"));
        }
    }

    #[test]
    fn unsupported_device_is_rejected_before_context_or_allocation() {
        let descriptor = DeviceDescriptor::new(
            DeviceId::new(eqiora_backend_cuda::CUDA_RUNTIME_ID, 7),
            "contract-only device",
            std::num::NonZeroU64::new(1024).unwrap(),
            [
                DeviceCapability::Float32,
                DeviceCapability::CsrMatrixVectorProduct,
                DeviceCapability::AsynchronousQueue,
            ],
        )
        .unwrap();
        let error = CudaLinearSolver::new(7)
            .admit_device(&descriptor)
            .unwrap_err();
        assert!(error.message().contains("Float64"));
    }

    #[test]
    fn device_admission_rejects_runtime_and_ordinal_before_capabilities() {
        let capabilities = [
            DeviceCapability::Float64,
            DeviceCapability::CsrMatrixVectorProduct,
            DeviceCapability::DenseVectorLevel1,
            DeviceCapability::AsynchronousQueue,
        ];
        let wrong_runtime = DeviceDescriptor::new(
            DeviceId::new(RuntimeId::new("test.other-runtime"), 7),
            "wrong runtime",
            std::num::NonZeroU64::new(1024).unwrap(),
            capabilities,
        )
        .unwrap();
        let error = CudaLinearSolver::new(7)
            .admit_device(&wrong_runtime)
            .unwrap_err();
        assert!(error.message().contains("test.other-runtime"));

        let wrong_ordinal = DeviceDescriptor::new(
            DeviceId::new(eqiora_backend_cuda::CUDA_RUNTIME_ID, 8),
            "wrong ordinal",
            std::num::NonZeroU64::new(1024).unwrap(),
            capabilities,
        )
        .unwrap();
        let error = CudaLinearSolver::new(7)
            .admit_device(&wrong_ordinal)
            .unwrap_err();
        assert!(error.message().contains("selected device 7"));
        assert!(error.message().contains("device 8"));
    }

    #[test]
    fn facade_versions_and_consuming_result_shape_are_compile_time_stable() {
        assert_eq!(CUDA_ADAPTER_VERSION, env!("CARGO_PKG_VERSION"));
        assert_eq!(CUDARC_VERSION, "0.18.2");
        assert_eq!(CUDA_BINDING_TOOLKIT, "12.0");
        let into_parts: fn(CudaLinearSolveResult) -> _ = CudaLinearSolveResult::into_parts;
        let _ = into_parts;
    }

    #[test]
    fn cg_property_is_rejected_before_runtime_loading() {
        let problem = canonical_system(
            vec![4.0, -1.0, -1.0, 3.0],
            vec![2.0, 5.0],
            LinearOperatorProperties::General,
        );
        let error = CudaLinearSolver::new(0)
            .solve(
                &problem,
                None,
                plan(LinearSolver::ConjugateGradient).with_reduction(ReductionPolicy::Fast),
            )
            .unwrap_err();
        assert_eq!(error.code(), codes::INVALID_REALIZATION);
        assert!(error.message().contains("ConjugateGradient"));
        assert!(error.message().contains("General"));
        assert!(error.message().contains("exact"));
    }

    #[test]
    fn invalid_jacobi_diagonal_is_rejected_before_runtime_loading() {
        let problem = canonical_system(
            vec![0.0, 1.0, 1.0, 2.0],
            vec![2.0, 5.0],
            LinearOperatorProperties::SymmetricPositiveDefinite,
        );
        let error = CudaLinearSolver::new(0)
            .solve(
                &problem,
                None,
                plan(LinearSolver::ConjugateGradient)
                    .with_reduction(ReductionPolicy::Fast)
                    .with_preconditioner(PreconditionerPolicy::Jacobi),
            )
            .unwrap_err();
        assert!(error.message().contains("diagonal entry 0"));
    }

    #[test]
    #[ignore = "requires an explicitly selected physical CUDA device"]
    fn physical_cuda_csr_action_matches_the_host_oracle() {
        let device = std::env::var("EQIORA_CUDA_DEVICE")
            .expect("set EQIORA_CUDA_DEVICE for the explicit hardware gate")
            .parse::<u16>()
            .expect("EQIORA_CUDA_DEVICE must be a u16 ordinal");
        let result = verify_csr_action(
            &matrix(),
            &[1.0, 2.0],
            device,
            SparseActionPolicy::Deterministic,
            SparseActionTolerance::new(1.0e-12, 1.0e-12).unwrap(),
        )
        .unwrap();

        assert_eq!(result.values(), &[2.0, 5.0]);
        let evidence = result.evidence();
        assert_eq!(evidence.device().id().ordinal(), device);
        assert!(evidence.compute_capability().major() > 0);
        assert_eq!(evidence.policy(), SparseActionPolicy::Deterministic);
        assert_eq!(evidence.versions().cudarc(), CUDARC_VERSION);
        assert_eq!(evidence.versions().binding_toolkit(), CUDA_BINDING_TOOLKIT);
        assert!(evidence.versions().driver() > 0);
        assert!(evidence.versions().cusparse() > 0);
        assert_eq!(evidence.versions().cublas(), None);
        assert_eq!(evidence.maximum_absolute_error(), 0.0);
        assert!(evidence.maximum_scaled_error() <= 1.0);
        assert!(evidence.transfers().host_to_device_bytes() > 0);
        assert_eq!(evidence.transfers().device_to_host_bytes(), 16);
        assert!(
            evidence
                .transfers()
                .input()
                .completion()
                .happens_before(evidence.action_completion())
                .unwrap()
        );
        assert!(
            evidence
                .action_completion()
                .happens_before(evidence.transfers().output().completion())
                .unwrap()
        );
        assert!(evidence.timings().total() >= evidence.timings().solve());
    }

    #[test]
    #[ignore = "requires an explicitly selected physical CUDA device"]
    fn physical_resident_rectangular_csr_reuses_one_matrix_and_dense_buffers() {
        let device = std::env::var("EQIORA_CUDA_DEVICE")
            .expect("set EQIORA_CUDA_DEVICE for the explicit hardware gate")
            .parse::<u16>()
            .expect("EQIORA_CUDA_DEVICE must be a u16 ordinal");
        let observed = CudaRuntime
            .observe()
            .unwrap()
            .into_iter()
            .find(|candidate| candidate.descriptor().id().ordinal() == device)
            .expect("the selected CUDA device is visible");
        let matrix = eqiora_assembly::CsrMatrix::from_sorted_csr(
            2,
            3,
            vec![0, 2, 4],
            vec![0, 1, 1, 2],
            vec![1.0, 2.0, -1.0, 3.0],
        )
        .unwrap();
        let queue = QueueSlot::new(observed.descriptor().id(), 0);
        let mut session = CudaResidentCsrActionSession::new(
            &matrix,
            &observed,
            queue,
            SparseActionPolicy::Deterministic,
        )
        .unwrap();
        let setup = session.setup_evidence().clone();
        assert_eq!(setup.device(), observed.descriptor());
        assert_eq!(setup.physical_uuid(), observed.physical_uuid());
        assert_eq!(setup.compute_capability(), observed.compute_capability());
        assert_eq!((setup.rows(), setup.columns(), setup.nonzeros()), (2, 3, 4));
        assert_eq!(setup.policy(), SparseActionPolicy::Deterministic);
        for transfer in [
            setup.row_offsets().completion(),
            setup.column_indices().completion(),
            setup.values().completion(),
        ] {
            assert!(
                transfer
                    .happens_before(setup.matrix_ready().completion())
                    .unwrap()
            );
        }

        let mut first_output = [0.0; 2];
        let first = session.apply(&[1.0, 2.0, 3.0], &mut first_output).unwrap();
        assert_eq!(first_output, [5.0, 7.0]);
        let mut second_output = [0.0; 2];
        let second = session
            .apply(&[2.0, -1.0, 4.0], &mut second_output)
            .unwrap();
        assert_eq!(second_output, [0.0, 13.0]);
        assert_eq!(session.action_count(), 2);
        assert_eq!(first.ordinal().get(), 1);
        assert_eq!(second.ordinal().get(), 2);
        assert_eq!(
            first.input_generation().buffer(),
            second.input_generation().buffer()
        );
        assert_eq!(
            first.output_generation().buffer(),
            second.output_generation().buffer()
        );
        assert_ne!(
            first.input_generation().buffer(),
            first.output_generation().buffer()
        );
        assert_eq!(first.input_generation().generation().get(), 1);
        assert_eq!(second.input_generation().generation().get(), 2);
        assert!(
            first
                .input_ready()
                .completion()
                .happens_before(first.action_completion())
                .unwrap()
        );
        assert!(
            first
                .action_completion()
                .happens_before(first.action_visible().completion())
                .unwrap()
        );
        assert!(
            first
                .action_visible()
                .completion()
                .happens_before(first.output().completion())
                .unwrap()
        );
        assert!(
            first
                .output()
                .completion()
                .happens_before(first.output_visible().completion())
                .unwrap()
        );
        assert!(
            first
                .output_visible()
                .completion()
                .happens_before(second.input().completion())
                .unwrap()
        );
        assert_eq!(
            setup.row_offsets().plan().destination().device(),
            Some(observed.descriptor().id())
        );
    }

    #[test]
    #[ignore = "requires an explicitly selected physical CUDA device"]
    fn physical_cuda_krylov_paths_are_independently_accepted() {
        let device = std::env::var("EQIORA_CUDA_DEVICE")
            .expect("set EQIORA_CUDA_DEVICE for the explicit hardware gate")
            .parse::<u16>()
            .expect("EQIORA_CUDA_DEVICE must be a u16 ordinal");

        let spd = canonical_system(
            vec![4.0, -1.0, -1.0, 3.0],
            vec![2.0, 5.0],
            LinearOperatorProperties::SymmetricPositiveDefinite,
        );
        let cg = CudaLinearSolver::new(device)
            .solve(
                &spd,
                None,
                plan(LinearSolver::ConjugateGradient)
                    .with_reduction(ReductionPolicy::Fast)
                    .with_preconditioner(PreconditionerPolicy::Jacobi),
            )
            .unwrap();
        assert_close(cg.solution().values(), &[1.0, 2.0]);
        assert_eq!(
            cg.solution().report().execution(),
            ExecutionReport::cuda(eqiora_backend_cuda::CUDA_LINEAR_EXECUTION, device)
        );
        assert!(cg.evidence().compute_capability().major() > 0);
        assert_eq!(
            cg.solution().report().verification(),
            ExecutionReport::host_serial()
        );
        assert!(cg.solution().report().true_residual_norm() <= 1.0e-12);
        assert!(cg.evidence().transfers().inverse_diagonal().is_some());

        let nonsymmetric = canonical_system(
            vec![4.0, 1.0, 2.0, 3.0],
            vec![6.0, 8.0],
            LinearOperatorProperties::General,
        );
        let bicgstab = CudaLinearSolver::new(device)
            .solve(
                &nonsymmetric,
                None,
                plan(LinearSolver::BiConjugateGradientStabilized)
                    .with_reduction(ReductionPolicy::Fast)
                    .with_preconditioner(PreconditionerPolicy::Identity),
            )
            .unwrap();
        assert_close(bicgstab.solution().values(), &[1.0, 2.0]);
        assert!(bicgstab.solution().report().true_residual_norm() <= 1.0e-12);
        assert!(bicgstab.evidence().versions().cublas().unwrap() > 0);
        assert!(bicgstab.evidence().transfers().inverse_diagonal().is_none());
        assert!(bicgstab.evidence().transfers().host_to_device_bytes() > 0);
        assert_eq!(bicgstab.evidence().transfers().device_to_host_bytes(), 16);
        assert!(
            bicgstab
                .evidence()
                .transfers()
                .initial_guess()
                .completion()
                .happens_before(bicgstab.evidence().solve_completion())
                .unwrap()
        );
        let uploaded = match bicgstab
            .evidence()
            .transfers()
            .initial_guess()
            .plan()
            .destination()
        {
            MemoryRegion::Device(buffer) => buffer.id(),
            MemoryRegion::Host(_) => panic!("initial guess must be uploaded"),
        };
        let downloaded = match bicgstab.evidence().transfers().solution().plan().source() {
            MemoryRegion::Device(buffer) => buffer.id(),
            MemoryRegion::Host(_) => panic!("solution must be downloaded"),
        };
        assert_eq!(uploaded, downloaded);
        assert!(
            bicgstab
                .evidence()
                .solve_completion()
                .happens_before(bicgstab.evidence().transfers().solution().completion())
                .unwrap()
        );

        let symmetric_indefinite = canonical_system(
            vec![1.0, 2.0, 2.0, -1.0],
            vec![5.0, 0.0],
            LinearOperatorProperties::SymmetricIndefinite,
        );
        let minres = CudaLinearSolver::new(device)
            .solve(
                &symmetric_indefinite,
                None,
                plan(LinearSolver::MinimumResidual)
                    .with_reduction(ReductionPolicy::Fast)
                    .with_preconditioner(PreconditionerPolicy::Identity),
            )
            .unwrap();
        assert_close(minres.solution().values(), &[1.0, 2.0]);
        assert!(minres.solution().report().true_residual_norm() <= 1.0e-12);
        assert!(minres.evidence().transfers().inverse_diagonal().is_none());
    }

    fn assert_close(actual: &[f64], expected: &[f64]) {
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!(
                (actual - expected).abs() <= 1.0e-12,
                "{actual} != {expected}"
            );
        }
    }
}
