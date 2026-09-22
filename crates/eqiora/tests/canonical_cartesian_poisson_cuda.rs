#![cfg(feature = "cuda")]

#[path = "support/canonical_cartesian_poisson.rs"]
mod canonical;

use eqiora::artifact::{
    ExecutionTopologyV1, LayoutArtifacts, ModelEnvelope, RealizationEnvelopeV1, RunManifestV2,
};
use eqiora::backends::cuda::{
    CUDA_ADAPTER_VERSION, CUDA_BINDING_TOOLKIT, CUDA_LINEAR_EXECUTION_PROVIDER,
    CUDA_LINEAR_SOLVER_PROVIDER, CUDARC_VERSION, CudaLinearSolveResult, CudaLinearSolver,
    CudaRuntime,
};
use eqiora::device::QueueSlot;
use eqiora::realization::{DiscretizationMethod, TargetCapabilities, resolve};
use eqiora::solver::ReductionPolicy;
use eqiora_backend_cuda::CudaAdmittedExecutionAdapter;
use eqiora_core::ScalarType;
use eqiora_execution::{AdmittedExecution, CudaExecutorDescriptor, DeploymentBinding};
use eqiora_numerics::scalar::finalize_resolved_scalar_elliptic_cartesian;

#[test]
fn cuda_facade_is_optional_and_admission_fails_closed_without_runtime_work() {
    assert!(!CUDA_ADAPTER_VERSION.is_empty());
    assert!(!CUDA_ADAPTER_VERSION.chars().any(char::is_control));
    assert_eq!(CUDARC_VERSION, "0.19.9");
    assert_eq!(CUDA_BINDING_TOOLKIT, "12.0");
    assert_eq!(
        CudaLinearSolver::capabilities(),
        canonical::cuda_solver_contract()
    );

    let reproducible = canonical::solver_plan(ReductionPolicy::Reproducible);
    let unsupported = CudaLinearSolver::capabilities()
        .require(reproducible, ScalarType::F64)
        .unwrap_err();
    assert_eq!(
        unsupported.code(),
        eqiora::diagnostic::codes::INVALID_REALIZATION
    );
    assert!(unsupported.message().contains("Reproducible"));

    let program = canonical::compile_program().unwrap();
    let request =
        canonical::request(&program, DiscretizationMethod::ContinuousGalerkin, 0, 1).unwrap();
    let no_device =
        canonical::exact_capabilities(CudaLinearSolver::capabilities(), TargetCapabilities::none());
    assert!(
        resolve(&request, canonical::requirements(), &no_device)
            .unwrap_err()
            .message()
            .contains("no executable CUDA device 0")
    );

    let into_parts: fn(CudaLinearSolveResult) -> _ = CudaLinearSolveResult::into_parts;
    let _ = into_parts;
    let admit: fn(
        CudaLinearSolver,
        &eqiora::device::DeviceDescriptor,
    ) -> Result<(), eqiora::Diagnostic> = CudaLinearSolver::admit_device;
    let _ = admit;
}

#[test]
#[ignore = "requires EQIORA_CUDA_DEVICE and an explicitly selected physical CUDA device"]
fn canonical_plane_poisson_runs_through_q1_and_tpfa_on_cuda() {
    let device_ordinal = std::env::var("EQIORA_CUDA_DEVICE")
        .expect("set EQIORA_CUDA_DEVICE for the explicit hardware gate")
        .parse::<u16>()
        .expect("EQIORA_CUDA_DEVICE must be a u16 ordinal");
    let device = CudaRuntime
        .discover()
        .unwrap()
        .into_iter()
        .find(|candidate| candidate.id().ordinal() == device_ordinal)
        .expect("selected CUDA device must be visible");
    CudaLinearSolver::new(device_ordinal)
        .admit_device(&device)
        .unwrap();

    let program = canonical::compile_program().unwrap();
    let model_artifact = ModelEnvelope::from_program(&program).unwrap();
    let capabilities = canonical::exact_capabilities(
        CudaLinearSolver::capabilities(),
        TargetCapabilities::none().with_cuda_device(device_ordinal),
    );

    for (revision, method, _) in canonical::METHODS {
        let resolved = resolve(
            &canonical::request(&program, method, device_ordinal, revision).unwrap(),
            canonical::requirements(),
            &capabilities,
        )
        .unwrap();
        let cpu_solution =
            canonical::reference_cpu_solution(&program, method, revision + 100).unwrap();
        let realization = RealizationEnvelopeV1::from_resolved(
            &model_artifact,
            &resolved,
            LayoutArtifacts::Replicated,
        )
        .unwrap();
        let (_, finalized) =
            finalize_resolved_scalar_elliptic_cartesian(&program, &resolved).unwrap();
        let binding = DeploymentBinding::bind_cuda(
            finalized.portable_realization(),
            CudaExecutorDescriptor::new(
                CUDA_LINEAR_SOLVER_PROVIDER,
                CUDA_LINEAR_EXECUTION_PROVIDER,
                device.clone(),
                QueueSlot::new(device.id(), 0),
                CudaLinearSolver::capabilities(),
            )
            .unwrap(),
        )
        .unwrap();
        let admitted = AdmittedExecution::admit_cuda_linear(
            finalized.portable_realization(),
            finalized.canonical_csr_system_view(),
            binding,
        )
        .unwrap();
        let cuda = CudaLinearSolver::new(device_ordinal)
            .execute_admitted(admitted)
            .unwrap();
        let (accepted, cuda_evidence) = cuda.into_parts();
        let (linear_solution, receipt) = accepted.into_parts();
        assert_eq!(
            receipt.operator(),
            finalized
                .canonical_csr_system_view()
                .agreement_fingerprint()
        );
        assert!(receipt.cuda_trace().is_some());
        let solution = finalized.finish(linear_solution).unwrap();

        canonical::method_metrics(method, &solution).unwrap();
        canonical::cpu_conformance(&cpu_solution, &solution).unwrap();

        let versions = cuda_evidence.versions();
        let compute = cuda_evidence.compute_capability();
        let native_libraries = [
            ("cusparse", versions.cusparse().to_string()),
            (
                "cublas",
                versions
                    .cublas()
                    .expect("CUDA Krylov always uses cuBLAS")
                    .to_string(),
            ),
        ];
        let execution = eqiora::artifact::ExecutionProvenanceV1::from_provider_releases(
            receipt.solver_provider(),
            receipt.execution_provider(),
            ExecutionTopologyV1::Cuda {
                device: device_ordinal,
                device_name: cuda_evidence.device().name().to_owned(),
                compute_capability_major: compute.major(),
                compute_capability_minor: compute.minor(),
                driver_version: versions.driver().to_string(),
            },
            ReductionPolicy::Fast,
            native_libraries,
        )
        .unwrap();
        let run = RunManifestV2::new(&realization, execution).unwrap();

        let realization_bytes = realization.canonical_json().unwrap();
        let decoded_realization =
            RealizationEnvelopeV1::from_json(&realization_bytes, Default::default()).unwrap();
        let run_bytes = run.canonical_json().unwrap();
        let decoded_run = RunManifestV2::from_json(&run_bytes, Default::default()).unwrap();
        decoded_run.validate_against(&decoded_realization).unwrap();
        assert_eq!(decoded_run.canonical_json().unwrap(), run_bytes);
        assert_eq!(decoded_run.model(), model_artifact.digest().unwrap());
        assert_eq!(decoded_run.realization(), realization.digest().unwrap());
    }
}
