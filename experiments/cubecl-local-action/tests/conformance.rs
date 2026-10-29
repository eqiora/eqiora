#[cfg(any(feature = "cuda", feature = "hip"))]
use eqiora_cubecl_local_action_experiment::{LocalActionPolicy, execute};
use eqiora_meshing::{CartesianMesh, QuadratureRule};
use eqiora_numerics::scalar::lower_cartesian_q1_diffusion_local_action;

fn action_case(dimension: usize) -> (eqiora_ir::LocalLinearActionIr, Vec<f64>) {
    let bounds = vec![[-0.5, 1.5]; dimension];
    let cells = vec![2; dimension];
    let mesh = CartesianMesh::uniform(&bounds, &cells).unwrap();
    let quadrature = QuadratureRule::tensor_product_gauss_legendre(dimension, 2).unwrap();
    let action =
        lower_cartesian_q1_diffusion_local_action(&mesh, &|_: &[f64]| 1.25, &quadrature).unwrap();
    let input = (0..action.input_len())
        .map(|index| {
            let signed = i32::try_from(index % 13).unwrap() - 6;
            f64::from(signed) / 7.0
        })
        .collect();
    (action, input)
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "requires a physical CUDA device"]
fn cuda_fails_closed_when_cubecl_omits_f64_buffers() {
    use cubecl::Device;
    use cubecl::device::CudaDevice;
    use eqiora_cubecl_local_action_experiment::ExperimentError;

    let (action, input) = action_case(3);
    let failure = execute(
        &Device::Cuda(CudaDevice::new(0)),
        &action,
        &input,
        LocalActionPolicy::Ordered,
    )
    .unwrap_err();

    assert_eq!(failure, ExperimentError::MissingF64Capability);
}

#[cfg(feature = "hip")]
#[test]
#[ignore = "requires a physical ROCm device"]
fn rocm_fails_closed_when_cubecl_omits_f64_buffers() {
    use cubecl::Device;
    use cubecl::device::AmdDevice;
    use eqiora_cubecl_local_action_experiment::ExperimentError;

    let (action, input) = action_case(3);
    let failure = execute(
        &Device::Hip(AmdDevice::new(0)),
        &action,
        &input,
        LocalActionPolicy::Ordered,
    )
    .unwrap_err();

    assert_eq!(failure, ExperimentError::MissingF64Capability);
}

#[cfg(not(any(feature = "cuda", feature = "hip")))]
#[test]
fn reference_case_retains_runtime_dimension() {
    for dimension in 1..=3 {
        let (action, input) = action_case(dimension);
        let mut output = vec![0.0; action.output_len()];
        action.apply_reference(&input, &mut output).unwrap();
        assert!(output.iter().all(|value| value.is_finite()));
        assert_eq!(action.columns(), 1 << dimension);
    }
}
