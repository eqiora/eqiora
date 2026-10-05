use super::*;
use crate::{
    CommonOdePlan, CommonOdePolicy, CommonOdeRunRequest, CommonOdeState, CommonTimeTolerance,
};
use eqiora_graph::{GraphStore, InMemoryGraphStore};
use eqiora_schema::kernel::KernelNode;
use eqiora_sem::KernelProgram;
use eqiora_time::{
    AcceptedTimeHistory, TimeBackendCapabilities, TimeBackendIdentity, TimeHistoryStep, TimeMethod,
};

fn fixture(
    expression: &str,
    complex: bool,
    times: Vec<f64>,
) -> (CommonTrajectory, ModelEnvelope, Id<kinds::Observable>) {
    let ty = if complex { "complex<m>" } else { "m" };
    typed_fixture(expression, ty, "", times)
}
fn typed_fixture(
    expression: &str,
    ty: &str,
    declarations: &str,
    times: Vec<f64>,
) -> (CommonTrajectory, ModelEnvelope, Id<kinds::Observable>) {
    let source = format!(
        "{declarations} model Signal() {{ state x:1; initial{{x=0;}} relation flow{{derivative(x)=0[1/s];}} observable signal:{ty}={expression}; }}"
    );
    let compiled = eqiora_compiler::compile("spectrum.eqi", &source)
        .unwrap()
        .pop()
        .unwrap();
    let (transaction, model, _) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    let kernel = KernelProgram::from_snapshot(&store.snapshot(), model).unwrap();
    let model = ModelEnvelope::from_program(&kernel).unwrap();
    let field = kernel
        .nodes()
        .find_map(|n| {
            if let KernelNode::Field(v) = n {
                Some(v.id())
            } else {
                None
            }
        })
        .unwrap();
    let observable = kernel
        .nodes()
        .find_map(|n| {
            if let KernelNode::Observable(v) = n {
                Some(v.id())
            } else {
                None
            }
        })
        .unwrap();
    let plan = CommonOdePlan::resolve(
        &model,
        &kernel,
        CommonOdePolicy::new(
            TimeMethod::Tsitouras45,
            0.01,
            1e-9,
            vec![
                CommonTimeTolerance::new(
                    eqiora_core::TimeStateCoordinate::new(field, 0, 0, false),
                    1e-11,
                )
                .unwrap(),
            ],
        )
        .unwrap(),
        TimeBackendCapabilities::new(
            TimeBackendIdentity::new("test.spectrum", "1"),
            &[ScalarDomain::Real],
            &[eqiora_core::ScalarType::F64],
        ),
    )
    .unwrap();
    let request = CommonOdeRunRequest::new(
        plan.clone(),
        plan.initial_state(0.).unwrap(),
        4.,
        times.clone(),
    )
    .unwrap();
    let states = times
        .into_iter()
        .map(|t| CommonOdeState::new(&plan, t, vec![0.], "result").unwrap())
        .collect();
    let history = AcceptedTimeHistory::accepted(
        1,
        vec![TimeHistoryStep::accepted(0., 4., vec![0.], vec![0.], vec![0.]).unwrap()],
        vec![],
    )
    .unwrap();
    (
        CommonTrajectory::accept_ode_states(request, states, history).unwrap(),
        model,
        observable,
    )
}
fn grid(window: SpectrumWindow) -> UniformDft {
    UniformDft::new(1., 0.5, 4, window).unwrap()
}
fn close(value: &ValueLiteral, re: f64, im: f64) {
    let (a, b) = value.component(0).unwrap();
    assert!(
        (a - re).abs() < 1e-12 && (b - im).abs() < 1e-12,
        "{a}+i{b} != {re}+i{im}"
    );
}
fn scalar(v: ValueLiteral) -> f64 {
    v.real_scalar_value().unwrap().value()
}

#[test]
fn accepted_complex_exponential_preserves_sign_units_phase_and_inverse() {
    // x(t)= (2+3i) exp(-i*pi*(t-1)), t=1+n/2: [z,-iz,-z,iz].
    // The four roots of unity sum to zero except k=1; C[1]=z.
    let (trajectory, model, id) = fixture(
        "math.complex(2[m],3[m])*math.exp(math.complex(0,-3.141592653589793)*(time()/1[s]-1))",
        true,
        vec![1., 1.5, 2., 2.5],
    );
    let spectrum = trajectory
        .observe_spectrum(&model, id, grid(SpectrumWindow::Rectangular), 16)
        .unwrap();
    assert_eq!(spectrum.trajectory_identity(), trajectory.identity());
    assert_eq!(spectrum.observable(), id);
    for (k, expected) in [(0, (0., 0.)), (1, (2., 3.)), (2, (0., 0.)), (3, (0., 0.))] {
        close(&spectrum.coefficients()[k], expected.0, expected.1);
    }
    assert_eq!(spectrum.frequency_hz(1).unwrap(), 0.5);
    assert_eq!(spectrum.frequency_hz(2).unwrap(), 1.);
    assert_eq!(spectrum.frequency_hz(3).unwrap(), -0.5);
    assert_eq!(
        spectrum.angular_frequency_rad_s(1).unwrap(),
        std::f64::consts::PI
    );
    assert!((spectrum.phase_rad(1, 0).unwrap() - 3_f64.atan2(2.)).abs() < 1e-12);
    assert!(spectrum.one_sided_amplitude(1, 0).is_err());
    assert_eq!(
        spectrum.coefficients()[1].value_type().dimension(),
        DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap()
    );
    let integral = spectrum.rectangle_transform_estimate(1).unwrap();
    close(&integral, 4., 6.);
    assert_eq!(
        integral.value_type().dimension(),
        DimExponents::from_integers([0, 1, 1, 0, 0, 0, 0]).unwrap()
    );
    assert!((scalar(spectrum.power(1, 0).unwrap()) - 13.).abs() < 1e-12);
    assert!((scalar(spectrum.power_density_per_hz(1, 0).unwrap()) - 26.).abs() < 1e-12);
    for (n, (re, im)) in [(2., 3.), (3., -2.), (-2., -3.), (-3., 2.)]
        .into_iter()
        .enumerate()
    {
        close(&spectrum.reconstruct_sample(n).unwrap(), re, im);
    }
}
#[test]
fn real_sinusoid_one_sided_factors_and_discrete_parseval() {
    // 4*sin(pi*(t-1)) has samples [0,4,0,-4]; with forward +i, C1=2i,C3=-2i.
    let (trajectory, model, id) = fixture(
        "4[m]*math.sin(3.141592653589793*(time()/1[s]-1))",
        false,
        vec![1., 1.5, 2., 2.5],
    );
    let spectrum = trajectory
        .observe_spectrum(&model, id, grid(SpectrumWindow::Rectangular), 16)
        .unwrap();
    close(&spectrum.coefficients()[1], 0., 2.);
    close(&spectrum.coefficients()[3], 0., -2.);
    assert!((scalar(spectrum.one_sided_amplitude(1, 0).unwrap()) - 4.).abs() < 1e-12);
    let power: f64 = (0..4).map(|k| scalar(spectrum.power(k, 0).unwrap())).sum();
    assert!((power - 8.).abs() < 1e-12); // mean [0,16,0,16] = 8.
    assert!(
        (scalar(spectrum.one_sided_power_density_per_hz(1, 0).unwrap()) * 0.5 - 8.).abs() < 1e-12
    );
    assert_eq!(
        spectrum
            .power_density_per_hz(1, 0)
            .unwrap()
            .value_type()
            .dimension(),
        DimExponents::from_integers([0, 2, 1, 0, 0, 0, 0]).unwrap()
    );
    assert!(spectrum.one_sided_amplitude(3, 0).is_err());
    let (trajectory, model, id) = fixture(
        "2[m]+3[m]*math.cos(6.283185307179586*(time()/1[s]-1))",
        false,
        vec![1., 1.5, 2., 2.5],
    );
    let spectrum = trajectory
        .observe_spectrum(&model, id, grid(SpectrumWindow::Rectangular), 16)
        .unwrap();
    assert!((scalar(spectrum.one_sided_amplitude(0, 0).unwrap()) - 2.).abs() < 1e-12);
    assert!((scalar(spectrum.one_sided_amplitude(2, 0).unwrap()) - 3.).abs() < 1e-12);
}
#[test]
fn off_bin_leakage_and_periodic_window_are_declared_not_corrected() {
    // Off-bin half-frequency signal [1,q,-i,-conj(q)], q=(1-i)/sqrt(2).
    // Independently sum each four-term coefficient with exact DFT roots [1,i,-1,-i].
    let (trajectory, model, id) = fixture(
        "1[m]*math.exp(math.complex(0,-1.5707963267948966)*(time()/1[s]-1))",
        true,
        vec![1., 1.5, 2., 2.5],
    );
    let spectrum = trajectory
        .observe_spectrum(&model, id, grid(SpectrumWindow::Rectangular), 16)
        .unwrap();
    let a = std::f64::consts::FRAC_1_SQRT_2;
    for (k, im) in [
        -(1. + 2. * a) / 4.,
        (1. + 2. * a) / 4.,
        (2. * a - 1.) / 4.,
        (1. - 2. * a) / 4.,
    ]
    .into_iter()
    .enumerate()
    {
        close(&spectrum.coefficients()[k], 0.25, im);
    }
    assert!((0..4).all(|k| scalar(spectrum.power(k, 0).unwrap()) > 0.01));
    let (trajectory, model, id) = fixture("2[m]", false, vec![1., 1.5, 2., 2.5]);
    let windowed = trajectory
        .observe_spectrum(&model, id, grid(SpectrumWindow::PeriodicHann), 16)
        .unwrap();
    // Periodic Hann N=4 is [0,1/2,1,1/2], distinct from symmetric [0,3/4,3/4,0].
    for (n, v) in [0., 1., 2., 1.].into_iter().enumerate() {
        close(&windowed.reconstruct_sample(n).unwrap(), v, 0.);
    }
    close(&windowed.coefficients()[0], 1., 0.);
    close(&windowed.coefficients()[1], -0.5, 0.);
}
#[test]
fn rejects_missing_irregular_foreign_forged_lineage_zero_phase_and_resource_exhaustion() {
    let (trajectory, model, id) = fixture("0[m]", false, vec![1., 1.5, 2., 2.5]);
    let spectrum = trajectory
        .observe_spectrum(&model, id, grid(SpectrumWindow::Rectangular), 16)
        .unwrap();
    assert!(spectrum.phase_rad(0, 0).is_err());
    assert!(spectrum.frequency_hz(4).is_err());
    assert!(spectrum.reconstruct_sample(4).is_err());
    assert!(
        trajectory
            .observe_spectrum(&model, id, grid(SpectrumWindow::Rectangular), 15)
            .is_err()
    );
    let (irregular, other, id2) = fixture("1[m]", false, vec![1., 1.5, 2., 2.6]);
    assert!(
        irregular
            .observe_spectrum(&other, id2, grid(SpectrumWindow::Rectangular), 16)
            .is_err()
    );
    assert!(
        trajectory
            .observe_spectrum(&other, id2, grid(SpectrumWindow::Rectangular), 16)
            .is_err()
    );
    let mut forged = trajectory.clone();
    if let CommonTrajectory::Ode { identity, .. } = &mut forged {
        *identity = "foreign".into();
    }
    assert!(
        forged
            .observe_spectrum(&model, id, grid(SpectrumWindow::Rectangular), 16)
            .is_err()
    );
    assert!(UniformDft::new(0., 0., 4, SpectrumWindow::Rectangular).is_err());
    assert!(UniformDft::new(1e30, 1., 4, SpectrumWindow::Rectangular).is_err());
    assert!(UniformDft::new(0., 1., 0, SpectrumWindow::Rectangular).is_err());
    // Endpoint sample at 3 cannot replace the included 2.5 sample in [1,3).
    let (endpoint, model, id) = fixture("1[m]", false, vec![1., 1.5, 2., 3.]);
    assert!(
        endpoint
            .observe_spectrum(&model, id, grid(SpectrumWindow::Rectangular), 16)
            .is_err()
    );
}

#[test]
fn shaped_and_finite_basis_components_retain_roles_without_size_caps() {
    // Six independent channels are constant multiples of the same four-sample carrier.
    let values = "[1[m],2[m],3[m],4[m],5[m],6[m]]*math.exp(math.complex(0,-3.141592653589793)*(time()/1[s]-1))";
    let (trajectory, model, id) =
        typed_fixture(values, "array<complex<m>,6>", "", vec![1., 1.5, 2., 2.5]);
    assert!(
        trajectory
            .observe_spectrum(&model, id, grid(SpectrumWindow::Rectangular), 95)
            .is_err()
    );
    let spectrum = trajectory
        .observe_spectrum(&model, id, grid(SpectrumWindow::Rectangular), 96)
        .unwrap();
    assert_eq!(spectrum.input_type().array_rank(), 1);
    assert_eq!(spectrum.coefficients()[1].component_count(), 6);
    for component in 0..6 {
        let (re, im) = spectrum.coefficients()[1].component(component).unwrap();
        assert!((re - (component + 1) as f64).abs() < 1e-12 && im.abs() < 1e-12);
        assert!(
            (scalar(spectrum.amplitude(1, component).unwrap()) - (component + 1) as f64).abs()
                < 1e-12
        );
    }
    assert!(spectrum.amplitude(1, 6).is_err());
    let (trajectory, model, id) = typed_fixture(
        "coordinates(S,[1[m],2[m],3[m],4[m],5[m]])",
        "coordinates<m,S>",
        "space S=orthonormal(a,b,c,d,e);",
        vec![1., 1.5, 2., 2.5],
    );
    let spectrum = trajectory
        .observe_spectrum(&model, id, grid(SpectrumWindow::Rectangular), 80)
        .unwrap();
    let ty = spectrum.coefficients()[0].value_type();
    assert_eq!(
        ty.coordinate_basis(),
        spectrum.input_type().coordinate_basis()
    );
    assert_eq!(ty.scalar_domain(), ScalarDomain::Complex);
    assert_eq!(spectrum.reconstruct_sample(1).unwrap().value_type(), ty);
    for (index, (re, im)) in spectrum.coefficients()[0].components().unwrap().enumerate() {
        assert!((re - (index + 1) as f64).abs() < 1e-12 && im.abs() < 1e-12);
    }
}

#[test]
fn finite_window_spectrum_claim() {
    accepted_complex_exponential_preserves_sign_units_phase_and_inverse();
    real_sinusoid_one_sided_factors_and_discrete_parseval();
    off_bin_leakage_and_periodic_window_are_declared_not_corrected();
    rejects_missing_irregular_foreign_forged_lineage_zero_phase_and_resource_exhaustion();
    shaped_and_finite_basis_components_retain_roles_without_size_caps();
    initial_sample_and_odd_count_keep_half_open_discrete_meaning();
}

#[test]
fn initial_sample_and_odd_count_keep_half_open_discrete_meaning() {
    // Samples of (2t-3)(t-2) at t=1,1.5,2 are [1,0,0]. For N=3 all Ck=1/3.
    let (trajectory, model, id) = fixture(
        "1[m]*(2*time()/1[s]-3)*(time()/1[s]-2)",
        false,
        vec![1., 1.5, 2.],
    );
    let odd = UniformDft::new(1., 0.5, 3, SpectrumWindow::Rectangular).unwrap();
    let spectrum = trajectory.observe_spectrum(&model, id, odd, 9).unwrap();
    for value in spectrum.coefficients() {
        close(value, 1. / 3., 0.);
    }
    assert!((scalar(spectrum.one_sided_amplitude(1, 0).unwrap()) - 2. / 3.).abs() < 1e-12);
    assert!((spectrum.frequency_hz(2).unwrap() + 2. / 3.).abs() < 1e-12);
    let (trajectory, model, id) = fixture("2[m]", false, vec![0.5, 1., 1.5]);
    let initial = UniformDft::new(0., 0.5, 4, SpectrumWindow::Rectangular).unwrap();
    let spectrum = trajectory
        .observe_spectrum(&model, id, initial, 16)
        .unwrap();
    close(&spectrum.coefficients()[0], 2., 0.);
}
