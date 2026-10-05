use eqiora::api::ModelDocument;
use eqiora::sem::{Interpreter, ReferenceConfig};
use eqiora::time::TimeSystem;

#[test]
fn mixed_complex_channels_initialize_through_original_equations() {
    // For each channel z'=i*omega*z, z(0)=1+2i gives z'(0)=-2+i
    // at omega=1/s. The real companion has w(0)=3 and w'(0)=6/s.
    // Six channels guard against treating a physical-space axis as a size limit.
    let source = r#"
model Oscillators() {
    parameter omega:1/s=1;
    state z:array<complex<1>,6>;
    state w:1;
    initial {
        z=[math.complex(1,2),math.complex(1,2),math.complex(1,2),
           math.complex(1,2),math.complex(1,2),math.complex(1,2)];
        w=3;
    }
    relation motion {
        derivative(z)=math.complex(0,1)*omega*z;
        derivative(w)=2*omega*w;
    }
}
"#;
    let document = ModelDocument::compile("complex-oscillators.eqi", source).unwrap();
    let initial = Interpreter::new()
        .initialize(
            document.program(),
            0.,
            ReferenceConfig::new(0., 1.).unwrap(),
        )
        .unwrap();
    let z = &initial.fields()[&document.aliases()["z"]];
    let dz = &initial.derivatives()[&(document.aliases()["z"], std::num::NonZeroU32::MIN)];
    assert_eq!(z.value_type().shape().component_count(), Some(6));
    assert_eq!(dz.value_type().shape(), z.value_type().shape());
    assert_eq!(
        dz.value_type().dimension(),
        eqiora_core::DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).unwrap()
    );
    for component in 0..6 {
        assert_eq!(z.component(component), Some((1., 2.)));
        let (real, imaginary) = dz.component(component).unwrap();
        assert!((real + 2.).abs() < 1e-8 && (imaginary - 1.).abs() < 1e-8);
    }
    assert_eq!(
        initial.fields()[&document.aliases()["w"]].component(0),
        Some((3., 0.))
    );
    let dw = &initial.derivatives()[&(document.aliases()["w"], std::num::NonZeroU32::MIN)];
    let (real, imaginary) = dw.component(0).unwrap();
    assert!((real - 6.).abs() < 1e-8);
    assert_eq!(imaginary, 0.);
    let flow = eqiora::runtime::FirstOrderProgram::lower(
        document.program(),
        document.aliases()["motion"].downcast().unwrap(),
    )
    .unwrap();
    let accepted = flow
        .initialize(0., ReferenceConfig::new(0., 1.).unwrap())
        .unwrap();
    assert_eq!(flow.dimension(), 13);
    let mut rhs = vec![0.; flow.dimension()];
    flow.rhs(0., accepted.state(), &mut rhs).unwrap();
    for (coordinate, value) in flow.state_coordinates().iter().zip(&rhs) {
        let expected = if coordinate.field().erase() == document.aliases()["w"] {
            6.
        } else if coordinate.is_imaginary() {
            1.
        } else {
            -2.
        };
        assert!((value - expected).abs() < 1e-8);
    }
    // JVP along z=1+2i, w=3 is the same linear vector field, without
    // holomorphic assumptions or a numerical difference of two RHS evaluations.
    let mut tangent = vec![0.; flow.dimension()];
    flow.rhs_jvp(0., accepted.state(), accepted.state(), &mut tangent)
        .unwrap();
    for (actual, expected) in tangent.iter().zip(rhs) {
        assert!((actual - expected).abs() < 1e-12);
    }
    let model = eqiora_artifact::ModelEnvelope::from_program(document.program()).unwrap();
    let envelope = eqiora_artifact::TimeLoweringEnvelopeV3::from_proof(
        &model,
        document.program(),
        flow.lowering_proof(),
    )
    .unwrap();
    let bytes = envelope.canonical_json().unwrap();
    let decoded =
        eqiora_artifact::TimeLoweringEnvelopeV3::from_json(&bytes, Default::default()).unwrap();
    decoded
        .validate_against(&model, document.program())
        .unwrap();
    assert_eq!(decoded.proof().unwrap(), *flow.lowering_proof());
    // A full-rank but changed mass coefficient cannot pass Model linkage.
    let mut forged_matrix = flow
        .lowering_proof()
        .derivative_matrix()
        .coefficients()
        .to_vec();
    let nonzero = forged_matrix
        .iter_mut()
        .find(|value| **value != 0.)
        .unwrap();
    *nonzero *= 2.;
    let forged = eqiora::time::TimeLoweringProof::new(
        flow.relation(),
        flow.state_coordinates().to_vec(),
        eqiora::time::ConstantDerivativeMatrixProof::new(flow.dimension(), forged_matrix).unwrap(),
    )
    .unwrap();
    assert!(
        eqiora_artifact::TimeLoweringEnvelopeV3::from_proof(&model, document.program(), &forged,)
            .is_err()
    );
    let temporal = eqiora_numerics::CommonOdePolicy::new(
        eqiora_time::TimeMethod::Tsitouras45,
        0.001,
        1e-9,
        flow.state_coordinates()
            .iter()
            .map(|&coordinate| {
                eqiora_numerics::CommonTimeTolerance::new(coordinate, 1e-11).unwrap()
            })
            .collect(),
    )
    .unwrap();
    let plan = eqiora_numerics::CommonOdePlan::resolve(
        &model,
        document.program(),
        temporal,
        eqiora::time::TimeBackendCapabilities::new(
            eqiora::time::TimeBackendIdentity::new("eqiora.test.time", "1"),
            &[eqiora::ScalarDomain::Real, eqiora::ScalarDomain::Complex],
            &[eqiora::ScalarType::F64],
        ),
    )
    .unwrap();
    assert_eq!(plan.initial_state(0.).unwrap().values(), accepted.state());
    let midpoint_controls = eqiora::time::TimePlan::new(
        eqiora::time::TimeMethod::ImplicitMidpoint,
        0.,
        0.05,
        1e-12,
        vec![1e-14; flow.dimension()],
        vec![0.013, 0.7, 1.],
    )
    .unwrap();
    let midpoint = eqiora::time::ImplicitMidpointTimeBackend::new()
        .solve(&flow.time_problem().unwrap(), &midpoint_controls)
        .unwrap();
    let angle = 40. * (0.025_f64).atan();
    for (coordinate, actual) in flow
        .state_coordinates()
        .iter()
        .zip(midpoint.state(2).unwrap())
    {
        let expected = if coordinate.field().erase() == document.aliases()["w"] {
            3. * (1.05_f64 / 0.95).powi(20)
        } else if coordinate.is_imaginary() {
            angle.sin() + 2. * angle.cos()
        } else {
            angle.cos() - 2. * angle.sin()
        };
        assert!((actual - expected).abs() < 1e-10);
    }
    assert_eq!(midpoint.history().unwrap().steps().len(), 20);
    let run =
        eqiora_artifact::TimeRunManifestV1::new(&envelope, &midpoint_controls, midpoint.report())
            .unwrap();
    let bytes = run.canonical_json().unwrap();
    let replay = eqiora_artifact::TimeRunManifestV1::from_json(&bytes, Default::default()).unwrap();
    assert_eq!(replay.canonical_json().unwrap(), bytes);

    #[cfg(feature = "diffsol")]
    {
        let problem = flow.time_problem().unwrap();
        let times = vec![0.1, 0.5, 1.];
        let controls = eqiora::time::TimePlan::new(
            eqiora::time::TimeMethod::Tsitouras45,
            0.,
            0.001,
            1e-10,
            vec![1e-12; flow.dimension()],
            times.clone(),
        )
        .unwrap();
        let solution = eqiora::backends::diffsol::DiffsolTimeBackend::new()
            .solve(&problem, &controls)
            .unwrap();
        // Independently, z(t)=(1+2i)(cos(t)+i sin(t)), w(t)=3 exp(2t).
        for (index, time) in times.iter().copied().enumerate() {
            for (coordinate, actual) in flow
                .state_coordinates()
                .iter()
                .zip(solution.state(index).unwrap())
            {
                let expected = if coordinate.field().erase() == document.aliases()["w"] {
                    3. * (2. * time).exp()
                } else if coordinate.is_imaginary() {
                    time.sin() + 2. * time.cos()
                } else {
                    time.cos() - 2. * time.sin()
                };
                assert!((actual - expected).abs() < 5e-8 * expected.abs().max(1.));
            }
        }
    }
}
