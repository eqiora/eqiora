use super::*;

pub(super) fn cyclic_frequency_conversion_and_peak_power_are_explicit() {
    // f=250 cycles/s, tau=R*C=0.001 s, hence omega*tau=pi/2.
    // Radian dimensionlessness cannot silently replace this explicit conversion.
    let converted = source().replace(
        "angular_frequency = omega",
        "angular_frequency = 2*math.pi*250[1/s]",
    );
    let missing_conversion =
        source().replace("angular_frequency = omega", "angular_frequency = 250[1/s]");
    let mut voltages = Vec::new();
    for (source, ratio) in [
        (converted, std::f64::consts::FRAC_PI_2),
        (missing_conversion, 0.25),
    ] {
        let plan = resolve(&source).unwrap();
        let result = plan
            .run_result(&plan.initial_state(&[]).unwrap(), &REFERENCE_LINEAR_SOLVER)
            .unwrap();
        let voltage = plan
            .harmonic_amplitudes()
            .find(|(name, _, _)| *name == "voltage_hat")
            .unwrap()
            .2;
        let actual = plan
            .field_values(result.finite_values().unwrap())
            .unwrap()
            .into_iter()
            .find(|(id, _)| *id == voltage)
            .unwrap()
            .1
            .component(0)
            .unwrap();
        // Independent circuit transfer 1/(1-i*x)=(1+i*x)/(1+x²).
        let denominator = 1. + ratio * ratio;
        assert!((actual.0 - 1. / denominator).abs() < 1e-10);
        assert!((actual.1 - ratio / denominator).abs() < 1e-10);
        assert!((plan.harmonic_angular_frequency().unwrap() * 0.001 - ratio).abs() < 1e-14);
        voltages.push(actual);
    }
    assert!(
        (voltages[0].0 - voltages[1].0).abs() > 0.5,
        "dropping 2*pi must not be indistinguishable from the declared cyclic drive"
    );

    let plan = resolve(source()).unwrap();
    let result = plan
        .run_result(&plan.initial_state(&[]).unwrap(), &REFERENCE_LINEAR_SOLVER)
        .unwrap();
    let current = plan
        .harmonic_amplitudes()
        .find(|(name, _, _)| *name == "current_hat")
        .unwrap()
        .1;
    let seconds = eqiora_core::DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).unwrap();
    let average_loss = (0..4)
        .map(|quarter| {
            let time = f64::from(quarter) * std::f64::consts::FRAC_PI_2 / 1000.;
            let value = plan
                .reconstruct_harmonic_fields(&result, eqiora_core::DynQuantity::new(time, seconds))
                .unwrap()
                .into_iter()
                .find(|(id, _)| *id == current)
                .unwrap()
                .1
                .real_scalar_value()
                .unwrap()
                .value();
            1000. * value * value / 4.
        })
        .sum::<f64>();
    // Four equally spaced phases integrate a squared sinusoid exactly. The peak phasor
    // I=(1-i)/2000 A gives <R*I(t)^2>=R*|Ihat|²/2=1/4000 W, not 1/2000 W.
    // The 1e-10 A component bound contributes <2e-10 W; allow 3e-10 W with roundoff.
    assert!((average_loss - 0.00025).abs() < 3e-10, "{average_loss}");
    assert!((average_loss - 0.0005).abs() > 0.00024);
}
