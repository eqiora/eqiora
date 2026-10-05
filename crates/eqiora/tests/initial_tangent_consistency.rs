//! Initial rates must be tangent to the regular equations, not initial conditions.
use eqiora::api::ModelDocument;
use eqiora::sem::{Interpreter, ReferenceConfig};

#[test]
fn nonlinear_descriptor_initial_rates_follow_the_constraint_tangent() {
    for (constraint, good_rate) in [("x-y*y=0;", "-4*rate/3"), ("x-y*y-rate*time()=0;", "-rate")] {
        for (rate, accepted) in [("0*rate", false), (good_rate, true)] {
            let model = ModelDocument::compile("initial-tangent.eqi", &format!(
                "model M(){{parameter rate:1/s=1; state x:1; state y:1; initial{{x=1;derivative(x)={rate};}} relation r{{derivative(x)+derivative(y)=-2*rate*x;{constraint}}}}}"
            )).unwrap();
            // The initial guess selects y=+1. The regular rate equation gives
            // x'+y'=-2. Differentiating x-y²-c*t=0 gives x'-2y'=c,
            // hence x'=(c-4)/3; c is independently 0 or 1 here.
            let result = Interpreter::new().initialize(
                model.program(),
                0.0,
                ReferenceConfig::new(0.0, 0.1)
                    .unwrap()
                    .with_initial_guess(1.0)
                    .unwrap(),
            );
            if accepted {
                let initial = result.unwrap();
                let dx = initial.derivatives()[&(model.aliases()["x"], std::num::NonZeroU32::MIN)];
                let dy = initial.derivatives()[&(model.aliases()["y"], std::num::NonZeroU32::MIN)];
                let c = if constraint.contains("time()") {
                    1.0
                } else {
                    0.0
                };
                assert!((dx - 2.0 * dy - c).abs() < 1e-7);
            } else {
                let errors = result.expect_err("inconsistent initial rates must reject");
                assert!(
                    errors
                        .iter()
                        .any(|error| error.message().contains("tangent")),
                    "{errors:?}"
                );
            }
        }
    }
}

#[test]
fn only_unprovided_algebraic_rates_are_free_in_tangent_compatibility() {
    for (condition, accepted) in [
        ("", true),
        ("derivative(z)=-2*rate;", true),
        ("derivative(z)=0*rate;", false),
    ] {
        let model = ModelDocument::compile("algebraic-rate.eqi", &format!(
            "model M(){{parameter rate:1/s=1;state x:1;state z:1;initial{{x=1;{condition}}}relation r{{derivative(x)=-rate*z;z=x*x;}}}}"
        )).unwrap();
        // At x=z=1, x'=-1 and z'=2*x*x'=-2. An unprovided z' is
        // an existential rate, whereas an authored z' must already agree.
        let config = ReferenceConfig::new(0.0, 0.1)
            .unwrap()
            .with_initial_guess(1.0)
            .unwrap();
        let result = Interpreter::new().initialize(model.program(), 0.0, config);
        if accepted {
            let initial = result.unwrap();
            if condition.is_empty() {
                assert!(
                    !initial
                        .derivatives()
                        .contains_key(&(model.aliases()["z"], std::num::NonZeroU32::MIN))
                );
            }
        } else {
            let errors = result.expect_err("authored algebraic rate must not be freed");
            assert!(
                errors
                    .iter()
                    .any(|error| error.message().contains("tangent")),
                "{errors:?}"
            );
        }
        let cpu = eqiora::runtime::CpuProgram::lower(model.program()).unwrap();
        assert_eq!(
            eqiora::runtime::CpuExecutor::new()
                .run(&cpu, config)
                .is_ok(),
            accepted
        );
    }
}

#[test]
fn tangent_ad_does_not_require_finite_acceleration_in_unconstrained_rows() {
    for (extra, initial, equations) in [
        ("", "x=0;", "derivative(x)=rate*math.sqrt(time()/tau);"),
        (
            "state y:1;",
            "x=0;derivative(x)=0*rate;",
            "derivative(x)+derivative(y)=rate*math.sqrt(time()/tau); x-y=0;",
        ),
        (
            "state y:1;",
            "x=0;derivative(x)=0*rate;",
            "derivative(x)+derivative(y)=rate*math.sqrt(time()/tau)-rate*x; derivative(x)+derivative(y)=rate*math.sqrt(time()/tau)-rate*y;",
        ),
        (
            "state y:1;",
            "x=0;derivative(x)=0*rate;",
            "derivative(x)+derivative(y)=rate*math.sqrt(time()/tau)-rate*x; 0.1*(0.1*(derivative(x)+derivative(y)))=0.1*(0.1*rate*math.sqrt(time()/tau))-rate*y;",
        ),
        (
            "state y:1;",
            "x=0;derivative(x)=0*rate;",
            "derivative(x)+derivative(y)=rate*math.sqrt(time()/tau)-rate*x; (0.1*0.1)*(derivative(x)+derivative(y))=0.1*(0.1*rate*math.sqrt(time()/tau))-rate*y;",
        ),
    ] {
        let source = format!(
            "model M(){{parameter rate:1/s=1;parameter tau:s=1;state x:1;{extra}initial{{{initial}}}relation r{{{equations}}}}}"
        );
        let model = ModelDocument::compile("finite-velocity.eqi", &source).unwrap();
        // Integral sqrt(t) is (2/3)*t^(3/2), with finite initial rate zero
        // but unbounded right acceleration. The first two descriptors imply
        // x=y; the final scaled form implies y=0.01*x. Their common forcing
        // cancels algebraically, including the non-binary-exact scale product.
        let initial = Interpreter::new()
            .initialize(
                model.program(),
                0.0,
                ReferenceConfig::new(0.0, 0.1).unwrap(),
            )
            .unwrap();
        assert_eq!(
            initial.derivatives()[&(model.aliases()["x"], std::num::NonZeroU32::MIN)],
            0.0
        );
    }
}

#[test]
fn rounded_coefficient_cancellation_cannot_hide_a_required_singular_derivative() {
    let model = ModelDocument::compile("rounded-term.eqi", "model M(){parameter tau:s=1;state z:1;initial{derivative(z)=0[1/s];}relation r{z=(9007199254740992.0*math.sqrt(time()/tau)+math.sqrt(time()/tau))-9007199254740992.0*math.sqrt(time()/tau);}}").unwrap();
    // The exact scalar coefficient is 2^53+1-2^53=1, never zero.
    let errors = Interpreter::new()
        .initialize(
            model.program(),
            0.0,
            ReferenceConfig::new(0.0, 0.1).unwrap(),
        )
        .unwrap_err();
    assert!(
        errors
            .iter()
            .any(|error| error.message().contains("square-root derivative")),
        "{errors:?}"
    );
}
