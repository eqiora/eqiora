//! A solvable joint initial system is not an index-one certificate.
use eqiora::api::ModelDocument;
use eqiora::runtime::{CpuExecutor, CpuProgram};
use eqiora::sem::{Interpreter, ReferenceConfig};

#[test]
fn common_execution_rejects_hidden_constraints_and_singular_rate_partitions() {
    for (declarations, initial, equations) in [
        ("variable z: 1;", "z=0;", "derivative(x)=rate*z; x=0;"),
        (
            "variable z: 1;",
            "z=0;",
            "(1+x*x)*derivative(x)=rate*z; x=0;",
        ),
        (
            "",
            "derivative(x)=0*rate;",
            "derivative(x)*derivative(x)=rate*rate*x;",
        ),
    ] {
        let model = ModelDocument::compile("hidden.eqi", &format!(
            "model M() {{ parameter rate: 1/s=1; state x: 1; {declarations} initial {{ {initial} }} relation r {{ {equations} }} }}"
        )).unwrap();
        let config = ReferenceConfig::new(0.2, 0.01).unwrap();
        let cpu = CpuProgram::lower(model.program()).unwrap();
        // Each joint initial system is regular at zero. In the hidden cases,
        // backward Euler is also invertible for every nonzero h. Those facts
        // cannot repair the continuous equation's missing rate constraint.
        for result in [
            Interpreter::new().run(model.program(), config),
            CpuExecutor::new().run(&cpu, config),
        ] {
            let diagnostics = result.expect_err("unsupported continuous partition must reject");
            assert!(
                diagnostics.iter().any(|error| {
                    error.message().contains("regularity") || error.message().contains("high-index")
                }),
                "{diagnostics:?}"
            );
        }
    }
}

#[test]
fn common_execution_preserves_a_coupled_constant_mass_descriptor() {
    for equations in [
        "derivative(x)+derivative(y)+2*rate*x=0; x-y=0;",
        "derivative(x)+derivative(y)+2*rate*x=0; derivative(x)+derivative(y)+2*rate*y=0;",
    ] {
        let model = ModelDocument::compile("descriptor.eqi", &format!(
            "model M() {{ parameter rate: 1/s=1; state x: 1; state y: 1; initial {{ x=1; }} relation r {{ {equations} }} }}"
        )).unwrap();
        let config = ReferenceConfig::new(0.2, 0.01).unwrap();
        let cpu = CpuProgram::lower(model.program()).unwrap();
        for trajectory in [
            Interpreter::new().run(model.program(), config).unwrap(),
            CpuExecutor::new().run(&cpu, config).unwrap(),
        ] {
            // Both row presentations imply x=y and x'=-x. Backward Euler
            // gives (1+h)^(-20); comparison here is to the continuous solution.
            // Nonexpansiveness and |x''|<=1 bound global error by T*h/2.
            let x = trajectory.last_value(model.aliases()["x"]).unwrap().value();
            let y = trajectory.last_value(model.aliases()["y"]).unwrap().value();
            assert!((x - (-0.2_f64).exp()).abs() <= 0.2 * 0.01 / 2.0 + 1e-8);
            assert!((x - y).abs() < 1e-9);
        }
    }
}

#[test]
fn a_time_dependent_mass_is_not_made_constant_at_the_initial_point() {
    let model = ModelDocument::compile(
        "time-mass.eqi",
        "model M(){state x:1; initial{derivative(x)=0[1/s];} relation r{time()*derivative(x)=x;}}",
    )
    .unwrap();
    let errors = Interpreter::new()
        .initialize(
            model.program(),
            0.0,
            ReferenceConfig::new(0.0, 0.1).unwrap(),
        )
        .unwrap_err();
    assert!(errors[0].message().contains("regularity"));
}

#[test]
fn nonlinear_value_terms_preserve_a_regular_constant_mass_descriptor() {
    let model = ModelDocument::compile("sin-descriptor.eqi", "model M(){parameter rate:1/s=1; state x:1; state y:1; initial{x=0;derivative(x)=0*rate;} relation r{derivative(x)+derivative(y)=-2*rate*x;x=math.sin(y);}}").unwrap();
    // At zero the constraint tangent is x'-y'=0; together with x'+y'=0
    // this determines both rates as zero. Nonlinearity is only in values.
    let initial = Interpreter::new()
        .initialize(
            model.program(),
            0.0,
            ReferenceConfig::new(0.0, 0.1).unwrap(),
        )
        .unwrap();
    for alias in ["x", "y"] {
        assert_eq!(
            initial.derivatives()[&(model.aliases()[alias], std::num::NonZeroU32::MIN)]
                .real_scalar_value()
                .unwrap()
                .value(),
            0.0
        );
    }
}

#[test]
fn an_independent_nonlinear_rate_does_not_change_descriptor_admission() {
    let model = ModelDocument::compile("separate.eqi", "model M(){parameter rate:1/s=1; state x:1; state y:1; state u:1; initial{x=1;derivative(x)=-rate;u=0;} relation r{derivative(x)+derivative(y)=-2*rate*x;x-y=0;derivative(u)+derivative(u)*derivative(u)*derivative(u)/(rate*rate)=0;}}").unwrap();
    // The independent monotone equation v+v^3=0 has unique rate v=0.
    let initial = Interpreter::new()
        .initialize(
            model.program(),
            0.0,
            ReferenceConfig::new(0.0, 0.1).unwrap(),
        )
        .unwrap();
    for alias in ["x", "y"] {
        assert!(
            (initial.derivatives()[&(model.aliases()[alias], std::num::NonZeroU32::MIN)]
                .real_scalar_value()
                .unwrap()
                .value()
                + 1.0)
                .abs()
                < 1e-9
        );
    }
    assert_eq!(
        initial.derivatives()[&(model.aliases()["u"], std::num::NonZeroU32::MIN)]
            .real_scalar_value()
            .unwrap()
            .value(),
        0.0
    );
}
