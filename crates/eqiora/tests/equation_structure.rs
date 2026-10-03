//! Structural incidence is necessary, but does not certify numerical rank or index.

use eqiora::api::ModelDocument;
use eqiora::diagnostic::codes;
use eqiora::runtime::{CpuExecutor, CpuProgram};
use eqiora::sem::{Interpreter, ReferenceConfig};

fn compile_model(source: &str) -> ModelDocument {
    ModelDocument::compile("equation-structure.eqi", source).unwrap()
}

#[test]
fn equal_counts_do_not_hide_a_structurally_deficient_block() {
    // Three rows, three columns, but two rows can only match x. Permuting
    // equations cannot create the missing second equation for {y,z}.
    for equations in ["x = 0; x = 0; y + z = 0;", "y + z = 0; x = 0; x = 0;"] {
        let model = compile_model(&format!(
            "model M() {{ variable x: 1; variable y: 1; variable z: 1; relation r {{ {equations} }} }}"
        ));
        let errors = Interpreter::new()
            .initialize(model.program(), ReferenceConfig::new(0.0, 0.1).unwrap())
            .unwrap_err();
        assert_eq!(errors[0].code(), codes::NONLINEAR_SOLVE_FAILED);
        assert!(
            errors[0].message().contains("incidence rank 2 for 3"),
            "{errors:?}"
        );
        assert!(errors[0].message().contains("unmatched equation"));
        assert!(
            errors[0]
                .graph_path()
                .unwrap()
                .to_string()
                .contains(&model.aliases()["r"].to_string())
        );
    }
}

#[test]
fn initial_equations_cannot_complete_an_incomplete_regular_system() {
    let model = compile_model(
        "model M() { variable x: 1; variable y: 1; relation r { x + y = 3; } initial { x = 1; } }",
    );
    let errors = Interpreter::new()
        .initialize(model.program(), ReferenceConfig::new(0.0, 0.1).unwrap())
        .unwrap_err();
    assert_eq!(errors[0].code(), codes::NONSQUARE_SYSTEM);
    assert!(errors[0].message().contains("1 equations and 2 unknowns"));
    assert!(
        errors[0]
            .message()
            .contains("overdetermined block: 0 equations [], 0 unknowns []")
    );
    assert!(
        errors[0]
            .message()
            .contains("underdetermined block: 1 equations")
    );
    assert!(
        errors[0]
            .message()
            .contains("initial equations are separate")
    );
}

#[test]
fn full_incidence_rank_does_not_accept_duplicate_numeric_equations() {
    // Every row references both unknowns: a perfect incidence matching exists,
    // but the numerical Jacobian has two identical rows even at residual zero.
    let model = compile_model(
        "model M() { variable x: 1; variable y: 1; relation r { x + y = 0; x + y = 0; } }",
    );
    let errors = Interpreter::new()
        .initialize(model.program(), ReferenceConfig::new(0.0, 0.1).unwrap())
        .unwrap_err();
    assert_eq!(errors[0].code(), codes::NONLINEAR_SOLVE_FAILED);
    assert!(errors[0].message().contains("Newton Jacobian is singular"));
}

#[test]
fn component_occurrences_and_expression_aliases_keep_distinct_unknowns() {
    let model = compile_model(
        r#"
        component Pair(parameter a: 1, parameter b: 1) {
            variable x: 1;
            variable y: 1;
            let sum = x + y;
            relation equations { sum = a; x - y = b; }
        }
        model M() {
            instance first: Pair(a = 3, b = 1);
            instance second: Pair(a = 8, b = 2);
        }
    "#,
    );
    let initial = Interpreter::new()
        .initialize(model.program(), ReferenceConfig::new(0.0, 0.1).unwrap())
        .unwrap();
    for (name, expected) in [
        ("first.x", 2.0),
        ("first.y", 1.0),
        ("second.x", 5.0),
        ("second.y", 3.0),
    ] {
        let actual = initial.fields()[&model.aliases()[name]]
            .real_scalar_value()
            .unwrap()
            .value();
        assert!((actual - expected).abs() < 1e-9);
    }
    assert_eq!(initial.fields().len(), 4, "let does not add an unknown");
}

#[test]
fn nonlinear_index_one_dae_uses_the_common_source_and_cpu_execution() {
    for (rate, initial) in [(1.0_f64, 1.0_f64), (0.5, 2.0)] {
        let model = compile_model(&format!(
            r#"
            model M() {{
                parameter rate: 1/s = {rate};
                state x: 1;
                variable z: 1;
                initial {{ x = {initial}; }}
                relation dynamics {{ derivative(x) = -rate*z; z = x*x; }}
            }}
        "#
        ));
        let end = 0.5;
        let step = 0.01;
        let config = ReferenceConfig::new(end, step)
            .unwrap()
            .with_nonlinear_tolerances(1e-12, 0.0)
            .unwrap();
        let accepted = Interpreter::new()
            .initialize(model.program(), config)
            .unwrap();
        let x = model.aliases()["x"];
        let z = model.aliases()["z"];
        assert!((accepted.derivatives()[&x] + rate * initial * initial).abs() < 1e-9);
        assert!(!accepted.derivatives().contains_key(&z));
        let cpu = CpuProgram::lower(model.program()).unwrap();
        for trajectory in [
            Interpreter::new().run(model.program(), config).unwrap(),
            CpuExecutor::new().run(&cpu, config).unwrap(),
        ] {
            let actual = trajectory.last_value(x).unwrap().value();
            let algebraic = trajectory.last_value(z).unwrap().value();
            // x'= -r*x^2 gives x(t)=x0/(1+r*x0*t), z=x^2.
            // On the positive decreasing branch, |x''| <= 2*r^2*x0^3.
            // Backward Euler's monotone resolvent is nonexpansive, so its
            // accumulated truncation error is <= T*h*r^2*x0^3. Residual
            // tolerance contributes less than 1e-8 over these 50 steps.
            let exact = initial / (1.0 + rate * initial * end);
            let bound = end * step * rate * rate * initial.powi(3) + 1e-8;
            assert!((actual - exact).abs() <= bound);
            assert!((algebraic - actual * actual).abs() <= 1e-9);
        }
        let inconsistent = compile_model(&format!(
            r#"
            model Bad() {{ parameter rate: 1/s = {rate}; state x: 1; variable z: 1;
                initial {{ x = {initial}; z = -1; }}
                relation dynamics {{ derivative(x) = -rate*z; z = x*x; }} }}
        "#
        ));
        assert!(
            Interpreter::new()
                .initialize(inconsistent.program(), config)
                .is_err()
        );
    }
}

#[test]
fn deficient_blocks_exclude_an_independent_balanced_component() {
    for equations in [
        "x = 0; x = 0; y + z = 0; w = 1;",
        "w = 1; y + z = 0; x = 0; x = 0;",
    ] {
        let model = compile_model(&format!(
            "model M() {{ variable x: 1; variable y: 1; variable z: 1; variable w: 1; relation r {{ {equations} }} }}"
        ));
        let errors = Interpreter::new()
            .initialize(model.program(), ReferenceConfig::new(0.0, 0.1).unwrap())
            .unwrap_err();
        let message = errors[0].message();
        let (over, under) = message.split_once("underdetermined block:").unwrap();
        assert!(
            over.contains("overdetermined block: 2 equations"),
            "{message}"
        );
        assert!(over.contains("1 unknowns"), "{message}");
        assert!(
            over.contains(&model.aliases()["x"].to_string()),
            "{message}"
        );
        assert!(under.contains("1 equations"), "{message}");
        assert!(under.contains("2 unknowns"), "{message}");
        for name in ["y", "z"] {
            assert!(
                under.contains(&model.aliases()[name].to_string()),
                "{message}"
            );
        }
        assert!(
            !message.contains(&model.aliases()["w"].to_string()),
            "{message}"
        );
    }
}

#[test]
fn deficient_component_occurrence_cannot_borrow_a_balanced_instances_equation() {
    // The broken instance has rank two for three unknowns. The independent
    // balanced instance must not enter either deficient block after elaboration.
    for equations in ["x = 0; x = 0; y + z = 0;", "y + z = 0; x = 0; x = 0;"] {
        let model = compile_model(&format!(
            "component Broken() {{ variable x:1; variable y:1; variable z:1; relation equations {{ {equations} }} }} component Balanced() {{ variable x:1; relation equations {{ x=1; }} }} model M() {{ instance broken:Broken(); instance balanced:Balanced(); }}"
        ));
        let errors = Interpreter::new()
            .initialize(model.program(), ReferenceConfig::new(0.0, 0.1).unwrap())
            .unwrap_err();
        assert_eq!(errors[0].code(), codes::NONLINEAR_SOLVE_FAILED);
        let message = errors[0].message();
        let (over, under) = message.split_once("underdetermined block:").unwrap();
        assert!(
            over.contains(&model.aliases()["broken.x"].to_string()),
            "{message}"
        );
        for name in ["broken.y", "broken.z"] {
            assert!(
                under.contains(&model.aliases()[name].to_string()),
                "{message}"
            );
        }
        assert!(
            !message.contains(&model.aliases()["balanced.x"].to_string()),
            "{message}"
        );
        assert!(
            errors[0]
                .graph_path()
                .unwrap()
                .to_string()
                .contains(&model.aliases()["broken.equations"].to_string()),
            "{errors:?}"
        );
    }
}
