//! Local constant-mass regularity is independent of equation counts and Newton convergence.

use eqiora::api::ModelDocument;
use eqiora::diagnostic::codes;
use eqiora::runtime::{CpuProgram, FirstOrderProgram};
use eqiora::sem::{Interpreter, ReferenceConfig};

#[test]
fn hidden_constraint_and_singular_algebraic_point_fail_after_consistent_initialization() {
    for (equations, initial) in [
        ("derivative(x) = rate*z; x = 0;", "z = 0;"),
        ("derivative(x) = -rate*z; z*z = 0;", "x = 1;"),
    ] {
        let model = ModelDocument::compile("irregular.eqi", &format!(
            "model M() {{ parameter rate: 1/s = 1; state x: 1; variable z: 1; initial {{ {initial} }} relation r {{ {equations} }} }}"
        )).unwrap();
        let config = ReferenceConfig::new(0.0, 1.0).unwrap();
        // These are consistent at residual zero. In the first case the full
        // initial equations have a nonsingular Jacobian; in the second, forward
        // finite differences produce a spurious nonzero derivative of z*z.
        Interpreter::new()
            .initialize(model.program(), config)
            .unwrap();
        let cpu = CpuProgram::lower(model.program()).unwrap();
        let relation = model.aliases()["r"].downcast().unwrap();
        let system = FirstOrderProgram::lower(&cpu, relation).unwrap();
        let error = system.initialize(config).unwrap_err();
        assert_eq!(error.code(), codes::INVALID_TIME_LOWERING);
        assert!(
            error
                .message()
                .contains("local regularity rank 2, required 3"),
            "{error:?}"
        );
        assert!(
            error
                .graph_path()
                .unwrap()
                .to_string()
                .contains(&relation.to_string())
        );
        assert!(system.time_problem().is_err());
    }
}

#[test]
fn nonlinear_index_one_initial_point_is_regular_under_equation_permutation() {
    for equations in [
        "derivative(x) = -rate*z; z = x*x;",
        "z = x*x; derivative(x) = -rate*z;",
    ] {
        let model = ModelDocument::compile("regular.eqi", &format!(
            "model M() {{ parameter rate: 1/s = 2; state x: 1; variable z: 1; initial {{ x = 1; }} relation r {{ {equations} }} }}"
        )).unwrap();
        let cpu = CpuProgram::lower(model.program()).unwrap();
        let system =
            FirstOrderProgram::lower(&cpu, model.aliases()["r"].downcast().unwrap()).unwrap();
        let initial = system
            .initialize(ReferenceConfig::new(0.0, 1.0).unwrap())
            .unwrap();
        for (index, field) in system.state_fields().iter().enumerate() {
            assert!((initial.state()[index] - 1.0).abs() < 1e-9);
            let derivative = if field.erase() == model.aliases()["x"] {
                -2.0
            } else {
                0.0
            };
            assert!((initial.derivative()[index] - derivative).abs() < 1e-9);
        }
        system.time_problem().unwrap();
    }
}
