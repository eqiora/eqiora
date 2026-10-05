//! Local constant-mass regularity is independent of equation counts and Newton convergence.

use eqiora::api::ModelDocument;
use eqiora::diagnostic::codes;
use eqiora::runtime::{CpuProgram, FirstOrderProgram};
use eqiora::sem::{Interpreter, ReferenceConfig};

#[test]
fn hidden_constraint_and_singular_algebraic_point_reject_at_their_regularity_boundary() {
    for (equations, initial) in [
        ("derivative(x) = rate*z; x = 0;", "z = 0;"),
        ("derivative(x) = -rate*z; z*z = 0;", "x = 1;"),
    ] {
        let model = ModelDocument::compile("irregular.eqi", &format!(
            "model M() {{ parameter rate: 1/s = 1; state x: 1; variable z: 1; initial {{ {initial} }} relation r {{ {equations} }} }}"
        )).unwrap();
        let config = ReferenceConfig::new(0.0, 1.0).unwrap();
        // The hidden constraint has a full-rank joint initial Jacobian.
        // The squared algebraic constraint is singular already at that gate:
        // d(z²)/dz = 0 at z = 0, independently of residual convergence.
        let common = Interpreter::new().initialize(model.program(), 0.0, config);
        if equations.contains("z*z") {
            let errors = common.unwrap_err();
            assert_eq!(errors[0].code(), codes::NONLINEAR_SOLVE_FAILED);
            assert!(errors[0].message().contains("initial Jacobian"));
        } else {
            let errors = common.unwrap_err();
            assert_eq!(errors[0].code(), codes::NONLINEAR_SOLVE_FAILED);
            assert!(errors[0].message().contains("local regularity"));
        }
        let cpu = CpuProgram::lower(model.program()).unwrap();
        let relation = model.aliases()["r"].downcast().unwrap();
        let system = FirstOrderProgram::lower(cpu.kernel(), relation).unwrap();
        let error = system.initialize(0.0, config).unwrap_err();
        if equations.contains("z*z") {
            assert_eq!(error.code(), codes::NONLINEAR_SOLVE_FAILED);
            assert!(error.message().contains("initial Jacobian"));
        } else {
            assert_eq!(error.code(), codes::NONLINEAR_SOLVE_FAILED);
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
        }
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
            FirstOrderProgram::lower(cpu.kernel(), model.aliases()["r"].downcast().unwrap())
                .unwrap();
        let initial = system
            .initialize(0.0, ReferenceConfig::new(0.0, 1.0).unwrap())
            .unwrap();
        for (index, coordinate) in system.state_coordinates().iter().enumerate() {
            let (field, order) = (coordinate.field(), coordinate.derivative_order());
            assert_eq!(order, 0);
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
