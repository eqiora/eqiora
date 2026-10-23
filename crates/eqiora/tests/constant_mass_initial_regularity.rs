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

#[test]
fn determined_initial_rate_tangent_does_not_differentiate_a_held_time_boundary() {
    let model = ModelDocument::compile("held-time.eqi", "model M() { parameter rate: 1/s=3; state x: 1; initial { x=0; } relation r { derivative(x)=if time()>0[s] then rate*(2+time()/1[s]) else 2*rate; } }").unwrap();
    let cpu = CpuProgram::lower(model.program()).unwrap();
    let system =
        FirstOrderProgram::lower(cpu.kernel(), model.aliases()["r"].downcast().unwrap()).unwrap();
    let problem = system.forward_sensitivity_problem(0.0).unwrap();
    // At fixed t=0 the else branch gives x_dot=2*p. Parameter changes do
    // not move the time selector, so dx_dot/dp=2 even though d/dt is undefined.
    assert_eq!(problem.primal().initial_state(), &[0.0]);
    assert_eq!(problem.primal().initial_derivative(), &[6.0]);
    let tangent = problem.initialization_parameter_jvp(0.0, &[1.0]).unwrap();
    assert_eq!(tangent.state(), &[0.0]);
    assert_eq!(tangent.derivative(), &[2.0]);
}

#[test]
fn supplied_initial_rate_does_not_erase_a_hidden_parameter_compatibility_condition() {
    let model = ModelDocument::compile("initial-rate-compatibility.eqi", "model M() { parameter p: 1/s=1; state x: 1; state y: 1; initial { x=0; derivative(x)=1[1/s]; } relation r { derivative(x)+derivative(y)=2[1/s]; derivative(x)+derivative(y)+p*x+1[1/s]*x*x-1[1/s]*y=2[1/s]; } }").unwrap();
    let cpu = CpuProgram::lower(model.program()).unwrap();
    let system =
        FirstOrderProgram::lower(cpu.kernel(), model.aliases()["r"].downcast().unwrap()).unwrap();
    let problem = system.forward_sensitivity_problem(0.0).unwrap();
    assert_eq!(problem.primal().initial_state(), &[0.0, 0.0]);
    assert_eq!(problem.primal().initial_derivative(), &[1.0, 1.0]);
    // Initial x_dot=1 and the first equation force both rate tangents to 0.
    // But y=p*x+x² implies y_dot=p*x_dot at x=0, so its p tangent is 1.
    // The primal point is consistent; the requested Parameter family is not.
    assert!(problem.initialization_parameter_jvp(0.0, &[1.0]).is_err());
}

#[test]
fn shared_nonsmooth_forcing_does_not_enter_a_hidden_constraint_tangent() {
    let forcing = "(if time()>0[s] then 4*p+p*time()/1[s] else 4*p)";
    for equations in [
        format!(
            "derivative(x)+derivative(y)={forcing}; derivative(x)+derivative(y)+p*(x-y)={forcing};"
        ),
        format!(
            "derivative(x)+derivative(y)+p*(x-y)={forcing}; derivative(x)+derivative(y)={forcing};"
        ),
        format!(
            "5*(derivative(x)+derivative(y))+3*p*(x-y)=5*{forcing}; 3*(derivative(x)+derivative(y))+2*p*(x-y)=3*{forcing};"
        ),
    ] {
        let model = ModelDocument::compile("shared-forcing.eqi", &format!("model M() {{ parameter p: 1/s=1; state x: 1; state y: 1; initial {{ x=0; derivative(x)=2*p; }} relation r {{ {equations} }} }}")).unwrap();
        let cpu = CpuProgram::lower(model.program()).unwrap();
        let system =
            FirstOrderProgram::lower(cpu.kernel(), model.aliases()["r"].downcast().unwrap())
                .unwrap();
        let problem = system.forward_sensitivity_problem(0.0).unwrap();
        assert_eq!(problem.primal().initial_state(), &[0.0, 0.0]);
        assert_eq!(problem.primal().initial_derivative(), &[2.0, 2.0]);
        // Subtraction removes the common forcing and yields x=y for p>0.
        // At fixed t=0, x_dot=y_dot=2*p and both rate tangents equal 2.
        let tangent = problem.initialization_parameter_jvp(0.0, &[1.0]).unwrap();
        assert_eq!(tangent.state(), &[0.0, 0.0]);
        assert_eq!(tangent.derivative(), &[2.0, 2.0]);
    }
}

#[test]
fn pointwise_time_cancellation_cannot_erase_parameter_dependence() {
    let forcing = "(if time()>0[s] then p*p*time() else 0[1/s])";
    let model = ModelDocument::compile("pointwise-forcing.eqi", &format!("model M() {{ parameter p: 1/s=1; state x: 1; state y: 1; initial {{ x=0; derivative(x)=p; }} relation r {{ derivative(x)+derivative(y)=2*p+{forcing}; derivative(x)+derivative(y)+p*(x-y)=2*p+p*1[s]*{forcing}; }} }}")).unwrap();
    let cpu = CpuProgram::lower(model.program()).unwrap();
    let system =
        FirstOrderProgram::lower(cpu.kernel(), model.aliases()["r"].downcast().unwrap()).unwrap();
    let problem = system.forward_sensitivity_problem(0.0).unwrap();
    assert_eq!(problem.primal().initial_derivative(), &[1.0, 1.0]);
    // The forcing cancels at p=1 only. Its Parameter derivative still enters
    // the hidden constraint and requires the undefined time derivative.
    let error = problem
        .initialization_parameter_jvp(0.0, &[1.0])
        .unwrap_err();
    assert!(
        error.message().contains("demanded comparison boundary"),
        "{error:?}"
    );
}
