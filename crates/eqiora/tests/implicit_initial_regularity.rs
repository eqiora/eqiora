//! Residual-native initialization must establish its declared local partition.
use eqiora::api::ModelDocument;
use eqiora::diagnostic::codes;
use eqiora::runtime::{CpuProgram, GeneralImplicitProgram};
use eqiora::sem::{Interpreter, ReferenceConfig};
use eqiora::time::{ReferenceImplicitTimeBackend, TimeMethod, TimePlan};

#[test]
fn hidden_and_singular_constraints_reject_after_a_consistent_initial_solve() {
    for (declarations, equations, initial, rank) in [
        (
            "variable z: 1;",
            "(1+x*x)*derivative(x)=rate*z; x=0;",
            "z=0;",
            "rank 1, required 2",
        ),
        (
            "variable z: 1;",
            "(1+x*x)*derivative(x)=-rate*z; z*z=0;",
            "x=1;",
            "rank 1, required 2",
        ),
        (
            "",
            "derivative(x)*derivative(x)=0*rate*rate;",
            "x=1;",
            "rank 0, required 1",
        ),
    ] {
        let model = ModelDocument::compile("irregular.eqi", &format!(
            "model M() {{ parameter rate: 1/s=1; state x: 1; {declarations} initial {{ {initial} }} relation r {{ {equations} }} }}"
        )).unwrap();
        let config = ReferenceConfig::new(0.0, 1.0).unwrap();
        // The hidden-constraint initial Jacobian is regular. For the squared
        // terms, forward secants instead give spurious nonzero derivatives.
        Interpreter::new()
            .initialize(model.program(), config)
            .unwrap();
        let cpu = CpuProgram::lower(model.program()).unwrap();
        let relation = model.aliases()["r"].downcast().unwrap();
        let system = GeneralImplicitProgram::lower(&cpu, relation).unwrap();
        let error = system.initialize(config).unwrap_err();
        assert_eq!(error.code(), codes::INVALID_TIME_LOWERING);
        assert!(error.message().contains(rank), "{error:?}");
        assert!(
            error
                .graph_path()
                .unwrap()
                .to_string()
                .contains(&relation.to_string())
        );
        assert!(system.implicit_problem().is_err());
    }
}

#[test]
fn state_dependent_mass_index_one_path_preserves_equation_order_independence() {
    for equations in [
        "(1+x*x)*derivative(x)=-rate*z; z=x*x;",
        "z=x*x; (1+x*x)*derivative(x)=-rate*z;",
    ] {
        let model = ModelDocument::compile("regular.eqi", &format!(
            "model M() {{ parameter rate: 1/s=1; state x: 1; variable z: 1; initial {{ x=1; }} relation r {{ {equations} }} }}"
        )).unwrap();
        let cpu = CpuProgram::lower(model.program()).unwrap();
        let system =
            GeneralImplicitProgram::lower(&cpu, model.aliases()["r"].downcast().unwrap()).unwrap();
        let initial = system
            .initialize(ReferenceConfig::new(0.0, 1.0).unwrap())
            .unwrap();
        let x = system
            .state_fields()
            .iter()
            .position(|field| field.erase() == model.aliases()["x"])
            .unwrap();
        let z = 1 - x;
        assert!((initial.derivative()[x] + 0.5).abs() < 1e-9);
        let end = 0.2;
        let step = 0.005;
        let plan = TimePlan::new(
            TimeMethod::ImplicitEuler,
            0.0,
            step,
            1e-11,
            vec![1e-13; 2],
            vec![end],
        )
        .unwrap();
        let problem = system.implicit_problem().unwrap();
        let solution = ReferenceImplicitTimeBackend::new()
            .solve(&problem, &plan)
            .unwrap();
        let state = solution.state(0).unwrap();
        // Separation gives 1/x-x=t for rate=x0=1, hence this positive root.
        // f=-x²/(1+x²) is decreasing: |f|<=1/2 and |f'|<=1 on [0,1].
        // The nonexpansive backward-Euler resolvent bounds accumulated error
        // by T*h*max|x''|/2 <= T*h/4, plus the stated solve residual allowance.
        let exact = ((end * end + 4.0).sqrt() - end) / 2.0;
        assert!((state[x] - exact).abs() <= end * step / 4.0 + 1e-8);
        assert!((state[z] - state[x] * state[x]).abs() < 1e-9);
    }
}
