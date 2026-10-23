use eqiora::api::ModelDocument;
use eqiora::backends::diffsol::DiffsolTimeBackend;
use eqiora::runtime::{CpuProgram, FirstOrderProgram};
use eqiora::time::{ForwardSensitivityPlan, TimeMethod, TimePlan};

#[test]
fn source_authored_initial_pairs_and_sensitivities_preserve_residual_row_changes() {
    for constrained in [false, true] {
        let (variables, initial, first, second) = if constrained {
            (
                "state x: 1; variable z: 1;",
                "x = 1;",
                "derivative(x) + rate*z",
                "rate*(z-x*x)",
            )
        } else {
            (
                "state x: 1; state z: 1;",
                "x = 1; z = 2;",
                "derivative(x) + rate*x",
                "derivative(z) + rate*z",
            )
        };
        // Identity, permutation, and determinant -2 mixing are equivalent
        // equation bases. Each branch starts at ordinary source compilation.
        for equations in [
            format!("{first}=0[1/s]; {second}=0[1/s];"),
            format!("{second}=0[1/s]; {first}=0[1/s];"),
            format!("({first})+({second})=0[1/s]; ({first})-({second})=0[1/s];"),
        ] {
            let source = format!(
                "model M() {{ parameter rate: 1/s=2; {variables} initial {{ {initial} }} relation r {{ {equations} }} }}"
            );
            let model = ModelDocument::compile("initial-pair-rows.eqi", &source).unwrap();
            let cpu = CpuProgram::lower(model.program()).unwrap();
            let system =
                FirstOrderProgram::lower(cpu.kernel(), model.aliases()["r"].downcast().unwrap())
                    .unwrap();
            let primal = system.time_problem().unwrap();
            let problem = system.forward_sensitivity_problem(0.0).unwrap();
            assert_eq!(primal.initialization(), problem.primal().initialization());
            let initial_tangent = problem.initialization_parameter_jvp(0.0, &[1.0]).unwrap();
            let plan =
                TimePlan::new(TimeMethod::Bdf, 0.0, 1e-5, 1e-8, vec![1e-10; 2], vec![0.1]).unwrap();
            let backend = DiffsolTimeBackend::new();
            let solution = backend.solve(&primal, &plan).unwrap();
            let sensitive = backend
                .solve_forward_sensitivities(
                    &problem,
                    &plan,
                    &ForwardSensitivityPlan::new(1e-8, vec![1e-10; 2]).unwrap(),
                )
                .unwrap();
            for (index, coordinate) in system.state_coordinates().iter().enumerate() {
                let is_x = coordinate.field().erase() == model.aliases()["x"];
                // Independent solutions: x=1/(1+p*t), z=x²; or
                // (x,z)=(1,2)*exp(-p*t). Differentiate these identities.
                let (value, sensitivity, initial_rate_tangent) = if constrained {
                    if is_x {
                        (1.0 / 1.2, -0.1 / 1.2_f64.powi(2), -1.0)
                    } else {
                        (1.0 / 1.2_f64.powi(2), -0.2 / 1.2_f64.powi(3), 0.0)
                    }
                } else {
                    let amplitude = if is_x { 1.0 } else { 2.0 };
                    let value = amplitude * (-0.2_f64).exp();
                    (value, -0.1 * value, -amplitude)
                };
                assert_eq!(initial_tangent.state()[index], 0.0);
                assert!((initial_tangent.derivative()[index] - initial_rate_tangent).abs() < 1e-9);
                assert!((solution.state(0).unwrap()[index] - value).abs() < 2e-6);
                assert!((sensitive.primal().state(0).unwrap()[index] - value).abs() < 2e-6);
                assert!((sensitive.sensitivity(0, 0).unwrap()[index] - sensitivity).abs() < 3e-6);
            }
        }
    }
}
