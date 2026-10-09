mod support;

use eqiora::api::DifferentiableProgram;
use eqiora::compiler::{ModelSymbols, compile};
use eqiora::graph::{GraphStore, InMemoryGraphStore};
use eqiora::ir::{
    DifferentiationRole, LinearizedRelation, RelationCotangent, RelationTangent, ScalarOperatorIr,
};
use eqiora::kernel::{KernelNode, SymbolRef};
use eqiora::runtime::{CpuExecutor, CpuProgram};
use eqiora::sem::{Interpreter, KernelProgram, ReferenceConfig};
use eqiora_numerics::CommonSpatialPolicy;
use support::common_scalar_plan::{COMPONENT, document_and_plan_with_source};

fn admit(source: &str) -> (KernelProgram, ModelSymbols) {
    let compiled = compile("runtime-aliases.eqi", source)
        .unwrap()
        .pop()
        .unwrap();
    let (transaction, model, symbols) = compiled.into_parts();
    let mut store = InMemoryGraphStore::new();
    store.commit(transaction).unwrap();
    (
        KernelProgram::from_snapshot(&store.snapshot(), model).unwrap(),
        symbols,
    )
}

const MIXED_CLOCKS: &str = r#"
model M() {
  clock a = periodic(1[s] / 1, phase = 1[s] / 1);
  clock b = periodic(1[s] / 1, phase = 1[s] / 1);
  state first: 1 at a;
  state second: 1 at b;
  state integral: s;
  initial { first = 2; second = 3; integral = 0[s]; }
  let sum = first + second;
  let identity = integral;
  relation flow { derivative(identity) = sum; }
  relation update_a at a { next(first) = pre(first); }
  relation update_b at b { next(second) = pre(second); }
}
"#;

#[test]
fn current_state_aliases_preserve_continuous_reads_across_distinct_equal_period_clocks() {
    let explicit = MIXED_CLOCKS
        .replace(
            "  let sum = first + second;\n  let identity = integral;\n",
            "",
        )
        .replace(
            "derivative(identity) = sum",
            "derivative(integral) = first + second",
        );
    let asserted = MIXED_CLOCKS.replace(
        "  let sum = first + second;",
        "  let observed_first at a = first;\n  let sum = observed_first + second;",
    );
    for source in [explicit.as_str(), MIXED_CLOCKS, asserted.as_str()] {
        let (program, symbols) = admit(source);
        assert_ne!(symbols.get("a"), symbols.get("b"));
        assert!(symbols.get("sum").is_none());
        assert!(symbols.get("identity").is_none());
        assert_eq!(
            program
                .nodes()
                .filter(|node| matches!(node, KernelNode::Field(_)))
                .count(),
            3
        );
        let config = ReferenceConfig::new(0.5, 0.1).unwrap();
        let reference = Interpreter::new().run(&program, config).unwrap();
        let cpu = CpuExecutor::new()
            .run(&CpuProgram::lower(&program).unwrap(), config)
            .unwrap();
        assert_eq!(reference, cpu);
        // Both memories remain constant until t=1: integral(0.5)=(2+3)*0.5.
        assert!(
            (reference
                .last_value(symbols.get("integral").unwrap())
                .unwrap()
                .value()
                - 2.5)
                .abs()
                < 1e-12
        );
        assert_eq!(
            reference
                .last_value(symbols.get("first").unwrap())
                .unwrap()
                .value(),
            2.0
        );
        assert_eq!(
            reference
                .last_value(symbols.get("second").unwrap())
                .unwrap()
                .value(),
            3.0
        );
    }
}

#[test]
fn alias_hidden_evolution_operators_keep_the_exact_context_checks() {
    let prefix = "clock a = periodic(1[s] / 1, phase = 0[s] / 1); clock b = periodic(1[s] / 1, phase = 0[s] / 1); state x: 1 at a;";
    for body in [
        "let old = pre(x); relation r at b { old = 0; }",
        "let old at a = pre(x); relation r at b { old = 0; }",
        "let future at a = next(x); initial { future = 0; }",
        "let old at a = pre(x); relation r { old = 0; }",
        "let future = next(x); initial { future = 0; }",
        "let old = pre(x); relation r { old = 0; }",
        "let composite = 2*x; relation r at a { next(composite) = 0; }",
    ] {
        assert!(
            compile(
                "bad-context.eqi",
                &format!("model M() {{ {prefix} {body} }}")
            )
            .is_err(),
            "{body}"
        );
    }
    // Initialization reads each memory's initial value without selecting one update clock.
    let source = format!(
        "model M() {{ {prefix} state y: 1 at b; let old_x = pre(x); let old_y = pre(y); initial {{ old_x = 2; old_y = 3; }} }}"
    );
    let explicit = source
        .replace("let old_x = pre(x); let old_y = pre(y);", "")
        .replace("old_x = 2", "pre(x) = 2")
        .replace("old_y = 3", "pre(y) = 3");
    admit(&explicit);
    admit(&source);
}

#[test]
fn runtime_polynomial_aliases_preserve_unknown_and_parameter_chain_rules_per_occurrence() {
    let aliased = r#"
component Polynomial(
  parameter target: 1,
  parameter y: 1
) {
  variable x: 1;
  variable output: 1;
  let f = z * y;
  let z = x * x;
  relation fix { x = target; }
  relation result { output = f; }
}
model M() {
  parameter y1: 1 = 3;
  parameter y2: 1 = -2;
  instance first: Polynomial(target = 2, y = y1);
  instance second: Polynomial(target = 5, y = y2);
}
"#;
    let explicit = aliased
        .replace("  let f = z * y;\n  let z = x * x;\n", "")
        .replace("output = f", "output = (x*x)*y");
    for source in [explicit.as_str(), aliased] {
        let (program, symbols) = admit(source);
        assert_eq!(
            program
                .nodes()
                .filter(|node| matches!(node, KernelNode::Field(_)))
                .count(),
            4
        );
        assert_eq!(
            program
                .nodes()
                .filter(|node| matches!(node, KernelNode::Relation(_)))
                .count(),
            4
        );
        for (occurrence, x, y) in [("first", 2.0, 3.0), ("second", 5.0, -2.0)] {
            assert!(symbols.get(&format!("{occurrence}.f")).is_none());
            let x_id = symbols.get(&format!("{occurrence}.x")).unwrap();
            let output_id = symbols.get(&format!("{occurrence}.output")).unwrap();
            let relation = symbols.get(&format!("{occurrence}.result")).unwrap();
            let Some(KernelNode::Relation(relation)) = program.node(relation) else {
                panic!("result Relation");
            };
            let ir = ScalarOperatorIr::lower(
                &program.numerical_residuals(relation.id().erase()).unwrap(),
            )
            .unwrap();
            let mut inputs = Vec::new();
            let mut roles = Vec::new();
            let mut derivatives = Vec::new();
            for symbol in ir.symbols() {
                match symbol {
                    SymbolRef::Field(field) => {
                        let is_x = field.erase() == x_id;
                        assert!(is_x || field.erase() == output_id);
                        inputs.push(if is_x { x } else { x * x * y });
                        roles.push(DifferentiationRole::Unknown);
                        derivatives.push(if is_x { -2.0 * x * y } else { 1.0 });
                    }
                    SymbolRef::Parameter(_) => {
                        inputs.push(y);
                        roles.push(DifferentiationRole::Parameter);
                    }
                    _ => panic!("only original Field and Parameter dependencies"),
                }
            }
            assert_eq!(ir.evaluate(&inputs).unwrap(), [0.0]);
            let linearized = ir.linearize(&inputs, &roles).unwrap();
            let mut unknown = [0.0; 2];
            let mut parameter = [0.0];
            linearized
                .vjp(
                    &[1.0],
                    RelationCotangent::Both {
                        unknown: &mut unknown,
                        parameter: &mut parameter,
                    },
                )
                .unwrap();
            assert_eq!(unknown.as_slice(), derivatives.as_slice());
            assert_eq!(parameter, [-x * x]);
            let mut tangent = [0.0];
            linearized
                .jvp(
                    RelationTangent::Both {
                        unknown: &[1.0, 1.0],
                        parameter: &[2.0],
                    },
                    &mut tangent,
                )
                .unwrap();
            assert_eq!(tangent, [1.0 - 2.0 * x * y - 2.0 * x * x]);
        }
    }
}

#[test]
fn runtime_heat_flux_alias_preserves_bounded_spatial_execution_and_parameter_chain_rule() {
    let aliased = COMPONENT
        .replace(
            "  relation balance on square {",
            "  let heat_flux = diffusion * grad(potential);\n  relation balance on square {",
        )
        .replace("-div(diffusion * grad(potential))", "-div(heat_flux)");
    let asserted = aliased.replace("let heat_flux =", "let heat_flux on square =");
    for policy in [
        CommonSpatialPolicy::Q1,
        CommonSpatialPolicy::CellCenteredTpfa,
    ] {
        let mut explicit_output = None;
        for source in [COMPONENT, aliased.as_str(), asserted.as_str()] {
            let (document, plan) = document_and_plan_with_source(policy, source);
            assert_eq!(
                document
                    .program()
                    .nodes()
                    .filter(|node| matches!(node, KernelNode::Field(_)))
                    .count(),
                1
            );
            let input = document.parameter_ref("diffusion").unwrap();
            let output = document
                .field_ref(&plan.fields().next().unwrap().0.ulid().to_string())
                .unwrap();
            let program = DifferentiableProgram::compile(
                eqiora_numerics::ResolvedCommonPlan::Linear(Box::new(plan)),
                &[input],
                &output,
                None,
                &eqiora_solver::REFERENCE_LINEAR_SOLVER,
            )
            .unwrap();
            let primal = program.primal();
            let values = primal.output();
            let tangent = program.jvp(&[1.0]).unwrap();
            // With zero boundary values, A(D)=D*A(1), hence du/dD=-u/D.
            // This fixture sets D=1. The identity follows from the linear equation.
            for (u, derivative) in values.iter().zip(tangent.tangent()) {
                assert!((u + derivative).abs() <= 1e-9 * (1.0 + u.abs()));
            }
            let reverse = program.vjp(&vec![1.0; values.len()]).unwrap();
            assert!(
                (reverse.input_cotangent()[0] + values.iter().sum::<f64>()).abs()
                    <= 1e-8 * (1.0 + values.iter().map(|u| u.abs()).sum::<f64>())
            );
            if let Some(expected) = &explicit_output {
                assert_eq!(values, expected);
            } else {
                explicit_output = Some(values.to_vec());
            }
        }
    }
}

#[test]
fn static_math_aliases_retain_spatial_derivatives_and_domain_failure_after_parameter_edit() {
    let source = COMPONENT
        .replace("  relation balance on square {", "  let coefficient = math.sqrt(diffusion) + math.sin(diffusion);\n  relation balance on square {")
        .replace("-div(diffusion * grad(potential))", "-div(coefficient * grad(potential))");
    for policy in [
        CommonSpatialPolicy::Q1,
        CommonSpatialPolicy::CellCenteredTpfa,
    ] {
        let (document, plan) = document_and_plan_with_source(policy, &source);
        let input = document.parameter_ref("diffusion").unwrap();
        let output = document
            .field_ref(&plan.fields().next().unwrap().0.ulid().to_string())
            .unwrap();
        let program = DifferentiableProgram::compile(
            eqiora_numerics::ResolvedCommonPlan::Linear(Box::new(plan)),
            &[input],
            &output,
            None,
            &eqiora_solver::REFERENCE_LINEAR_SOLVER,
        )
        .unwrap();
        for p in [1.0_f64, 9.0] {
            let point = program.evaluate(&[p]).unwrap();
            let primal = point.primal();
            let tangent = point.jvp(&[1.0]).unwrap();
            // A(p)=k(p)*A(1), k(p)=sqrt(p)+sin(p), so du/dp=-u*k'(p)/k(p).
            let factor = -(0.5 / p.sqrt() + p.cos()) / (p.sqrt() + p.sin());
            for (u, derivative) in primal.output().iter().zip(tangent.tangent()) {
                assert!((derivative - factor * u).abs() <= 1e-9 * (1.0 + u.abs()));
            }
            let reverse = point.vjp(&vec![1.0; primal.output().len()]).unwrap();
            let expected = factor * primal.output().iter().sum::<f64>();
            assert!(
                (reverse.input_cotangent()[0] - expected).abs() <= 1e-8 * (1.0 + expected.abs())
            );
        }
        // Rebind only the original Parameter; retained sqrt must still reject its negative domain.
        assert!(program.evaluate(&[-1.0]).is_err());
    }
}
