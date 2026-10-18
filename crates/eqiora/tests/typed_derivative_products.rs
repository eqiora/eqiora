//! Exact base spaces and dual roles survive total first-order actions.

use eqiora::api::{
    DifferentiableCotangent, DifferentiableProgram, DifferentiableTangent, ModelDocument,
};
use eqiora::artifact::ModelEnvelope;
use eqiora::solver::{LinearSolver, ReductionPolicy, SolverPlan};
use eqiora_backend_faer::FaerLinearSolver;
use eqiora_numerics::{
    CommonAlgebraicPlan, CommonLinearRequest, CommonSolvePolicy, ResolvedCommonPlan,
};
use std::num::NonZeroUsize;

fn program(a: u32) -> DifferentiableProgram {
    let source = format!(
        "model M(){{parameter a:1={a};parameter b:1=3;variable x:1;relation r{{x=a+2*b;}}observable y:1=3*x+b;}}"
    );
    let document = ModelDocument::compile("typed-actions.eqi", &source).unwrap();
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let solver = SolverPlan::new(
        LinearSolver::SparseLu,
        1e-13,
        1e-15,
        NonZeroUsize::new(8).unwrap(),
    )
    .unwrap()
    .with_reduction(ReductionPolicy::Fast);
    let plan = CommonAlgebraicPlan::resolve(
        &model,
        CommonSolvePolicy::Linear(
            CommonLinearRequest::exact(solver, FaerLinearSolver.provider()).unwrap(),
        ),
        None,
        &[],
        None,
        &FaerLinearSolver,
    )
    .unwrap();
    DifferentiableProgram::compile(
        ResolvedCommonPlan::Algebraic(Box::new(plan)),
        &[
            document.parameter_ref("a").unwrap(),
            document.parameter_ref("b").unwrap(),
        ],
        &[
            document.field_ref("x").unwrap().erase(),
            document.observable_ref("y").unwrap().erase(),
        ],
        Default::default(),
        None,
        &FaerLinearSolver,
    )
    .unwrap()
}

#[test]
fn actions_retain_the_exact_base_spaces_and_reject_equal_sized_substitutions() {
    let program = program(2);
    let inputs = program.identity().inputs();
    let outputs = program.identity().outputs();
    let tangent =
        DifferentiableTangent::from_components(inputs, &[&[(1.0, 0.0)], &[(-1.0, 0.0)]]).unwrap();
    let cotangent =
        DifferentiableCotangent::from_components(outputs, &[&[(2.0, 0.0)], &[(-1.0, 0.0)]])
            .unwrap();
    let forward = program.jvp(&tangent).unwrap();
    let reverse = program.vjp(&cotangent).unwrap();
    // Authored x=a+2b and y=3a+7b give J=[[1,2],[3,7]].
    assert_eq!(forward.derivative().base_signatures(), outputs);
    assert_eq!(reverse.derivative().base_signatures(), inputs);
    for (index, expected) in [-1.0, -4.0].into_iter().enumerate() {
        assert_eq!(
            forward
                .derivative()
                .components(index)
                .unwrap()
                .collect::<Vec<_>>(),
            [(expected, 0.0)]
        );
    }
    for (index, expected) in [-1.0, -3.0].into_iter().enumerate() {
        assert_eq!(
            reverse
                .derivative()
                .components(index)
                .unwrap()
                .collect::<Vec<_>>(),
            [(expected, 0.0)]
        );
    }
    assert!(forward.derivative().components(2).is_none());
    assert!(reverse.derivative().components(2).is_none());
    assert_eq!(forward.output(), program.primal().output());
    assert_eq!(reverse.output(), forward.output());

    let components = &[&[(1.0, 0.0)][..], &[(1.0, 0.0)][..]];
    for signatures in [outputs.to_vec(), vec![inputs[1].clone(), inputs[0].clone()]] {
        let wrong = DifferentiableTangent::from_components(&signatures, components).unwrap();
        assert!(
            program
                .jvp(&wrong)
                .unwrap_err()
                .message()
                .contains("ordered base-space signatures")
        );
    }
    for signatures in [
        inputs.to_vec(),
        vec![outputs[1].clone(), outputs[0].clone()],
    ] {
        let wrong = DifferentiableCotangent::from_components(&signatures, components).unwrap();
        assert!(
            program
                .vjp(&wrong)
                .unwrap_err()
                .message()
                .contains("ordered base-space signatures")
        );
    }
    // A changed immutable Model has the same dimensions and Jacobian, but cannot
    // lend either base space to this Program.
    let foreign = self::program(4);
    assert_ne!(foreign.identity().model(), program.identity().model());
    let wrong =
        DifferentiableTangent::from_components(foreign.identity().inputs(), components).unwrap();
    assert!(program.jvp(&wrong).is_err());
    let wrong =
        DifferentiableCotangent::from_components(foreign.identity().outputs(), components).unwrap();
    assert!(program.vjp(&wrong).is_err());
    assert_eq!(program.jvp(&tangent).unwrap(), forward);
    assert_eq!(program.vjp(&cotangent).unwrap(), reverse);
}

#[test]
fn derivative_components_use_the_existing_finite_real_domain_admission() {
    let program = program(2);
    let inputs = program.identity().inputs();
    let outputs = program.identity().outputs();
    assert!(DifferentiableTangent::from_components(inputs, &[&[(1.0, 0.0)]]).is_err());
    for invalid in [(f64::NAN, 0.0), (1.0, f64::INFINITY), (1.0, 1.0)] {
        assert!(
            DifferentiableTangent::from_components(inputs, &[&[invalid], &[(0.0, 0.0)]]).is_err()
        );
        assert!(
            DifferentiableCotangent::from_components(outputs, &[&[invalid], &[(0.0, 0.0)]])
                .is_err()
        );
    }
}
