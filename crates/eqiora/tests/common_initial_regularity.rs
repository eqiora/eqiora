//! The common initializer must not confuse a forward secant with a derivative.
use eqiora::api::ModelDocument;
use eqiora::runtime::{CpuExecutor, CpuProgram};
use eqiora::sem::{Interpreter, ReferenceConfig};

#[test]
fn singular_algebraic_and_initial_points_reject_in_reference_and_cpu_execution() {
    for source in [
        "model M() { variable x: 1; relation r { x*x=0; } }",
        "model M() { parameter rate: 1/s=1; state x: 1; initial { x*x=0; } relation r { derivative(x)=-rate*x; } }",
    ] {
        let model = ModelDocument::compile("singular-point.eqi", source).unwrap();
        let config = ReferenceConfig::new(0.0, 0.1).unwrap();
        let errors = Interpreter::new()
            .initialize(model.program(), 0.0, config)
            .unwrap_err();
        assert!(
            errors[0].message().contains("initial Jacobian"),
            "{errors:?}"
        );
        let cpu = CpuProgram::lower(model.program()).unwrap();
        let errors = CpuExecutor::new().run(&cpu, config).unwrap_err();
        assert!(
            errors[0].message().contains("initial Jacobian"),
            "{errors:?}"
        );
    }
}

#[test]
fn regular_initial_equations_retain_the_unique_zero_state() {
    let model = ModelDocument::compile("regular-point.eqi", "model M() { parameter rate: 1/s=1; state x: 1; initial { x=0; } relation r { derivative(x)=-rate*x; } }").unwrap();
    let initial = Interpreter::new()
        .initialize(
            model.program(),
            0.0,
            ReferenceConfig::new(0.0, 0.1).unwrap(),
        )
        .unwrap();
    assert_eq!(
        initial.fields()[&model.aliases()["x"]]
            .real_scalar_value()
            .unwrap()
            .value(),
        0.0
    );
    assert_eq!(
        initial.derivatives()[&(model.aliases()["x"], std::num::NonZeroU32::MIN)],
        0.0
    );
}

#[test]
fn frozen_typed_expressions_keep_their_regular_scalar_solution() {
    for (source, expected) in [
        (
            "model M() { parameter p: array<1,2>=[2,3]; variable x: 1; relation r { x=p[0]; } }",
            2.0,
        ),
        (
            "model M() { parameter n: integer=7; variable x: 1; relation r { x=to_real(quotient(n, 2)); } }",
            3.0,
        ),
        (
            "model M() { variable x: 1; relation r { x=math.sqrt(0); } }",
            0.0,
        ),
        (
            "model M() { parameter n: integer=7; variable x: 1; relation r { x=if n > 0 then 2 else math.sqrt(-1); } }",
            2.0,
        ),
    ] {
        let model = ModelDocument::compile("frozen-point.eqi", source).unwrap();
        let initial = Interpreter::new()
            .initialize(
                model.program(),
                0.0,
                ReferenceConfig::new(0.0, 0.1).unwrap(),
            )
            .unwrap();
        assert_eq!(
            initial.fields()[&model.aliases()["x"]]
                .real_scalar_value()
                .unwrap()
                .value(),
            expected
        );
    }
}

#[test]
fn scalar_equations_can_select_from_constructed_active_arrays() {
    for expression in [
        "[2*x,3][0]",
        "([[2*x,3],[4,5]][0])[0]",
        "(if true then [2*x,3] else [4*x,5])[0]",
        "math.sqrt([0,x][0])",
    ] {
        // The first three equations reduce to -x=0; the last to x=0.
        let model = ModelDocument::compile(
            "active-array.eqi",
            &format!("model M() {{ variable x: 1; relation r {{ x={expression}; }} }}"),
        )
        .unwrap();
        let initial = Interpreter::new()
            .initialize(
                model.program(),
                0.0,
                ReferenceConfig::new(0.0, 0.1).unwrap(),
            )
            .unwrap();
        assert_eq!(
            initial.fields()[&model.aliases()["x"]]
                .real_scalar_value()
                .unwrap()
                .value(),
            0.0
        );
    }
}

#[test]
fn unindexed_channel_array_arithmetic_retains_its_existing_rejection() {
    let model = ModelDocument::compile(
        "array-arithmetic.eqi",
        "model M() { variable x: 1; relation r { x=([x,3]+[x,4])[0]; } }",
    )
    .unwrap();
    let errors = Interpreter::new()
        .initialize(
            model.program(),
            0.0,
            ReferenceConfig::new(0.0, 0.1).unwrap(),
        )
        .unwrap_err();
    assert!(
        errors[0]
            .message()
            .contains("channel arrays require explicit indexing before arithmetic")
    );
}
