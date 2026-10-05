//! Exact coordinate binding is Model meaning; numerical reconstruction stays with the Result.
#[path = "point_evaluation/fields.rs"]
mod fields;
mod support;
use eqiora::api::ModelDocument;
use eqiora::compiler::StaticBindingValue;
use eqiora::kernel::AxisBounds;
use eqiora::{DimExponents, DynQuantity};

const ANALYTIC: &str = r#"
operator jump(input x:m):m=if x<0.25[m] then 0[m] else 1[m];
model Probe(support position:interval(m)) {
  coordinate x:m on position from position;
  parameter sample:m=0.25;
  parameter gain:1=2;
  variable anchor:1;
  relation retain { anchor=1; }
  observable ramp:m=evaluate(2*x+1[m],at=(x=sample));
  observable analytic_slope:1=evaluate(partial(gain*x+1[m],wrt=x),at=(x=sample));
  observable gain_slope:m=evaluate(partial(gain*x,wrt=gain),at=(x=sample));
  observable jump_side:m=evaluate(if x<0.25[m] then 0[m] else 1[m],at=(x=0.25[m]),side=lower);
  observable operator_side:m=evaluate(jump(x=x),at=(x=0.25[m]),side=lower);
  observable sinusoid:1=evaluate(math.sin(x/1[m]),at=(x=sample));
  observable difference:m=evaluate(x,at=(x=0.25[m]))-evaluate(x,at=(x=0.75[m]));
  observable lazy:m=if true then evaluate(x,at=(x=sample)) else evaluate(x,at=(x=-1[m]));
  observable reference_lazy:m=if true then ramp else outside;
  observable center:m=evaluate(x,at=(x=0[m]),side=upper);
  observable outside:m=evaluate(x,at=(x=-1[m]));
  observable exterior_side:m=evaluate(x,at=(x=0[m]),side=lower);
}
"#;

fn document() -> ModelDocument {
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let support = StaticBindingValue::CoordinateInterval(
        AxisBounds::new(DynQuantity::new(0.0, length), DynQuantity::new(2.0, length)).unwrap(),
    );
    ModelDocument::compile_selected(
        "point-evaluation.eqi",
        ANALYTIC,
        "Probe",
        &[("position", support)],
    )
    .unwrap()
}

#[test]
fn analytic_point_bindings_retain_exact_support_through_model_replay() {
    let model = document();
    let replay = ModelDocument::replay(&model.canonical_json().unwrap()).unwrap();
    assert_eq!(model.program(), replay.program());
}

#[test]
fn analytic_ramp_and_sinusoid_execute_on_the_ordinary_result() {
    use eqiora_artifact::ModelEnvelope;
    use eqiora_backend_faer::FaerLinearSolver;
    use eqiora_numerics::{CommonAlgebraicPlan, CommonLinearRequest, CommonSolvePolicy};
    use eqiora_solver::{LinearSolver, LinearSolverBackend, ReductionPolicy, SolverPlan};
    use std::{collections::HashMap, num::NonZeroUsize};

    let document = document();
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let linear = CommonLinearRequest::exact(
        SolverPlan::new(
            LinearSolver::SparseLu,
            1e-12,
            1e-12,
            NonZeroUsize::new(10).unwrap(),
        )
        .unwrap()
        .with_reduction(ReductionPolicy::Fast),
        FaerLinearSolver.provider(),
    )
    .unwrap();
    let plan = CommonAlgebraicPlan::resolve(
        &model,
        CommonSolvePolicy::Linear(linear),
        None,
        None,
        &FaerLinearSolver,
    )
    .unwrap();
    let result = plan
        .run_result(&plan.initial_state(&[]).unwrap(), &FaerLinearSolver)
        .unwrap();
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    // Alternating Taylor series at 1/4; the next term is below 3e-18.
    // This reference does not call the library's trigonometric implementation.
    let mut term = 0.25;
    let mut sine = term;
    for k in 1..6 {
        term *= -0.0625 / ((2 * k) * (2 * k + 1)) as f64;
        sine += term;
    }
    for (name, expected, unit) in [
        ("ramp", 1.5, length),
        ("analytic_slope", 2.0, DimExponents::DIMENSIONLESS),
        ("gain_slope", 0.25, length),
        ("sinusoid", sine, DimExponents::DIMENSIONLESS),
        ("difference", -0.5, length),
        ("lazy", 0.25, length),
        ("reference_lazy", 1.5, length),
        ("center", 0.0, length),
    ] {
        let id = document.aliases()[name].downcast().unwrap();
        let observed = result.observe(&model, id, &HashMap::new()).unwrap();
        let value = observed.value().real_scalar_value().unwrap();
        assert_eq!(value.dim(), unit);
        assert!((value.value() - expected).abs() < 1e-14);
    }
    for (name, gate) in [
        ("outside", "outside its exact support"),
        ("exterior_side", "side approaches from outside"),
        ("jump_side", "piecewise or branch limits are unavailable"),
        (
            "operator_side",
            "piecewise or branch limits are unavailable",
        ),
    ] {
        let id = document.aliases()[name].downcast().unwrap();
        assert!(
            result
                .observe(&model, id, &HashMap::new())
                .unwrap_err()
                .message()
                .contains(gate)
        );
    }
}

#[test]
fn point_bindings_reject_wrong_units_support_time_side_and_output_type() {
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let interval = StaticBindingValue::CoordinateInterval(
        AxisBounds::new(DynQuantity::new(0.0, length), DynQuantity::new(1.0, length)).unwrap(),
    );
    let compile = |expression: &str, unit: &str| {
        let source = format!(
            "model Probe(support position:interval(m), support other:interval(m)) {{
            coordinate x:m on position from position;
            coordinate z:m on other from other;
            parameter parameter_x:m=0.5;
            variable anchor:1; relation hold {{anchor=1;}}
            observable probe:{unit}={expression};
        }}"
        );
        ModelDocument::compile_selected(
            "point-bindings.eqi",
            &source,
            "Probe",
            &[("position", interval), ("other", interval)],
        )
    };
    assert!(compile("evaluate(x,at=(x=0.5[m]))", "m").is_ok());
    for (expression, unit) in [
        ("evaluate(x,at=(x=0.5[s]))", "m"),
        ("evaluate(x,at=(z=0.5[m]))", "m"),
        ("evaluate(x,at=(parameter_x=0.5[m]))", "m"),
        ("evaluate(x,at=(x=0.5[m]),time=0[s])", "m"),
        ("evaluate(x,at=(x=0.5[m]),side=nearest)", "m"),
        ("evaluate(x,at=(x=0.5[m],x=0.25[m]))", "m"),
        ("evaluate(x,at=(x=0.5[m]))", "s"),
    ] {
        assert!(
            compile(expression, unit).is_err(),
            "unexpectedly admitted {expression}:{unit}"
        );
    }
}

#[test]
fn point_fingerprint_retains_bindings_and_side_and_rejects_displaced_wire() {
    let source = "model P(){domain body=box(0,2); coordinate x:m on body from body[0];
        variable anchor:1; relation hold{anchor=1;}
        observable probe:m=evaluate(x,at=(x=0.25[m]),side=lower);}";
    let model = ModelDocument::compile("point.eqi", source).unwrap();
    let renamed = ModelDocument::compile(
        "renamed.eqi",
        &source
            .replace("coordinate x:", "coordinate position:")
            .replace("evaluate(x,at=(x=", "evaluate(position,at=(position="),
    )
    .unwrap();
    assert!(model.structurally_equivalent(&renamed).unwrap());
    for source in [
        source.replace("0.25[m]", "0.75[m]"),
        source.replace("side=lower", "side=upper"),
    ] {
        let changed = ModelDocument::compile("changed.eqi", &source).unwrap();
        assert!(!model.structurally_equivalent(&changed).unwrap());
    }
    let bytes = model.canonical_json().unwrap();
    let displaced = String::from_utf8(bytes.clone())
        .unwrap()
        .replace("eqiora.model-envelope/v39", "eqiora.model-envelope/v36");
    assert_ne!(displaced.as_bytes(), bytes);
    assert!(ModelDocument::replay(displaced.as_bytes()).is_err());
}

#[test]
fn point_bindings_cannot_hide_physical_port_ownership() {
    for probe in [
        "evaluate(left.position,at=(x=0.5[m]))",
        "evaluate(x,at=(x=left.position))",
    ] {
        let source = format!(
            r#"
model Pair() {{
    domain line=box(0,1);
    coordinate x:m on line from line[0];
    domain mechanical=scalar_physical(across position:m,through flow:1);
    port left:mechanical;
    port right:mechanical;
    relation left_owner {{ {probe}=0[m]; }}
    relation right_owner {{ right.flow=0; }}
    connect left,right;
}}
"#
        );
        eqiora::compiler::compile("point-port-owner.eqi", &source).unwrap();
        let duplicate = source.replace(
            "connect left,right;",
            "relation extra { left.position=0[m]; } connect left,right;",
        );
        let errors = eqiora::compiler::compile("point-port-duplicate.eqi", &duplicate).unwrap_err();
        assert!(
            errors
                .iter()
                .any(|error| error.message().contains("more than one owning Relation")),
            "{errors:?}"
        );
    }
}
