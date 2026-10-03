//! Independent two-node conductance oracle through the ordinary finite lifecycle.
use eqiora::api::ModelDocument;
use eqiora::artifact::ModelEnvelope;
use eqiora::solver::{LinearSolver, LinearSolverBackend, ReductionPolicy, SolverPlan};
use eqiora_backend_faer::FaerLinearSolver;
use eqiora_compiler::StaticBindingValue;
use eqiora_numerics::{
    CommonAlgebraicPlan, CommonLinearRequest, CommonResult, CommonSolvePolicy, ResolvedCommonPlan,
};
use std::num::NonZeroUsize;

const SOURCE: &str = r#"public component Network(parameter g:1, parameter i1:1, parameter i2:1, parameter offset:1) {
    variable v1:1; variable v2:1;
    relation first { g*(v1-v2)=i1; }
    relation second { g*(v2-v1)=i2; }
    observable drop:1=v1-v2;
    form floating for first,second {
        finite voltage(v1,v2);
        gauge voltage { reference v1=offset; compatibility i1+i2=0; }
        g*(v1-v2)=i1;
        g*(v2-v1)=i2;
    }
}"#;
fn fixture(source: &str, offset: f64, load: f64) -> (ModelDocument, CommonAlgebraicPlan) {
    let document = document(source, offset, load);
    let plan = resolve(&document, true).unwrap();
    (document, plan)
}
fn document(source: &str, offset: f64, load: f64) -> ModelDocument {
    let values = eqiora_lang::parse("values.eqi", &format!("model V(){{parameter g:1=2;parameter i1:1=6;parameter i2:1={load};parameter offset:1={offset};}}"))
        .into_document().unwrap();
    let bindings = values.models()[0]
        .items()
        .iter()
        .filter_map(|item| {
            if let eqiora_lang::Item::Parameter(p) = item {
                Some((p.name(), StaticBindingValue::Expression(p.value())))
            } else {
                None
            }
        })
        .collect::<Vec<_>>();
    ModelDocument::compile_selected("network.eqi", source, "Network", &bindings).unwrap()
}
fn resolve(
    document: &ModelDocument,
    authored: bool,
) -> Result<CommonAlgebraicPlan, eqiora::Diagnostic> {
    let model = ModelEnvelope::from_program(document.program()).unwrap();
    let solver = SolverPlan::new(
        LinearSolver::SparseLu,
        1e-13,
        1e-15,
        NonZeroUsize::new(8).unwrap(),
    )
    .unwrap()
    .with_reduction(ReductionPolicy::Fast);
    CommonAlgebraicPlan::resolve(
        &model,
        CommonSolvePolicy::Linear(
            CommonLinearRequest::exact(solver, FaerLinearSolver.provider()).unwrap(),
        ),
        None,
        if authored {
            document.authored_formulation_projection().unwrap()
        } else {
            None
        },
        &FaerLinearSolver,
    )
}
#[test]
fn explicit_reference_changes_coordinates_but_not_voltage_drop_and_replays() {
    for offset in [0., 4., -2.] {
        let (document, plan) = fixture(SOURCE, offset, -6.);
        let initial = plan.initial_state(&[]).unwrap();
        let resolved = ResolvedCommonPlan::Algebraic(Box::new(plan.clone()));
        let replayed = ResolvedCommonPlan::from_bytes(
            &resolved.to_bytes().unwrap(),
            &FaerLinearSolver,
            eqiora::time::TimeBackendIdentity::new("eqiora.test.time", "1"),
        )
        .unwrap();
        assert_eq!(replayed, resolved);
        let result = replayed
            .as_algebraic()
            .unwrap()
            .run_result(&initial, &FaerLinearSolver)
            .unwrap();
        // 2(v1-v2)=6 gives the gauge-independent drop 3; v1=offset
        // independently fixes v2=offset-3. No solver output defines this oracle.
        for (index, expected) in [(0, offset), (1, offset - 3.)] {
            let field = document.authored_formulations().next().unwrap().trials()[index];
            let coordinate = plan
                .symbols()
                .iter()
                .position(|symbol| *symbol == eqiora_schema::kernel::SymbolRef::Field(field))
                .unwrap();
            assert!((result.finite_values().unwrap()[coordinate] - expected).abs() < 1e-12);
        }
        let drop = result
            .observe(
                plan.model_artifact(),
                document
                    .program()
                    .nodes()
                    .find_map(|node| match node {
                        eqiora_schema::kernel::KernelNode::Observable(observable) => {
                            Some(observable.id())
                        }
                        _ => None,
                    })
                    .unwrap(),
                None,
            )
            .unwrap();
        assert!((drop.value().real_scalar_value().unwrap().value() - 3.).abs() < 1e-12);
        assert!(result.original_residual_norm().unwrap() < 1e-12);
        assert!(result.compatibility_residual().unwrap().abs() < 1e-12);
        assert!(result.gauge_residual().unwrap().abs() < 1e-12);
        assert!(result.gauge_multiplier().unwrap().abs() < 1e-12);
        let bytes = result.to_bytes().unwrap();
        assert_eq!(CommonResult::from_bytes(&bytes, &resolved).unwrap(), result);
        for mutation in ["shift", "multiplier", "missing"] {
            let mut wire: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
            let payload = &mut wire["content"]["payload"];
            match mutation {
                "shift" => {
                    for value in payload["values"].as_array_mut().unwrap() {
                        *value = serde_json::json!(value.as_f64().unwrap() + 1.);
                    }
                }
                "multiplier" => payload["nullspace"][0] = serde_json::json!(2.),
                _ => payload["nullspace"] = serde_json::Value::Null,
            }
            let error = CommonResult::from_bytes(&serde_json::to_vec(&wire).unwrap(), &resolved)
                .unwrap_err();
            assert!(!error.message().contains("digest"), "{error:?}");
        }
    }
}
#[test]
fn incompatible_load_or_absent_reference_is_not_repaired() {
    let (_, plan) = fixture(SOURCE, 4., -5.);
    let error = plan
        .run_result(&plan.initial_state(&[]).unwrap(), &FaerLinearSolver)
        .unwrap_err();
    assert!(error.message().contains("incompatible"), "{error:?}");
    let model = document(SOURCE, 4., -6.);
    let plan = resolve(&model, false).unwrap();
    assert!(
        plan.run_result(&plan.initial_state(&[]).unwrap(), &FaerLinearSolver)
            .is_err()
    );
}

#[test]
fn hidden_pin_and_wrong_authored_balance_are_rejected() {
    let pinned = SOURCE.replace("g*(v1-v2)", "g*(v1-v2)+v1");
    let (_, plan) = fixture(&pinned, 4., -6.);
    let error = plan
        .run_result(&plan.initial_state(&[]).unwrap(), &FaerLinearSolver)
        .unwrap_err();
    assert!(error.message().contains("nullspace vector"), "{error:?}");
    let wrong = document(
        &SOURCE.replace("compatibility i1+i2=0", "compatibility i1-i2=0"),
        4.,
        -6.,
    );
    let error = resolve(&wrong, true).unwrap_err();
    assert!(error.message().contains("compatibility"), "{error:?}");
}

#[test]
fn dimensioned_network_retains_the_physical_reference() {
    let source = SOURCE
        .replace("parameter g:1", "parameter g:A^2*s^3/kg/m^2")
        .replace("parameter i1:1", "parameter i1:A")
        .replace("parameter i2:1", "parameter i2:A")
        .replace("parameter offset:1", "parameter offset:kg*m^2/s^3/A")
        .replace("variable v1:1", "variable v1:kg*m^2/s^3/A")
        .replace("variable v2:1", "variable v2:kg*m^2/s^3/A")
        .replace("observable drop:1", "observable drop:kg*m^2/s^3/A");
    let (document, plan) = fixture(&source, 4., -6.);
    let result = plan
        .run_result(&plan.initial_state(&[]).unwrap(), &FaerLinearSolver)
        .unwrap();
    let drop = result
        .observe(
            plan.model_artifact(),
            document.aliases()["definition.drop"].downcast().unwrap(),
            None,
        )
        .unwrap();
    let expected = eqiora::DimExponents::from_integers([1, 2, -3, -1, 0, 0, 0]).unwrap();
    assert_eq!(drop.value().value_type().dimension(), expected);
    assert!((drop.value().real_scalar_value().unwrap().value() - 3.).abs() < 1e-12);
}

#[test]
fn three_node_rows_follow_the_declared_coordinate_correspondence() {
    let source = r#"public component Network(parameter g:1, parameter i1:1, parameter i2:1, parameter offset:1) {
        variable v1:1; variable v2:1; variable v3:1;
        relation first { g*(v1-v2)=i1; }
        relation second { g*(v2-v1)+3*g/2*(v2-v3)=i2/2; }
        relation third { 3*g/2*(v3-v2)=i2/2; }
        form floating for third,first,second {
            finite voltage(v3,v1,v2);
            gauge voltage { reference v1=offset; compatibility i2/2+i1+i2/2=0; }
            3*g/2*(v3-v2)=i2/2;
            g*(v1-v2)=i1;
            g*(v2-v1)+3*g/2*(v2-v3)=i2/2;
        }
    }"#;
    let (model, plan) = fixture(source, 4., -6.);
    let result = plan
        .run_result(&plan.initial_state(&[]).unwrap(), &FaerLinearSolver)
        .unwrap();
    // Node 1 injects 6 through conductance 2: drop 3. Node 3 draws
    // 3 through conductance 3: drop 1. The chosen reference gives [4,1,0].
    for (index, expected) in [(0, 0.), (1, 4.), (2, 1.)] {
        let field = model.authored_formulations().next().unwrap().trials()[index];
        let coordinate = plan
            .symbols()
            .iter()
            .position(|symbol| *symbol == eqiora_schema::kernel::SymbolRef::Field(field))
            .unwrap();
        assert!((result.finite_values().unwrap()[coordinate] - expected).abs() < 1e-12);
    }
    let wrong = document(
        &source.replace("finite voltage(v3,v1,v2)", "finite voltage(v3,v2,v1)"),
        4.,
        -6.,
    );
    assert!(resolve(&wrong, true).is_err());
}
