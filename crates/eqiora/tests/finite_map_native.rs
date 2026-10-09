//! Native local constitutive maps share the source and Result owners.
use eqiora::api::ModelDocument;
use eqiora::artifact::{ModelEnvelope, ModelTransactionEnvelope};
use eqiora::solver::{LinearSolver, LinearSolverBackend, ReductionPolicy, SolverPlan};
use eqiora_backend_faer::FaerLinearSolver;
use eqiora_core::{DimExponents, Id, ScalarDomain, ValueLiteral, ValueType};
use eqiora_lang::{
    DraftDeclaration, DraftField, DraftObservable, DraftParameter, DraftRelation, FieldRoleSyntax,
    Module,
};
use eqiora_numerics::{
    CommonAlgebraicPlan, CommonLinearRequest, CommonResult, CommonSolvePolicy, ResolvedCommonPlan,
};
use eqiora_schema::kernel::FiniteSpaceDef;
use std::{collections::HashMap, num::NonZeroUsize};

#[test]
fn six_component_constitutive_map_uses_native_authoring_and_exact_replay() {
    let space = FiniteSpaceDef::new(Id::new(), (0..6).map(|i| format!("s{i}"))).unwrap();
    let basis = space.basis();
    let unit = DimExponents::DIMENSIONLESS;
    let pressure = DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0]).unwrap();
    let map =
        |dimension| ValueType::linear_map(basis, basis, ScalarDomain::Real, dimension).unwrap();
    let coordinates =
        |dimension| ValueType::coordinates(basis, ScalarDomain::Real, dimension).unwrap();
    let scalar = |dimension| ValueType::scalar(ScalarDomain::Real, dimension).unwrap();
    // The non-symmetric 2x2 block has determinant -1 and inverse [-7,3;5,-2].
    // The other four coordinates have identity stiffness, all coefficients in Pa.
    let entries = (0..36)
        .map(|i| {
            let (r, c) = (i / 6, i % 6);
            let value = match (r, c) {
                (0, 0) => 2.0,
                (0, 1) => 3.0,
                (1, 0) => 5.0,
                (1, 1) => 7.0,
                _ => f64::from(r == c),
            };
            (value, 0.0)
        })
        .collect::<Vec<_>>();
    let a = DraftParameter::new(
        "a",
        ValueLiteral::new(map(pressure), entries.clone()).unwrap(),
    );
    let identity = DraftParameter::new(
        "unit_map",
        ValueLiteral::new(map(unit), (0..36).map(|i| (f64::from(i / 6 == i % 6), 0.0))).unwrap(),
    );
    let rhs = DraftParameter::new(
        "stress",
        ValueLiteral::new(
            coordinates(pressure),
            [61.0, 146.0, 1.0, 2.0, 3.0, 4.0].map(|x| (x, 0.0)),
        )
        .unwrap(),
    );
    let state = DraftField::new("strain", coordinates(unit), FieldRoleSyntax::Variable);
    let declarations: Vec<DraftDeclaration> = vec![
        DraftDeclaration::FiniteSpace {
            name: "S".into(),
            definition: space,
        },
        a.clone().into(),
        identity.clone().into(),
        rhs.clone().into(),
        state.clone().into(),
        DraftRelation::continuous(
            "constitutive",
            [(
                state.expression(),
                a.expression().inverse().apply(rhs.expression()),
            )],
        )
        .into(),
        DraftObservable::new("solution", coordinates(unit), state.expression()).into(),
        DraftObservable::new(
            "determinant_value",
            scalar(pressure.pow(6, 1).unwrap()),
            a.expression().determinant(),
        )
        .into(),
        DraftObservable::new(
            "trace_value",
            scalar(pressure),
            a.expression().matrix_trace(),
        )
        .into(),
        DraftObservable::new(
            "inverse_map",
            map(pressure.pow(-1, 1).unwrap()),
            a.expression().inverse(),
        )
        .into(),
        DraftObservable::new(
            "composition",
            map(pressure),
            a.expression().compose_map(identity.expression()),
        )
        .into(),
    ];
    let module = Module::new("M", declarations).unwrap();
    let document = ModelDocument::compile_module(&module, None, &[]).unwrap();
    let source = eqiora_lang::format(module.document());
    let parsed = ModelDocument::compile("native-map.eqi", &source).unwrap();
    assert_eq!(
        document.structural_fingerprint().unwrap(),
        parsed.structural_fingerprint().unwrap()
    );
    let artifact = ModelEnvelope::from_program(document.program()).unwrap();
    assert_eq!(
        document
            .structural_fingerprint()
            .unwrap()
            .generation()
            .as_str(),
        "eqiora.structural-semantic-fingerprint/v37"
    );
    let bytes = artifact.canonical_json().unwrap();
    let mut displaced: serde_json::Value = serde_json::from_slice(&bytes).unwrap();
    assert_eq!(displaced["schema"], "eqiora.model-envelope/v42");
    displaced["schema"] = serde_json::json!("eqiora.model-envelope/v38");
    let error =
        ModelEnvelope::from_json(&serde_json::to_vec(&displaced).unwrap(), Default::default())
            .unwrap_err();
    assert!(
        error
            .message()
            .contains("unsupported eqiora.model-envelope/v42 schema")
    );
    let (transaction, _) = artifact.to_transaction().unwrap();
    let wire = ModelTransactionEnvelope::from_transaction(&transaction)
        .unwrap()
        .canonical_json()
        .unwrap();
    let mut displaced: serde_json::Value = serde_json::from_slice(&wire).unwrap();
    assert_eq!(displaced["schema"], "eqiora.model-transaction-envelope/v42");
    ModelTransactionEnvelope::from_json(&wire, Default::default())
        .unwrap()
        .to_transaction()
        .unwrap();
    displaced["schema"] = serde_json::json!("eqiora.model-transaction-envelope/v38");
    let error = ModelTransactionEnvelope::from_json(
        &serde_json::to_vec(&displaced).unwrap(),
        Default::default(),
    )
    .unwrap_err();
    assert!(
        error
            .message()
            .contains("unsupported eqiora.model-transaction-envelope/v42 schema")
    );

    let artifact =
        ModelEnvelope::from_json(&artifact.canonical_json().unwrap(), Default::default()).unwrap();
    let policy = CommonSolvePolicy::Linear(
        CommonLinearRequest::exact(
            SolverPlan::new(
                LinearSolver::SparseLu,
                1e-12,
                1e-14,
                NonZeroUsize::new(16).unwrap(),
            )
            .unwrap()
            .with_reduction(ReductionPolicy::Fast),
            FaerLinearSolver.provider(),
        )
        .unwrap(),
    );
    let plan = CommonAlgebraicPlan::resolve(&artifact, policy, None, &[], None, &FaerLinearSolver)
        .unwrap();
    let result = plan
        .run_result(&plan.initial_state(&[]).unwrap(), &FaerLinearSolver)
        .unwrap();
    let replay = ResolvedCommonPlan::from_bytes(
        &result.plan().to_bytes().unwrap(),
        &FaerLinearSolver,
        eqiora::time::TimeBackendCapabilities::new(
            eqiora::time::TimeBackendIdentity::new("eqiora.test.time", "1"),
            &[eqiora::ScalarDomain::Real, eqiora::ScalarDomain::Complex],
            &[eqiora::ScalarType::F64],
        ),
    )
    .unwrap();
    let result = CommonResult::from_bytes(&result.to_bytes().unwrap(), &replay).unwrap();
    for (name, expected, dimension) in [
        ("solution", vec![11.0, 13.0, 1.0, 2.0, 3.0, 4.0], unit),
        ("determinant_value", vec![-1.0], pressure.pow(6, 1).unwrap()),
        ("trace_value", vec![13.0], pressure),
        (
            "composition",
            entries.iter().map(|x| x.0).collect(),
            pressure,
        ),
        (
            "inverse_map",
            (0..36)
                .map(|i| match (i / 6, i % 6) {
                    (0, 0) => -7.0,
                    (0, 1) => 3.0,
                    (1, 0) => 5.0,
                    (1, 1) => -2.0,
                    (r, c) => f64::from(r == c),
                })
                .collect(),
            pressure.pow(-1, 1).unwrap(),
        ),
    ] {
        let id = document.aliases()[name].downcast().unwrap();
        let value = result.observe(&artifact, id, &HashMap::new()).unwrap();
        assert_eq!(value.value().value_type().dimension(), dimension);
        for (i, expected) in expected.into_iter().enumerate() {
            assert!(
                (value.value().component(i).unwrap().0 - expected).abs() < 1e-10,
                "{name}[{i}]"
            );
        }
    }
}
