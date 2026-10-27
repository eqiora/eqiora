//! Independent RC values are fixed by V=1/(1-i omega RC), I=-i omega C V.
//! The one-element undamped Q1 wave has u1=(1+omega²/6)/(1-omega²/3).
use super::*;
use eqiora_backend_faer::FaerLinearSolver;
use eqiora_numerics::{CommonAlgebraicPlan, CommonLinearRequest, CommonSolvePolicy};
use eqiora_solver::{
    LinearProblem, LinearSolution, ReplicatedLinearExecution, SolverCapabilities, SolverPlan,
    SolverProvider,
};
use std::num::NonZeroUsize;

const LIMIT: usize = 1 << 28;
const MEMBER: usize = 1 << 22;
// Existing harmonic-response acceptance boundary, fixed before observing these Runs.
const TOLERANCE: f64 = 1e-10;

fn solve() -> CommonSolvePolicy {
    CommonSolvePolicy::Linear(
        CommonLinearRequest::exact(
            SolverPlan::new(
                eqiora_solver::LinearSolver::SparseLu,
                1e-12,
                1e-14,
                NonZeroUsize::new(32).unwrap(),
            )
            .unwrap()
            .with_reduction(eqiora_solver::ReductionPolicy::Fast),
            eqiora_backend_faer::FAER_SOLVER_PROVIDER,
        )
        .unwrap(),
    )
}
fn fixture() -> (Arc<ResolvedCommonPlan>, Parameter) {
    let source = include_str!("../../../../../docs/language/harmonic-rc.md")
        .split("```eqiora\n")
        .nth(1)
        .unwrap()
        .split("```")
        .next()
        .unwrap();
    let original = crate::ModelDocument::compile("mapped-rc.eqi", source).unwrap();
    let omega = original
        .parameter_ref("omega")
        .unwrap_or_else(|error| panic!("{error:?}; aliases={:?}", original.aliases()))
        .id();
    let model = ModelEnvelope::from_program(original.program()).unwrap();
    let plan = CommonAlgebraicPlan::resolve(
        &model,
        solve(),
        None,
        &[],
        original.authored_formulation_projection().unwrap(),
        &FaerLinearSolver,
    )
    .unwrap();
    (
        Arc::new(ResolvedCommonPlan::Algebraic(Box::new(plan))),
        omega,
    )
}
fn points(plan: &ResolvedCommonPlan, omega: Parameter, values: &[f64]) -> Vec<Vec<ValueLiteral>> {
    let original = document(plan).unwrap();
    let value_type = original
        .program()
        .typed_value(omega.erase())
        .unwrap()
        .value_type()
        .clone();
    values
        .iter()
        .map(|value| vec![ValueLiteral::from_real(value_type.clone(), *value).unwrap()])
        .collect()
}
fn map(plan: Arc<ResolvedCommonPlan>, omega: Parameter, values: &[f64]) -> EvaluationMapPlan {
    let points = points(&plan, omega, values);
    EvaluationMapPlan::from_parameters(
        plan,
        &[omega],
        &points,
        EvaluationMapExecutionPolicy::retained(LIMIT),
        MEMBER,
        Arc::new(FaerLinearSolver),
    )
    .unwrap()
}
fn check_rc(result: &CommonResult, expected: [(f64, f64); 2]) {
    let plan = result.plan().as_algebraic().unwrap();
    for ((_, _, amplitude), expected) in plan.harmonic_amplitudes().zip(expected) {
        let values = result.finite_field_values().unwrap();
        let value = &values
            .iter()
            .find(|(field, _)| *field == amplitude)
            .unwrap()
            .1;
        let actual = value.component(0).unwrap();
        assert!(
            (actual.0 - expected.0).abs() < TOLERANCE && (actual.1 - expected.1).abs() < TOLERANCE,
            "{actual:?} != {expected:?}"
        );
    }
}

#[derive(Debug)]
struct NoSolve;
impl LinearSolverBackend for NoSolve {
    fn provider(&self) -> SolverProvider {
        FaerLinearSolver.provider()
    }
    fn capabilities(&self) -> SolverCapabilities {
        FaerLinearSolver.capabilities()
    }
    fn complex_backend(&self) -> Option<&dyn LinearSolverBackend<num_complex::Complex64>> {
        Some(self)
    }
    fn solve_with_execution(
        &self,
        _: &LinearProblem<'_>,
        _: SolverPlan,
        _: &dyn ReplicatedLinearExecution,
    ) -> Result<LinearSolution, Diagnostic> {
        panic!("offline reopening must not solve")
    }
}

impl LinearSolverBackend<num_complex::Complex64> for NoSolve {
    fn provider(&self) -> SolverProvider {
        FaerLinearSolver.provider()
    }
    fn capabilities(&self) -> SolverCapabilities {
        FaerLinearSolver.capabilities()
    }
    fn solve_with_execution(
        &self,
        _: &LinearProblem<'_, num_complex::Complex64>,
        _: SolverPlan,
        _: &dyn ReplicatedLinearExecution<num_complex::Complex64>,
    ) -> Result<LinearSolution<num_complex::Complex64>, Diagnostic> {
        panic!("offline reopening must not solve complex systems")
    }
}

#[test]
fn rc_ordered_runs_rederive_frequency_and_preserve_default_duplicate_and_offline_plans() {
    let (template, omega) = fixture();
    let plan = map(template.clone(), omega, &[500., 1000., 500.]);
    let complete = plan.execute().unwrap();
    let first = complete.run_result(0).unwrap();
    let encoded_charge = first.to_bytes().unwrap().len() + first.plan().to_bytes().unwrap().len();
    assert!(check_member_size(first, encoded_charge).is_ok());
    assert!(check_member_size(first, encoded_charge - 1).is_err());
    assert!(check_member_size(first, first.to_bytes().unwrap().len()).is_err());
    for (index, expected) in [
        [(0.8, 0.4), (0.0002, -0.0004)],
        [(0.5, 0.5), (0.0005, -0.0005)],
        [(0.8, 0.4), (0.0002, -0.0004)],
    ]
    .into_iter()
    .enumerate()
    {
        check_rc(complete.run_result(index).unwrap(), expected);
    }
    assert_eq!(
        complete.run_result(1).unwrap().plan(),
        template.as_ref(),
        "unchanged/default point retains original identity"
    );
    assert_eq!(
        complete.run_result(0).unwrap().plan(),
        complete.run_result(2).unwrap().plan()
    );
    assert_ne!(complete.occurrence_id(0), complete.occurrence_id(2));
    let amplitude = |plan: &ResolvedCommonPlan| {
        plan.as_algebraic()
            .unwrap()
            .harmonic_amplitudes()
            .next()
            .unwrap()
            .2
    };
    assert_ne!(
        amplitude(complete.run_result(0).unwrap().plan()),
        amplitude(&template)
    );
    let collection = complete.record().unwrap();
    let bytes = collection.to_bytes().unwrap();
    let reopened = EvaluationCollection::from_bytes(&bytes, &NoSolve, &[]).unwrap();
    assert_eq!(collection, reopened);
    assert_eq!(reopened.groups()[0].member_plan(1), Some(template.as_ref()));
    let cold = map(template, omega, &[500.]).execute().unwrap();
    assert_eq!(
        time::acceptance_fingerprint(cold.run_result(0).unwrap()).unwrap(),
        time::acceptance_fingerprint(complete.run_result(0).unwrap()).unwrap()
    );
}

#[test]
fn partial_cancellation_retry_and_corruption_keep_occurrence_authority() {
    let (template, omega) = fixture();
    let plan = map(template.clone(), omega, &[500., 1000., 500.]);
    let mut calls = 0;
    let partial = plan
        .execute_recorded(|| {
            calls += 1;
            calls == 2
        })
        .unwrap();
    let group = &partial.groups()[0];
    assert!(group.occurrence(0).unwrap().1.is_accepted());
    assert!(matches!(
        group.occurrence(1).unwrap().1,
        RecordedEvaluationOutcome::Cancelled(None)
    ));
    let retry_points = group
        .parameter_points()
        .unwrap()
        .iter()
        .enumerate()
        .filter(|(index, _)| !group.occurrence(*index).unwrap().1.is_accepted())
        .map(|(_, point)| point.clone())
        .collect::<Vec<_>>();
    let retry = EvaluationMapPlan::from_parameters(
        template.clone(),
        &[omega],
        &retry_points,
        EvaluationMapExecutionPolicy::retained(LIMIT),
        MEMBER,
        Arc::new(FaerLinearSolver),
    )
    .unwrap()
    .execute_recorded(|| false)
    .unwrap();
    assert_eq!(retry.groups()[0].len(), 2);
    let combined = EvaluationCollection::combine(vec![partial.clone(), retry]).unwrap();
    assert_eq!(&combined.groups()[0], group);
    assert_ne!(
        combined.groups()[0].execution_id(),
        combined.groups()[1].execution_id()
    );
    assert_eq!(
        combined,
        EvaluationCollection::from_bytes(&combined.to_bytes().unwrap(), &NoSolve, &[]).unwrap()
    );
    let complete = plan.execute_recorded(|| false).unwrap();
    let mut corrupt: serde_json::Value =
        serde_json::from_slice(&complete.to_bytes().unwrap()).unwrap();
    // Stale frequency Plan and Result cannot be relabeled as a different requested member.
    corrupt["groups"][0]["members"][1]["outcome"] =
        corrupt["groups"][0]["members"][0]["outcome"].clone();
    let errors =
        EvaluationCollection::from_bytes(&serde_json::to_vec(&corrupt).unwrap(), &NoSolve, &[])
            .unwrap_err();
    assert!(errors[0].message().contains("requested Parameter point"));
    let failed = map(template.clone(), omega, &[500., 0., 1000.])
        .execute_recorded(|| false)
        .unwrap();
    assert!(failed.groups()[0].occurrence(0).unwrap().1.is_accepted());
    assert!(matches!(
        failed.groups()[0].occurrence(1).unwrap().1,
        RecordedEvaluationOutcome::Failed(_)
    ));
    assert!(matches!(
        failed.groups()[0].occurrence(2).unwrap().1,
        RecordedEvaluationOutcome::NotStarted
    ));
    assert_eq!(
        failed,
        EvaluationCollection::from_bytes(&failed.to_bytes().unwrap(), &NoSolve, &[]).unwrap()
    );
    assert!(map(template.clone(), omega, &[-1.]).execute().is_err());
    let wrong = vec![vec![
        ValueLiteral::from_real(
            eqiora_core::ValueType::scalar(
                eqiora_core::ScalarDomain::Real,
                eqiora_core::DimExponents::DIMENSIONLESS,
            )
            .unwrap(),
            500.,
        )
        .unwrap(),
    ]];
    assert!(
        EvaluationMapPlan::from_parameters(
            template,
            &[omega],
            &wrong,
            EvaluationMapExecutionPolicy::retained(LIMIT),
            MEMBER,
            Arc::new(FaerLinearSolver)
        )
        .is_err()
    );
}

fn wave_fixture() -> (Arc<ResolvedCommonPlan>, Parameter) {
    use eqiora_artifact::{
        CartesianMeshCellsV2, GeometryMeshCorrespondenceEnvelopeV1, MeshProductionLineageEnvelopeV1,
    };
    use eqiora_numerics::{AuthenticatedCommonMesh, CommonSpatialPolicy};
    let source = include_str!("../../../../../verify/numerics/harmonic-response/models/wave.eqi")
        .replace("public component Wave", "public model Wave")
        .replace(" state u:", " parameter omega:1/s=1[1/s];\n state u:")
        .replace("+1[1/s]*derivative(u)", "")
        .replace("angular_frequency=1[1/s]", "angular_frequency=omega")
        .replace(
            "math.complex(-1[1/s^2],-3[1/s^2])",
            "math.complex(0[1/s^2],0[1/s^2])",
        )
        .replace("math.complex(2,1)", "math.complex(1,0)");
    let graph = eqiora_geometry::GeometryGraph::new();
    let interval = graph.interval([0., 1.]).unwrap();
    let geometry = graph
        .build(
            &interval,
            &std::collections::BTreeMap::from([
                ("body".into(), vec![interval.region().into()]),
                ("left".into(), vec![interval.boundaries()[0].into()]),
                ("right".into(), vec![interval.boundaries()[1].into()]),
            ]),
        )
        .unwrap();
    let bindings = ["body", "left", "right"].map(|name| {
        (
            name,
            eqiora_compiler::StaticBindingValue::GeometrySupport {
                geometry: &geometry,
                selection: geometry.entity_set(name).unwrap(),
                parent: (name != "body").then(|| geometry.entity_set("body").unwrap()),
            },
        )
    });
    let original =
        crate::ModelDocument::compile_selected("mapped-wave.eqi", &source, "Wave", &bindings)
            .unwrap();
    let omega = original
        .parameter_ref("omega")
        .unwrap_or_else(|error| panic!("{error:?}; aliases={:?}", original.aliases()))
        .id();
    let policy = CartesianMeshCellsV2::new(vec![1]).unwrap();
    let (mesh, correspondence) =
        GeometryMeshCorrespondenceEnvelopeV1::from_cartesian_box_v1(&geometry, policy.cells())
            .unwrap();
    let production = MeshProductionLineageEnvelopeV1::from_structured_cartesian_v2_resources(
        &policy,
        &geometry,
        &mesh,
        &correspondence,
    )
    .unwrap();
    let mesh =
        AuthenticatedCommonMesh::structured_cartesian(geometry, mesh, correspondence, production)
            .unwrap();
    let model = ModelEnvelope::from_program(original.program()).unwrap();
    let plan = ResolvedCommonPlan::resolve(
        &model,
        mesh,
        CommonSpatialPolicy::Q1,
        solve(),
        None,
        None,
        &FaerLinearSolver,
        None,
        original.authored_formulation_projection().unwrap(),
    )
    .unwrap();
    (Arc::new(plan), omega)
}

#[test]
fn scalar_wave_uses_the_same_ordered_run_collection_and_invalidates_frequency_dependent_matrix() {
    let (template, omega) = wave_fixture();
    let complete = map(template.clone(), omega, &[1., 2., 1.])
        .execute()
        .unwrap();
    for (index, expected) in [[1., 1.75], [1., -5.], [1., 1.75]].iter().enumerate() {
        let result = complete.run_result(index).unwrap();
        let (_, values, shape) = result.field_block(0, 0).unwrap();
        assert_eq!(shape, &[2]);
        for (node, expected) in values.as_chunks::<2>().0.iter().zip(expected) {
            assert!((node[0] - expected).abs() < TOLERANCE && node[1].abs() < TOLERANCE);
        }
    }
    let cold = map(template, omega, &[2.]).execute().unwrap();
    assert_eq!(
        time::acceptance_fingerprint(cold.run_result(0).unwrap()).unwrap(),
        time::acceptance_fingerprint(complete.run_result(1).unwrap()).unwrap()
    );
    let collection = complete.record().unwrap();
    assert_eq!(
        collection,
        EvaluationCollection::from_bytes(&collection.to_bytes().unwrap(), &NoSolve, &[]).unwrap()
    );
}

#[test]
fn ordinary_finite_runs_use_revision_local_parameter_values_after_value_edits() {
    let original = crate::ModelDocument::compile(
        "finite-edited-defaults.eqi",
        "model M(){parameter p:1=2;variable u:1;relation r{u=p;}observable observed:1=u+p;}",
    )
    .unwrap();
    let observable = original.observable_ref("observed").unwrap().id();
    let parameter = original.parameter_ref("p").unwrap().id();
    let before = original.program().typed_value(parameter.erase()).unwrap();
    let value = ValueLiteral::from_real(before.value_type().clone(), 3.).unwrap();
    let edit = original
        .preview_value_edit(parameter.erase(), value.clone())
        .unwrap();
    let edited = original.commit_value_edit(edit).unwrap().into_document();
    let model = ModelEnvelope::from_program(edited.program()).unwrap();
    assert_eq!(
        model.to_program().unwrap().typed_value(parameter.erase()),
        Some(&value)
    );
    let plan =
        CommonAlgebraicPlan::resolve(&model, solve(), None, &[], None, &FaerLinearSolver).unwrap();
    let state = plan.initial_state(&[]).unwrap();
    let result = plan.run_result(&state, &FaerLinearSolver).unwrap();
    assert_eq!(
        result.finite_field_values().unwrap()[0].1.component(0),
        Some((3., 0.))
    );
    let observed = result
        .observe(&model, observable, &Default::default(), None)
        .unwrap();
    assert_eq!(observed.value().component(0), Some((6., 0.)));
}

#[test]
fn edited_finite_gauge_and_observable_use_the_same_revision_local_reference() {
    let source = r#"model Network(){
        parameter g:1=2;parameter i1:1=6;parameter i2:1=-6;parameter offset:1=0;
        variable v1:1;variable v2:1;
        relation first{g*(v1-v2)=i1;}relation second{g*(v2-v1)=i2;}
        observable shifted_drop:1=v1-v2+offset;
        form floating for first,second{
            finite voltage(v1,v2);
            gauge voltage{reference v1=offset;compatibility i1+i2=0;}
            g*(v1-v2)=i1;g*(v2-v1)=i2;
        }
    }"#;
    let original = crate::ModelDocument::compile("edited-gauge.eqi", source).unwrap();
    let parameter = original.parameter_ref("offset").unwrap().id();
    let field1 = original.field_ref("v1").unwrap().id();
    let field2 = original.field_ref("v2").unwrap().id();
    let observable = original.observable_ref("shifted_drop").unwrap().id();
    let before = original.program().typed_value(parameter.erase()).unwrap();
    let edit = original
        .preview_value_edit(
            parameter.erase(),
            ValueLiteral::from_real(before.value_type().clone(), 4.).unwrap(),
        )
        .unwrap();
    let edited = original.commit_value_edit(edit).unwrap().into_document();
    let model = ModelEnvelope::from_program(edited.program()).unwrap();
    let plan = CommonAlgebraicPlan::resolve(
        &model,
        solve(),
        None,
        &[],
        original.authored_formulation_projection().unwrap(),
        &FaerLinearSolver,
    )
    .unwrap();
    let result = plan
        .run_result(&plan.initial_state(&[]).unwrap(), &FaerLinearSolver)
        .unwrap();
    let values = result.finite_field_values().unwrap();
    // 2(v1-v2)=6, v1=4 gives v2=1; the shifted drop is 3+4=7.
    for (field, wanted) in [(field1, 4.), (field2, 1.)] {
        let value = &values.iter().find(|(id, _)| *id == field).unwrap().1;
        assert!((value.component(0).unwrap().0 - wanted).abs() < TOLERANCE);
    }
    let observed = result
        .observe(&model, observable, &Default::default(), None)
        .unwrap();
    assert!((observed.value().component(0).unwrap().0 - 7.).abs() < TOLERANCE);
}
