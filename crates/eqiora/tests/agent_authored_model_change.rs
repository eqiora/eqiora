use std::num::{NonZeroU16, NonZeroUsize};

use eqiora::api::ModelDocument;
use eqiora::control::{CompileOutcomeV2, CompileRequestV2, execute_compile_v2};
use eqiora::diagnostic::codes;
use eqiora::realization::{
    Discretization, DiscretizationMethod, ExecutionSchedule, MeshPolicy, QuadraturePolicy,
    RealizationCapabilities, RealizationPlan, RealizationRequest, RealizationRequirements,
    RealizationRevision, SemanticRevision, Space, Target, VectorLayoutKind, resolve,
};
use eqiora::solver::{LinearSolver, REFERENCE_LINEAR_SOLVER, ScalarType, SolverPlan};
use eqiora_numerics::{
    scalar::ResolvedScalarEllipticSolution1d, scalar::solve_resolved_scalar_elliptic_1d,
};
use serde::Deserialize;

const SOURCE: &str =
    include_str!("../../../verify/interfaces/agent-authored-model-change/models/poisson.eqi");
const OBJECTIVE: &[u8] =
    include_bytes!("../../../verify/interfaces/agent-authored-model-change/expected/oracle.json");
const ACCEPTED_PROPOSAL: &[u8] = include_bytes!(
    "../../../verify/interfaces/agent-authored-model-change/proposals/accepted.json"
);
const REJECTED_PROPOSAL: &[u8] = include_bytes!(
    "../../../verify/interfaces/agent-authored-model-change/proposals/rejected.json"
);

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ScientificObjective {
    schema: String,
    target_alias: String,
    expected_field_values: Vec<f64>,
    absolute_tolerance: f64,
}

#[derive(Debug, Deserialize)]
#[serde(deny_unknown_fields)]
struct ProposedEdit {
    proposed_value_si: f64,
}

fn objective() -> ScientificObjective {
    let objective: ScientificObjective = serde_json::from_slice(OBJECTIVE).unwrap();
    assert_eq!(
        objective.schema,
        "eqiora.verify.agent-authored-model-change-oracle/v1"
    );
    assert!(objective.absolute_tolerance.is_finite());
    assert!(objective.absolute_tolerance >= 0.0);
    objective
}

fn proposal(bytes: &[u8]) -> ProposedEdit {
    let proposal: ProposedEdit = serde_json::from_slice(bytes).unwrap();
    assert!(proposal.proposed_value_si.is_finite());
    proposal
}

fn compile_base() -> ModelDocument {
    let request =
        CompileRequestV2::new("verify.agent-authored-model-change", "poisson.eqi", SOURCE).unwrap();
    let (response, document) = execute_compile_v2(&request).into_parts();
    assert!(matches!(
        response.outcome(),
        CompileOutcomeV2::Accepted { .. }
    ));
    document.expect("accepted compilation must expose one immutable Model")
}

fn execute(
    document: &ModelDocument,
) -> Result<ResolvedScalarEllipticSolution1d, eqiora::Diagnostic> {
    let plan = RealizationPlan::new(
        Space::continuous_lagrange(NonZeroU16::MIN),
        Discretization::new(
            DiscretizationMethod::ContinuousGalerkin,
            MeshPolicy::GeneratedUniform {
                cells_per_axis: NonZeroUsize::new(4).unwrap(),
            },
            QuadraturePolicy::GaussLegendre {
                points_per_axis: NonZeroUsize::new(2).unwrap(),
            },
        ),
        SolverPlan::new(
            LinearSolver::ConjugateGradient,
            1.0e-12,
            1.0e-14,
            NonZeroUsize::new(128).unwrap(),
        )?,
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
        ExecutionSchedule::Offline,
    )?;
    let realization = resolve(
        &RealizationRequest::explicit(
            document.program().model(),
            SemanticRevision::new(document.program().revision().0),
            RealizationRevision::new(1),
            plan,
        ),
        RealizationRequirements::new(
            NonZeroUsize::MIN,
            ScalarType::F64,
            VectorLayoutKind::Replicated,
        ),
        &RealizationCapabilities::scalar_elliptic_reference(),
    )?;
    solve_resolved_scalar_elliptic_1d(document.program(), &realization, &REFERENCE_LINEAR_SOLVER)
        .map(|(_, result)| result)
}

fn verify_scientific_objective(
    result: &ResolvedScalarEllipticSolution1d,
    objective: &ScientificObjective,
) -> Result<(), String> {
    let ResolvedScalarEllipticSolution1d::FiniteElement(result) = result else {
        return Err("the verification-only P1 policy returned a non-FEM solution".to_owned());
    };
    let observed = result.field().values();
    if observed.len() != objective.expected_field_values.len() {
        return Err(format!(
            "oracle expected {} values, execution produced {}",
            objective.expected_field_values.len(),
            observed.len()
        ));
    }
    for (index, (&actual, &expected)) in observed
        .iter()
        .zip(&objective.expected_field_values)
        .enumerate()
    {
        if (actual - expected).abs() > objective.absolute_tolerance {
            return Err(format!(
                "oracle mismatch at field value {index}: expected {expected}, observed {actual}"
            ));
        }
    }
    Ok(())
}

#[test]
fn offline_agent_proposal_uses_the_ordinary_exact_edit_and_execution_path() {
    let base = compile_base();
    let objective = objective();
    let proposal = proposal(ACCEPTED_PROPOSAL);
    let base_bytes = base.canonical_json().unwrap();
    let base_digest = base.digest().unwrap();
    let base_revision = base.program().revision();
    let target = base.aliases()[&objective.target_alias];

    let agent_plan = base
        .preview_value_edit(
            target,
            scalar_edit_value(&base, target, proposal.proposed_value_si),
        )
        .unwrap();
    let ordinary_client_plan = base
        .preview_value_edit(
            target,
            scalar_edit_value(&base, target, proposal.proposed_value_si),
        )
        .unwrap();

    assert_eq!(agent_plan, ordinary_client_plan);
    assert_eq!(
        agent_plan.transaction_json().unwrap(),
        ordinary_client_plan.transaction_json().unwrap()
    );
    assert_eq!(agent_plan.base_digest(), base_digest);
    assert_eq!(agent_plan.base_revision(), base_revision);
    assert_eq!(
        agent_plan.before().real_scalar_value().unwrap().value(),
        1.0
    );
    assert_eq!(agent_plan.after().real_scalar_value().unwrap().value(), 2.0);
    assert!(
        String::from_utf8(agent_plan.transaction_json().unwrap())
            .unwrap()
            .contains("eqiora.model-transaction-envelope/v26")
    );
    assert_eq!(base.canonical_json().unwrap(), base_bytes);
    assert_eq!(base.digest().unwrap(), base_digest);
    assert_eq!(base.program().revision(), base_revision);

    let agent_change = base.commit_value_edit(agent_plan).unwrap();
    let ordinary_change = base.commit_value_edit(ordinary_client_plan).unwrap();
    assert_eq!(
        agent_change.document().canonical_json().unwrap(),
        ordinary_change.document().canonical_json().unwrap()
    );
    assert_eq!(
        agent_change.result_digest(),
        ordinary_change.result_digest()
    );
    assert_ne!(agent_change.result_digest(), base_digest);
    assert_eq!(agent_change.result_revision().0, base_revision.0 + 1);
    assert_eq!(base.canonical_json().unwrap(), base_bytes);
    assert_eq!(base.digest().unwrap(), base_digest);
    assert_eq!(base.program().revision(), base_revision);
    assert_eq!(base.program().value(target).unwrap().value(), 1.0);

    let accepted = execute(agent_change.document()).unwrap();
    verify_scientific_objective(&accepted, &objective).unwrap();
    let ResolvedScalarEllipticSolution1d::FiniteElement(accepted_fem) = &accepted else {
        panic!("the verification-only P1 policy returned a non-FEM solution");
    };
    assert!(
        accepted_fem.solve_report().true_residual_norm()
            <= accepted_fem.solve_report().residual_target()
    );

    let child_bytes = agent_change.document().canonical_json().unwrap();
    let replayed_child = ModelDocument::replay(&child_bytes).unwrap();
    assert_eq!(replayed_child.canonical_json().unwrap(), child_bytes);
    assert_eq!(
        replayed_child.digest().unwrap(),
        agent_change.result_digest()
    );

    let replayed_result = execute(&replayed_child).unwrap();
    assert_eq!(replayed_result, accepted);
}

#[test]
fn independent_evidence_rejects_a_valid_but_scientifically_wrong_proposal() {
    let base = compile_base();
    let objective = objective();
    let proposal = proposal(REJECTED_PROPOSAL);
    let base_bytes = base.canonical_json().unwrap();
    let base_digest = base.digest().unwrap();
    let base_revision = base.program().revision();
    let target = base.aliases()[&objective.target_alias];

    let plan = base
        .preview_value_edit(
            target,
            scalar_edit_value(&base, target, proposal.proposed_value_si),
        )
        .unwrap();
    let candidate = base.commit_value_edit(plan).unwrap();
    let result = execute(candidate.document()).unwrap();

    let ResolvedScalarEllipticSolution1d::FiniteElement(fem) = &result else {
        panic!("the verification-only P1 policy returned a non-FEM solution");
    };
    assert!(fem.solve_report().true_residual_norm() <= fem.solve_report().residual_target());
    assert!(
        verify_scientific_objective(&result, &objective).is_err(),
        "solver acceptance cannot substitute for an independent scientific oracle"
    );
    assert_eq!(base.canonical_json().unwrap(), base_bytes);
    assert_eq!(base.digest().unwrap(), base_digest);
    assert_eq!(base.program().revision(), base_revision);
    assert_eq!(base.program().value(target).unwrap().value(), 1.0);
}

#[test]
fn stale_foreign_forged_and_unsupported_inputs_fail_closed() {
    // This unused Parameter retains the same old value on both distinct bases,
    // isolating the artifact-identity check from the transaction payload.
    let source = SOURCE.replace(
        "model poisson() {",
        "model poisson() { parameter probe: 1 = 0;",
    );
    let base = ModelDocument::compile("poisson.eqi", &source).unwrap();
    let probe = base.aliases()["probe"];
    let objective = objective();
    let target = base.aliases()[&objective.target_alias];
    let field = base.aliases()["potential"];
    let relation = base.aliases()["balance"];
    let base_bytes = base.canonical_json().unwrap();
    let base_digest = base.digest().unwrap();
    let base_revision = base.program().revision();

    let stale = base
        .preview_value_edit(target, scalar_edit_value(&base, target, 2.0))
        .unwrap();
    let child = base
        .commit_value_edit(
            base.preview_value_edit(target, scalar_edit_value(&base, target, 3.0))
                .unwrap(),
        )
        .unwrap();
    assert_eq!(
        child.document().commit_value_edit(stale).unwrap_err()[0].code(),
        codes::PRECONDITION_FAILED
    );

    let left = base
        .commit_value_edit(
            base.preview_value_edit(target, scalar_edit_value(&base, target, 2.0))
                .unwrap(),
        )
        .unwrap();
    let right = base
        .commit_value_edit(
            base.preview_value_edit(target, scalar_edit_value(&base, target, 3.0))
                .unwrap(),
        )
        .unwrap();
    let left_plan = left
        .document()
        .preview_value_edit(probe, scalar_edit_value(left.document(), probe, 1.0))
        .unwrap();
    let right_plan = right
        .document()
        .preview_value_edit(
            probe,
            eqiora_core::ValueLiteral::from_real(
                right
                    .document()
                    .program()
                    .typed_value(probe)
                    .unwrap()
                    .value_type()
                    .clone(),
                1.0,
            )
            .unwrap(),
        )
        .unwrap();
    assert_eq!(left_plan.base_revision(), right_plan.base_revision());
    assert_eq!(
        left_plan.transaction_digest(),
        right_plan.transaction_digest()
    );
    assert_ne!(left_plan.base_digest(), right_plan.base_digest());
    assert_eq!(
        left.document().commit_value_edit(right_plan).unwrap_err()[0].code(),
        codes::PRECONDITION_FAILED
    );

    assert_ne!(
        execute(left.document()).unwrap(),
        execute(right.document()).unwrap()
    );

    assert_eq!(
        base.preview_value_edit(target, scalar_edit_value(&base, target, 1.0))
            .unwrap_err()
            .code(),
        codes::INVALID_OPERATION
    );
    assert!(
        eqiora_core::ValueLiteral::from_real(
            base.program()
                .typed_value(target)
                .unwrap()
                .value_type()
                .clone(),
            f64::NAN
        )
        .is_err()
    );
    assert_eq!(
        base.preview_value_edit(field, scalar_edit_value(&base, target, 1.0))
            .unwrap_err()
            .code(),
        codes::INVALID_OPERATION
    );
    assert_eq!(
        base.preview_value_edit(relation, scalar_edit_value(&base, target, 2.0))
            .unwrap_err()
            .code(),
        codes::INVALID_OPERATION
    );
    assert_eq!(base.canonical_json().unwrap(), base_bytes);
    assert_eq!(base.digest().unwrap(), base_digest);
    assert_eq!(base.program().revision(), base_revision);
}

fn scalar_edit_value(
    document: &ModelDocument,
    target: eqiora_core::RawId,
    value: f64,
) -> eqiora_core::ValueLiteral {
    eqiora_core::ValueLiteral::from_real(
        document
            .program()
            .typed_value(target)
            .unwrap()
            .value_type()
            .clone(),
        value,
    )
    .unwrap()
}
