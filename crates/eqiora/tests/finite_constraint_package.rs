//! Exact packages retain mathematical condition kinds and operand order.

use std::num::NonZeroUsize;

use eqiora::DimExponents;
use eqiora::api::ModelDocument;
use eqiora::artifact::ModelEnvelope;
use eqiora::kernel::{KernelNode, RelationConditionKind};
use eqiora::package::{
    BundleEntryV1, BundleRoleV1, ExactVersion, InMemoryPackageStore, NormalizedRelativePath,
    PackageManifestV1, PackageSourcesV1, PackagedModelDocument, QualifiedName, ResolutionRecordV1,
    SourceFileV1, prepare_package_release_v1,
};
use eqiora::solver::{
    LinearSolver, LinearSolverBackend, PreconditionerPolicy, ReductionPolicy, SolverPlan,
};
use eqiora_backend_faer::FaerLinearSolver;
use eqiora_numerics::finite_constraints::{
    ConstraintActivity, ConstraintRef, ConstraintTolerance, FiniteConstraintEnforcement,
};
use eqiora_numerics::{
    CommonAlgebraicPlan, CommonLinearRequest, CommonResult, CommonSolvePolicy, ResolvedCommonPlan,
};

const SOURCE: &str = r#"
model Contact() {
  variable gap: m;
  variable force: N;
  relation contact {
    2[N/m] * gap - force = 6[N];
    complementarity(0[m] <= gap, force >= 0[N]);
    inequality(gap <= 4[m]);
  }
}
"#;

#[test]
fn exact_package_replay_retains_equality_complementarity_and_inequality() {
    let direct = ModelDocument::compile("contact.eqi", SOURCE).unwrap();
    let path = NormalizedRelativePath::parse("src/contact.eqi").unwrap();
    let manifest = PackageManifestV1::new(
        "contact",
        QualifiedName::parse("org.eqiora.test.FiniteConstraint").unwrap(),
        ExactVersion::parse("1.0.0").unwrap(),
        vec![],
        vec![BundleEntryV1::new(path.clone(), BundleRoleV1::ModelSource)],
    )
    .unwrap();
    let sources = PackageSourcesV1::new(
        manifest,
        vec![SourceFileV1::new(
            path,
            BundleRoleV1::ModelSource,
            SOURCE.as_bytes().to_vec(),
        )],
    )
    .unwrap();
    let release = prepare_package_release_v1(sources, &[]).unwrap();
    let mut store = InMemoryPackageStore::default();
    store.insert(&release).unwrap();
    let resolution = ResolutionRecordV1::from_exact_releases(&release, &[]).unwrap();
    let packaged = PackagedModelDocument::compile_locked(&store, &resolution, "Contact").unwrap();
    assert!(packaged.model().structurally_equivalent(&direct).unwrap());

    let relation = packaged
        .model()
        .program()
        .nodes()
        .find_map(|node| match node {
            KernelNode::Relation(relation) => Some(relation),
            _ => None,
        })
        .unwrap();
    assert_eq!(
        relation.conditions().unwrap(),
        &[
            RelationConditionKind::Equality,
            RelationConditionKind::Complementarity,
            RelationConditionKind::Inequality,
        ]
    );

    let model = ModelEnvelope::from_program(packaged.model().program()).unwrap();
    let length = DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).unwrap();
    let force = DimExponents::from_integers([1, 1, -2, 0, 0, 0, 0]).unwrap();
    let enforcement = FiniteConstraintEnforcement::active_set(
        vec![
            ConstraintTolerance::complementarity(
                ConstraintRef::new(relation.id(), 1),
                eqiora::DynQuantity::new(1e-10, length),
                eqiora::DynQuantity::new(1e-10, force),
            )
            .unwrap(),
            ConstraintTolerance::inequality(
                ConstraintRef::new(relation.id(), 2),
                eqiora::DynQuantity::new(1e-10, length),
            )
            .unwrap(),
        ],
        2,
    )
    .unwrap();
    let solver = SolverPlan::new(
        LinearSolver::SparseLu,
        1e-12,
        1e-14,
        NonZeroUsize::new(8).unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Identity)
    .with_reduction(ReductionPolicy::Fast);
    let request = CommonSolvePolicy::Linear(
        CommonLinearRequest::exact(solver, FaerLinearSolver.provider()).unwrap(),
    );
    let plan = CommonAlgebraicPlan::resolve(&model, request, Some(enforcement), &FaerLinearSolver)
        .unwrap();
    let state = plan.initial_state().unwrap();
    let result = plan.run_result(&state, &FaerLinearSolver).unwrap();
    let measurements = result.constraint_measurements();
    assert_eq!(measurements.len(), 2);
    assert_eq!(measurements[0].activity(), ConstraintActivity::Inactive);
    assert_eq!(measurements[0].left().value(), 3.0);
    assert_eq!(measurements[0].right().value(), 0.0);
    assert_eq!(measurements[1].activity(), ConstraintActivity::Inequality);
    let resolved = ResolvedCommonPlan::Algebraic(Box::new(plan));
    let reopened = ResolvedCommonPlan::from_bytes(
        &resolved.to_bytes().unwrap(),
        &FaerLinearSolver,
        eqiora::time::TimeBackendIdentity::new("eqiora.test.time", "1"),
    )
    .unwrap();
    assert_eq!(reopened, resolved);
    assert_eq!(
        CommonResult::from_bytes(&result.to_bytes().unwrap(), &reopened).unwrap(),
        result
    );
}
