use std::num::{NonZeroU16, NonZeroUsize};

use eqiora_core::diagnostic::codes;
use eqiora_core::entity::kinds;
use eqiora_core::{Id, OntologyId, ScalarType};
use eqiora_schema::Model;
use eqiora_solver::{LinearSolver, SolverPlan};

use super::*;

fn scalar_interval_requirements() -> RealizationRequirements {
    RealizationRequirements::new(
        NonZeroUsize::MIN,
        ScalarType::F64,
        VectorLayoutKind::Replicated,
    )
}

fn explicit_fvm() -> RealizationPlan {
    RealizationPlan::new(
        Space::cell_constant(),
        Discretization::new(
            DiscretizationMethod::CellCenteredFiniteVolume,
            MeshPolicy::GeneratedUniform {
                cells_per_axis: NonZeroUsize::new(32).unwrap(),
            },
            QuadraturePolicy::CellCentroid,
        ),
        SolverPlan::new(
            LinearSolver::ConjugateGradient,
            1.0e-10,
            0.0,
            NonZeroUsize::new(10_000).unwrap(),
        )
        .unwrap(),
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
        ExecutionSchedule::Offline,
    )
    .unwrap()
}

fn imported_p1() -> RealizationPlan {
    RealizationPlan::new(
        Space::continuous_lagrange(NonZeroU16::MIN),
        Discretization::new(
            DiscretizationMethod::ContinuousGalerkin,
            MeshPolicy::ImportedSimplicial {
                artifact: MeshArtifactReference::from_sha256([7; 32]),
            },
            QuadraturePolicy::SimplexCentroid,
        ),
        SolverPlan::new(
            LinearSolver::ConjugateGradient,
            1.0e-10,
            0.0,
            NonZeroUsize::new(10_000).unwrap(),
        )
        .unwrap(),
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
        ExecutionSchedule::Offline,
    )
    .unwrap()
}

#[test]
fn default_and_equal_explicit_fem_resolve_to_the_same_plan() {
    let model = OntologyId::<Model>::new();
    let semantic_revision = SemanticRevision::new(41);
    let capabilities = RealizationCapabilities::scalar_elliptic_reference();
    let default = resolve(
        &RealizationRequest::default(model, semantic_revision, DefaultPolicyVersion::V0),
        scalar_interval_requirements(),
        &capabilities,
    )
    .unwrap();
    let explicit = resolve(
        &RealizationRequest::explicit(
            model,
            semantic_revision,
            RealizationRevision::new(7),
            default_plan_v0().unwrap(),
        ),
        scalar_interval_requirements(),
        &capabilities,
    )
    .unwrap();

    assert_eq!(default.model(), explicit.model());
    assert_eq!(default.semantic_revision(), explicit.semantic_revision());
    assert_eq!(default.plan(), explicit.plan());
    assert_eq!(
        default.source(),
        ResolutionSource::Default(DefaultPolicyVersion::V0)
    );
    assert_eq!(
        explicit.source(),
        ResolutionSource::Explicit(RealizationRevision::new(7))
    );

    let domain = Id::<kinds::Domain>::new();
    let field = Id::<kinds::Field>::new();
    let graph = explicit
        .portable_graph(
            domain,
            field,
            eqiora_solver::LinearOperatorProperties::SymmetricPositiveDefinite,
        )
        .unwrap();
    assert_eq!(graph.domains()[0].domain(), domain);
    assert_eq!(graph.fields()[0].field(), field);
    assert!(matches!(graph.root(), SolveRoot::Linear(_)));
    assert_eq!(
        graph.placements(),
        [PlacementRequirementNode::HostWorkers {
            workers_per_partition: NonZeroUsize::MIN,
        }]
    );
}

#[test]
fn same_semantic_model_accepts_distinct_fvm_realization() {
    let model = OntologyId::<Model>::new();
    let semantic_revision = SemanticRevision::new(12);
    let capabilities = RealizationCapabilities::scalar_elliptic_reference();
    let default = resolve(
        &RealizationRequest::default(model, semantic_revision, DefaultPolicyVersion::V0),
        scalar_interval_requirements(),
        &capabilities,
    )
    .unwrap();
    let fvm = resolve(
        &RealizationRequest::explicit(
            model,
            semantic_revision,
            RealizationRevision::new(2),
            explicit_fvm(),
        ),
        scalar_interval_requirements(),
        &capabilities,
    )
    .unwrap();

    assert_eq!(default.model(), fvm.model());
    assert_eq!(default.semantic_revision(), fvm.semantic_revision());
    assert_ne!(default.plan(), fvm.plan());
    assert_eq!(
        fvm.plan().discretization().method(),
        DiscretizationMethod::CellCenteredFiniteVolume
    );
}

#[test]
fn portable_projection_rejects_solver_operator_contradiction() {
    let model = OntologyId::<Model>::new();
    let resolved = resolve(
        &RealizationRequest::default(model, SemanticRevision::new(1), DefaultPolicyVersion::V0),
        scalar_interval_requirements(),
        &RealizationCapabilities::scalar_elliptic_reference(),
    )
    .unwrap();
    assert_eq!(
        resolved
            .portable_graph(
                Id::new(),
                Id::new(),
                eqiora_solver::LinearOperatorProperties::General,
            )
            .unwrap_err()
            .code(),
        codes::INVALID_REALIZATION
    );
}

#[test]
fn imported_mesh_capability_is_paired_with_its_verified_dimension() {
    let model = OntologyId::<Model>::new();
    let request = RealizationRequest::explicit(
        model,
        SemanticRevision::new(12),
        RealizationRevision::new(3),
        imported_p1(),
    );
    let capabilities = RealizationCapabilities::scalar_elliptic_reference();
    resolve(
        &request,
        RealizationRequirements::new(
            NonZeroUsize::new(2).unwrap(),
            ScalarType::F64,
            VectorLayoutKind::Replicated,
        ),
        &capabilities,
    )
    .unwrap();
    for dimension in [1, 3] {
        assert_eq!(
            resolve(
                &request,
                RealizationRequirements::new(
                    NonZeroUsize::new(dimension).unwrap(),
                    ScalarType::F64,
                    VectorLayoutKind::Replicated,
                ),
                &capabilities,
            )
            .unwrap_err()
            .code(),
            codes::INVALID_REALIZATION,
        );
    }
}

#[test]
fn contradictory_and_unsupported_explicit_plans_never_fall_back() {
    let solver = SolverPlan::new(
        LinearSolver::ConjugateGradient,
        1.0e-10,
        0.0,
        NonZeroUsize::new(100).unwrap(),
    )
    .unwrap();
    let contradiction = RealizationPlan::new(
        Space::cell_constant(),
        Discretization::new(
            DiscretizationMethod::ContinuousGalerkin,
            MeshPolicy::GeneratedUniform {
                cells_per_axis: NonZeroUsize::MIN,
            },
            QuadraturePolicy::GaussLegendre {
                points_per_axis: NonZeroUsize::MIN,
            },
        ),
        solver,
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
        ExecutionSchedule::Offline,
    )
    .unwrap_err();
    assert_eq!(contradiction.code(), codes::INVALID_REALIZATION);

    let cuda = RealizationPlan::new(
        default_plan_v0().unwrap().space(),
        default_plan_v0().unwrap().discretization(),
        solver,
        Target::CudaGpu { device: 0 },
        ExecutionSchedule::Offline,
    )
    .unwrap();
    let request = RealizationRequest::explicit(
        OntologyId::new(),
        SemanticRevision::new(1),
        RealizationRevision::new(1),
        cuda,
    );
    assert_eq!(
        resolve(
            &request,
            scalar_interval_requirements(),
            &RealizationCapabilities::scalar_elliptic_reference()
        )
        .unwrap_err()
        .code(),
        codes::INVALID_REALIZATION
    );
}

#[test]
fn unknown_default_policy_is_an_error_not_a_fallback() {
    let request = RealizationRequest::default(
        OntologyId::new(),
        SemanticRevision::new(1),
        DefaultPolicyVersion::new(99),
    );
    assert_eq!(
        resolve(
            &request,
            scalar_interval_requirements(),
            &RealizationCapabilities::scalar_elliptic_reference()
        )
        .unwrap_err()
        .code(),
        codes::INVALID_REALIZATION
    );
}

#[test]
fn revision_types_and_schedule_ownership_remain_distinct() {
    let request = RealizationRequest::explicit(
        OntologyId::new(),
        SemanticRevision::new(8),
        RealizationRevision::new(13),
        explicit_fvm(),
    );
    assert_eq!(request.semantic_revision().get(), 8);
    assert_eq!(request.realization_revision().unwrap().get(), 13);
    assert!(matches!(request.selection(), Selection::Explicit { .. }));
    assert!(matches!(
        explicit_fvm().schedule(),
        ExecutionSchedule::Offline
    ));
}

#[test]
fn reference_capability_is_exact_about_verified_dimensions_scalar_layout_and_threads() {
    let capabilities = RealizationCapabilities::scalar_elliptic_reference();
    let request = RealizationRequest::default(
        OntologyId::new(),
        SemanticRevision::new(1),
        DefaultPolicyVersion::V0,
    );
    resolve(
        &request,
        RealizationRequirements::new(
            NonZeroUsize::new(3).unwrap(),
            ScalarType::F64,
            VectorLayoutKind::Replicated,
        ),
        &capabilities,
    )
    .expect("the Cartesian reference path admits verified dimension three");
    for unsupported in [
        RealizationRequirements::new(
            NonZeroUsize::new(4).unwrap(),
            ScalarType::F64,
            VectorLayoutKind::Replicated,
        ),
        RealizationRequirements::new(
            NonZeroUsize::MIN,
            ScalarType::F32,
            VectorLayoutKind::Replicated,
        ),
        RealizationRequirements::new(
            NonZeroUsize::MIN,
            ScalarType::F64,
            VectorLayoutKind::Distributed,
        ),
    ] {
        assert_eq!(
            resolve(&request, unsupported, &capabilities)
                .unwrap_err()
                .code(),
            codes::INVALID_REALIZATION
        );
    }

    let default = default_plan_v0().unwrap();
    let plan = RealizationPlan::new(
        default.space(),
        default.discretization(),
        default.solver(),
        Target::HostCpu {
            threads: NonZeroUsize::new(2).unwrap(),
        },
        default.schedule(),
    )
    .unwrap();
    let threaded = RealizationRequest::explicit(
        OntologyId::new(),
        SemanticRevision::new(1),
        RealizationRevision::new(1),
        plan,
    );
    assert_eq!(
        resolve(&threaded, scalar_interval_requirements(), &capabilities)
            .unwrap_err()
            .code(),
        codes::INVALID_REALIZATION
    );
}

#[test]
fn elasticity_reference_capability_is_exactly_2d_cartesian_q1() {
    let model = OntologyId::new();
    let semantic_revision = SemanticRevision::new(1);
    let capabilities = RealizationCapabilities::isotropic_elasticity_2d_reference();
    let request = RealizationRequest::default(model, semantic_revision, DefaultPolicyVersion::V0);
    let requirements = RealizationRequirements::new(
        NonZeroUsize::new(2).unwrap(),
        ScalarType::F64,
        VectorLayoutKind::Replicated,
    );
    resolve(&request, requirements, &capabilities)
        .expect("elasticity reference path admits its exact 2D Q1 envelope");

    for dimension in [1, 3] {
        assert_eq!(
            resolve(
                &request,
                RealizationRequirements::new(
                    NonZeroUsize::new(dimension).unwrap(),
                    ScalarType::F64,
                    VectorLayoutKind::Replicated,
                ),
                &capabilities,
            )
            .unwrap_err()
            .code(),
            codes::INVALID_REALIZATION,
        );
    }

    let fvm = RealizationRequest::explicit(
        model,
        semantic_revision,
        RealizationRevision::new(2),
        explicit_fvm(),
    );
    assert_eq!(
        resolve(&fvm, requirements, &capabilities)
            .unwrap_err()
            .code(),
        codes::INVALID_REALIZATION,
    );
}
