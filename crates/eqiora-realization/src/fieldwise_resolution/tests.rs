use std::num::{NonZeroU16, NonZeroUsize};

use eqiora_core::diagnostic::codes;
use eqiora_core::entity::kinds;
use eqiora_core::{DimExponents, DynQuantity, Id, OntologyId, ScalarType};
use eqiora_solver::{LinearOperatorProperties, LinearSolver, SolverPlan};

use super::*;
use crate::{
    AlgebraicBlock, AlgebraicBlockScale, AlgebraicConstraint, Discretization, DiscretizationMethod,
    ExecutionSchedule, FieldSpaceBinding, FieldwiseSpatialDiscretization, MeshArtifactReference,
    MeshPolicy, PositivePhysicalScale, QuadraturePolicy, Space, SymmetricCongruenceScaling, Target,
    VectorLayoutKind,
};

#[test]
fn exact_requirements_are_order_invariant_and_resolve() {
    let domain = Id::<kinds::Domain>::new();
    let velocity = Id::<kinds::Field>::new();
    let pressure = Id::<kinds::Field>::new();
    let request = request(domain, velocity, pressure, LinearSolver::MinimumResidual);
    let forward = requirements(domain, [velocity, pressure]);
    let reverse = requirements(domain, [pressure, velocity]);

    assert_eq!(forward, reverse);
    let resolved = resolve_fieldwise(
        &request,
        reverse,
        &RealizationCapabilities::symmetric_mixed_simplicial_2d_reference(),
    )
    .unwrap();
    assert_eq!(resolved.model(), request.model());
    assert_eq!(resolved.semantic_revision(), SemanticRevision::new(4));
    assert_eq!(resolved.realization_revision(), RealizationRevision::new(9));
    assert_eq!(resolved.plan(), request.plan());

    let graph = resolved.portable_graph().unwrap();
    assert_eq!(
        crate::PortableRealizationGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(),
        graph
    );
    assert!(matches!(graph.root(), crate::SolveRoot::Linear(_)));
    assert!(graph.transformations().is_empty());
    assert_eq!(graph.domains()[0].domain(), domain);
    assert_eq!(graph.fields().len(), 2);
    assert_eq!(
        graph.placements(),
        [crate::PlacementRequirementNode::HostWorkers {
            workers_per_partition: NonZeroUsize::MIN,
        }]
    );
}

#[test]
fn exact_domain_and_unknown_field_drift_are_rejected() {
    let domain = Id::<kinds::Domain>::new();
    let velocity = Id::<kinds::Field>::new();
    let pressure = Id::<kinds::Field>::new();
    let request = request(domain, velocity, pressure, LinearSolver::MinimumResidual);
    let capabilities = RealizationCapabilities::symmetric_mixed_simplicial_2d_reference();

    for wrong in [
        requirements(Id::new(), [velocity, pressure]),
        requirements(domain, [velocity, Id::new()]),
    ] {
        assert_eq!(
            resolve_fieldwise(&request, wrong, &capabilities)
                .unwrap_err()
                .code(),
            codes::INVALID_REALIZATION
        );
    }
    assert_eq!(
        FieldwiseRealizationRequirements::new(
            domain,
            [velocity, velocity],
            execution_requirements(),
        )
        .unwrap_err()
        .code(),
        codes::INVALID_REALIZATION
    );
}

#[test]
fn operator_property_participates_in_exact_solver_capability() {
    let domain = Id::<kinds::Domain>::new();
    let velocity = Id::<kinds::Field>::new();
    let pressure = Id::<kinds::Field>::new();
    let request = request(domain, velocity, pressure, LinearSolver::ConjugateGradient);

    assert_eq!(
        resolve_fieldwise(
            &request,
            requirements(domain, [velocity, pressure]),
            &RealizationCapabilities::symmetric_mixed_simplicial_2d_reference(),
        )
        .unwrap_err()
        .code(),
        codes::INVALID_REALIZATION
    );
}

fn request(
    domain: Id<kinds::Domain>,
    velocity: Id<kinds::Field>,
    pressure: Id<kinds::Field>,
    algorithm: LinearSolver,
) -> FieldwiseRealizationRequest {
    let spatial = FieldwiseSpatialDiscretization::new(
        domain,
        physical_scale(length_dimension()),
        [
            FieldSpaceBinding::new(velocity, Space::simplex_p1_bubble()),
            FieldSpaceBinding::new(pressure, Space::continuous_lagrange(NonZeroU16::MIN)),
        ],
        [AlgebraicConstraint::ZeroIntegral { field: pressure }],
        Discretization::new(
            DiscretizationMethod::ContinuousGalerkin,
            MeshPolicy::ImportedSimplicial {
                artifact: MeshArtifactReference::from_sha256([5; 32]),
            },
            QuadraturePolicy::TriangleDuffyGaussLegendre {
                points_per_axis: NonZeroUsize::new(3).unwrap(),
            },
        ),
    )
    .unwrap();
    let scaling = SymmetricCongruenceScaling::new(
        [
            AlgebraicBlockScale::new(
                AlgebraicBlock::Field(velocity),
                physical_scale(velocity_dimension()),
            ),
            AlgebraicBlockScale::new(
                AlgebraicBlock::Field(pressure),
                physical_scale(pressure_dimension()),
            ),
            AlgebraicBlockScale::new(
                AlgebraicBlock::ConstraintMultiplier { field: pressure },
                physical_scale(gauge_dimension()),
            ),
        ],
        physical_scale(functional_dimension()),
    )
    .unwrap();
    let solver = SolverPlan::new(
        algorithm,
        1.0e-11,
        1.0e-13,
        NonZeroUsize::new(10_000).unwrap(),
    )
    .unwrap();
    let plan = FieldwiseRealizationPlan::new(
        spatial,
        scaling,
        LinearOperatorProperties::SymmetricIndefinite,
        solver,
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
        ExecutionSchedule::Offline,
    )
    .unwrap();
    FieldwiseRealizationRequest::explicit(
        OntologyId::new(),
        SemanticRevision::new(4),
        RealizationRevision::new(9),
        plan,
    )
}

fn requirements<const N: usize>(
    domain: Id<kinds::Domain>,
    fields: [Id<kinds::Field>; N],
) -> FieldwiseRealizationRequirements {
    FieldwiseRealizationRequirements::new(domain, fields, execution_requirements()).unwrap()
}

fn execution_requirements() -> RealizationRequirements {
    RealizationRequirements::new(
        NonZeroUsize::new(2).unwrap(),
        ScalarType::F64,
        VectorLayoutKind::Replicated,
    )
}

fn physical_scale(dimension: DimExponents) -> PositivePhysicalScale {
    PositivePhysicalScale::new(DynQuantity::new(1.0, dimension)).unwrap()
}

fn length_dimension() -> DimExponents {
    DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension")
}

fn velocity_dimension() -> DimExponents {
    DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).expect("bounded dimension")
}

fn pressure_dimension() -> DimExponents {
    DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0]).expect("bounded dimension")
}

fn gauge_dimension() -> DimExponents {
    DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).expect("bounded dimension")
}

fn functional_dimension() -> DimExponents {
    DimExponents::from_integers([1, 1, -3, 0, 0, 0, 0]).expect("bounded dimension")
}
