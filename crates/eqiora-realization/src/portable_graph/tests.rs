use std::num::{NonZeroU16, NonZeroUsize};

use eqiora_core::diagnostic::codes;
use eqiora_core::entity::kinds;
use eqiora_core::{DimExponents, DynQuantity, Id, OntologyId};
use eqiora_solver::{
    LinearOperatorProperties, LinearSolver, PreconditionerPolicy, ReductionPolicy,
    SolverCapabilities, SolverCapability, SolverPlan,
};

use super::*;
use crate::{
    AlgebraicBlockScale, BackwardEulerRelationStep, DiscretizationMethod, FieldSpaceBinding,
    FieldwiseRealizationPlan, FieldwiseRealizationRequirements, FieldwiseSpatialDiscretization,
    MeshArtifactReference, MeshKind, MeshPolicy, QuadraturePolicy, RealizationCapabilities,
    RealizationRequirements, RealizationRevision, ResolutionSource,
    ResolvedTransientFieldwiseRealization, SemanticRevision, SpatialDimensionSupport,
    TargetCapabilities, TransientFieldwiseRealizationPlan, TransientFieldwiseRealizationRequest,
    TransientFieldwiseRealizationRequirements, resolve_transient_fieldwise,
};

const LENGTH: DimExponents =
    DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension");
const TIME: DimExponents =
    DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).expect("bounded dimension");
const VELOCITY: DimExponents =
    DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).expect("bounded dimension");
const PRESSURE: DimExponents =
    DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0]).expect("bounded dimension");
const GAUGE: DimExponents =
    DimExponents::from_integers([0, 0, -1, 0, 0, 0, 0]).expect("bounded dimension");
const FUNCTIONAL: DimExponents =
    DimExponents::from_integers([1, 1, -3, 0, 0, 0, 0]).expect("bounded dimension");

#[test]
fn dimensional_linear_fields_have_complete_canonical_bindings() {
    let lineage = RealizationLineage::explicit(
        OntologyId::from_ulid("01ARZ3NDEKTSV4RRFFQ69G5FAV".parse::<ulid::Ulid>().unwrap()),
        SemanticRevision::new(1),
        RealizationRevision::new(1),
    );
    let domain = Id::new();
    let space = Space::continuous_lagrange(NonZeroU16::MIN);
    let bindings = [
        FieldSpaceBinding::new(Id::new(), space),
        FieldSpaceBinding::new(Id::new(), space),
    ];
    let build = |bindings: Vec<FieldSpaceBinding>| {
        PortableRealizationGraph::linear_regions(
            lineage,
            [crate::DomainFieldDiscretization::new(domain, bindings, [])?],
            [],
            crate::Discretization::new(
                DiscretizationMethod::ContinuousGalerkin,
                MeshPolicy::GeneratedUniform {
                    cells_per_axis: NonZeroUsize::new(2).unwrap(),
                },
                QuadraturePolicy::GaussLegendre {
                    points_per_axis: NonZeroUsize::new(2).unwrap(),
                },
            ),
            LinearOperatorProperties::General,
            ScalarType::F64,
            VectorLayoutKind::Replicated,
            SolverPlan::new(
                LinearSolver::BiConjugateGradientStabilized,
                1.0e-8,
                1.0e-12,
                NonZeroUsize::new(100).unwrap(),
            )
            .unwrap(),
            Target::HostCpu {
                threads: NonZeroUsize::MIN,
            },
            ExecutionSchedule::Offline,
        )
    };
    let graph = build(bindings.to_vec()).unwrap();
    let reversed = build(bindings.into_iter().rev().collect()).unwrap();
    assert_eq!(graph, reversed);
    assert_eq!(graph.digest().unwrap(), reversed.digest().unwrap());
    assert_eq!(graph.fields().len(), 2);
    assert_eq!(graph.systems()[0].blocks.len(), 2);
    assert_eq!(graph.systems()[0].scaling, SystemScaling::Dimensional);
    assert_eq!(
        PortableRealizationGraph::from_bytes(&graph.to_bytes().unwrap()).unwrap(),
        graph
    );
    assert!(build(Vec::new()).is_err());
    assert!(build(vec![bindings[0], bindings[0]]).is_err());
    let mut missing = graph;
    missing.systems[0].blocks.pop();
    assert!(missing.validate().is_err());
}

#[test]
fn transient_projection_is_one_connected_typed_solve_dag() {
    let fixture = Fixture::new();
    let resolved = fixture.resolve();
    let graph = resolved.portable_graph().unwrap();

    assert_eq!(graph.lineage().model(), resolved.model());
    assert_eq!(
        graph.lineage().source(),
        ResolutionSource::Explicit(RealizationRevision::new(9))
    );
    assert_eq!(graph.domains().len(), 1);
    assert_eq!(graph.domains()[0].domain(), fixture.domain);
    assert_eq!(graph.fields().len(), 2);
    assert_eq!(graph.systems().len(), 1);
    assert_eq!(graph.linear_solves().len(), 1);
    assert_eq!(graph.nonlinear_solves().len(), 1);
    assert_eq!(
        graph.placements(),
        [PlacementRequirementNode::HostWorkers {
            workers_per_partition: NonZeroUsize::MIN,
        }]
    );

    let SolveRoot::Nonlinear(root) = graph.root() else {
        panic!("transient projection must have a nonlinear root");
    };
    let nonlinear = graph.nonlinear_solve(root).unwrap();
    let linear = graph.linear_solve(nonlinear.linearization()).unwrap();
    assert_eq!(linear.plan(), resolved.plan().fieldwise().solver());
    assert_eq!(linear.schedule(), ExecutionSchedule::Offline);
    assert_eq!(nonlinear.plan(), resolved.plan().nonlinear());

    let velocity = graph
        .fields()
        .iter()
        .position(|field| field.field() == fixture.velocity)
        .map(FieldRepresentationId::new)
        .unwrap();
    assert_eq!(
        graph.transformations(),
        [
            TransformationNode::BackwardEulerDerivative {
                relation: fixture.relation,
                state: velocity,
                duration: DynQuantity::new(0.01, TIME),
            },
            TransformationNode::EnergySkewConvection {
                relation: fixture.relation,
                velocity,
            },
        ]
    );
}

#[test]
fn malformed_or_disconnected_projection_fails_closed() {
    let fixture = Fixture::new();
    let graph = fixture.resolve().portable_graph().unwrap();

    let mut orphan_domain = graph.clone();
    orphan_domain.domains.push(DomainDiscretizationNode {
        domain: Id::new(),
        coordinates: CoordinateTreatment::Physical,
        configuration: DomainConfiguration::FixedGeometry,
        discretization: orphan_domain.domains[0].discretization,
    });
    orphan_domain
        .domains
        .sort_by_key(|domain| domain.domain.ulid());
    assert_eq!(
        orphan_domain.validate().unwrap_err().code(),
        codes::INVALID_REALIZATION
    );

    let mut missing_state = graph.clone();
    missing_state.transformations[0] = TransformationNode::BackwardEulerDerivative {
        relation: fixture.relation,
        state: FieldRepresentationId::new(usize::MAX),
        duration: DynQuantity::new(0.01, TIME),
    };
    assert_eq!(
        missing_state.validate().unwrap_err().code(),
        codes::INVALID_REALIZATION
    );

    let mut orphan_transformation = graph;
    orphan_transformation.systems[0].transformations.pop();
    assert_eq!(
        orphan_transformation.validate().unwrap_err().code(),
        codes::INVALID_REALIZATION
    );
}

#[test]
fn portable_wire_round_trips_and_has_a_domain_separated_digest() {
    use sha2::{Digest, Sha256};

    let graph = Fixture::new().resolve().portable_graph().unwrap();
    let bytes = graph.to_bytes().unwrap();
    let decoded = PortableRealizationGraph::from_bytes(&bytes).unwrap();

    assert_eq!(decoded, graph);
    assert_eq!(decoded.to_bytes().unwrap(), bytes);

    let mut expected = Sha256::new();
    expected.update(b"eqiora.portable-realization-graph/v2\0");
    expected.update(&bytes);
    assert_eq!(
        graph.digest().unwrap(),
        <[u8; 32]>::from(expected.finalize())
    );
}

#[test]
fn portable_wire_rejects_noncanonical_unknown_and_disconnected_payloads() {
    let graph = Fixture::new().resolve().portable_graph().unwrap();
    let bytes = graph.to_bytes().unwrap();

    let mut noncanonical = bytes.clone();
    noncanonical.push(b'\n');
    assert_eq!(
        PortableRealizationGraph::from_bytes(&noncanonical)
            .unwrap_err()
            .code(),
        codes::INVALID_REALIZATION
    );

    let unknown = String::from_utf8(bytes.clone()).unwrap().replacen(
        "{\"schema\":",
        "{\"unknown\":true,\"schema\":",
        1,
    );
    assert_eq!(
        PortableRealizationGraph::from_bytes(unknown.as_bytes())
            .unwrap_err()
            .code(),
        codes::INVALID_REALIZATION
    );

    let unsupported = String::from_utf8(bytes.clone()).unwrap().replace(
        "eqiora.portable-realization-graph/v2",
        "eqiora.portable-realization-graph/v1",
    );
    assert_eq!(
        PortableRealizationGraph::from_bytes(unsupported.as_bytes())
            .unwrap_err()
            .code(),
        codes::INVALID_REALIZATION
    );

    let unknown_default = String::from_utf8(bytes.clone()).unwrap().replace(
        "\"source\":{\"kind\":\"explicit\",\"realization_revision\":9}",
        "\"source\":{\"kind\":\"default\",\"policy_version\":999}",
    );
    assert_eq!(
        PortableRealizationGraph::from_bytes(unknown_default.as_bytes())
            .unwrap_err()
            .code(),
        codes::INVALID_REALIZATION
    );

    let lowercase_ulid = String::from_utf8(bytes.clone())
        .unwrap()
        .replace("01ARZ3NDEKTSV4RRFFQ69G5FAV", "01arz3ndektsv4rrffq69g5fav");
    assert_eq!(
        PortableRealizationGraph::from_bytes(lowercase_ulid.as_bytes())
            .unwrap_err()
            .code(),
        codes::INVALID_REALIZATION
    );

    let uppercase_digest =
        String::from_utf8(bytes.clone())
            .unwrap()
            .replacen(&"ab".repeat(32), &"AB".repeat(32), 1);
    assert_eq!(
        PortableRealizationGraph::from_bytes(uppercase_digest.as_bytes())
            .unwrap_err()
            .code(),
        codes::INVALID_REALIZATION
    );

    let negative_zero = String::from_utf8(bytes.clone()).unwrap().replacen(
        "\"absolute_tolerance\":1e-13",
        "\"absolute_tolerance\":-0.0",
        1,
    );
    assert_eq!(
        PortableRealizationGraph::from_bytes(negative_zero.as_bytes())
            .unwrap_err()
            .code(),
        codes::INVALID_REALIZATION
    );

    let disconnected = String::from_utf8(bytes).unwrap().replace(
        "\"root\":{\"kind\":\"nonlinear\",\"solve\":0}",
        "\"root\":{\"kind\":\"nonlinear\",\"solve\":99}",
    );
    assert_eq!(
        PortableRealizationGraph::from_bytes(disconnected.as_bytes())
            .unwrap_err()
            .code(),
        codes::INVALID_REALIZATION
    );
}

#[test]
fn portable_wire_rejects_oversized_input_and_invalid_graphs() {
    let oversized = vec![b' '; 8 * 1024 * 1024 + 1];
    assert_eq!(
        PortableRealizationGraph::from_bytes(&oversized)
            .unwrap_err()
            .code(),
        codes::INVALID_REALIZATION
    );

    let mut graph = Fixture::new().resolve().portable_graph().unwrap();
    graph.systems[0].transformations.clear();
    assert_eq!(
        graph.to_bytes().unwrap_err().code(),
        codes::INVALID_REALIZATION
    );
}

struct Fixture {
    domain: Id<kinds::Domain>,
    velocity: Id<kinds::Field>,
    relation: Id<kinds::Relation>,
    request: TransientFieldwiseRealizationRequest,
    requirements: TransientFieldwiseRealizationRequirements,
}

impl Fixture {
    fn new() -> Self {
        let domain = Id::new();
        let velocity = Id::new();
        let pressure = Id::new();
        let relation = Id::new();
        let spatial = FieldwiseSpatialDiscretization::new(
            domain,
            scale(LENGTH),
            [
                FieldSpaceBinding::new(velocity, Space::simplex_p1_bubble()),
                FieldSpaceBinding::new(pressure, Space::continuous_lagrange(NonZeroU16::MIN)),
            ],
            [AlgebraicConstraint::ZeroIntegral { field: pressure }],
            Discretization::new(
                DiscretizationMethod::ContinuousGalerkin,
                MeshPolicy::ImportedSimplicial {
                    artifact: MeshArtifactReference::from_sha256([0xab; 32]),
                },
                QuadraturePolicy::TriangleDuffyGaussLegendre {
                    points_per_axis: NonZeroUsize::new(5).unwrap(),
                },
            ),
        )
        .unwrap();
        let scaling = SymmetricCongruenceScaling::new(
            [
                AlgebraicBlockScale::new(AlgebraicBlock::Field(velocity), scale(VELOCITY)),
                AlgebraicBlockScale::new(AlgebraicBlock::Field(pressure), scale(PRESSURE)),
                AlgebraicBlockScale::new(
                    AlgebraicBlock::ConstraintMultiplier { field: pressure },
                    scale(GAUGE),
                ),
            ],
            scale(FUNCTIONAL),
        )
        .unwrap();
        let solver = SolverPlan::new(
            LinearSolver::BiConjugateGradientStabilized,
            1.0e-11,
            1.0e-13,
            NonZeroUsize::new(2_000).unwrap(),
        )
        .unwrap()
        .with_reduction(ReductionPolicy::Fast);
        let fieldwise = FieldwiseRealizationPlan::new(
            spatial,
            scaling,
            LinearOperatorProperties::General,
            solver,
            Target::HostCpu {
                threads: NonZeroUsize::MIN,
            },
            ExecutionSchedule::Offline,
        )
        .unwrap();
        let plan = TransientFieldwiseRealizationPlan::new(
            fieldwise,
            BackwardEulerRelationStep::new(relation, velocity, DynQuantity::new(0.01, TIME))
                .unwrap(),
            crate::EnergySkewConvection::new(relation, velocity),
            NonlinearSolvePlan::new(1.0e-9, 1.0e-11, NonZeroUsize::new(12).unwrap(), 12).unwrap(),
        )
        .unwrap();
        let execution = RealizationRequirements::new(
            NonZeroUsize::new(2).unwrap(),
            ScalarType::F64,
            VectorLayoutKind::Replicated,
        );
        let requirements = TransientFieldwiseRealizationRequirements::new(
            FieldwiseRealizationRequirements::new(domain, [velocity, pressure], execution).unwrap(),
            relation,
            velocity,
        )
        .unwrap();
        let request = TransientFieldwiseRealizationRequest::explicit(
            OntologyId::from_ulid("01ARZ3NDEKTSV4RRFFQ69G5FAV".parse::<ulid::Ulid>().unwrap()),
            SemanticRevision::new(4),
            RealizationRevision::new(9),
            plan,
        );
        Self {
            domain,
            velocity,
            relation,
            request,
            requirements,
        }
    }

    fn resolve(&self) -> ResolvedTransientFieldwiseRealization {
        resolve_transient_fieldwise(&self.request, self.requirements.clone(), &capabilities())
            .unwrap()
    }
}

fn capabilities() -> RealizationCapabilities {
    let solver = SolverCapabilities::exact([SolverCapability {
        algorithm: LinearSolver::BiConjugateGradientStabilized,
        operator_properties: LinearOperatorProperties::General,
        preconditioner: PreconditionerPolicy::Identity,
        reduction: ReductionPolicy::Fast,
        scalar_type: ScalarType::F64,
    }])
    .unwrap();
    RealizationCapabilities::cartesian_product(
        [DiscretizationMethod::ContinuousGalerkin],
        [(
            MeshKind::ImportedAffineSimplicial,
            SpatialDimensionSupport::exact(NonZeroUsize::new(2).unwrap()),
        )],
        [VectorLayoutKind::Replicated],
        solver,
        TargetCapabilities::none().with_host_cpu(NonZeroUsize::MIN),
    )
    .unwrap()
}

fn scale(dimension: DimExponents) -> PositivePhysicalScale {
    PositivePhysicalScale::new(DynQuantity::new(1.0, dimension)).unwrap()
}
