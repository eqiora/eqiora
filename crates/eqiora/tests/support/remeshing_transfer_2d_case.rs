use eqiora_core::ScalarType;
use std::num::{NonZeroU16, NonZeroUsize};

use eqiora::api::ModelDocument;
use eqiora::artifact::SimplicialMeshEnvelopeV1;
use eqiora::backends::faer::FaerLinearSolver;
use eqiora::meshing::{
    CellId, FacetId, MeshEntity, MeshQualityGate, MeshTopology, SimplicialMesh,
    triangle_duffy_gauss_legendre,
};
use eqiora::realization::{
    AleFsiRemeshScaleProfile2d, AleFsiRemeshTransferPlan2d, AleGeometryQualityGate,
    AlgebraicBlockScale, BackwardEulerRelationStep, BackwardEulerStateBinding,
    BackwardEulerStatePair, BackwardEulerStep, ConformingTraceQuotient,
    CoupledFieldwiseRealizationPlan, CoupledFieldwiseSpatialDiscretization, Discretization,
    DiscretizationMethod, DomainFieldDiscretization, ExecutionSchedule, FieldSpaceBinding,
    FixedTopologyAleCoupledRealizationPlan, FixedTopologyAleCoupledRealizationRequest,
    GclCompatibleAlePullback, MeshArtifactReference, MeshKind, MeshPolicy, NonlinearSolvePlan,
    P1HarmonicMeshMotionPolicy, PositivePhysicalScale, QuadraturePolicy, RealizationCapabilities,
    RealizationRevision, ResolvedFixedTopologyAleCoupledRealization, SemanticRevision, Space,
    SpatialDimensionSupport, SymmetricCongruenceScaling, Target, TargetCapabilities,
    TraceFieldEndpoint, VectorLayoutKind, resolve_fixed_topology_ale_coupled,
};
use eqiora::solver::{
    AlgebraicBlock, LinearOperatorProperties, LinearSolveRequest, LinearSolver,
    PreconditionerPolicy, REFERENCE_LINEAR_SOLVER, ReductionPolicy, SolverCapabilities,
    SolverCapability, SolverPlan,
};
use eqiora::{DimExponents, DynQuantity, Id, kinds};
use eqiora_numerics::{
    ale::AcceptedAleFsiRemeshProjection2d, ale::AleFsiBoundary, ale::AleFsiCartesianModel,
    ale::AleFsiState, ale::FinalizedResolvedFixedTopologyAleFsi,
    ale::fixed_topology_ale_fsi_requirements_2d, ale::lower_ale_fsi_cartesian_2d,
    ale::project_simplicial_ale_fsi_remesh_2d, ale::remesh_resolved_fixed_topology_ale_fsi_2d,
    common::NonZeroStepCount, fsi::FixedReferenceFsiPartition, fsi::FixedReferenceFsiScale,
};

pub(super) const COMPONENTS: usize = 2;
pub(super) const TIME_STEP: f64 = 1.0 / 512.0;
pub(super) const LENGTH: DimExponents =
    DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension");
const TIME: DimExponents =
    DimExponents::from_integers([0, 0, 1, 0, 0, 0, 0]).expect("bounded dimension");
pub(super) const VELOCITY: DimExponents =
    DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).expect("bounded dimension");
pub(super) const PRESSURE: DimExponents =
    DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0]).expect("bounded dimension");
pub(super) const WEAK_FUNCTIONAL: DimExponents =
    DimExponents::from_integers([1, 1, -3, 0, 0, 0, 0]).expect("bounded dimension");
const DIRECT_SOURCE: &str =
    include_str!("../../../../verify/fsi/remeshing-transfer-2d/models/direct.eqi");
const CASE_CONTRACT: &str = include_str!("../../../../verify/fsi/remeshing-transfer-2d/case.toml");

pub(super) struct Case {
    pub(super) document: ModelDocument,
    pub(super) canonical: AleFsiCartesianModel<2>,
    pub(super) source_mesh_artifact: SimplicialMeshEnvelopeV1,
    pub(super) source_reference: MeshArtifactReference,
    pub(super) source_mesh: SimplicialMesh,
    pub(super) source_partition: FixedReferenceFsiPartition<2>,
    pub(super) source_boundary: AleFsiBoundary<2>,
    pub(super) target_mesh_artifact: SimplicialMeshEnvelopeV1,
    pub(super) target_reference: MeshArtifactReference,
    pub(super) target_mesh: SimplicialMesh,
    pub(super) target_partition: FixedReferenceFsiPartition<2>,
    pub(super) target_boundary: AleFsiBoundary<2>,
}

impl Case {
    pub(super) fn new() -> Self {
        let document =
            eqiora::api::ModelDocument::compile("remeshing-transfer-2d.eqi", DIRECT_SOURCE)
                .unwrap();
        let canonical = lower_ale_fsi_cartesian_2d(document.program()).unwrap();
        let source_mesh = two_domain_mesh(false);
        let target_mesh = two_domain_mesh(true);
        assert_ne!(source_mesh.vertices().len(), target_mesh.vertices().len());
        assert_ne!(source_mesh.cells().len(), target_mesh.cells().len());
        assert_ne!(source_mesh.entity_count(1), target_mesh.entity_count(1));
        let source_mesh_artifact = SimplicialMeshEnvelopeV1::from_mesh(&source_mesh).unwrap();
        let target_mesh_artifact = SimplicialMeshEnvelopeV1::from_mesh(&target_mesh).unwrap();
        let source_reference = source_mesh_artifact.artifact_reference().unwrap();
        let target_reference = target_mesh_artifact.artifact_reference().unwrap();
        assert_ne!(source_reference, target_reference);
        let source_partition = partition(&source_mesh);
        let target_partition = partition(&target_mesh);
        assert_eq!(source_partition.interface_facets().len(), 2);
        assert_eq!(target_partition.interface_facets().len(), 4);
        let source_boundary = AleFsiBoundary::<2>::homogeneous_exterior(&source_mesh).unwrap();
        let target_boundary = AleFsiBoundary::<2>::homogeneous_exterior(&target_mesh).unwrap();
        Self {
            document,
            canonical,
            source_mesh_artifact,
            source_reference,
            source_mesh,
            source_partition,
            source_boundary,
            target_mesh_artifact,
            target_reference,
            target_mesh,
            target_partition,
            target_boundary,
        }
    }

    pub(super) fn initial_physical(&self) -> eqiora_numerics::ale::AleFsiInitialPhysicalState<2> {
        let mut displacement = vec![[0.0; COMPONENTS]; self.source_mesh.vertices().len()];
        for vertex in self.source_partition.solid_vertices() {
            let y = self.source_mesh.vertices()[vertex.index()][1];
            if y.to_bits() == 0.5_f64.to_bits() {
                displacement[vertex.index()] = [0.0, 1.0 / 1024.0];
            }
        }
        eqiora_numerics::ale::AleFsiInitialPhysicalState::<2>::new(
            0.0,
            vec![[0.0; COMPONENTS]; self.source_mesh.vertices().len()],
            self.source_partition
                .fluid_cells()
                .iter()
                .copied()
                .map(|cell| (cell, [0.0; COMPONENTS]))
                .collect(),
            vec![0.0; self.source_partition.fluid_vertices().len()],
            displacement,
        )
        .unwrap()
    }

    pub(super) fn resolve(
        &self,
        mesh: MeshArtifactReference,
        revision: u64,
        time_step: f64,
        semantic_revision: Option<u64>,
    ) -> ResolvedFixedTopologyAleCoupledRealization {
        resolve_fixed_topology_ale_coupled(
            &FixedTopologyAleCoupledRealizationRequest::explicit(
                self.canonical.model(),
                SemanticRevision::new(
                    semantic_revision.unwrap_or(self.canonical.semantic_revision()),
                ),
                RealizationRevision::new(revision),
                realization_plan(&self.canonical, mesh, time_step),
            ),
            fixed_topology_ale_fsi_requirements_2d(&self.canonical),
            &capabilities(),
        )
        .unwrap()
    }

    fn resolve_with_scale(
        &self,
        mesh: MeshArtifactReference,
        revision: u64,
        length: f64,
        velocity: f64,
        pressure: f64,
    ) -> ResolvedFixedTopologyAleCoupledRealization {
        resolve_fixed_topology_ale_coupled(
            &FixedTopologyAleCoupledRealizationRequest::explicit(
                self.canonical.model(),
                SemanticRevision::new(self.canonical.semantic_revision()),
                RealizationRevision::new(revision),
                realization_plan_with_scales(
                    &self.canonical,
                    mesh,
                    TIME_STEP,
                    length,
                    velocity,
                    pressure,
                ),
            ),
            fixed_topology_ale_fsi_requirements_2d(&self.canonical),
            &capabilities(),
        )
        .unwrap()
    }
}

pub(super) fn assert_numerical_falsifiers(
    case: &Case,
    source: &FinalizedResolvedFixedTopologyAleFsi<2>,
    source_state: &AleFsiState<2>,
    target: &ResolvedFixedTopologyAleCoupledRealization,
) {
    let stale_mesh = remesh_resolved_fixed_topology_ale_fsi_2d(
        &case.canonical,
        source,
        source_state,
        target,
        case.source_reference,
        &case.target_mesh,
        &case.target_partition,
        &case.target_boundary,
        transfer_plan(),
        &FaerLinearSolver,
        &REFERENCE_LINEAR_SOLVER,
    )
    .expect_err("a stale target mesh identity must fail before transfer");
    assert_eq!(
        stale_mesh.code(),
        eqiora::diagnostic::codes::INVALID_REALIZATION
    );

    let changed_policy = case.resolve(case.target_reference, 3, 2.0 * TIME_STEP, None);
    assert!(
        remesh_resolved_fixed_topology_ale_fsi_2d(
            &case.canonical,
            source,
            source_state,
            &changed_policy,
            case.target_reference,
            &case.target_mesh,
            &case.target_partition,
            &case.target_boundary,
            transfer_plan(),
            &FaerLinearSolver,
            &REFERENCE_LINEAR_SOLVER,
        )
        .is_err(),
        "a changed positive-time policy must not cross the remesh seam"
    );

    let changed_identity = case.resolve(
        case.target_reference,
        4,
        TIME_STEP,
        Some(case.canonical.semantic_revision() + 1),
    );
    assert!(
        remesh_resolved_fixed_topology_ale_fsi_2d(
            &case.canonical,
            source,
            source_state,
            &changed_identity,
            case.target_reference,
            &case.target_mesh,
            &case.target_partition,
            &case.target_boundary,
            transfer_plan(),
            &FaerLinearSolver,
            &REFERENCE_LINEAR_SOLVER,
        )
        .is_err(),
        "a changed Semantic revision must fail before transfer"
    );

    assert!(
        remesh_resolved_fixed_topology_ale_fsi_2d(
            &case.canonical,
            source,
            source_state,
            target,
            case.target_reference,
            &case.target_mesh,
            &case.target_partition,
            &case.target_boundary,
            scale_invariance_transfer_plan(),
            &FaerLinearSolver,
            &REFERENCE_LINEAR_SOLVER,
        )
        .is_err(),
        "a transfer scale profile different from either Realization must fail admission"
    );

    let changed_realization_scale =
        case.resolve_with_scale(case.target_reference, 5, 4.0, 2.0, 4.0);
    assert!(
        remesh_resolved_fixed_topology_ale_fsi_2d(
            &case.canonical,
            source,
            source_state,
            &changed_realization_scale,
            case.target_reference,
            &case.target_mesh,
            &case.target_partition,
            &case.target_boundary,
            scale_invariance_transfer_plan(),
            &FaerLinearSolver,
            &REFERENCE_LINEAR_SOLVER,
        )
        .is_err(),
        "source and target Realizations with different L/U/P profiles must fail admission"
    );

    let wrong_solver = SolverPlan::new(
        LinearSolver::ConjugateGradient,
        1.0e-11,
        1.0e-13,
        NonZeroUsize::new(2_000).unwrap(),
    )
    .unwrap();
    assert!(
        AleFsiRemeshTransferPlan2d::new(
            QuadraturePolicy::TriangleDuffyGaussLegendre {
                points_per_axis: NonZeroUsize::new(5).unwrap(),
            },
            transfer_scales(),
            wrong_solver,
        )
        .is_err()
    );
}

pub(super) fn assert_strong_source_witness(case: &Case, state: &AleFsiState<2>) {
    assert!(
        state
            .fluid_cell_bubble_velocity()
            .values()
            .flatten()
            .any(|value| value.abs() > 1.0e-12),
        "the source must exercise the MINI bubble rather than a disguised P1/P0 path"
    );
    let pressure_min = state
        .fluid_pressure()
        .iter()
        .copied()
        .fold(f64::INFINITY, f64::min);
    let pressure_max = state
        .fluid_pressure()
        .iter()
        .copied()
        .fold(f64::NEG_INFINITY, f64::max);
    assert!(
        pressure_min.abs().max(pressure_max.abs()) > 1.0e-12
            && pressure_max - pressure_min > 1.0e-12,
        "the absolute-pressure source witness must be nonzero and nonconstant"
    );
    assert!(
        case.source_partition
            .interface_vertices()
            .iter()
            .flat_map(|vertex| state.vertex_velocity()[vertex.index()])
            .any(|value| value.abs() > 1.0e-12),
        "the conserving interface must carry nonzero shared velocity"
    );
    assert!(
        case.source_partition
            .solid_vertices()
            .iter()
            .flat_map(|vertex| state.solid_displacement()[vertex.index()])
            .any(|value| value.abs() > 1.0e-12),
        "the material displacement transfer must not be a zero-field witness"
    );
    assert_interface_witness(
        &case.source_mesh,
        state.vertex_velocity(),
        state.solid_displacement(),
        "source",
    );
}

pub(super) fn assert_interface_witness(
    mesh: &SimplicialMesh,
    vertex_velocity: &[[f64; COMPONENTS]],
    solid_displacement: &[[f64; COMPONENTS]],
    label: &str,
) {
    let lower = find_vertex(mesh, [1.0, 0.0]);
    let kink = find_vertex(mesh, [1.0, 0.5]);
    let upper = find_vertex(mesh, [1.0, 1.0]);
    let affine_midpoint = 0.5 * (solid_displacement[lower][1] + solid_displacement[upper][1]);
    assert!(
        (solid_displacement[kink][1] - affine_midpoint).abs() > 1.0e-12,
        "{label} interface must retain a non-affine transverse P1 trace kink"
    );
    assert!(
        vertex_velocity[kink][1].abs() > 1.0e-12,
        "{label} shared transverse velocity witness must remain nonzero"
    );
}

pub(super) fn assert_exact_interface_bisection(
    source_mesh: &SimplicialMesh,
    source_displacement: &[[f64; COMPONENTS]],
    target_mesh: &SimplicialMesh,
    target_current_coordinates: &[Vec<f64>],
) {
    // Material P1 trace is evaluated first; current geometry then adds X + d.
    // An exact-real midpoint of already rounded current endpoints need not be
    // representable. It is not the retained geometry-action contract.
    let matches = |candidate: &[Vec<f64>]| {
        for (lower_y, upper_y) in [(0.0, 0.5), (0.5, 1.0)] {
            let lower = find_vertex(source_mesh, [1.0, lower_y]);
            let upper = find_vertex(source_mesh, [1.0, upper_y]);
            let midpoint = find_vertex(target_mesh, [1.0, 0.5 * (lower_y + upper_y)]);
            let lower_current: [f64; COMPONENTS] = std::array::from_fn(|axis| {
                source_mesh.vertices()[lower][axis] + source_displacement[lower][axis]
            });
            let upper_current: [f64; COMPONENTS] = std::array::from_fn(|axis| {
                source_mesh.vertices()[upper][axis] + source_displacement[upper][axis]
            });
            if !(lower_current[1] < candidate[midpoint][1]
                && candidate[midpoint][1] < upper_current[1])
            {
                return false;
            }
            for axis in 0..COMPONENTS {
                if !midpoint_rounding_cell_contains(
                    lower_current[axis],
                    upper_current[axis],
                    candidate[midpoint][axis],
                ) {
                    return false;
                }
                let expected = target_mesh.vertices()[midpoint][axis]
                    + (0.5 * source_displacement[lower][axis]
                        + 0.5 * source_displacement[upper][axis]);
                if candidate[midpoint][axis].to_bits() != expected.to_bits() {
                    return false;
                }
                for (source, material_y) in [(lower, lower_y), (upper, upper_y)] {
                    let target = find_vertex(target_mesh, [1.0, material_y]);
                    let endpoint =
                        source_mesh.vertices()[source][axis] + source_displacement[source][axis];
                    if candidate[target][axis].to_bits() != endpoint.to_bits() {
                        return false;
                    }
                }
            }
        }
        true
    };
    assert!(
        matches(target_current_coordinates),
        "target interface must preserve the material P1 trace, certified rounded segment support, and every source breakpoint"
    );
    let midpoint = find_vertex(target_mesh, [1.0, 0.25]);
    for axis in 0..COMPONENTS {
        let mut wrong = target_current_coordinates.to_vec();
        wrong[midpoint][axis] = wrong[midpoint][axis].next_up();
        assert!(
            !matches(&wrong),
            "one-ULP support/interpolation drift must fail"
        );
    }
}

fn midpoint_rounding_cell_contains(left: f64, right: f64, midpoint: f64) -> bool {
    let terms = [
        left,
        right,
        midpoint.next_down(),
        midpoint,
        midpoint.next_up(),
    ]
    .map(dyadic);
    let minimum_exponent = terms
        .iter()
        .filter_map(|(mantissa, exponent)| (*mantissa != 0).then_some(*exponent))
        .min()
        .unwrap_or(0);
    let aligned = terms.map(|(mantissa, exponent)| {
        if mantissa == 0 {
            return Some(0);
        }
        let factor = 2_i128.checked_pow((exponent - minimum_exponent) as u32)?;
        mantissa.checked_mul(factor)
    });
    match aligned {
        [
            Some(left),
            Some(right),
            Some(previous),
            Some(midpoint),
            Some(next),
        ] => match (
            previous.checked_add(midpoint),
            left.checked_add(right),
            midpoint.checked_add(next),
        ) {
            (Some(lower), Some(value), Some(upper)) => lower <= value && value <= upper,
            _ => false,
        },
        _ => false,
    }
}

fn dyadic(value: f64) -> (i128, i32) {
    if value == 0.0 {
        return (0, 0);
    }
    let bits = value.to_bits();
    let sign = if bits >> 63 == 0 { 1_i128 } else { -1_i128 };
    let exponent_bits = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & ((1_u64 << 52) - 1);
    if exponent_bits == 0 {
        (sign * i128::from(fraction), -1074)
    } else {
        (
            sign * i128::from((1_u64 << 52) | fraction),
            exponent_bits - 1023 - 52,
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub(super) fn assert_scale_invariant_projection(
    case: &Case,
    source: &FinalizedResolvedFixedTopologyAleFsi<2>,
    source_state: &AleFsiState<2>,
    accepted: &eqiora_numerics::ale::AcceptedResolvedAleFsiRemesh2d,
    base: &AcceptedAleFsiRemeshProjection2d,
    transfer_plan: AleFsiRemeshTransferPlan2d,
) {
    let alternative_scale = FixedReferenceFsiScale::<2>::new(4.0, 2.0, 4.0).unwrap();
    let alternative = project_simplicial_ale_fsi_remesh_2d(
        &case.source_mesh,
        &case.source_partition,
        source.motion(),
        source_state,
        &case.target_mesh,
        &case.target_partition,
        accepted.target().motion(),
        source.step_plan().material(),
        alternative_scale,
        &triangle_duffy_gauss_legendre(5).unwrap(),
        LinearSolveRequest::new(&REFERENCE_LINEAR_SOLVER, transfer_plan.solver()),
    )
    .unwrap();

    let base_evidence = base.evidence();
    let alternative_evidence = alternative.evidence();
    assert_ne!(alternative_evidence.scale(), base_evidence.scale());
    assert_eq!(alternative.time(), base.time());
    assert_eq!(
        alternative_evidence.independent_velocity_constraint_count(),
        base_evidence.independent_velocity_constraint_count()
    );
    assert_independently_accepted_projection(base_evidence);
    assert_independently_accepted_projection(alternative_evidence);

    // Both calls independently assemble and replay the same dimensional
    // projection and constraints. Their computational normalization differs;
    // compare the resulting physical Fields in the Realization's common L/U/P
    // units. This is a deliberately non-semantic observation bound, not a
    // coefficient-error bound inferred from physical conservation residuals.
    assert_eq!(
        alternative
            .fluid_cell_bubble_velocity()
            .keys()
            .collect::<Vec<_>>(),
        base.fluid_cell_bubble_velocity().keys().collect::<Vec<_>>()
    );
    let report = ScaleInvarianceReport {
        vertex_velocity_drift: maximum_vector_field_defect(
            alternative.vertex_velocity(),
            base.vertex_velocity(),
        ) / transfer_plan.scales().velocity().value(),
        bubble_velocity_drift: maximum_vector_field_defect(
            &alternative
                .fluid_cell_bubble_velocity()
                .values()
                .copied()
                .collect::<Vec<_>>(),
            &base
                .fluid_cell_bubble_velocity()
                .values()
                .copied()
                .collect::<Vec<_>>(),
        ) / transfer_plan.scales().velocity().value(),
        pressure_drift: maximum_scalar_field_defect(
            alternative.fluid_pressure(),
            base.fluid_pressure(),
        ) / transfer_plan.scales().pressure().value(),
        displacement_drift: maximum_vector_field_defect(
            alternative.solid_displacement(),
            base.solid_displacement(),
        ) / transfer_plan.scales().length().value(),
        geometry_drift: maximum_coordinate_defect(
            alternative_evidence.target_geometry().coordinates(),
            base_evidence.target_geometry().coordinates(),
        ) / transfer_plan.scales().length().value(),
        base_physical_limit: base_evidence.dimensionless_physical_acceptance_limit(),
        alternative_physical_limit: alternative_evidence.dimensionless_physical_acceptance_limit(),
    };
    report.assert_within_observation_bound(scale_profile_observation_bound());
    assert_eq!(
        alternative_evidence.solid_reference_overlap(),
        base_evidence.solid_reference_overlap()
    );
    assert_eq!(
        overlap_combinatorial_signature(alternative_evidence.fluid_current_overlap()),
        overlap_combinatorial_signature(base_evidence.fluid_current_overlap())
    );
}

#[derive(Debug)]
struct ScaleInvarianceReport {
    vertex_velocity_drift: f64,
    bubble_velocity_drift: f64,
    pressure_drift: f64,
    displacement_drift: f64,
    geometry_drift: f64,
    base_physical_limit: f64,
    alternative_physical_limit: f64,
}

impl ScaleInvarianceReport {
    fn assert_within_observation_bound(&self, observation_bound: f64) {
        let drifts = [
            self.vertex_velocity_drift,
            self.bubble_velocity_drift,
            self.pressure_drift,
            self.displacement_drift,
            self.geometry_drift,
        ];
        assert!(
            self.base_physical_limit.is_finite()
                && self.base_physical_limit > 0.0
                && self.alternative_physical_limit.is_finite()
                && self.alternative_physical_limit > 0.0
                && drifts
                    .iter()
                    .all(|drift| drift.is_finite() && *drift <= observation_bound),
            "independently accepted scale profiles disagree in physical L/U/P units: {self:?}; registered non-semantic observation bound={observation_bound:e}"
        );
    }
}

fn scale_profile_observation_bound() -> f64 {
    CASE_CONTRACT
        .lines()
        .find_map(|line| {
            line.trim()
                .strip_prefix("maximum_dimensionless_physical_field_drift = ")
        })
        .and_then(|value| value.parse::<f64>().ok())
        .filter(|bound| bound.is_finite() && *bound > 0.0)
        .expect("the registered case must declare one positive finite scale observation bound")
}

fn assert_independently_accepted_projection(
    evidence: &eqiora_numerics::ale::AleFsiRemeshProjectionEvidence2d,
) {
    assert!(
        evidence.dimensionless_displacement_projection_residual_norm()
            <= evidence.dimensionless_displacement_projection_acceptance_limit()
    );
    assert!(
        evidence.dimensionless_velocity_projection_residual_norm()
            <= evidence.dimensionless_velocity_projection_acceptance_limit()
    );
    assert!(
        evidence.dimensionless_pressure_projection_residual_norm()
            <= evidence.dimensionless_pressure_projection_acceptance_limit()
    );
    for report in evidence.displacement_solve_reports().iter().chain([
        evidence.velocity_solve_report(),
        evidence.pressure_solve_report(),
    ]) {
        assert!(report.true_residual_norm() <= report.residual_target());
    }
    let physical_limit = evidence.dimensionless_physical_acceptance_limit();
    for defect in [
        evidence.dimensionless_displacement_trace_defect(),
        evidence.dimensionless_shared_velocity_trace_defect(),
        evidence.dimensionless_exterior_velocity_trace_defect(),
        evidence.dimensionless_weak_incompressibility_defect(),
        evidence.dimensionless_momentum_defect(),
        evidence.dimensionless_pressure_zeroth_moment_defect(),
    ] {
        assert!(defect <= physical_limit);
    }
}

fn maximum_vector_field_defect(left: &[[f64; COMPONENTS]], right: &[[f64; COMPONENTS]]) -> f64 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .flatten()
        .zip(right.iter().flatten())
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max)
}

fn maximum_scalar_field_defect(left: &[f64], right: &[f64]) -> f64 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max)
}

pub(super) fn maximum_coordinate_defect(left: &[Vec<f64>], right: &[Vec<f64>]) -> f64 {
    assert_eq!(left.len(), right.len());
    left.iter()
        .zip(right)
        .flat_map(|(left, right)| {
            assert_eq!(left.len(), right.len());
            left.iter().zip(right)
        })
        .map(|(left, right)| (left - right).abs())
        .fold(0.0_f64, f64::max)
}

type OverlapCombinatorialSignature = (Vec<(CellId, CellId)>, Vec<(FacetId, FacetId)>);

fn overlap_combinatorial_signature(
    overlap: &eqiora::meshing::SimplicialRevisionOverlap2d,
) -> OverlapCombinatorialSignature {
    (
        overlap
            .cell_fragments()
            .iter()
            .map(|fragment| (fragment.source_cell(), fragment.target_cell()))
            .collect(),
        overlap
            .retained_facet_fragments()
            .iter()
            .map(|fragment| (fragment.source_facet(), fragment.target_facet()))
            .collect(),
    )
}

pub(super) fn assert_genuine_many_to_many(
    overlap: &eqiora::meshing::SimplicialRevisionOverlap2d,
    label: &str,
) {
    use std::collections::{BTreeMap, BTreeSet};

    let mut source_targets = BTreeMap::<CellId, BTreeSet<CellId>>::new();
    let mut target_sources = BTreeMap::<CellId, BTreeSet<CellId>>::new();
    for fragment in overlap.cell_fragments() {
        source_targets
            .entry(fragment.source_cell())
            .or_default()
            .insert(fragment.target_cell());
        target_sources
            .entry(fragment.target_cell())
            .or_default()
            .insert(fragment.source_cell());
    }
    assert!(
        source_targets.values().any(|targets| targets.len() > 1),
        "{label} overlap must split at least one source cell across target cells"
    );
    assert!(
        target_sources.values().any(|sources| sources.len() > 1),
        "{label} overlap must combine at least two source cells into one target cell"
    );
}

#[allow(clippy::too_many_arguments)]
pub(super) fn one_step() -> NonZeroStepCount {
    NonZeroStepCount::new(NonZeroUsize::MIN)
}

pub(super) fn transfer_plan() -> AleFsiRemeshTransferPlan2d {
    let solver = SolverPlan::new(
        LinearSolver::MinimumResidual,
        1.0e-11,
        1.0e-13,
        NonZeroUsize::new(2_000).unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Identity)
    .with_reduction(ReductionPolicy::Reproducible);
    AleFsiRemeshTransferPlan2d::new(
        QuadraturePolicy::TriangleDuffyGaussLegendre {
            points_per_axis: NonZeroUsize::new(5).unwrap(),
        },
        transfer_scales(),
        solver,
    )
    .unwrap()
}

fn transfer_scales() -> AleFsiRemeshScaleProfile2d {
    AleFsiRemeshScaleProfile2d::new(
        DynQuantity::new(2.0, LENGTH),
        DynQuantity::new(1.0, VELOCITY),
        DynQuantity::new(1.0, PRESSURE),
    )
    .unwrap()
}

pub(super) fn scale_invariance_transfer_plan() -> AleFsiRemeshTransferPlan2d {
    let base = transfer_plan();
    AleFsiRemeshTransferPlan2d::new(
        base.quadrature(),
        AleFsiRemeshScaleProfile2d::new(
            DynQuantity::new(4.0, LENGTH),
            DynQuantity::new(2.0, VELOCITY),
            DynQuantity::new(4.0, PRESSURE),
        )
        .unwrap(),
        base.solver(),
    )
    .unwrap()
}

fn harmonic_solver_plan() -> SolverPlan {
    SolverPlan::new(
        LinearSolver::ConjugateGradient,
        1.0e-12,
        1.0e-14,
        NonZeroUsize::new(500).unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Identity)
    .with_reduction(ReductionPolicy::Fast)
}

fn nonlinear_solver_plan() -> SolverPlan {
    SolverPlan::new(
        LinearSolver::BiConjugateGradientStabilized,
        1.0e-9,
        1.0e-11,
        NonZeroUsize::new(2_000).unwrap(),
    )
    .unwrap()
    .with_preconditioner(PreconditionerPolicy::Identity)
    .with_reduction(ReductionPolicy::Fast)
}

fn realization_plan(
    model: &AleFsiCartesianModel<2>,
    mesh_artifact: MeshArtifactReference,
    time_step: f64,
) -> FixedTopologyAleCoupledRealizationPlan {
    realization_plan_with_scales(model, mesh_artifact, time_step, 2.0, 1.0, 1.0)
}

#[allow(clippy::too_many_arguments)]
fn realization_plan_with_scales(
    model: &AleFsiCartesianModel<2>,
    mesh_artifact: MeshArtifactReference,
    time_step: f64,
    length_value: f64,
    velocity_value: f64,
    pressure_value: f64,
) -> FixedTopologyAleCoupledRealizationPlan {
    let p1 = Space::continuous_lagrange(NonZeroU16::MIN);
    let length = physical_scale(length_value, LENGTH);
    let velocity = physical_scale(velocity_value, VELOCITY);
    let pressure = physical_scale(pressure_value, PRESSURE);
    let duration = DynQuantity::new(time_step, TIME);
    let coupled = CoupledFieldwiseRealizationPlan::new(
        CoupledFieldwiseSpatialDiscretization::new(
            length,
            [
                DomainFieldDiscretization::new(
                    fluid_domain(model),
                    [
                        FieldSpaceBinding::new(fluid_velocity(model), Space::simplex_p1_bubble()),
                        FieldSpaceBinding::new(fluid_pressure(model), p1),
                    ],
                    [],
                )
                .unwrap(),
                DomainFieldDiscretization::new(
                    solid_domain(model),
                    [FieldSpaceBinding::new(solid_velocity(model), p1)],
                    [],
                )
                .unwrap(),
            ],
            [trace_quotient(model)],
            Discretization::new(
                DiscretizationMethod::ContinuousGalerkin,
                MeshPolicy::ImportedSimplicial {
                    artifact: mesh_artifact,
                },
                QuadraturePolicy::TriangleDuffyGaussLegendre {
                    points_per_axis: NonZeroUsize::new(5).unwrap(),
                },
            ),
        )
        .unwrap(),
        BackwardEulerStep::new(
            duration,
            [BackwardEulerStateBinding::new(
                state_pair(model),
                p1,
                length,
            )],
        )
        .unwrap(),
        SymmetricCongruenceScaling::new(
            [
                AlgebraicBlockScale::new(AlgebraicBlock::Field(fluid_velocity(model)), velocity),
                AlgebraicBlockScale::new(AlgebraicBlock::Field(fluid_pressure(model)), pressure),
                AlgebraicBlockScale::new(AlgebraicBlock::Field(solid_velocity(model)), velocity),
            ],
            physical_scale(2.0, WEAK_FUNCTIONAL),
        )
        .unwrap(),
        LinearOperatorProperties::General,
        nonlinear_solver_plan(),
        Target::HostCpu {
            threads: NonZeroUsize::MIN,
        },
        ExecutionSchedule::Offline,
    )
    .unwrap();
    FixedTopologyAleCoupledRealizationPlan::new(
        coupled,
        BackwardEulerRelationStep::new(fluid_relation(model), fluid_velocity(model), duration)
            .unwrap(),
        solid_kinematic_relation(model),
        P1HarmonicMeshMotionPolicy::new(
            fluid_domain(model),
            solid_domain(model),
            solid_displacement(model),
            connection(model),
            AleGeometryQualityGate::new(0.3).unwrap(),
            harmonic_solver_plan(),
        )
        .unwrap(),
        GclCompatibleAlePullback::new(fluid_relation(model), fluid_velocity(model)),
        NonlinearSolvePlan::new(1.0e-7, 1.0e-10, NonZeroUsize::new(20).unwrap(), 16).unwrap(),
    )
    .unwrap()
}

fn capabilities() -> RealizationCapabilities {
    RealizationCapabilities::cartesian_product(
        [DiscretizationMethod::ContinuousGalerkin],
        [(
            MeshKind::ImportedAffineSimplicial,
            SpatialDimensionSupport::exact(NonZeroUsize::new(2).unwrap()),
        )],
        [VectorLayoutKind::Replicated],
        SolverCapabilities::exact([
            SolverCapability {
                algorithm: LinearSolver::BiConjugateGradientStabilized,
                operator_properties: LinearOperatorProperties::General,
                preconditioner: PreconditionerPolicy::Identity,
                reduction: ReductionPolicy::Fast,
                scalar_type: ScalarType::F64,
            },
            SolverCapability {
                algorithm: LinearSolver::ConjugateGradient,
                operator_properties: LinearOperatorProperties::SymmetricPositiveDefinite,
                preconditioner: PreconditionerPolicy::Identity,
                reduction: ReductionPolicy::Fast,
                scalar_type: ScalarType::F64,
            },
        ])
        .unwrap(),
        TargetCapabilities::none().with_host_cpu(NonZeroUsize::MIN),
    )
    .unwrap()
}

fn physical_scale(value: f64, dimension: DimExponents) -> PositivePhysicalScale {
    PositivePhysicalScale::new(DynQuantity::new(value, dimension)).unwrap()
}

pub(super) fn fluid_domain(model: &AleFsiCartesianModel<2>) -> Id<kinds::Domain> {
    model.fluid().domain().downcast().unwrap()
}

pub(super) fn solid_domain(model: &AleFsiCartesianModel<2>) -> Id<kinds::Domain> {
    model.solid().continuum().domain().downcast().unwrap()
}

pub(super) fn fluid_velocity(model: &AleFsiCartesianModel<2>) -> Id<kinds::Field> {
    model.fluid().velocity().downcast().unwrap()
}

pub(super) fn fluid_pressure(model: &AleFsiCartesianModel<2>) -> Id<kinds::Field> {
    model.fluid().pressure().downcast().unwrap()
}

pub(super) fn solid_velocity(model: &AleFsiCartesianModel<2>) -> Id<kinds::Field> {
    model.solid().velocity().downcast().unwrap()
}

pub(super) fn solid_displacement(model: &AleFsiCartesianModel<2>) -> Id<kinds::Field> {
    model.solid().continuum().displacement().downcast().unwrap()
}

fn fluid_relation(model: &AleFsiCartesianModel<2>) -> Id<kinds::Relation> {
    model.fluid().momentum_relation().downcast().unwrap()
}

fn solid_kinematic_relation(model: &AleFsiCartesianModel<2>) -> Id<kinds::Relation> {
    fixed_topology_ale_fsi_requirements_2d(model).solid_kinematic_relation()
}

fn connection(model: &AleFsiCartesianModel<2>) -> Id<kinds::Connection> {
    model.interface().connection().downcast().unwrap()
}

fn trace_quotient(model: &AleFsiCartesianModel<2>) -> ConformingTraceQuotient {
    ConformingTraceQuotient::new(
        connection(model),
        TraceFieldEndpoint::new(fluid_domain(model), fluid_velocity(model)),
        TraceFieldEndpoint::new(solid_domain(model), solid_velocity(model)),
    )
    .unwrap()
}

fn state_pair(model: &AleFsiCartesianModel<2>) -> BackwardEulerStatePair {
    BackwardEulerStatePair::new(
        solid_kinematic_relation(model),
        solid_displacement(model),
        solid_velocity(model),
    )
    .unwrap()
}

fn two_domain_mesh(flip_diagonal: bool) -> SimplicialMesh {
    if flip_diagonal {
        return unstructured_target_mesh();
    }
    let x_coordinates = vec![0.0, 0.5, 1.0, 1.5, 2.0];
    let y_coordinates = vec![0.0, 0.5, 1.0];
    let mut vertices = Vec::new();
    for y in y_coordinates {
        for &x in &x_coordinates {
            vertices.push(vec![x, y]);
        }
    }
    let width = x_coordinates.len();
    let mut cells = Vec::new();
    let rows = vertices.len() / width;
    for row in 0..rows - 1 {
        for column in 0..width - 1 {
            let lower_left = row * width + column;
            let lower_right = lower_left + 1;
            let upper_left = lower_left + width;
            let upper_right = upper_left + 1;
            cells.push(vec![lower_left, lower_right, upper_right]);
            cells.push(vec![lower_left, upper_right, upper_left]);
        }
    }
    SimplicialMesh::new(2, vertices, cells, MeshQualityGate::new(0.3).unwrap()).unwrap()
}

fn unstructured_target_mesh() -> SimplicialMesh {
    let vertices = vec![
        vec![1.0, 0.0],
        vec![1.0, 0.25],
        vec![1.0, 0.5],
        vec![1.0, 0.75],
        vec![1.0, 1.0],
        vec![0.0, 0.0],
        vec![0.0, 0.25],
        vec![0.0, 0.5],
        vec![0.0, 0.75],
        vec![0.0, 1.0],
        vec![2.0, 0.0],
        vec![2.0, 0.25],
        vec![2.0, 0.5],
        vec![2.0, 0.75],
        vec![2.0, 1.0],
        vec![0.5, 0.5],
        vec![1.5, 0.5],
    ];
    let fluid_boundary = [5, 0, 1, 2, 3, 4, 9, 8, 7, 6];
    let solid_boundary = [0, 10, 11, 12, 13, 14, 4, 3, 2, 1];
    let mut cells = Vec::with_capacity(fluid_boundary.len() + solid_boundary.len());
    for index in 0..fluid_boundary.len() {
        cells.push(vec![
            15,
            fluid_boundary[index],
            fluid_boundary[(index + 1) % fluid_boundary.len()],
        ]);
    }
    for index in 0..solid_boundary.len() {
        cells.push(vec![
            16,
            solid_boundary[index],
            solid_boundary[(index + 1) % solid_boundary.len()],
        ]);
    }
    SimplicialMesh::new(2, vertices, cells, MeshQualityGate::new(0.3).unwrap()).unwrap()
}

fn partition(mesh: &SimplicialMesh) -> FixedReferenceFsiPartition<2> {
    let mut fluid = Vec::new();
    let mut solid = Vec::new();
    for (index, cell) in mesh.cells().iter().enumerate() {
        let centroid_x = cell
            .iter()
            .map(|vertex| mesh.vertices()[*vertex][0])
            .sum::<f64>()
            / 3.0;
        if centroid_x < 1.0 {
            fluid.push(CellId::new(index));
        } else {
            solid.push(CellId::new(index));
        }
    }
    let interface = (0..mesh.entity_count(1).unwrap())
        .filter(|&facet| {
            mesh.entity_vertices(MeshEntity::new(1, facet))
                .unwrap()
                .iter()
                .all(|vertex| mesh.vertices()[vertex.index()][0] == 1.0)
        })
        .map(FacetId::new)
        .collect();
    FixedReferenceFsiPartition::<2>::new(mesh, fluid, solid, interface).unwrap()
}

fn find_vertex(mesh: &SimplicialMesh, target: [f64; COMPONENTS]) -> usize {
    mesh.vertices()
        .iter()
        .position(|coordinates| coordinates.as_slice() == target)
        .unwrap()
}
