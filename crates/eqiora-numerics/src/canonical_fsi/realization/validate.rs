//! Exact replay gates between canonical FSI meaning and numerical realization.

use eqiora_solver::AlgebraicBlock;
use std::collections::BTreeSet;

use eqiora_core::diagnostic::codes;
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, DimExponents, DynQuantity, Id};
use eqiora_meshing::{CellId, MeshEntity, MeshTopology, SimplicialMesh};
use eqiora_realization::{
    BackwardEulerStatePair, ConformingTraceQuotient, MeshArtifactReference,
    PortableRealizationGraph, ResolvedCoupledFieldwiseRealization, SolveRoot, Target,
    TraceFieldEndpoint, TransformationNode, VectorLayoutKind,
};
use eqiora_schema::kernel::BoundarySide;
use eqiora_solver::{LinearSolver, PreconditionerPolicy, SolverPlan};

use super::super::{FixedReferenceFsiCartesianModel2d, FsiInterfaceSide};
use super::result::FixedReferenceFsiFieldIdentities2d;
use super::{
    DIMENSION, FixedReferenceFsiExecutionProfile, FixedReferenceFsiScaleProfile2d,
    fixed_reference_fsi_plan_2d_for_profile, fixed_reference_fsi_requirements_2d_for_layout,
};
use crate::canonical_boundary::PhysicalBoundaryDisposition;
use crate::simplicial_fsi::FixedReferenceFsiPartition;

pub(super) fn exact_graph_inventory(
    plan: &eqiora_realization::CoupledFieldwiseRealizationPlan,
    graph: &PortableRealizationGraph,
) -> bool {
    let expected_domains = plan
        .spatial()
        .domains()
        .iter()
        .map(|domain| domain.domain().erase())
        .collect::<BTreeSet<_>>();
    let actual_domains = graph
        .domains()
        .iter()
        .map(|domain| domain.domain().erase())
        .collect::<BTreeSet<_>>();
    if expected_domains != actual_domains || actual_domains.len() != graph.domains().len() {
        return false;
    }
    let mut expected = plan
        .spatial()
        .domains()
        .iter()
        .flat_map(|domain| {
            domain.field_spaces().iter().map(move |field| {
                (
                    field.field().erase(),
                    (domain.domain().erase(), field.space()),
                )
            })
        })
        .collect::<std::collections::BTreeMap<_, _>>();
    let state = plan.time_step().eliminated_state();
    let Some(&(domain, _)) = expected.get(&state.pair().rate().erase()) else {
        return false;
    };
    expected.insert(state.pair().state().erase(), (domain, state.state_space()));
    let actual = graph
        .fields()
        .iter()
        .map(|field| {
            graph.domain(field.domain()).map(|domain| {
                (
                    field.field().erase(),
                    (domain.domain().erase(), field.space()),
                )
            })
        })
        .collect::<Option<std::collections::BTreeMap<_, _>>>();
    if !actual.is_some_and(|actual| actual.len() == graph.fields().len() && actual == expected) {
        return false;
    }
    let expected = plan
        .spatial()
        .trace_quotients()
        .iter()
        .map(|quotient| {
            (
                quotient.connection().erase(),
                quotient
                    .endpoints()
                    .map(|endpoint| endpoint.field().erase()),
            )
        })
        .collect::<BTreeSet<_>>();
    let actual = graph
        .transformations()
        .iter()
        .filter_map(|transformation| match transformation {
            TransformationNode::ConformingTraceQuotient {
                connection,
                endpoints,
            } => Some((connection, endpoints)),
            _ => None,
        })
        .map(|(connection, endpoints)| {
            let [Some(first), Some(second)] = endpoints.map(|endpoint| graph.field(endpoint))
            else {
                return None;
            };
            Some((
                connection.erase(),
                [first.field().erase(), second.field().erase()],
            ))
        })
        .collect::<Option<Vec<_>>>();
    actual.is_some_and(|actual| {
        let unique = actual.iter().copied().collect::<BTreeSet<_>>();
        unique.len() == actual.len() && unique == expected
    })
}

pub(super) fn require_exact_plan(
    model: &FixedReferenceFsiCartesianModel2d,
    resolved: &ResolvedCoupledFieldwiseRealization,
    graph: &PortableRealizationGraph,
    mesh_artifact: MeshArtifactReference,
) -> Result<FixedReferenceFsiScaleProfile2d, Diagnostic> {
    if resolved.model() != model.model()
        || resolved.semantic_revision().get() != model.semantic_revision()
    {
        return Err(invalid_realization(
            "resolved coupled realization does not reference the exact lowered FSI Semantic Model revision",
        ));
    }
    let vector_layout = resolved.requirements().execution().vector_layout();
    if resolved.requirements()
        != &fixed_reference_fsi_requirements_2d_for_layout(model, vector_layout)
    {
        return Err(invalid_realization(
            "resolved coupled requirements differ from the exact fixed-reference FSI Domain, Field, Connection, state, or execution inventory",
        ));
    }
    if graph.lineage().model() != resolved.model()
        || graph.lineage().semantic_revision() != resolved.semantic_revision()
        || !exact_graph_inventory(resolved.plan(), graph)
        || graph.systems().len() != 1
    {
        return Err(invalid_realization(
            "fixed-reference FSI portable graph lineage or exact Domain/Field inventory drifted",
        ));
    }
    let SolveRoot::Linear(root) = graph.root() else {
        return Err(invalid_realization(
            "fixed-reference FSI portable graph requires one linear solve root",
        ));
    };
    let linear = graph
        .linear_solve(root)
        .ok_or_else(|| invalid_realization("fixed-reference FSI graph linear root is absent"))?;
    let execution = require_execution_profile(resolved)?;
    if graph.placement(linear.placement()) != Some(execution.placement())
        || linear.plan() != resolved.plan().solver()
    {
        return Err(invalid_realization(
            "fixed-reference FSI portable graph solver or exact admitted placement drifted",
        ));
    }
    let state = graph
        .fields()
        .iter()
        .position(|field| field.field() == solid_displacement(model))
        .ok_or_else(|| invalid_realization("fixed-reference FSI graph omits displacement"))?;
    let rate = graph
        .fields()
        .iter()
        .position(|field| field.field() == solid_velocity(model))
        .ok_or_else(|| invalid_realization("fixed-reference FSI graph omits solid velocity"))?;
    let transformation_matches = graph.transformations().iter().any(|transformation| {
        matches!(
            transformation,
            TransformationNode::BackwardEulerElimination {
                relation,
                state: selected_state,
                rate: selected_rate,
                duration,
                ..
            } if *relation == solid_kinematic_relation(model)
                && selected_state.index() == state
                && selected_rate.index() == rate
                && *duration == resolved.plan().time_step().duration()
        )
    });
    if !transformation_matches {
        return Err(invalid_realization(
            "fixed-reference FSI portable transformations differ from the exact kinematic Relation or interface Connection",
        ));
    }
    let plan = resolved.plan();
    let scale_for = |block| {
        graph.systems()[0]
            .congruence_scaling()
            .ok_or_else(|| {
                invalid_realization("fixed-reference FSI graph requires congruence scaling")
            })?
            .block_scales()
            .iter()
            .find(|entry| entry.block() == block)
            .map(|entry| entry.scale().quantity())
            .ok_or_else(|| {
                invalid_realization("fixed-reference FSI plan omits an exact block scale")
            })
    };
    let scales = FixedReferenceFsiScaleProfile2d::new(
        plan.spatial().coordinate_length_scale().quantity(),
        scale_for(AlgebraicBlock::Field(fluid_velocity(model)))?,
        scale_for(AlgebraicBlock::Field(fluid_pressure(model)))?,
    )?;
    let expected = fixed_reference_fsi_plan_2d_for_profile(
        model,
        mesh_artifact,
        plan.time_step().duration(),
        scales,
        plan.solver(),
        execution,
    )?;
    if plan != &expected {
        return Err(invalid_realization(
            "resolved coupled plan differs from the exact coherent-SI fixed-reference FSI contract",
        ));
    }
    Ok(scales)
}

pub(super) fn require_zero_load(
    model: &FixedReferenceFsiCartesianModel2d,
) -> Result<(), Diagnostic> {
    if model.fluid().force_potential_expression().constant_value() != Some(0.0)
        || model
            .solid()
            .continuum()
            .load_potential_expression()
            .constant_value()
            != Some(0.0)
    {
        return Err(invalid_realization(
            "fixed-reference FSI v1 requires exact zero canonical fluid and solid load potentials",
        ));
    }
    Ok(())
}

pub(super) fn require_boundary_meaning(
    model: &FixedReferenceFsiCartesianModel2d,
) -> Result<(), Diagnostic> {
    let interface = model.interface();
    require_physics_boundary(
        model.fluid().boundary_inventory(),
        interface.axis(),
        interface.fluid(),
        interface.connection(),
        "fluid",
    )?;
    require_physics_boundary(
        model.solid().continuum().boundary_inventory(),
        interface.axis(),
        interface.solid(),
        interface.connection(),
        "solid",
    )
}

fn require_physics_boundary(
    inventory: &crate::canonical_boundary::CartesianBoundaryInventory<2>,
    interface_axis: usize,
    interface_side: FsiInterfaceSide,
    connection: eqiora_core::RawId,
    physics: &str,
) -> Result<(), Diagnostic> {
    for axis in 0..DIMENSION {
        for side in [BoundarySide::Lower, BoundarySide::Upper] {
            let entry = inventory.boundary(axis, side).ok_or_else(|| {
                invalid_realization(format!(
                    "fixed-reference FSI {physics} boundary inventory omits axis {axis} {side:?}"
                ))
            })?;
            if axis == interface_axis && side == interface_side.side() {
                if entry.boundary() != interface_side.boundary()
                    || entry.disposition()
                        != (PhysicalBoundaryDisposition::PortBinding {
                            connection,
                            port: interface_side.port(),
                        })
                {
                    return Err(invalid_realization(format!(
                        "fixed-reference FSI {physics} interface boundary identity or live Port binding drifted"
                    )));
                }
            } else if entry.disposition() != PhysicalBoundaryDisposition::TraceZero {
                return Err(invalid_realization(format!(
                    "fixed-reference FSI v1 requires TraceZero on every exterior {physics} side"
                )));
            }
        }
    }
    Ok(())
}

pub(super) fn require_mesh_partition(
    model: &FixedReferenceFsiCartesianModel2d,
    mesh: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<2>,
) -> Result<(), Diagnostic> {
    if mesh.topological_dimension() != DIMENSION
        || mesh.vertices().iter().any(|point| point.len() != DIMENSION)
    {
        return Err(invalid_realization(
            "fixed-reference FSI canonical bridge requires an intrinsic 2D mesh",
        ));
    }
    require_cells_in_bounds(
        mesh,
        partition.fluid_cells(),
        model.fluid().bounds(),
        "fluid",
    )?;
    require_cells_in_bounds(
        mesh,
        partition.solid_cells(),
        model.solid().continuum().bounds(),
        "solid",
    )?;

    let interface = model.interface();
    let interface_coordinate = match interface.fluid().side() {
        BoundarySide::Lower => model.fluid().bounds()[interface.axis()][0],
        BoundarySide::Upper => model.fluid().bounds()[interface.axis()][1],
    };
    for facet in partition.interface_facets() {
        let vertices = mesh
            .entity_vertices(MeshEntity::new(DIMENSION - 1, facet.index()))
            .ok_or_else(|| {
                invalid_realization(
                    "fixed-reference FSI interface facet is outside the mesh revision",
                )
            })?;
        if vertices
            .iter()
            .any(|vertex| mesh.vertices()[vertex.index()][interface.axis()] != interface_coordinate)
        {
            return Err(invalid_realization(
                "fixed-reference FSI partition interface does not lie on the exact semantic interface",
            ));
        }
    }

    let fluid_cells = partition
        .fluid_cells()
        .iter()
        .map(|cell| cell.index())
        .collect::<BTreeSet<_>>();
    let mut fluid_coverage = [[false; 2]; DIMENSION];
    let mut solid_coverage = [[false; 2]; DIMENSION];
    let facet_count = mesh
        .entity_count(DIMENSION - 1)
        .ok_or_else(|| invalid_realization("fixed-reference FSI mesh omits its facet stratum"))?;
    for facet_index in 0..facet_count {
        let facet = MeshEntity::new(DIMENSION - 1, facet_index);
        if !mesh.is_boundary_entity(facet).ok_or_else(|| {
            invalid_realization("fixed-reference FSI facet is outside the mesh revision")
        })? {
            continue;
        }
        let adjacent = mesh.incidence(facet, DIMENSION).ok_or_else(|| {
            invalid_realization("fixed-reference FSI facet has no cell-incidence relation")
        })?;
        let [cell] = adjacent.as_slice() else {
            return Err(invalid_realization(
                "fixed-reference FSI exterior facet must have exactly one adjacent cell",
            ));
        };
        let (bounds, coverage, interface_side) = if fluid_cells.contains(&cell.entity.index()) {
            (
                model.fluid().bounds(),
                &mut fluid_coverage,
                interface.fluid().side(),
            )
        } else {
            (
                model.solid().continuum().bounds(),
                &mut solid_coverage,
                interface.solid().side(),
            )
        };
        let vertices = mesh.entity_vertices(facet).ok_or_else(|| {
            invalid_realization("fixed-reference FSI exterior facet has no vertex closure")
        })?;
        let mut matched = None;
        for (axis, axis_bounds) in bounds.iter().enumerate() {
            for (side_index, bound) in axis_bounds.iter().enumerate() {
                if vertices
                    .iter()
                    .all(|vertex| mesh.vertices()[vertex.index()][axis] == *bound)
                {
                    if matched.is_some() {
                        return Err(invalid_realization(
                            "fixed-reference FSI exterior facet ambiguously belongs to multiple semantic sides",
                        ));
                    }
                    matched = Some((axis, side_index));
                }
            }
        }
        let Some((axis, side_index)) = matched else {
            return Err(invalid_realization(
                "fixed-reference FSI mesh exterior does not lie on an exact semantic side",
            ));
        };
        let side = if side_index == 0 {
            BoundarySide::Lower
        } else {
            BoundarySide::Upper
        };
        if axis == interface.axis() && side == interface_side {
            return Err(invalid_realization(
                "fixed-reference FSI semantic interface appeared on the mesh exterior",
            ));
        }
        coverage[axis][side_index] = true;
    }
    require_exterior_coverage(
        fluid_coverage,
        interface.axis(),
        interface.fluid().side(),
        "fluid",
    )?;
    require_exterior_coverage(
        solid_coverage,
        interface.axis(),
        interface.solid().side(),
        "solid",
    )?;
    Ok(())
}

fn require_cells_in_bounds(
    mesh: &SimplicialMesh,
    cells: &[CellId],
    bounds: &[[f64; 2]; DIMENSION],
    physics: &str,
) -> Result<(), Diagnostic> {
    for cell in cells {
        let vertices = mesh
            .entity_vertices(MeshEntity::new(DIMENSION, cell.index()))
            .ok_or_else(|| {
                invalid_realization(format!(
                    "fixed-reference FSI {physics} cell is outside the mesh revision"
                ))
            })?;
        if vertices.iter().any(|vertex| {
            mesh.vertices()[vertex.index()]
                .iter()
                .enumerate()
                .any(|(axis, value)| *value < bounds[axis][0] || *value > bounds[axis][1])
        }) {
            return Err(invalid_realization(format!(
                "fixed-reference FSI {physics} cell lies outside its exact semantic Domain"
            )));
        }
    }
    Ok(())
}

fn require_exterior_coverage(
    coverage: [[bool; 2]; DIMENSION],
    interface_axis: usize,
    interface_side: BoundarySide,
    physics: &str,
) -> Result<(), Diagnostic> {
    for (axis, sides) in coverage.iter().enumerate() {
        for (side_index, covered) in sides.iter().enumerate() {
            let side = if side_index == 0 {
                BoundarySide::Lower
            } else {
                BoundarySide::Upper
            };
            if !(*covered || axis == interface_axis && side == interface_side) {
                return Err(invalid_realization(format!(
                    "fixed-reference FSI mesh does not cover the exact exterior {physics} side on axis {axis} {side:?}"
                )));
            }
        }
    }
    Ok(())
}

pub(super) fn require_solver(
    solver: SolverPlan,
    execution: FixedReferenceFsiExecutionProfile,
) -> Result<(), Diagnostic> {
    if solver.algorithm() != LinearSolver::MinimumResidual
        || solver.preconditioner() != PreconditionerPolicy::Identity
        || solver.reduction() != execution.reduction()
    {
        return Err(invalid_realization(match execution {
            FixedReferenceFsiExecutionProfile::HostReproducible => {
                "fixed-reference FSI host execution requires reproducible identity-preconditioned MINRES"
            }
            FixedReferenceFsiExecutionProfile::CudaFast { .. } => {
                "fixed-reference FSI CUDA execution requires fast identity-preconditioned MINRES"
            }
            FixedReferenceFsiExecutionProfile::DistributedCudaReproducible { .. } => {
                "fixed-reference FSI distributed CUDA execution requires reproducible identity-preconditioned MINRES"
            }
        }));
    }
    Ok(())
}

fn require_execution_profile(
    resolved: &ResolvedCoupledFieldwiseRealization,
) -> Result<FixedReferenceFsiExecutionProfile, Diagnostic> {
    let layout = resolved.requirements().execution().vector_layout();
    match (layout, resolved.plan().target()) {
        (
            VectorLayoutKind::Replicated | VectorLayoutKind::Distributed,
            Target::HostCpu { threads },
        ) if threads == std::num::NonZeroUsize::MIN => {
            Ok(FixedReferenceFsiExecutionProfile::HostReproducible)
        }
        (VectorLayoutKind::Replicated, Target::CudaGpu { device }) => {
            Ok(FixedReferenceFsiExecutionProfile::CudaFast { device })
        }
        (VectorLayoutKind::Distributed, Target::CudaGpu { device }) if device == 0 => {
            Ok(FixedReferenceFsiExecutionProfile::DistributedCudaReproducible { device })
        }
        (VectorLayoutKind::Distributed, Target::CudaGpu { .. }) => Err(invalid_realization(
            "fixed-reference FSI distributed CUDA execution requires deployment-local device ordinal zero",
        )),
        (VectorLayoutKind::Replicated | VectorLayoutKind::Distributed, _) => {
            Err(invalid_realization(
                "fixed-reference FSI host execution requires exactly one worker per partition",
            ))
        }
    }
}

pub(super) fn require_dimension(
    value: DynQuantity,
    expected: DimExponents,
    label: &str,
) -> Result<(), Diagnostic> {
    if value.dim() != expected {
        return Err(invalid_realization(format!(
            "{label} has incompatible physical dimension {:?}",
            value.dim()
        )));
    }
    Ok(())
}

pub(super) fn trace_quotient(model: &FixedReferenceFsiCartesianModel2d) -> ConformingTraceQuotient {
    ConformingTraceQuotient::new(
        connection_id(model),
        TraceFieldEndpoint::new(fluid_domain(model), fluid_velocity(model)),
        TraceFieldEndpoint::new(solid_domain(model), solid_velocity(model)),
    )
    .expect("lowered FSI interface joins distinct Domains")
}

pub(super) fn state_pair(model: &FixedReferenceFsiCartesianModel2d) -> BackwardEulerStatePair {
    BackwardEulerStatePair::new(solid_displacement(model), solid_velocity(model))
        .expect("lowered solid displacement and velocity are distinct Fields")
}

pub(super) fn field_identities(
    model: &FixedReferenceFsiCartesianModel2d,
) -> FixedReferenceFsiFieldIdentities2d {
    FixedReferenceFsiFieldIdentities2d::new(
        fluid_velocity(model),
        fluid_pressure(model),
        solid_velocity(model),
        solid_displacement(model),
    )
}

pub(super) fn fluid_domain(model: &FixedReferenceFsiCartesianModel2d) -> Id<kinds::Domain> {
    model
        .fluid()
        .domain()
        .downcast()
        .expect("lowered fluid Domain retains its entity kind")
}

pub(super) fn solid_domain(model: &FixedReferenceFsiCartesianModel2d) -> Id<kinds::Domain> {
    model
        .solid()
        .continuum()
        .domain()
        .downcast()
        .expect("lowered solid Domain retains its entity kind")
}

pub(super) fn fluid_velocity(model: &FixedReferenceFsiCartesianModel2d) -> Id<kinds::Field> {
    model
        .fluid()
        .velocity()
        .downcast()
        .expect("lowered fluid velocity retains its Field kind")
}

pub(super) fn fluid_pressure(model: &FixedReferenceFsiCartesianModel2d) -> Id<kinds::Field> {
    model
        .fluid()
        .pressure()
        .downcast()
        .expect("lowered fluid pressure retains its Field kind")
}

pub(super) fn solid_displacement(model: &FixedReferenceFsiCartesianModel2d) -> Id<kinds::Field> {
    model
        .solid()
        .continuum()
        .displacement()
        .downcast()
        .expect("lowered solid displacement retains its Field kind")
}

pub(super) fn solid_velocity(model: &FixedReferenceFsiCartesianModel2d) -> Id<kinds::Field> {
    model
        .solid()
        .velocity()
        .downcast()
        .expect("lowered solid velocity retains its Field kind")
}

pub(super) fn solid_kinematic_relation(
    model: &FixedReferenceFsiCartesianModel2d,
) -> Id<kinds::Relation> {
    model
        .solid()
        .kinematic_relation()
        .downcast()
        .expect("lowered solid kinematic relation retains its Relation kind")
}

fn connection_id(model: &FixedReferenceFsiCartesianModel2d) -> Id<kinds::Connection> {
    model
        .interface()
        .connection()
        .downcast()
        .expect("lowered FSI Connection retains its entity kind")
}

pub(super) fn realization_error(error: Diagnostic) -> Diagnostic {
    invalid_realization(error.message())
}

pub(super) fn invalid_realization(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_REALIZATION, message)
}
