//! Exact replay gates between canonical FSI meaning and numerical realization.

use eqiora_solver::AlgebraicBlock;
use std::collections::BTreeSet;

use eqiora_core::diagnostic::codes;
use eqiora_core::{Diagnostic, DimExponents, DynQuantity};
use eqiora_meshing::{CellId, MeshEntity, MeshTopology, SimplicialMesh};
use eqiora_realization::{
    BackwardEulerStatePair, ConformingTraceQuotient, MeshArtifactReference,
    PortableRealizationGraph, ResolvedCoupledFieldwiseRealization, SolveRoot, Target,
    TraceFieldEndpoint, TransformationNode, VectorLayoutKind,
};
use eqiora_schema::kernel::BoundarySide;
use eqiora_solver::{LinearSolver, PreconditionerPolicy, SolverPlan};

use super::super::FixedReferenceFsiCartesianModel2d;
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
    for state in plan.time_step().eliminated_states() {
        let Some(&(domain, _)) = expected.get(&state.pair().rate().erase()) else {
            return false;
        };
        expected.insert(state.pair().state().erase(), (domain, state.state_space()));
    }
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
    let pairs = state_pairs(model);
    for pair in &pairs {
        let state = graph
            .fields()
            .iter()
            .position(|field| field.field() == pair.state())
            .ok_or_else(|| invalid_realization("graph omits exact eliminated state"))?;
        let rate = graph
            .fields()
            .iter()
            .position(|field| field.field() == pair.rate())
            .ok_or_else(|| invalid_realization("graph omits exact algebraic rate"))?;
        if !graph.transformations().iter().any(|transformation| matches!(transformation,
            TransformationNode::BackwardEulerElimination { relation, state: selected_state,
                rate: selected_rate, duration, .. }
                if *relation == pair.relation() && selected_state.index() == state
                    && selected_rate.index() == rate && *duration == resolved.plan().time_step().duration())) {
            return Err(invalid_realization("portable state transformation differs from exact Relation/Field/duration"));
        }
    }
    if graph
        .transformations()
        .iter()
        .filter(|transformation| {
            matches!(
                transformation,
                TransformationNode::BackwardEulerElimination { .. }
            )
        })
        .count()
        != pairs.len()
    {
        return Err(invalid_realization(
            "portable graph has an extra state transformation",
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
    let uniform_scale = |fields: Vec<eqiora_core::RawId>| -> Result<DynQuantity, Diagnostic> {
        let values = fields
            .into_iter()
            .map(|field| scale_for(AlgebraicBlock::Field(field.downcast().expect("Field"))))
            .collect::<Result<Vec<_>, _>>()?;
        let [first, rest @ ..] = values.as_slice() else {
            return Err(invalid_realization("scale role inventory is empty"));
        };
        if rest.iter().any(|value| value != first) {
            return Err(invalid_realization(
                "coupled velocity/pressure scales must agree within each exact role inventory",
            ));
        }
        Ok(*first)
    };
    let scales = FixedReferenceFsiScaleProfile2d::new(
        plan.spatial().coordinate_length_scale().quantity(),
        uniform_scale(
            model
                .fluids()
                .map(|fluid| fluid.velocity())
                .chain(model.solids().map(|solid| solid.velocity()))
                .collect(),
        )?,
        uniform_scale(model.fluids().map(|fluid| fluid.pressure()).collect())?,
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
    if model
        .fluids()
        .any(|fluid| fluid.force_potential_expression().constant_value() != Some(0.0))
        || model.solids().any(|solid| {
            solid
                .continuum()
                .load_potential_expression()
                .constant_value()
                != Some(0.0)
        })
    {
        return Err(invalid_realization(
            "current transient energy acceptance requires exact zero load potentials in every Region",
        ));
    }
    Ok(())
}

pub(super) fn require_boundary_meaning(
    model: &FixedReferenceFsiCartesianModel2d,
) -> Result<(), Diagnostic> {
    for (domain, inventory) in model
        .fluids()
        .map(|fluid| (fluid.domain(), fluid.boundary_inventory()))
        .chain(model.solids().map(|solid| {
            (
                solid.continuum().domain(),
                solid.continuum().boundary_inventory(),
            )
        }))
    {
        for axis in 0..DIMENSION {
            for side in [BoundarySide::Lower, BoundarySide::Upper] {
                let entry = inventory.boundary(axis, side).ok_or_else(|| {
                    invalid_realization("Region omits an exact Cartesian boundary")
                })?;
                match entry.disposition() {
                    PhysicalBoundaryDisposition::TraceZero => {}
                    PhysicalBoundaryDisposition::PortBinding { connection, port } => {
                        let interface = model.interfaces.get(&connection).ok_or_else(|| {
                            invalid_realization("Boundary has a foreign Connection")
                        })?;
                        let endpoint = interface.endpoint(domain).ok_or_else(|| {
                            invalid_realization("Connection has a foreign parent Domain")
                        })?;
                        if interface.axis() != axis
                            || endpoint.side() != side
                            || endpoint.boundary() != entry.boundary()
                            || endpoint.port() != port
                        {
                            return Err(invalid_realization(
                                "Boundary differs from its exact Connection endpoint",
                            ));
                        }
                    }
                    _ => {
                        return Err(invalid_realization(
                            "current transient energy acceptance requires zero velocity on every exterior side",
                        ));
                    }
                }
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
            "canonical bridge requires intrinsic two-dimensional mesh",
        ));
    }
    let bounds = model
        .fluids()
        .map(|fluid| (fluid.domain(), fluid.bounds()))
        .chain(
            model
                .solids()
                .map(|solid| (solid.continuum().domain(), solid.continuum().bounds())),
        )
        .collect::<std::collections::BTreeMap<_, _>>();
    if bounds.keys().copied().collect::<BTreeSet<_>>()
        != partition.domains().map(|domain| domain.erase()).collect()
    {
        return Err(invalid_realization(
            "partition differs from complete exact Model Domain inventory",
        ));
    }
    let replay = FixedReferenceFsiPartition::<2>::new(
        mesh,
        partition.domains().map(|domain| {
            (
                domain,
                partition
                    .domain_cells(domain)
                    .expect("exact Domain")
                    .to_vec(),
            )
        }),
        &trace_quotients(model),
    )?;
    if &replay != partition {
        return Err(invalid_realization(
            "partition quotient or topology differs from exact Model replay",
        ));
    }
    for (&domain, &bounds) in &bounds {
        require_cells_in_bounds(
            mesh,
            partition
                .domain_cells(domain.downcast().expect("Domain"))
                .expect("exact Domain"),
            bounds,
            "Region",
        )?;
        let mut coverage = [[false; 2]; DIMENSION];
        for index in 0..mesh.entity_count(DIMENSION - 1).expect("facet stratum") {
            let facet = MeshEntity::new(DIMENSION - 1, index);
            let adjacent = mesh.incidence(facet, DIMENSION).expect("exact facet");
            let own = adjacent
                .iter()
                .filter(|side| partition.cell_domains()[side.entity.index()] == domain)
                .count();
            if own != 1 {
                continue;
            }
            let vertices = mesh.entity_vertices(facet).expect("exact facet closure");
            let mut matched = None;
            for (axis, axis_bounds) in bounds.iter().enumerate() {
                for (side, &bound) in axis_bounds.iter().enumerate() {
                    if vertices
                        .iter()
                        .all(|vertex| mesh.vertices()[vertex.index()][axis] == bound)
                        && matched.replace((axis, side)).is_some()
                    {
                        return Err(invalid_realization(
                            "Region facet has ambiguous Cartesian support",
                        ));
                    }
                }
            }
            let (axis, side_index) = matched.ok_or_else(|| {
                invalid_realization("Region frontier is outside its exact Cartesian sides")
            })?;
            let side = if side_index == 0 {
                BoundarySide::Lower
            } else {
                BoundarySide::Upper
            };
            let expected_interface = model.interfaces().find(|interface| {
                interface.axis() == axis
                    && interface
                        .endpoint(domain)
                        .is_some_and(|endpoint| endpoint.side() == side)
            });
            if (adjacent.len() == 2) != expected_interface.is_some() {
                return Err(invalid_realization(
                    "semantic Connection and exact exterior facet ownership differ",
                ));
            }
            coverage[axis][side_index] = true;
        }
        if coverage.into_iter().flatten().any(|covered| !covered) {
            return Err(invalid_realization("mesh omits an exact Region side"));
        }
    }
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

pub(super) fn trace_quotients(
    model: &FixedReferenceFsiCartesianModel2d,
) -> Vec<ConformingTraceQuotient> {
    model
        .interfaces()
        .map(|interface| {
            let endpoints = interface.endpoints().map(|endpoint| {
                TraceFieldEndpoint::new(
                    endpoint.domain().downcast().expect("Domain"),
                    endpoint.field().downcast().expect("Field"),
                )
            });
            ConformingTraceQuotient::new(
                interface.connection().downcast().expect("Connection"),
                endpoints[0],
                endpoints[1],
            )
            .expect("exact distinct Domain endpoints")
        })
        .collect()
}

pub(super) fn state_pairs(
    model: &FixedReferenceFsiCartesianModel2d,
) -> Vec<BackwardEulerStatePair> {
    model
        .solids()
        .map(|solid| {
            BackwardEulerStatePair::new(
                solid.kinematic_relation().downcast().expect("Relation"),
                solid.continuum().displacement().downcast().expect("Field"),
                solid.velocity().downcast().expect("Field"),
            )
            .expect("exact distinct state/rate Fields")
        })
        .collect()
}

pub(super) fn realization_error(error: Diagnostic) -> Diagnostic {
    invalid_realization(error.message())
}

pub(super) fn invalid_realization(message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_REALIZATION, message)
}
