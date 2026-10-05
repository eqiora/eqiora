//! Exact two-subdomain elasticity interface recognition.

use std::collections::BTreeSet;

use eqiora_assembly::{AssemblyBackend, REFERENCE_ASSEMBLY_BACKEND};
use eqiora_core::{Diagnostic, RawId};
use eqiora_graph::EdgeKind;
use eqiora_meshing::QuadratureRule;
use eqiora_realization::ResolvedRealization;
use eqiora_schema::kernel::{BoundarySide, DomainKind, KernelNode};
use eqiora_sem::KernelProgram;
use eqiora_solver::LinearSolverBackend;

use super::{
    IsotropicElasticityContinuum, lower_isotropic_elasticity_subdomain_2d, lowering_error,
    model_lowering_error, require_closed_elasticity_models,
    require_resolved_cartesian_elasticity_q1_plan_2d,
};
use crate::canonical_boundary::PhysicalBoundaryDisposition;
use crate::cartesian_elasticity::ConformingCartesianLinearElasticityPair2dSolution;
use crate::cartesian_elasticity::{
    CartesianEssentialSides2d, finalize_conforming_cartesian_q1_linear_elasticity_pair_2d,
};
use crate::finalized_spatial::FinalizedConformingIsotropicElasticityCartesianPair2dProblem;
use eqiora_meshing::CartesianMesh;

type CartesianBounds2d = [[f64; 2]; 2];

/// One body-local end of an exact conforming elasticity interface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConformingElasticityInterfaceSide2d {
    boundary: RawId,
    port: RawId,
    side: BoundarySide,
}

impl ConformingElasticityInterfaceSide2d {
    /// Exact semantic Boundary supporting this side of the interface.
    #[must_use]
    pub const fn boundary(&self) -> RawId {
        self.boundary
    }

    /// Exact field-valued Port owned by this body's boundary law.
    #[must_use]
    pub const fn port(&self) -> RawId {
        self.port
    }

    /// Outward Cartesian side of the owning body.
    #[must_use]
    pub const fn side(&self) -> BoundarySide {
        self.side
    }
}

/// Exact semantic witness for one coincident two-body mechanical interface.
///
/// `negative` is the body on the lower-coordinate side and therefore exposes
/// its `Upper` boundary. `positive` exposes its `Lower` boundary. This order
/// is geometric and independent of declaration, package, or Connection-member
/// order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ConformingElasticityInterface2d {
    connection: RawId,
    axis: usize,
    negative: ConformingElasticityInterfaceSide2d,
    positive: ConformingElasticityInterfaceSide2d,
}

impl ConformingElasticityInterface2d {
    /// Exact conserving Connection carrying continuity and flux balance.
    #[must_use]
    pub const fn connection(&self) -> RawId {
        self.connection
    }

    /// Cartesian normal axis of the coincident interface.
    #[must_use]
    pub const fn axis(&self) -> usize {
        self.axis
    }

    /// Interface side belonging to the lower-coordinate body.
    #[must_use]
    pub const fn negative(&self) -> ConformingElasticityInterfaceSide2d {
        self.negative
    }

    /// Interface side belonging to the upper-coordinate body.
    #[must_use]
    pub const fn positive(&self) -> ConformingElasticityInterfaceSide2d {
        self.positive
    }
}

/// Two canonical elasticity bodies joined by one exact coincident interface.
///
/// The subdomain order follows geometry along [`Self::interface`]: index zero
/// is the lower-coordinate body and index one is the upper-coordinate body.
/// Meshes, trace nodes, quotient DOFs, assembly, and solve policy are absent.
#[derive(Debug, Clone, PartialEq)]
pub struct ConformingIsotropicElasticityCartesianPair2d {
    subdomains: [IsotropicElasticityContinuum<2>; 2],
    interface: ConformingElasticityInterface2d,
}

impl ConformingIsotropicElasticityCartesianPair2d {
    /// Geometrically ordered canonical subdomain contracts.
    #[must_use]
    pub const fn subdomains(&self) -> &[IsotropicElasticityContinuum<2>; 2] {
        &self.subdomains
    }

    /// Exact semantic interface joining the two subdomains.
    #[must_use]
    pub const fn interface(&self) -> ConformingElasticityInterface2d {
        self.interface
    }
}

#[derive(Debug, Clone, Copy)]
struct LiveSide {
    subdomain: usize,
    axis: usize,
    side: BoundarySide,
    boundary: RawId,
    connection: RawId,
    port: RawId,
}

/// Lower exactly two Cartesian isotropic-elasticity bodies joined by one
/// exact two-Port coincident mechanical Connection.
///
/// The result is still method-neutral. In particular, semantic coincidence
/// does not imply matching mesh nodes or select a conforming, mortar, Nitsche,
/// multiplier, monolithic, or partitioned Realization.
///
/// # Errors
/// Returns `EQ0703` unless the complete flat Model is exactly the admitted
/// two-body network and its sole live interface is an opposite-side pair.
pub fn lower_conforming_isotropic_elasticity_cartesian_pair_2d(
    program: &KernelProgram,
) -> Result<ConformingIsotropicElasticityCartesianPair2d, Diagnostic> {
    let boxes = cartesian_boxes_2d(program)?;
    if boxes.len() != 2 {
        return Err(model_lowering_error(
            program,
            format!(
                "conforming elasticity pair requires exactly two Cartesian boxes, found {}",
                boxes.len()
            ),
        ));
    }
    let mut lowered = [
        lower_isotropic_elasticity_subdomain_2d(program, boxes[0].0, boxes[0].1)?,
        lower_isotropic_elasticity_subdomain_2d(program, boxes[1].0, boxes[1].1)?,
    ];
    for subdomain in &lowered {
        if let Some(relation) = subdomain
            .boundary
            .uninterpreted_live_relations
            .iter()
            .next()
        {
            return Err(lowering_error(
                *relation,
                "conforming elasticity interface contains an additional live Port Relation that this Realization does not interpret",
            ));
        }
    }

    let mut live = Vec::new();
    for (subdomain, model) in lowered.iter().enumerate() {
        for axis in 0..2 {
            for side in [BoundarySide::Lower, BoundarySide::Upper] {
                let entry = model
                    .model
                    .boundary_inventory()
                    .boundary(axis, side)
                    .expect("a lowered 2D subdomain owns every Cartesian side");
                if let PhysicalBoundaryDisposition::PortBinding { connection, port } =
                    entry.disposition()
                {
                    live.push(LiveSide {
                        subdomain,
                        axis,
                        side,
                        boundary: entry.boundary(),
                        connection,
                        port,
                    });
                }
            }
        }
    }
    let [first, second] = live.as_slice() else {
        return Err(model_lowering_error(
            program,
            format!(
                "conforming elasticity pair requires exactly two body-local live sides, found {}",
                live.len()
            ),
        ));
    };
    if first.subdomain == second.subdomain
        || first.connection != second.connection
        || first.axis != second.axis
        || first.side == second.side
    {
        return Err(lowering_error(
            first.connection,
            "conforming elasticity interface must join one opposite side from each body through one Connection",
        ));
    }
    let member_ports = program
        .edges()
        .iter()
        .filter(|edge| edge.kind() == EdgeKind::Connects && edge.from() == first.connection)
        .map(|edge| edge.to())
        .collect::<BTreeSet<_>>();
    if member_ports != BTreeSet::from([first.port, second.port]) {
        return Err(lowering_error(
            first.connection,
            "conforming elasticity interface requires exactly the two recognized body Ports",
        ));
    }

    let (negative, positive) = match (first.side, second.side) {
        (BoundarySide::Upper, BoundarySide::Lower) => (*first, *second),
        (BoundarySide::Lower, BoundarySide::Upper) => (*second, *first),
        _ => unreachable!("equal sides were rejected"),
    };
    if negative.subdomain != 0 {
        lowered.swap(0, 1);
    }
    require_adjacent_bounds(
        &lowered[0].model,
        &lowered[1].model,
        negative.axis,
        first.connection,
    )?;
    require_closed_elasticity_models(program, &lowered)?;

    Ok(ConformingIsotropicElasticityCartesianPair2d {
        subdomains: [lowered[0].model.clone(), lowered[1].model.clone()],
        interface: ConformingElasticityInterface2d {
            connection: first.connection,
            axis: negative.axis,
            negative: ConformingElasticityInterfaceSide2d {
                boundary: negative.boundary,
                port: negative.port,
                side: negative.side,
            },
            positive: ConformingElasticityInterfaceSide2d {
                boundary: positive.boundary,
                port: positive.port,
                side: positive.side,
            },
        },
    })
}

/// Execute one exact resolved monolithic Q1 plan over a conforming pair.
///
/// The two semantic Domains remain distinct. The selected Realization creates
/// two meshes and an exact topological interface quotient before assembly.
///
/// # Errors
/// Preserves canonical pair recognition, Realization, assembly, and solver
/// diagnostics.
pub fn solve_resolved_conforming_isotropic_elasticity_cartesian_pair_2d(
    program: &KernelProgram,
    resolved: &ResolvedRealization,
    backend: &dyn LinearSolverBackend,
) -> Result<
    (
        ConformingIsotropicElasticityCartesianPair2d,
        ConformingCartesianLinearElasticityPair2dSolution,
    ),
    Diagnostic,
> {
    solve_resolved_conforming_isotropic_elasticity_cartesian_pair_2d_with_assembly(
        program,
        resolved,
        &REFERENCE_ASSEMBLY_BACKEND,
        backend,
    )
}

/// Execute the conforming pair through explicit assembly and solver adapters.
///
/// # Errors
/// Preserves the reference entry point diagnostics and each adapter's
/// complete-operation failure.
pub fn solve_resolved_conforming_isotropic_elasticity_cartesian_pair_2d_with_assembly(
    program: &KernelProgram,
    resolved: &ResolvedRealization,
    assembly: &dyn AssemblyBackend<f64>,
    backend: &dyn LinearSolverBackend,
) -> Result<
    (
        ConformingIsotropicElasticityCartesianPair2d,
        ConformingCartesianLinearElasticityPair2dSolution,
    ),
    Diagnostic,
> {
    let (model, finalized) =
        finalize_resolved_conforming_isotropic_elasticity_cartesian_pair_2d_with_assembly(
            program, resolved, assembly,
        )?;
    let solved = backend.solve(&finalized.linear_problem()?, finalized.solver_plan())?;
    Ok((model, finalized.finish(solved)?))
}

/// Finalize a conforming monolithic pair without selecting a solver adapter.
///
/// # Errors
/// Preserves canonical pair recognition, Realization, discretization, boundary,
/// and assembly diagnostics.
pub fn finalize_resolved_conforming_isotropic_elasticity_cartesian_pair_2d(
    program: &KernelProgram,
    resolved: &ResolvedRealization,
) -> Result<
    (
        ConformingIsotropicElasticityCartesianPair2d,
        FinalizedConformingIsotropicElasticityCartesianPair2dProblem,
    ),
    Diagnostic,
> {
    finalize_resolved_conforming_isotropic_elasticity_cartesian_pair_2d_with_assembly(
        program,
        resolved,
        &REFERENCE_ASSEMBLY_BACKEND,
    )
}

/// Finalize the two-body Q1 quotient system through one assembly adapter.
///
/// Only the recognized two-Port semantic interface may remain live. Exterior
/// zero trace/traction laws are normalized before mesh construction, and the
/// coupled system as a whole must be anchored.
///
/// # Errors
/// Returns `EQ0807` for an inadmissible plan or live boundary and preserves
/// exact lowering, meshing, and assembly diagnostics.
pub fn finalize_resolved_conforming_isotropic_elasticity_cartesian_pair_2d_with_assembly(
    program: &KernelProgram,
    resolved: &ResolvedRealization,
    assembly: &dyn AssemblyBackend<f64>,
) -> Result<
    (
        ConformingIsotropicElasticityCartesianPair2d,
        FinalizedConformingIsotropicElasticityCartesianPair2dProblem,
    ),
    Diagnostic,
> {
    let execution = require_resolved_cartesian_elasticity_q1_plan_2d(program, resolved)?;
    let model = lower_conforming_isotropic_elasticity_cartesian_pair_2d(program)?;
    let essential_sides = pair_essential_sides(&model)?;
    let meshes = [
        CartesianMesh::uniform(model.subdomains[0].bounds(), &[execution.cells; 2])?,
        CartesianMesh::uniform(model.subdomains[1].bounds(), &[execution.cells; 2])?,
    ];
    let block_system =
        super::block::conforming_elasticity_pair_block_system(&model, resolved, &meshes)?;
    let checked_assembly = block_system.checked_backend(assembly);
    let quadrature = QuadratureRule::tensor_product_gauss_legendre(2, execution.points_per_axis)?;
    let assembly = finalize_conforming_cartesian_q1_linear_elasticity_pair_2d(
        meshes,
        model.subdomains.each_ref().map(|body| body.material()),
        model
            .subdomains
            .each_ref()
            .map(|body| body.load_potential_expression()),
        &quadrature,
        model.interface.axis,
        essential_sides,
        &checked_assembly,
    )?;
    let finalized = FinalizedConformingIsotropicElasticityCartesianPair2dProblem::new(
        execution.solver,
        execution.vector_layout,
        execution.target,
        assembly,
        block_system,
    )?;
    Ok((model, finalized))
}

fn pair_essential_sides(
    pair: &ConformingIsotropicElasticityCartesianPair2d,
) -> Result<[CartesianEssentialSides2d; 2], Diagnostic> {
    let interface_sides = [pair.interface.negative, pair.interface.positive];
    let mut result = Vec::with_capacity(2);
    for (subdomain, model) in pair.subdomains.iter().enumerate() {
        let mut essential = [[false; 2]; 2];
        for (axis, axis_sides) in essential.iter_mut().enumerate() {
            for (side_index, side) in [BoundarySide::Lower, BoundarySide::Upper]
                .into_iter()
                .enumerate()
            {
                let entry = model
                    .boundary_inventory()
                    .boundary(axis, side)
                    .expect("a lowered 2D inventory owns every Cartesian side");
                axis_sides[side_index] = match entry.disposition() {
                    PhysicalBoundaryDisposition::TraceZero => true,
                    PhysicalBoundaryDisposition::FluxZero => false,
                    PhysicalBoundaryDisposition::Prescribed(law) => {
                        return Err(super::invalid_realization(format!(
                            "prescribed elasticity {:?} law {} on subdomain {subdomain} axis {axis} {side:?} requires an explicit boundary-data Realization",
                            law.quantity(),
                            law.relation()
                        )));
                    }
                    PhysicalBoundaryDisposition::PortBinding { connection, port }
                        if axis == pair.interface.axis
                            && side == interface_sides[subdomain].side
                            && entry.boundary() == interface_sides[subdomain].boundary
                            && port == interface_sides[subdomain].port
                            && connection == pair.interface.connection =>
                    {
                        false
                    }
                    PhysicalBoundaryDisposition::PortBinding { connection, .. } => {
                        return Err(super::invalid_realization(format!(
                            "live elasticity PortBinding {connection} on subdomain {subdomain} axis {axis} {side:?} is not the exact admitted conforming interface"
                        )));
                    }
                };
            }
        }
        result.push(CartesianEssentialSides2d::new(essential));
    }
    let result: [CartesianEssentialSides2d; 2] = result.try_into().map_err(|_| {
        super::invalid_realization(
            "conforming elasticity pair did not yield exactly two boundary inventories",
        )
    })?;
    if !result
        .iter()
        .copied()
        .any(|sides| sides.has_essential_side())
    {
        return Err(super::invalid_realization(
            "conforming Cartesian elasticity pair requires at least one external homogeneous essential side to remove global rigid modes",
        ));
    }
    Ok(result)
}

fn cartesian_boxes_2d(
    program: &KernelProgram,
) -> Result<Vec<(RawId, CartesianBounds2d)>, Diagnostic> {
    program
        .nodes()
        .filter_map(|node| match node {
            KernelNode::Domain(domain)
                if matches!(domain.kind(), DomainKind::CartesianBox { .. }) =>
            {
                Some((domain.id().erase(), domain.id()))
            }
            _ => None,
        })
        .map(|(domain, typed_domain)| {
            let bounds = program.resolved_cartesian_bounds(typed_domain)?;
            if bounds.len() != 2 {
                return Err(lowering_error(
                    domain,
                    format!(
                        "conforming elasticity pair requires dimension two, received {}",
                        bounds.len()
                    ),
                ));
            }
            Ok((
                domain,
                [
                    [bounds[0].lower().value(), bounds[0].upper().value()],
                    [bounds[1].lower().value(), bounds[1].upper().value()],
                ],
            ))
        })
        .collect()
}

fn require_adjacent_bounds(
    negative: &IsotropicElasticityContinuum<2>,
    positive: &IsotropicElasticityContinuum<2>,
    axis: usize,
    connection: RawId,
) -> Result<(), Diagnostic> {
    if negative.bounds()[axis][1] != positive.bounds()[axis][0] {
        return Err(lowering_error(
            connection,
            "conforming elasticity interface does not separate adjacent body interiors",
        ));
    }
    let tangent = 1 - axis;
    if negative.bounds()[tangent] != positive.bounds()[tangent] {
        return Err(lowering_error(
            connection,
            "conforming elasticity interface bodies do not share one exact tangential interval",
        ));
    }
    Ok(())
}
