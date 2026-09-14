//! Method-neutral recognition of one exact fixed-reference fluid-solid pair.

mod ale;
mod ale_realization;
mod geometry_regions;
mod network;
mod realization;

use std::collections::{BTreeMap, BTreeSet};

use eqiora_core::diagnostic::codes;
use eqiora_core::{Diagnostic, GraphPath, OntologyId, RawId};
use eqiora_geometry::CanonicalGeometryV1;
use eqiora_graph::EdgeKind;
use eqiora_schema::Model;
use eqiora_schema::kernel::{BoundarySide, DomainKind, ExprNode, KernelNode, SymbolRef};
use eqiora_sem::KernelProgram;

use crate::canonical_boundary::{CartesianBoundaryInventory, PhysicalBoundaryDisposition};
use crate::canonical_elasticity::{
    IsotropicElastodynamicsCartesianModel, LoweredIsotropicElastodynamicsSubdomain,
    lower_isotropic_elastodynamics_subdomain_2d,
    lower_isotropic_elastodynamics_subdomain_2d_with_boundaries,
};
use crate::canonical_stokes::{
    InertialIncompressibleNewtonianCartesianModel2d, LoweredStokesBoundary,
    lower_inertial_incompressible_newtonian_subdomain_2d_with_boundaries,
};

pub use ale::{AleFsiCartesianModel, lower_ale_fsi_cartesian_2d, lower_ale_fsi_cartesian_3d};
pub use ale_realization::{
    AcceptedResolvedAleFsiRemesh2d, AleFsiFieldIdentities, AleFsiInitialPhysicalState,
    FinalizedResolvedFixedTopologyAleFsi, finalize_resolved_fixed_topology_ale_fsi_2d,
    finalize_resolved_fixed_topology_ale_fsi_3d, fixed_topology_ale_fsi_requirements_2d,
    fixed_topology_ale_fsi_requirements_3d, remesh_resolved_fixed_topology_ale_fsi_2d,
    solve_resolved_fixed_topology_ale_fsi_2d,
    solve_resolved_fixed_topology_ale_fsi_2d_with_assembly,
    solve_resolved_fixed_topology_ale_fsi_3d,
    solve_resolved_fixed_topology_ale_fsi_3d_with_assembly,
};
pub use realization::{
    AcceptedDistributedFixedReferenceFsiStep2d, FinalizedResolvedFixedReferenceFsiStep2d,
    FixedReferenceFsiScaleProfile2d, PreparedDistributedFixedReferenceFsiStep2d,
    ResolvedFixedReferenceFsiSolution2d, finalize_resolved_fixed_reference_fsi_step_2d,
    finalize_resolved_fixed_reference_fsi_step_2d_with_assembly, fixed_reference_fsi_cuda_plan_2d,
    fixed_reference_fsi_distributed_cuda_plan_2d, fixed_reference_fsi_plan_2d,
    fixed_reference_fsi_requirements_2d, fixed_reference_fsi_requirements_2d_for_layout,
};
pub(crate) use realization::{
    PreparedResolvedFixedReferenceFsiRun2d, prepare_resolved_fixed_reference_fsi_run_2d,
};

type CartesianBounds<const D: usize> = [[f64; 2]; D];

/// One exact physics-local end of an FSI interface.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FsiInterfaceSide {
    domain: RawId,
    field: RawId,
    boundary: RawId,
    port: RawId,
    side: BoundarySide,
}

impl FsiInterfaceSide {
    /// Exact parent Domain of this endpoint.
    pub const fn domain(self) -> RawId {
        self.domain
    }
    /// Exact trace Field carried by this endpoint.
    pub const fn field(self) -> RawId {
        self.field
    }

    /// Exact semantic Boundary supporting the interface law.
    #[must_use]
    pub const fn boundary(self) -> RawId {
        self.boundary
    }

    /// Exact velocity/traction Port owned by that boundary law.
    #[must_use]
    pub const fn port(self) -> RawId {
        self.port
    }

    /// Parent-outward Cartesian side of the owning Domain.
    #[must_use]
    pub const fn side(self) -> BoundarySide {
        self.side
    }
}

/// Exact semantic witness for one compatible fluid-solid interface.
///
/// The roles are physical rather than geometric: `fluid` always belongs to
/// the inertial Newtonian Domain and `solid` to the dynamic elastic Domain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FsiInterface {
    connection: RawId,
    axis: usize,
    endpoints: [FsiInterfaceSide; 2],
}

impl FsiInterface {
    /// Exact conserving Connection carrying continuity and traction balance.
    #[must_use]
    pub const fn connection(self) -> RawId {
        self.connection
    }

    /// Cartesian normal axis shared by the coincident sides.
    #[must_use]
    pub const fn axis(self) -> usize {
        self.axis
    }

    /// Exact endpoint for one parent Domain; no endpoint position owns a role.
    pub fn endpoint(self, domain: RawId) -> Option<FsiInterfaceSide> {
        self.endpoints
            .into_iter()
            .find(|endpoint| endpoint.domain == domain)
    }
    /// Both exact endpoints in canonical Domain order.
    pub const fn endpoints(self) -> [FsiInterfaceSide; 2] {
        self.endpoints
    }
}

/// One exact fixed-reference inertial-fluid/dynamic-solid semantic network.
///
/// The interface contract proves matching velocity traces and the sum of the
/// two parent-outward constitutive tractions. It does not select a mesh,
/// trace quotient, time method, monolithic or partitioned coupling, pressure
/// policy, assembly, solver, or execution target.
#[derive(Debug, Clone, PartialEq)]
pub struct FixedReferenceFsiCartesianModel2d {
    model: OntologyId<Model>,
    semantic_revision: u64,
    fluids: BTreeMap<RawId, InertialIncompressibleNewtonianCartesianModel2d>,
    solids: BTreeMap<RawId, IsotropicElastodynamicsCartesianModel<2>>,
    interfaces: BTreeMap<RawId, FsiInterface>,
    test_orientations: BTreeMap<RawId, f64>,
    equation_roles: crate::form_compiler::equation_roles::EquationRoles,
    region_forms: BTreeMap<RawId, crate::form_compiler::region::CompiledRegionForm>,
}

impl FixedReferenceFsiCartesianModel2d {
    /// Exact Semantic Model identity from which this closed projection was lowered.
    #[must_use]
    pub const fn model(&self) -> OntologyId<Model> {
        self.model
    }

    /// Exact Semantic Model revision number used during closed lowering.
    #[must_use]
    pub const fn semantic_revision(&self) -> u64 {
        self.semantic_revision
    }

    /// Every exact admitted inertial incompressible submodel.
    pub fn fluids(&self) -> impl Iterator<Item = &InertialIncompressibleNewtonianCartesianModel2d> {
        self.fluids.values()
    }
    /// Every exact admitted first-order elastic submodel.
    pub fn solids(&self) -> impl Iterator<Item = &IsotropicElastodynamicsCartesianModel<2>> {
        self.solids.values()
    }
    /// Every exact conserving velocity/traction Connection.
    pub fn interfaces(&self) -> impl Iterator<Item = FsiInterface> + '_ {
        self.interfaces.values().copied()
    }
}

#[derive(Debug, Clone, Copy)]
struct LiveSide {
    axis: usize,
    side: BoundarySide,
    boundary: RawId,
    connection: RawId,
    port: RawId,
}

/// Lower one complete, flat fixed-reference 2D FSI semantic network.
///
/// Recognition is identity-parametric and package-neutral. Exactly one
/// Cartesian Domain must have inertial incompressible Newtonian meaning and
/// exactly one must have first-order isotropic elastodynamic meaning. Their
/// only live sides must be coincident, opposite, and members of the same
/// exact two-Port conserving velocity/traction Connection.
///
/// # Errors
/// Returns `EQ0703` when the typed physics assignment is not unique, either
/// submodel is incomplete, the interface is not exact, or whole-model closure
/// would ignore any semantic node.
pub fn lower_fixed_reference_fsi_cartesian_2d(
    program: &KernelProgram,
) -> Result<FixedReferenceFsiCartesianModel2d, Diagnostic> {
    network::lower(
        program,
        cartesian_boxes_2d(program)?
            .into_iter()
            .map(|(domain, bounds)| (domain, bounds, None))
            .collect(),
    )
}

/// Lower the same exact FSI meaning from two external GeometryRegion supports.
pub fn lower_fixed_reference_fsi_geometry_2d(
    program: &KernelProgram,
    geometry: &CanonicalGeometryV1,
) -> Result<FixedReferenceFsiCartesianModel2d, Diagnostic> {
    network::lower(
        program,
        geometry_regions::cartesian_regions(program, geometry)?
            .into_iter()
            .map(|(domain, bounds, sides)| (domain, bounds, Some(sides)))
            .collect(),
    )
}

fn reject_uninterpreted_live_relation_sets<const D: usize>(
    fluid_boundary: &LoweredStokesBoundary<D>,
    solid: &LoweredIsotropicElastodynamicsSubdomain<D>,
) -> Result<(), Diagnostic> {
    if let Some(relation) = fluid_boundary
        .uninterpreted_live_relations
        .iter()
        .chain(solid.boundary.uninterpreted_live_relations.iter())
        .next()
    {
        return Err(lowering_error(
            *relation,
            "fixed-reference FSI interface contains an additional live Port Relation outside the canonical velocity/traction law",
        ));
    }
    Ok(())
}

fn unique_live_side<const D: usize>(
    inventory: &CartesianBoundaryInventory<D>,
    physics: &str,
) -> Result<LiveSide, Diagnostic> {
    let live = inventory
        .entries()
        .filter_map(|(&(axis, side), entry)| match entry.disposition() {
            PhysicalBoundaryDisposition::PortBinding { connection, port } => Some(LiveSide {
                axis,
                side,
                boundary: entry.boundary(),
                connection,
                port,
            }),
            _ => None,
        })
        .collect::<Vec<_>>();
    let [side] = live.as_slice() else {
        let owner = inventory
            .entries()
            .next()
            .map(|(_, entry)| entry.boundary())
            .expect("complete Cartesian inventory is nonempty");
        return Err(lowering_error(
            owner,
            format!(
                "fixed-reference FSI requires exactly one live {physics} side, found {}",
                live.len()
            ),
        ));
    };
    Ok(*side)
}

fn require_exact_interface(
    program: &KernelProgram,
    fluid: LiveSide,
    solid: LiveSide,
) -> Result<(), Diagnostic> {
    if fluid.connection != solid.connection || fluid.axis != solid.axis || fluid.side == solid.side
    {
        return Err(lowering_error(
            fluid.connection,
            "fixed-reference FSI must join opposite fluid and solid sides on one axis through one Connection",
        ));
    }
    let member_ports = program
        .edges()
        .iter()
        .filter(|edge| edge.kind() == EdgeKind::Connects && edge.from() == fluid.connection)
        .map(|edge| edge.to())
        .collect::<BTreeSet<_>>();
    if member_ports != BTreeSet::from([fluid.port, solid.port]) {
        return Err(lowering_error(
            fluid.connection,
            "fixed-reference FSI interface requires exactly the recognized fluid and solid Ports",
        ));
    }
    Ok(())
}

fn require_coincident_bounds<const D: usize>(
    fluid_bounds: &CartesianBounds<D>,
    solid_bounds: &CartesianBounds<D>,
    fluid: LiveSide,
    solid: LiveSide,
) -> Result<(), Diagnostic> {
    let fluid_coordinate = match fluid.side {
        BoundarySide::Lower => fluid_bounds[fluid.axis][0],
        BoundarySide::Upper => fluid_bounds[fluid.axis][1],
    };
    let solid_coordinate = match solid.side {
        BoundarySide::Lower => solid_bounds[solid.axis][0],
        BoundarySide::Upper => solid_bounds[solid.axis][1],
    };
    if fluid_coordinate != solid_coordinate {
        return Err(lowering_error(
            fluid.connection,
            "fixed-reference FSI sides do not share one exact interface coordinate",
        ));
    }
    for tangent in 0..D {
        if tangent != fluid.axis && fluid_bounds[tangent] != solid_bounds[tangent] {
            return Err(lowering_error(
                fluid.connection,
                format!(
                    "fixed-reference FSI sides do not share one exact tangential interval on axis {tangent}"
                ),
            ));
        }
    }
    let interiors_are_opposite = match (fluid.side, solid.side) {
        (BoundarySide::Upper, BoundarySide::Lower) => {
            fluid_bounds[fluid.axis][0] < fluid_coordinate
                && solid_coordinate < solid_bounds[solid.axis][1]
        }
        (BoundarySide::Lower, BoundarySide::Upper) => {
            solid_bounds[solid.axis][0] < solid_coordinate
                && fluid_coordinate < fluid_bounds[fluid.axis][1]
        }
        _ => false,
    };
    if !interiors_are_opposite {
        return Err(lowering_error(
            fluid.connection,
            "fixed-reference FSI interface does not separate opposite Domain interiors",
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn require_closed_fsi_model_parts<const D: usize>(
    program: &KernelProgram,
    fluid_domain: RawId,
    fluid_fields: [RawId; 3],
    fluid_representation: RawId,
    fluid_volume_relations: &[RawId],
    fluid_boundary: &LoweredStokesBoundary<D>,
    solid: &LoweredIsotropicElastodynamicsSubdomain<D>,
    projection: &str,
) -> Result<(), Diagnostic> {
    let mut domains = BTreeSet::from([fluid_domain, solid.model.continuum().domain()]);
    domains.extend(
        fluid_boundary
            .inventory
            .entries()
            .chain(solid.model.continuum().boundary_inventory().entries())
            .map(|(_, entry)| entry.boundary()),
    );
    domains.extend(fluid_boundary.connector_domains.iter().copied());
    domains.extend(solid.boundary.connector_domains.iter().copied());

    let fields = BTreeSet::from([
        fluid_fields[0],
        fluid_fields[1],
        fluid_fields[2],
        solid.model.continuum().displacement(),
        solid.model.velocity(),
        solid.model.continuum().load_potential(),
    ]);
    let representations = BTreeSet::from([fluid_representation, solid.representation]);
    let mut relations = fluid_volume_relations
        .iter()
        .chain(solid.volume_relations.iter())
        .copied()
        .collect::<BTreeSet<_>>();
    relations.extend(fluid_boundary.relations.iter().copied());
    relations.extend(solid.boundary.relations.iter().copied());
    let mut ports = fluid_boundary.ports.clone();
    ports.extend(solid.boundary.ports.iter().copied());
    let mut connections = fluid_boundary.connections.clone();
    connections.extend(solid.boundary.connections.iter().copied());
    let allowed = domains
        .into_iter()
        .chain(fields)
        .chain(representations)
        .chain(relations)
        .chain(ports)
        .chain(connections)
        .collect();
    network::require_closed(program, allowed, projection)
}

fn cartesian_boxes_2d(
    program: &KernelProgram,
) -> Result<Vec<(RawId, CartesianBounds<2>)>, Diagnostic> {
    cartesian_boxes::<2>(program)
}

fn cartesian_boxes<const D: usize>(
    program: &KernelProgram,
) -> Result<Vec<(RawId, CartesianBounds<D>)>, Diagnostic> {
    if !matches!(D, 2 | 3) {
        return Err(model_lowering_error(
            program,
            format!("Cartesian FSI lowering supports dimension two or three, received {D}"),
        ));
    }
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
            if bounds.len() != D {
                return Err(lowering_error(
                    domain,
                    format!(
                        "fixed-reference FSI requires dimension {D}, received {}",
                        bounds.len()
                    ),
                ));
            }
            let bounds = bounds
                .iter()
                .map(|bound| [bound.lower().value(), bound.upper().value()])
                .collect::<Vec<_>>()
                .try_into()
                .expect("dimension equality establishes Cartesian bound count");
            Ok((domain, bounds))
        })
        .collect()
}

fn lowering_error(owner: RawId, message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_SPATIAL_LOWERING, message).with_graph_path(GraphPath::new([
        owner.kind().graph().name().to_owned(),
        format!("{:?}", owner.kind()),
        owner.to_string(),
    ]))
}

fn model_lowering_error(program: &KernelProgram, message: impl Into<String>) -> Diagnostic {
    Diagnostic::error(codes::INVALID_SPATIAL_LOWERING, message).with_graph_path(GraphPath::new([
        "ontology-view".to_owned(),
        "eqiora.model/v1".to_owned(),
        program.model().to_string(),
    ]))
}

#[cfg(test)]
mod tests;

impl FixedReferenceFsiCartesianModel2d {
    pub(crate) fn algebraic_structure(
        &self,
    ) -> Result<eqiora_solver::AlgebraicStructure, Diagnostic> {
        eqiora_solver::AlgebraicStructure::new(
            self.equation_roles
                .relations
                .values()
                .filter_map(|relation| match relation.kind {
                    crate::form_compiler::equation_roles::Role::Residual { tested } => Some(
                        tested
                            .downcast::<eqiora_core::entity::kinds::Field>()
                            .expect("admitted Field"),
                    ),
                    _ => None,
                }),
            [],
        )
    }
}
