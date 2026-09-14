//! Method-neutral recognition of one exact fixed-reference fluid-solid pair.

mod ale;
mod ale_realization;
mod geometry_regions;
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
    LoweredIsotropicElastodynamicsSubdomain2d, lower_isotropic_elastodynamics_subdomain_2d,
    lower_isotropic_elastodynamics_subdomain_2d_with_boundaries,
};
use crate::canonical_stokes::{
    InertialIncompressibleNewtonianCartesianModel2d,
    LoweredInertialIncompressibleNewtonianSubdomain2d, LoweredStokesBoundary,
    lower_inertial_incompressible_newtonian_subdomain_2d,
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
    FixedReferenceFsiFieldIdentities2d, FixedReferenceFsiScaleProfile2d,
    PreparedDistributedFixedReferenceFsiStep2d, ResolvedFixedReferenceFsiSolution2d,
    finalize_resolved_fixed_reference_fsi_step_2d,
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
    boundary: RawId,
    port: RawId,
    side: BoundarySide,
}

impl FsiInterfaceSide {
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
    fluid: FsiInterfaceSide,
    solid: FsiInterfaceSide,
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

    /// Fluid-local interface identity.
    #[must_use]
    pub const fn fluid(self) -> FsiInterfaceSide {
        self.fluid
    }

    /// Solid-local interface identity.
    #[must_use]
    pub const fn solid(self) -> FsiInterfaceSide {
        self.solid
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
    fluid: InertialIncompressibleNewtonianCartesianModel2d,
    solid: IsotropicElastodynamicsCartesianModel<2>,
    interface: FsiInterface,
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

    /// Exact canonical inertial-fluid submodel.
    #[must_use]
    pub const fn fluid(&self) -> &InertialIncompressibleNewtonianCartesianModel2d {
        &self.fluid
    }

    /// Exact canonical dynamic-solid submodel.
    #[must_use]
    pub const fn solid(&self) -> &IsotropicElastodynamicsCartesianModel<2> {
        &self.solid
    }

    /// Exact compatible live interface joining the two submodels.
    #[must_use]
    pub const fn interface(&self) -> FsiInterface {
        self.interface
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
    let boxes = cartesian_boxes_2d(program)?;
    if boxes.len() != 2 {
        return Err(model_lowering_error(
            program,
            format!(
                "fixed-reference FSI requires exactly two Cartesian boxes, found {}",
                boxes.len()
            ),
        ));
    }

    let mut candidates = Vec::new();
    for fluid_index in 0..2 {
        let solid_index = 1 - fluid_index;
        let fluid = lower_inertial_incompressible_newtonian_subdomain_2d(
            program,
            boxes[fluid_index].0,
            boxes[fluid_index].1,
        );
        let solid = lower_isotropic_elastodynamics_subdomain_2d(
            program,
            boxes[solid_index].0,
            boxes[solid_index].1,
        );
        if let (Ok(fluid), Ok(solid)) = (fluid, solid) {
            candidates.push((fluid, solid));
        }
    }
    if candidates.len() != 1 {
        return Err(model_lowering_error(
            program,
            format!(
                "fixed-reference FSI requires one unique inertial-fluid/dynamic-solid Domain assignment, found {}",
                candidates.len()
            ),
        ));
    }
    finish_fixed_reference_fsi(program, candidates)
}

/// Lower the same exact FSI meaning from two external GeometryRegion supports.
pub fn lower_fixed_reference_fsi_geometry_2d(
    program: &KernelProgram,
    geometry: &CanonicalGeometryV1,
) -> Result<FixedReferenceFsiCartesianModel2d, Diagnostic> {
    let domains = geometry_regions::cartesian_regions(program, geometry)?;
    if domains.len() != 2 {
        return Err(model_lowering_error(
            program,
            "fixed-reference numerical state projection currently requires two exact Regions",
        ));
    }
    let mut candidates = Vec::new();
    for fluid_index in 0..2 {
        let solid_index = 1 - fluid_index;
        let (fluid_domain, fluid_bounds, fluid_sides) = &domains[fluid_index];
        let (solid_domain, solid_bounds, solid_sides) = &domains[solid_index];
        let fluid = lower_inertial_incompressible_newtonian_subdomain_2d_with_boundaries(
            program,
            *fluid_domain,
            *fluid_bounds,
            Some(
                fluid_sides
                    .iter()
                    .map(|(&(axis, side), &id)| ((side, axis), id))
                    .collect(),
            ),
        );
        let solid = lower_isotropic_elastodynamics_subdomain_2d_with_boundaries(
            program,
            *solid_domain,
            *solid_bounds,
            solid_sides.clone(),
        );
        if let (Ok(fluid), Ok(solid)) = (fluid, solid) {
            candidates.push((fluid, solid));
        }
    }
    finish_fixed_reference_fsi(program, candidates)
}

fn finish_fixed_reference_fsi(
    program: &KernelProgram,
    mut candidates: Vec<(
        LoweredInertialIncompressibleNewtonianSubdomain2d,
        LoweredIsotropicElastodynamicsSubdomain2d,
    )>,
) -> Result<FixedReferenceFsiCartesianModel2d, Diagnostic> {
    if candidates.len() != 1 {
        return Err(model_lowering_error(
            program,
            format!(
                "fixed-reference FSI requires one unique inertial-fluid/dynamic-solid Domain assignment, found {}",
                candidates.len()
            ),
        ));
    }
    let (fluid, solid) = candidates
        .pop()
        .expect("one unique typed FSI assignment was established");

    reject_uninterpreted_live_relations(&fluid, &solid)?;
    let fluid_side = unique_live_side(fluid.model.boundary_inventory(), "fluid")?;
    let solid_side = unique_live_side(solid.model.continuum().boundary_inventory(), "solid")?;
    require_exact_interface(program, fluid_side, solid_side)?;
    require_coincident_bounds(
        fluid.model.bounds(),
        solid.model.continuum().bounds(),
        fluid_side,
        solid_side,
    )?;
    require_closed_fsi_model(program, &fluid, &solid)?;

    let domains = program
        .nodes()
        .filter_map(|node| match node {
            KernelNode::Domain(domain)
                if !crate::canonical::continuum_fields_on(program, domain.id().erase())
                    .is_empty() =>
            {
                Some(domain.id().erase())
            }
            _ => None,
        })
        .collect::<BTreeSet<_>>();
    let equation_roles = crate::form_compiler::equation_roles::EquationRoles::derive(
        program,
        domains.iter().copied(),
    )?;
    let region_forms = domains
        .into_iter()
        .map(|domain| {
            crate::form_compiler::region::CompiledRegionForm::derive(program, domain, 2)
                .map(|form| (domain, form))
        })
        .collect::<Result<_, _>>()?;
    Ok(FixedReferenceFsiCartesianModel2d {
        equation_roles,
        region_forms,
        model: program.model(),
        semantic_revision: program.revision().0,
        fluid: fluid.model,
        solid: solid.model,
        interface: FsiInterface {
            connection: fluid_side.connection,
            axis: fluid_side.axis,
            fluid: FsiInterfaceSide {
                boundary: fluid_side.boundary,
                port: fluid_side.port,
                side: fluid_side.side,
            },
            solid: FsiInterfaceSide {
                boundary: solid_side.boundary,
                port: solid_side.port,
                side: solid_side.side,
            },
        },
    })
}

fn reject_uninterpreted_live_relations(
    fluid: &LoweredInertialIncompressibleNewtonianSubdomain2d,
    solid: &LoweredIsotropicElastodynamicsSubdomain2d,
) -> Result<(), Diagnostic> {
    reject_uninterpreted_live_relation_sets(&fluid.boundary, solid)
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

fn require_closed_fsi_model(
    program: &KernelProgram,
    fluid: &LoweredInertialIncompressibleNewtonianSubdomain2d,
    solid: &LoweredIsotropicElastodynamicsSubdomain2d,
) -> Result<(), Diagnostic> {
    require_closed_fsi_model_parts(
        program,
        fluid.model.domain(),
        [
            fluid.model.velocity(),
            fluid.model.pressure(),
            fluid.model.force_potential(),
        ],
        fluid.representation,
        &fluid.volume_relations,
        &fluid.boundary,
        solid,
        "fixed-reference FSI",
    )
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
    let activations = program
        .edges()
        .iter()
        .filter(|edge| edge.kind() == EdgeKind::Activates && relations.contains(&edge.to()))
        .map(|edge| edge.from())
        .collect::<BTreeSet<_>>();
    let parameters = relations
        .iter()
        .copied()
        .flat_map(|relation| match program.node(relation) {
            Some(KernelNode::Relation(definition)) => definition.expression().nodes().iter(),
            _ => unreachable!("admitted Relations were already inspected"),
        })
        .filter_map(|node| match node {
            ExprNode::Symbol(SymbolRef::Parameter(parameter)) => Some(parameter.erase()),
            _ => None,
        })
        .collect::<BTreeSet<_>>();

    for node in program.nodes() {
        let admitted = match node {
            KernelNode::Domain(value) => domains.contains(&value.id().erase()),
            KernelNode::Representation(value) => representations.contains(&value.id().erase()),
            KernelNode::Field(value) => fields.contains(&value.id().erase()),
            KernelNode::Parameter(value) => parameters.contains(&value.id().erase()),
            KernelNode::Relation(value) => relations.contains(&value.id().erase()),
            KernelNode::Activation(value) => activations.contains(&value.id().erase()),
            KernelNode::Port(value) => ports.contains(&value.id().erase()),
            KernelNode::Connection(value) => connections.contains(&value.id().erase()),
            _ => false,
        };
        if !admitted {
            return Err(model_lowering_error(
                program,
                format!(
                    "closed {projection} lowering would ignore unexpected {:?} node {}",
                    node.kind(),
                    node.id()
                ),
            ));
        }
    }
    Ok(())
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
            [
                self.fluid().velocity(),
                self.fluid().pressure(),
                self.solid().velocity(),
            ]
            .map(|field| {
                field
                    .downcast::<eqiora_core::entity::kinds::Field>()
                    .expect("admitted Field")
            }),
            [],
        )
    }
}
