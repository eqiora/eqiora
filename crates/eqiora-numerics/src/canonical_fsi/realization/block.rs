//! Exact discrete block projection of the accepted fixed-reference FSI slice.

use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, Id, RawId};
use eqiora_meshing::{MeshTopology, SimplicialMesh};
use eqiora_realization::{MeshArtifactReference, ResolvedCoupledFieldwiseRealization};
use eqiora_solver::AlgebraicBlock;
use eqiora_solver::LinearOperatorProperties;

use super::super::FixedReferenceFsiCartesianModel2d;
use super::validate::{
    fluid_domain, fluid_pressure, fluid_velocity, solid_displacement, solid_domain, solid_velocity,
    trace_quotient,
};
use crate::canonical_boundary::BoundaryRelationBinding;
use crate::canonical_boundary::{CartesianBoundaryInventory, PhysicalBoundaryDisposition};
use crate::discrete_block::{
    AlgebraicClosure, BlockRealizationIdentity, BlockSupport, BlockTransformation,
    ContributionBatch, ContributionTerm, DiscreteBlockContext, DiscreteBlockSystem, RelationBlock,
    RelationDisposition, ResidualOrigin, boundary_treatment, conforming_interface_relations,
};
use crate::simplicial_fsi::FixedReferenceFsiPartition;

mod roles;

pub(super) fn fixed_reference_fsi_block_system(
    model: &FixedReferenceFsiCartesianModel2d,
    resolved: &ResolvedCoupledFieldwiseRealization,
    mesh_artifact: MeshArtifactReference,
    mesh: &SimplicialMesh,
    partition: &FixedReferenceFsiPartition<2>,
) -> Result<DiscreteBlockSystem, Diagnostic> {
    let fluid_domain = fluid_domain(model);
    let solid_domain = solid_domain(model);
    let fluid_velocity = fluid_velocity(model);
    let fluid_pressure = fluid_pressure(model);
    let solid_velocity = solid_velocity(model);
    let solid_displacement = solid_displacement(model);
    let roles::VolumeBlocks {
        fields,
        mut relations,
        residuals,
    } = roles::volume_blocks(model, resolved.plan())?;
    let fluid_definition = roles::coefficient_relation(model, fluid_domain)?;
    let solid_definition = roles::coefficient_relation(model, solid_domain)?;
    let fluid_momentum = roles::residual_relation(model, fluid_velocity)?;
    let incompressibility = roles::residual_relation(model, fluid_pressure)?;
    let solid_momentum = roles::residual_relation(model, solid_velocity)?;
    let solid_kinematic = roles::kinematic_relation(model, solid_displacement, solid_velocity)?;
    let plan = resolved.plan();
    relations.extend(boundary_relation_blocks(
        model.fluid().boundary_inventory(),
        model.fluid().boundary_relations(),
        fluid_velocity,
    )?);
    relations.extend(boundary_relation_blocks(
        model.solid().continuum().boundary_inventory(),
        model.solid().continuum().boundary_relations(),
        solid_velocity,
    )?);

    let quotient = trace_quotient(model);
    let interface_relations = conforming_interface_relations(
        [
            (
                model.fluid().boundary_inventory(),
                model.fluid().boundary_relations(),
            ),
            (
                model.solid().continuum().boundary_inventory(),
                model.solid().continuum().boundary_relations(),
            ),
        ],
        quotient.connection(),
    )?;
    let mut transformations = vec![
        BlockTransformation::BackwardEulerElimination {
            relation: solid_kinematic,
            state: solid_displacement,
            rate: solid_velocity,
            duration: plan.time_step().duration(),
        },
        BlockTransformation::ConformingTraceQuotient {
            quotient,
            interface_relations,
        },
    ];
    let fluid_essential = essential_relations(
        model.fluid().boundary_inventory(),
        model.fluid().boundary_relations(),
    )?;
    let solid_essential = essential_relations(
        model.solid().continuum().boundary_inventory(),
        model.solid().continuum().boundary_relations(),
    )?;
    if !fluid_essential.is_empty() {
        transformations.push(BlockTransformation::EssentialElimination {
            field: fluid_velocity,
            boundary_relations: fluid_essential.clone(),
        });
    }
    if !solid_essential.is_empty() {
        transformations.push(BlockTransformation::EssentialElimination {
            field: solid_velocity,
            boundary_relations: solid_essential.clone(),
        });
    }

    let fluid_packets = partition
        .fluid_cells()
        .iter()
        .map(|cell| cell.index())
        .collect::<Vec<_>>();
    let solid_packets = partition
        .solid_cells()
        .iter()
        .map(|cell| cell.index())
        .collect::<Vec<_>>();
    let contributions = vec![
        ContributionBatch::new(
            [BlockSupport::Volume(fluid_domain)],
            fluid_packets,
            [0, 1],
            [
                ResidualOrigin::Relation(fluid_definition),
                ResidualOrigin::Relation(fluid_momentum),
                ResidualOrigin::Relation(incompressibility),
            ],
            parameter_inventory([
                model.fluid().mass_density_expression().parameter_fields(),
                model
                    .fluid()
                    .dynamic_viscosity_expression()
                    .parameter_fields(),
                model
                    .fluid()
                    .force_potential_expression()
                    .parameter_fields(),
            ]),
            [
                AlgebraicBlock::Field(fluid_velocity),
                AlgebraicBlock::Field(fluid_pressure),
            ],
            [
                AlgebraicBlock::Field(fluid_velocity),
                AlgebraicBlock::Field(fluid_pressure),
            ],
            [
                ContributionTerm::Mass,
                ContributionTerm::Stiffness,
                ContributionTerm::MixedConstraint,
                ContributionTerm::Load,
            ],
        )?,
        ContributionBatch::new(
            [BlockSupport::Volume(solid_domain)],
            solid_packets,
            [0, 1],
            [
                ResidualOrigin::Relation(solid_definition),
                ResidualOrigin::Relation(solid_kinematic),
                ResidualOrigin::Relation(solid_momentum),
            ],
            parameter_inventory([
                model.solid().mass_density_expression().parameter_fields(),
                model
                    .solid()
                    .continuum()
                    .shear_modulus_expression()
                    .parameter_fields(),
                model
                    .solid()
                    .continuum()
                    .first_lame_parameter_expression()
                    .parameter_fields(),
                model
                    .solid()
                    .continuum()
                    .load_potential_expression()
                    .parameter_fields(),
            ]),
            [AlgebraicBlock::Field(solid_velocity)],
            [AlgebraicBlock::Field(solid_velocity)],
            [
                ContributionTerm::Mass,
                ContributionTerm::Stiffness,
                ContributionTerm::Load,
            ],
        )?,
    ];

    let packet_count = mesh
        .entity_count(2)
        .expect("accepted intrinsic-2D mesh owns cells");
    let closures = vec![
        AlgebraicClosure::EssentialBoundary {
            field: fluid_velocity,
            relations: fluid_essential,
        },
        AlgebraicClosure::EssentialBoundary {
            field: solid_velocity,
            relations: solid_essential,
        },
        AlgebraicClosure::CompleteOperator {
            field: fluid_pressure,
            relations: vec![fluid_momentum, incompressibility, solid_momentum],
        },
    ];
    DiscreteBlockSystem::new(
        DiscreteBlockContext::new(
            resolved.model(),
            resolved.semantic_revision(),
            BlockRealizationIdentity::Explicit(resolved.realization_revision()),
            Some(mesh_artifact),
        ),
        fields,
        vec![],
        relations,
        residuals,
        transformations,
        closures,
        contributions,
        packet_count,
        2,
        0,
        LinearOperatorProperties::SymmetricIndefinite,
    )
}

fn parameter_inventory<'a>(
    fields: impl IntoIterator<Item = &'a [Id<kinds::Parameter>]>,
) -> Vec<Id<kinds::Parameter>> {
    let mut result = fields.into_iter().flatten().copied().collect::<Vec<_>>();
    result.sort_by_key(Id::ulid);
    result.dedup();
    result
}

fn boundary_relation_blocks(
    inventory: &CartesianBoundaryInventory<2>,
    bindings: &[BoundaryRelationBinding],
    field: Id<kinds::Field>,
) -> Result<Vec<RelationBlock>, Diagnostic> {
    bindings
        .iter()
        .map(|binding| {
            Ok(RelationBlock::new(
                relation(binding.relation())?,
                BlockSupport::Boundary(domain(binding.boundary())?),
                RelationDisposition::BoundaryCondition {
                    field,
                    treatment: boundary_treatment(inventory, *binding)?,
                },
            ))
        })
        .collect()
}

fn essential_relations(
    inventory: &crate::canonical_boundary::CartesianBoundaryInventory<2>,
    bindings: &[BoundaryRelationBinding],
) -> Result<Vec<Id<kinds::Relation>>, Diagnostic> {
    bindings
        .iter()
        .filter(|binding| {
            inventory.entries().any(|(_, entry)| {
                entry.boundary() == binding.boundary()
                    && entry.disposition() == PhysicalBoundaryDisposition::TraceZero
            })
        })
        .map(|binding| relation(binding.relation()))
        .collect()
}

fn domain(id: RawId) -> Result<Id<kinds::Domain>, Diagnostic> {
    id.downcast::<kinds::Domain>()
        .ok_or_else(|| invalid_identity("Domain", id))
}

fn field(id: RawId) -> Result<Id<kinds::Field>, Diagnostic> {
    id.downcast::<kinds::Field>()
        .ok_or_else(|| invalid_identity("Field", id))
}

fn relation(id: RawId) -> Result<Id<kinds::Relation>, Diagnostic> {
    id.downcast::<kinds::Relation>()
        .ok_or_else(|| invalid_identity("Relation", id))
}

fn invalid_identity(expected: &str, id: RawId) -> Diagnostic {
    Diagnostic::error(
        eqiora_core::diagnostic::codes::INVALID_REALIZATION,
        format!("fixed-reference FSI block inventory expected {expected} identity, received {id}"),
    )
}
