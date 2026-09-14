//! Exact discrete block projection of the accepted fixed-reference FSI slice.

use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, Id, RawId};
use eqiora_meshing::{MeshTopology, SimplicialMesh};
use eqiora_realization::{MeshArtifactReference, ResolvedCoupledFieldwiseRealization};
use eqiora_solver::AlgebraicBlock;
use eqiora_solver::LinearOperatorProperties;

use super::super::FixedReferenceFsiCartesianModel2d;

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
    let roles::VolumeBlocks {
        fields,
        mut relations,
        residuals,
    } = roles::volume_blocks(model, resolved.plan())?;
    let mut inventories = model
        .fluids()
        .map(|fluid| {
            (
                fluid.domain(),
                fluid.velocity(),
                fluid.boundary_inventory(),
                fluid.boundary_relations(),
                parameter_inventory([
                    fluid.mass_density_expression().parameter_fields(),
                    fluid.dynamic_viscosity_expression().parameter_fields(),
                    fluid.force_potential_expression().parameter_fields(),
                ]),
            )
        })
        .chain(model.solids().map(|solid| {
            (
                solid.continuum().domain(),
                solid.velocity(),
                solid.continuum().boundary_inventory(),
                solid.continuum().boundary_relations(),
                parameter_inventory([
                    solid.mass_density_expression().parameter_fields(),
                    solid
                        .continuum()
                        .shear_modulus_expression()
                        .parameter_fields(),
                    solid
                        .continuum()
                        .first_lame_parameter_expression()
                        .parameter_fields(),
                    solid
                        .continuum()
                        .load_potential_expression()
                        .parameter_fields(),
                ]),
            )
        }))
        .collect::<Vec<_>>();
    inventories.sort_by_key(|entry| entry.0);
    let plan = resolved.plan();
    let mut transformations = plan
        .time_step()
        .eliminated_states()
        .iter()
        .map(|binding| {
            let pair = binding.pair();
            BlockTransformation::BackwardEulerElimination {
                relation: pair.relation(),
                state: pair.state(),
                rate: pair.rate(),
                duration: plan.time_step().duration(),
            }
        })
        .collect::<Vec<_>>();
    for &quotient in plan.spatial().trace_quotients() {
        let interface_relations = conforming_interface_relations(
            inventories.iter().map(|entry| (entry.2, entry.3)),
            quotient.connection(),
        )?;
        transformations.push(BlockTransformation::ConformingTraceQuotient {
            quotient,
            interface_relations,
        });
    }
    let mut closures = Vec::new();
    let mut contributions = Vec::new();
    let all_residuals = model
        .equation_roles
        .relations
        .iter()
        .filter_map(|(&id, role)| {
            matches!(
                role.kind,
                crate::form_compiler::equation_roles::Role::Residual { .. }
            )
            .then_some(relation(id))
        })
        .collect::<Result<Vec<_>, _>>()?;
    for (owner, velocity, inventory, bindings, parameters) in inventories {
        let velocity = field(velocity)?;
        relations.extend(boundary_relation_blocks(inventory, bindings, velocity)?);
        let essential = essential_relations(inventory, bindings)?;
        if !essential.is_empty() {
            transformations.push(BlockTransformation::EssentialElimination {
                field: velocity,
                boundary_relations: essential.clone(),
            });
        }
        closures.push(AlgebraicClosure::EssentialBoundary {
            field: velocity,
            relations: essential,
        });
        let domain = domain(owner)?;
        let fields = plan
            .spatial()
            .domains()
            .iter()
            .find(|entry| entry.domain() == domain)
            .ok_or_else(|| invalid_identity("admitted Domain", owner))?
            .field_spaces()
            .iter()
            .map(|binding| AlgebraicBlock::Field(binding.field()))
            .collect::<Vec<_>>();
        let origins = model
            .equation_roles
            .relations
            .iter()
            .filter(|(_, role)| role.domain == owner)
            .map(|(&id, _)| relation(id).map(ResidualOrigin::Relation))
            .collect::<Result<Vec<_>, _>>()?;
        let mut terms = vec![
            ContributionTerm::Mass,
            ContributionTerm::Stiffness,
            ContributionTerm::Load,
        ];
        for &(constrained, multiplier) in model.equation_roles.constraints.values() {
            if model.equation_roles.fields[&constrained].0 == owner {
                terms.push(ContributionTerm::MixedConstraint);
                closures.push(AlgebraicClosure::CompleteOperator {
                    field: field(multiplier)?,
                    relations: all_residuals.clone(),
                });
            }
        }
        contributions.push(ContributionBatch::new(
            [BlockSupport::Volume(domain)],
            partition
                .domain_cells(domain)
                .ok_or_else(|| invalid_identity("partition Domain", owner))?
                .iter()
                .map(|cell| cell.index()),
            [0, 1],
            origins,
            parameters,
            fields.clone(),
            fields,
            terms,
        )?);
    }
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
        mesh.entity_count(2).expect("intrinsic 2D mesh owns cells"),
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
