//! Shared discrete block projection for the conforming elasticity pair.

use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, DimExponents, DynQuantity, Id, RawId, ValueShape};
use eqiora_core::{ScalarDomain, ValueFrame, ValueType};
use eqiora_meshing::MeshTopology;
use eqiora_realization::{
    ConformingTraceQuotient, ResolutionSource, ResolvedRealization, TraceFieldEndpoint,
};
use eqiora_solver::AlgebraicBlock;
use eqiora_solver::LinearOperatorProperties;

use super::ConformingIsotropicElasticityCartesianPair2d;
use crate::canonical_boundary::BoundaryRelationBinding;
use crate::canonical_boundary::{CartesianBoundaryInventory, PhysicalBoundaryDisposition};
use crate::discrete_block::{
    AlgebraicClosure, BlockRealizationIdentity, BlockSupport, BlockTransformation,
    ContributionBatch, ContributionTerm, DiscreteBlockContext, DiscreteBlockSystem, FieldBlock,
    FieldBlockRole, RelationBlock, RelationDisposition, ResidualBlock, ResidualOrigin,
    boundary_treatment, conforming_interface_relations,
};
use eqiora_meshing::CartesianMesh;

const LENGTH: DimExponents =
    DimExponents::from_integers([0, 1, 0, 0, 0, 0, 0]).expect("bounded dimension");
const PRESSURE: DimExponents =
    DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0]).expect("bounded dimension");

pub(super) fn conforming_elasticity_pair_block_system(
    model: &ConformingIsotropicElasticityCartesianPair2d,
    resolved: &ResolvedRealization,
    meshes: &[CartesianMesh; 2],
) -> Result<DiscreteBlockSystem, Diagnostic> {
    let domains = model
        .subdomains()
        .each_ref()
        .map(|body| domain(body.domain()))
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    let displacements = model
        .subdomains()
        .each_ref()
        .map(|body| field(body.displacement()))
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    let loads = model
        .subdomains()
        .each_ref()
        .map(|body| field(body.load_potential()))
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    let vector = ValueShape::new([2]).expect("two-component vector is representable");
    let mut fields = Vec::new();
    for subdomain in 0..2 {
        fields.push(FieldBlock::discrete(
            domains[subdomain],
            displacements[subdomain],
            resolved.plan().space(),
            ValueType::shaped(
                ScalarDomain::Real,
                LENGTH,
                vector.clone(),
                ValueFrame::SpatialCartesian,
            )
            .expect("admitted spatial type"),
            DynQuantity::new(1.0, LENGTH),
            FieldBlockRole::Algebraic,
        )?);
        fields.push(FieldBlock::coefficient(
            domains[subdomain],
            loads[subdomain],
            ValueType::scalar(ScalarDomain::Real, PRESSURE).expect("numeric scalar type"),
        ));
    }

    let definitions = model
        .subdomains()
        .each_ref()
        .map(|body| relation(body.load_definition_relation()))
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    let balances = model
        .subdomains()
        .each_ref()
        .map(|body| relation(body.equilibrium_relation()))
        .into_iter()
        .collect::<Result<Vec<_>, _>>()?;
    let mut relations = Vec::new();
    let mut residuals = Vec::new();
    let mut transformations = Vec::new();
    let mut closures = Vec::new();
    let mut contributions = Vec::new();
    let mut packet_cursor = 0;
    for subdomain in 0..2 {
        relations.push(RelationBlock::new(
            definitions[subdomain],
            BlockSupport::Volume(domains[subdomain]),
            RelationDisposition::CoefficientDefinition {
                field: loads[subdomain],
            },
        ));
        relations.push(RelationBlock::new(
            balances[subdomain],
            BlockSupport::Volume(domains[subdomain]),
            RelationDisposition::Residual {
                tested: AlgebraicBlock::Field(displacements[subdomain]),
            },
        ));
        relations.extend(boundary_relation_blocks(
            model.subdomains()[subdomain].boundary_inventory(),
            model.subdomains()[subdomain].boundary_relations(),
            displacements[subdomain],
        )?);
        residuals.push(ResidualBlock::new(
            AlgebraicBlock::Field(displacements[subdomain]),
            BlockSupport::Volume(domains[subdomain]),
            [ResidualOrigin::Relation(balances[subdomain])],
        )?);
        let essential = essential_relations(&model.subdomains()[subdomain])?;
        if !essential.is_empty() {
            transformations.push(BlockTransformation::EssentialElimination {
                field: displacements[subdomain],
                boundary_relations: essential.clone(),
            });
            closures.push(AlgebraicClosure::EssentialBoundary {
                field: displacements[subdomain],
                relations: essential,
            });
        }
        let cell_count = meshes[subdomain]
            .entity_count(2)
            .expect("accepted Cartesian 2D mesh owns cells");
        contributions.push(ContributionBatch::new(
            [BlockSupport::Volume(domains[subdomain])],
            packet_cursor..packet_cursor + cell_count,
            [0, 1, 2 + subdomain],
            [
                ResidualOrigin::Relation(definitions[subdomain]),
                ResidualOrigin::Relation(balances[subdomain]),
            ],
            parameter_inventory([
                model.subdomains()[subdomain]
                    .shear_modulus_expression()
                    .parameter_fields(),
                model.subdomains()[subdomain]
                    .first_lame_parameter_expression()
                    .parameter_fields(),
                model.subdomains()[subdomain]
                    .load_potential_expression()
                    .parameter_fields(),
            ]),
            [AlgebraicBlock::Field(displacements[subdomain])],
            [AlgebraicBlock::Field(displacements[subdomain])],
            [ContributionTerm::Stiffness, ContributionTerm::Load],
        )?);
        packet_cursor += cell_count;
    }
    let connection = model
        .interface()
        .connection()
        .downcast::<kinds::Connection>()
        .ok_or_else(|| invalid_identity("Connection", model.interface().connection()))?;
    let quotient = ConformingTraceQuotient::new(
        connection,
        TraceFieldEndpoint::new(domains[0], displacements[0]),
        TraceFieldEndpoint::new(domains[1], displacements[1]),
    )?;
    let interface_relations = conforming_interface_relations(
        model
            .subdomains()
            .iter()
            .map(|body| (body.boundary_inventory(), body.boundary_relations())),
        connection,
    )?;
    transformations.push(BlockTransformation::ConformingTraceQuotient {
        quotient,
        interface_relations,
    });
    let realization = match resolved.source() {
        ResolutionSource::Default(policy) => BlockRealizationIdentity::Default(policy),
        ResolutionSource::Explicit(revision) => BlockRealizationIdentity::Explicit(revision),
    };
    DiscreteBlockSystem::new(
        DiscreteBlockContext::new(
            resolved.model(),
            resolved.semantic_revision(),
            realization,
            None,
        ),
        fields,
        vec![],
        relations,
        residuals,
        transformations,
        closures,
        contributions,
        packet_cursor,
        4,
        0,
        LinearOperatorProperties::SymmetricPositiveDefinite,
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
    model: &super::IsotropicElasticityContinuum<2>,
) -> Result<Vec<Id<kinds::Relation>>, Diagnostic> {
    model
        .boundary_relations()
        .iter()
        .filter(|binding| {
            model.boundary_inventory().entries().any(|(_, entry)| {
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
        format!("elasticity-pair block inventory expected {expected} identity, received {id}"),
    )
}
