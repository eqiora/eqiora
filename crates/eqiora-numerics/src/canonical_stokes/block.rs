//! Shared discrete block projection for the accepted steady MINI path.

use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, DimExponents, Id, RawId, ValueShape};
use eqiora_core::{ScalarDomain, ValueFrame, ValueType};
use eqiora_meshing::{MeshTopology, SimplicialMesh};
use eqiora_realization::{
    MeshArtifactReference, ResolvedFieldwiseRealization, ResolvedTransientFieldwiseRealization,
    Space,
};
use eqiora_sem::KernelProgram;
use eqiora_solver::AlgebraicBlock;
use eqiora_solver::LinearOperatorProperties;

use super::navier_stokes::TransientIncompressibleNavierStokesModel2d;
use super::{IncompressibleFlowScaleProfile2d, SteadyStokesScaleProfile2d};
use crate::canonical_boundary::BoundaryRelationBinding;
use crate::canonical_boundary::{PhysicalBoundaryDisposition, PhysicalBoundaryQuantity};
use crate::canonical_stokes::api::SteadyIncompressibleStokesModel2d;
use crate::discrete_block::{
    AlgebraicClosure, AuxiliaryBlock, BlockRealizationIdentity, BlockSupport, BlockTransformation,
    ContributionBatch, ContributionTerm, DiscreteBlockContext, DiscreteBlockSystem, FieldBlock,
    FieldBlockRole, RelationBlock, RelationDisposition, ResidualBlock, ResidualOrigin,
    boundary_treatment_for,
};
use crate::simplicial_stokes::{
    SimplicialMiniStokesBoundary2d, SimplicialMiniStokesBoundaryCondition2d,
};

const VELOCITY: DimExponents =
    DimExponents::from_integers([0, 1, -1, 0, 0, 0, 0]).expect("bounded dimension");
const PRESSURE: DimExponents =
    DimExponents::from_integers([1, -1, -2, 0, 0, 0, 0]).expect("bounded dimension");

pub(super) fn steady_stokes_block_system(
    model: &SteadyIncompressibleStokesModel2d,
    resolved: &ResolvedFieldwiseRealization,
    mesh_artifact: MeshArtifactReference,
    mesh: &SimplicialMesh,
    boundary: &SimplicialMiniStokesBoundary2d,
    scales: SteadyStokesScaleProfile2d,
) -> Result<DiscreteBlockSystem, Diagnostic> {
    model.replay_correspondence().map_err(|message| {
        Diagnostic::error(eqiora_core::diagnostic::codes::INVALID_REALIZATION, message)
    })?;
    let domain = domain_id(model.domain())?;
    let velocity = field(model.velocity())?;
    let pressure = field(model.pressure())?;
    let force = field(model.force_potential())?;
    let plan = resolved.plan();
    let space_for = |field| {
        plan.spatial()
            .field_spaces()
            .iter()
            .find(|binding| binding.field() == field)
            .map(|binding| binding.space())
            .expect("accepted Stokes plan contains both exact Field spaces")
    };
    let mut fields = vec![
        FieldBlock::discrete(
            domain,
            velocity,
            space_for(velocity),
            ValueType::shaped(
                ScalarDomain::Real,
                VELOCITY,
                ValueShape::new([2]).expect("two-component vector is representable"),
                ValueFrame::SpatialCartesian,
            )
            .expect("admitted spatial type"),
            scales.velocity(),
            FieldBlockRole::Algebraic,
        )?,
        FieldBlock::discrete(
            domain,
            pressure,
            space_for(pressure),
            ValueType::scalar(ScalarDomain::Real, PRESSURE).expect("numeric scalar type"),
            scales.pressure(),
            FieldBlockRole::Algebraic,
        )?,
        FieldBlock::coefficient(
            domain,
            force,
            ValueType::scalar(ScalarDomain::Real, PRESSURE).expect("numeric scalar type"),
        ),
    ];

    let mut boundary_coefficients = Vec::new();
    for pressure_law in model.normal_pressures() {
        let (Some(raw_field), Some(raw_definition)) = (
            pressure_law.coefficient_field(),
            pressure_law.definition_relation(),
        ) else {
            continue;
        };
        let coefficient = field(raw_field)?;
        let definition = relation(raw_definition)?;
        if let Some((_, existing)) = boundary_coefficients
            .iter()
            .find(|(field, _)| *field == coefficient)
        {
            if *existing != definition {
                return Err(invalid_identity(
                    "one definition Relation for a normal-pressure Field",
                    raw_definition,
                ));
            }
        } else {
            boundary_coefficients.push((coefficient, definition));
            fields.push(FieldBlock::coefficient(
                domain,
                coefficient,
                ValueType::scalar(ScalarDomain::Real, PRESSURE).expect("numeric scalar type"),
            ));
        }
    }
    let mut boundary_velocity_coefficients = Vec::new();
    for (raw_field, raw_definition) in model.normal_velocity_coefficients() {
        let coefficient = field(raw_field)?;
        let definition = relation(raw_definition)?;
        if let Some((_, existing)) = boundary_velocity_coefficients
            .iter()
            .find(|(field, _)| *field == coefficient)
        {
            if *existing != definition {
                return Err(invalid_identity(
                    "one definition Relation for a normal-velocity Field",
                    raw_definition,
                ));
            }
        } else {
            boundary_velocity_coefficients.push((coefficient, definition));
            fields.push(FieldBlock::coefficient(
                domain,
                coefficient,
                ValueType::scalar(ScalarDomain::Real, VELOCITY).expect("numeric scalar type"),
            ));
        }
    }

    let force_definition = relation(model.force_potential_definition())?;
    let momentum = relation(model.momentum_relation())?;
    let incompressibility = relation(model.incompressibility_relation())?;
    let mut relations = vec![
        RelationBlock::new(
            force_definition,
            BlockSupport::Volume(domain),
            RelationDisposition::CoefficientDefinition { field: force },
        ),
        RelationBlock::new(
            momentum,
            BlockSupport::Volume(domain),
            RelationDisposition::Residual {
                tested: AlgebraicBlock::Field(velocity),
            },
        ),
        RelationBlock::new(
            incompressibility,
            BlockSupport::Volume(domain),
            RelationDisposition::Residual {
                tested: AlgebraicBlock::Field(pressure),
            },
        ),
    ];
    relations.extend(boundary_coefficients.iter().map(|(field, definition)| {
        RelationBlock::new(
            *definition,
            BlockSupport::Volume(domain),
            RelationDisposition::CoefficientDefinition { field: *field },
        )
    }));
    relations.extend(
        boundary_velocity_coefficients
            .iter()
            .map(|(field, definition)| {
                RelationBlock::new(
                    *definition,
                    BlockSupport::Volume(domain),
                    RelationDisposition::CoefficientDefinition { field: *field },
                )
            }),
    );
    relations.extend(steady_boundary_relation_blocks(
        model,
        model.boundary_relations(),
        velocity,
    )?);

    let constraint = plan.spatial().constraints().first().copied();
    let auxiliaries = constraint
        .map(|constraint| AuxiliaryBlock::new(constraint, scales.gauge()))
        .transpose()?
        .into_iter()
        .collect::<Vec<_>>();
    let mut residuals = vec![
        ResidualBlock::new(
            AlgebraicBlock::Field(velocity),
            BlockSupport::Volume(domain),
            [ResidualOrigin::Relation(momentum)],
        )?,
        ResidualBlock::new(
            AlgebraicBlock::Field(pressure),
            BlockSupport::Volume(domain),
            [ResidualOrigin::Relation(incompressibility)],
        )?,
    ];
    if let Some(constraint) = constraint {
        residuals.push(ResidualBlock::new(
            AlgebraicBlock::ConstraintMultiplier { field: pressure },
            BlockSupport::Volume(domain),
            [ResidualOrigin::AlgebraicConstraint(constraint)],
        )?);
    }

    let essential = essential_relations(model)?;
    let transformations = (!essential.is_empty())
        .then_some(BlockTransformation::EssentialElimination {
            field: velocity,
            boundary_relations: essential.clone(),
        })
        .into_iter()
        .collect::<Vec<_>>();

    let cell_count = mesh
        .entity_count(2)
        .expect("accepted 2D Stokes mesh owns cells");
    let mut packet_cursor = cell_count;
    let mut contributions = vec![ContributionBatch::new(
        [BlockSupport::Volume(domain)],
        0..cell_count,
        [0, 1, 2],
        [
            ResidualOrigin::Relation(force_definition),
            ResidualOrigin::Relation(momentum),
            ResidualOrigin::Relation(incompressibility),
        ],
        parameter_inventory([
            model.dynamic_viscosity_expression().parameter_fields(),
            model.force_potential_expression().parameter_fields(),
        ]),
        [
            AlgebraicBlock::Field(velocity),
            AlgebraicBlock::Field(pressure),
        ],
        [
            AlgebraicBlock::Field(velocity),
            AlgebraicBlock::Field(pressure),
        ],
        [
            ContributionTerm::Stiffness,
            ContributionTerm::MixedConstraint,
            ContributionTerm::Load,
        ],
    )?];
    if let Some(constraint) = constraint {
        let end = packet_cursor + cell_count;
        contributions.push(ContributionBatch::new(
            [BlockSupport::Volume(domain)],
            packet_cursor..end,
            [0, 1, 2],
            [ResidualOrigin::AlgebraicConstraint(constraint)],
            [],
            [
                AlgebraicBlock::Field(pressure),
                AlgebraicBlock::ConstraintMultiplier { field: pressure },
            ],
            [
                AlgebraicBlock::Field(pressure),
                AlgebraicBlock::ConstraintMultiplier { field: pressure },
            ],
            [ContributionTerm::AlgebraicConstraint],
        )?);
        packet_cursor = end;
    }
    let traction_count = boundary
        .facets()
        .iter()
        .filter(|entry| {
            matches!(
                entry.condition(),
                SimplicialMiniStokesBoundaryCondition2d::ConstantTraction { .. }
            )
        })
        .count();
    if traction_count > 0 {
        let traction_relations = traction_relations(model)?;
        let boundary_origins = traction_relations
            .iter()
            .copied()
            .map(ResidualOrigin::Relation)
            .chain(
                boundary_coefficients
                    .iter()
                    .map(|(_, definition)| ResidualOrigin::Relation(*definition)),
            )
            .collect::<Vec<_>>();
        let mut supports = vec![BlockSupport::Volume(domain)];
        for binding in model.boundary_relations() {
            if traction_relations.contains(&relation(binding.relation())?) {
                let boundary = binding
                    .boundary()
                    .downcast::<kinds::Domain>()
                    .ok_or_else(|| invalid_identity("Boundary Domain", binding.boundary()))?;
                let support = BlockSupport::Boundary(boundary);
                if !supports.contains(&support) {
                    supports.push(support);
                }
            }
        }
        contributions.push(ContributionBatch::new(
            supports,
            packet_cursor..packet_cursor + traction_count,
            [0, 1],
            boundary_origins,
            boundary_parameter_inventory(model),
            [AlgebraicBlock::Field(velocity)],
            [AlgebraicBlock::Field(velocity)],
            [ContributionTerm::Boundary],
        )?);
        packet_cursor += traction_count;
    }

    let mut closures = vec![AlgebraicClosure::EssentialBoundary {
        field: velocity,
        relations: essential,
    }];
    if constraint.is_some() {
        closures.push(AlgebraicClosure::ZeroIntegral { field: pressure });
    } else {
        closures.push(AlgebraicClosure::BoundaryTraction {
            field: pressure,
            relations: traction_relations(model)?,
        });
    }

    DiscreteBlockSystem::new(
        DiscreteBlockContext::new(
            resolved.model(),
            resolved.semantic_revision(),
            BlockRealizationIdentity::Explicit(resolved.realization_revision()),
            Some(mesh_artifact),
        ),
        fields,
        auxiliaries,
        relations,
        residuals,
        transformations,
        closures,
        contributions,
        packet_cursor,
        3,
        0,
        LinearOperatorProperties::SymmetricIndefinite,
    )
}

#[allow(clippy::too_many_arguments)]
pub(super) fn transient_navier_stokes_block_system(
    program: &KernelProgram,
    model: &TransientIncompressibleNavierStokesModel2d,
    mesh_artifact: MeshArtifactReference,
    mesh: &SimplicialMesh,
    boundary: &SimplicialMiniStokesBoundary2d,
    resolved: &ResolvedTransientFieldwiseRealization,
    scales: IncompressibleFlowScaleProfile2d,
) -> Result<DiscreteBlockSystem, Diagnostic> {
    let domain = domain_id(model.domain)?;
    let velocity = field(model.velocity)?;
    let pressure = field(model.pressure)?;
    let force = field(model.force_potential)?;
    let momentum = relation(model.momentum_relation)?;
    let incompressibility = relation(model.incompressibility_relation)?;
    let force_definition = relation(model.force_potential_definition)?;
    let fields = vec![
        FieldBlock::discrete(
            domain,
            velocity,
            Space::simplex_p1_bubble(),
            ValueType::shaped(
                ScalarDomain::Real,
                VELOCITY,
                ValueShape::new([2]).expect("two-component vector is representable"),
                ValueFrame::SpatialCartesian,
            )
            .expect("admitted spatial type"),
            scales.velocity(),
            FieldBlockRole::Algebraic,
        )?,
        FieldBlock::discrete(
            domain,
            pressure,
            Space::continuous_lagrange(std::num::NonZeroU16::MIN),
            ValueType::scalar(ScalarDomain::Real, PRESSURE).expect("numeric scalar type"),
            scales.pressure(),
            FieldBlockRole::Algebraic,
        )?,
        FieldBlock::coefficient(
            domain,
            force,
            ValueType::scalar(ScalarDomain::Real, PRESSURE).expect("numeric scalar type"),
        ),
    ];
    let mut relations = vec![
        RelationBlock::new(
            force_definition,
            BlockSupport::Volume(domain),
            RelationDisposition::CoefficientDefinition { field: force },
        ),
        RelationBlock::new(
            momentum,
            BlockSupport::Volume(domain),
            RelationDisposition::Residual {
                tested: AlgebraicBlock::Field(velocity),
            },
        ),
        RelationBlock::new(
            incompressibility,
            BlockSupport::Volume(domain),
            RelationDisposition::Residual {
                tested: AlgebraicBlock::Field(pressure),
            },
        ),
    ];
    relations.extend(boundary_relation_blocks(
        &model.boundary_dispositions,
        &model.boundary_relations,
        velocity,
    )?);

    let constraint = resolved
        .plan()
        .fieldwise()
        .spatial()
        .constraints()
        .first()
        .copied();
    let auxiliaries = constraint
        .map(|constraint| AuxiliaryBlock::new(constraint, scales.gauge()))
        .transpose()?
        .into_iter()
        .collect::<Vec<_>>();
    let mut residuals = vec![
        ResidualBlock::new(
            AlgebraicBlock::Field(velocity),
            BlockSupport::Volume(domain),
            [ResidualOrigin::Relation(momentum)],
        )?,
        ResidualBlock::new(
            AlgebraicBlock::Field(pressure),
            BlockSupport::Volume(domain),
            [ResidualOrigin::Relation(incompressibility)],
        )?,
    ];
    if let Some(constraint) = constraint {
        residuals.push(ResidualBlock::new(
            AlgebraicBlock::ConstraintMultiplier { field: pressure },
            BlockSupport::Volume(domain),
            [ResidualOrigin::AlgebraicConstraint(constraint)],
        )?);
    }

    let essential = transient_boundary_relations(model, |disposition| {
        matches!(disposition, PhysicalBoundaryDisposition::TraceZero)
            || matches!(
                disposition,
                PhysicalBoundaryDisposition::Prescribed(law)
                    if law.quantity() == PhysicalBoundaryQuantity::Trace
            )
    })?;
    let traction = transient_boundary_relations(model, |disposition| {
        matches!(disposition, PhysicalBoundaryDisposition::FluxZero)
            || matches!(
                disposition,
                PhysicalBoundaryDisposition::Prescribed(law)
                    if law.quantity() == PhysicalBoundaryQuantity::Flux
            )
    })?;
    let mut transformations = vec![BlockTransformation::EssentialElimination {
        field: velocity,
        boundary_relations: essential.clone(),
    }];
    transformations.extend([
        BlockTransformation::BackwardEulerDerivative {
            relation: momentum,
            state: velocity,
            duration: resolved.plan().time_step().duration(),
        },
        BlockTransformation::EnergySkewConvection {
            relation: momentum,
            velocity,
        },
    ]);

    let cell_count = mesh
        .entity_count(2)
        .expect("accepted 2D transient mesh owns cells");
    let mut packet_cursor = cell_count;
    let mut contributions = vec![ContributionBatch::new(
        [BlockSupport::Volume(domain)],
        0..cell_count,
        [0],
        [
            ResidualOrigin::Relation(force_definition),
            ResidualOrigin::Relation(momentum),
            ResidualOrigin::Relation(incompressibility),
        ],
        parameter_inventory([
            model.mass_density.parameter_fields(),
            model.dynamic_viscosity.parameter_fields(),
            model.force_potential_expression.parameter_fields(),
        ]),
        [
            AlgebraicBlock::Field(velocity),
            AlgebraicBlock::Field(pressure),
        ],
        [
            AlgebraicBlock::Field(velocity),
            AlgebraicBlock::Field(pressure),
        ],
        [
            ContributionTerm::Mass,
            ContributionTerm::Advection,
            ContributionTerm::Stiffness,
            ContributionTerm::MixedConstraint,
            ContributionTerm::Load,
        ],
    )?];
    if let Some(constraint) = constraint {
        let end = packet_cursor + cell_count;
        contributions.push(ContributionBatch::new(
            [BlockSupport::Volume(domain)],
            packet_cursor..end,
            [0],
            [ResidualOrigin::AlgebraicConstraint(constraint)],
            [],
            [
                AlgebraicBlock::Field(pressure),
                AlgebraicBlock::ConstraintMultiplier { field: pressure },
            ],
            [
                AlgebraicBlock::Field(pressure),
                AlgebraicBlock::ConstraintMultiplier { field: pressure },
            ],
            [ContributionTerm::AlgebraicConstraint],
        )?);
        packet_cursor = end;
    }
    let traction_count = boundary
        .facets()
        .iter()
        .filter(|facet| {
            matches!(
                facet.condition(),
                SimplicialMiniStokesBoundaryCondition2d::ConstantTraction { .. }
            )
        })
        .count();
    if traction_count > 0 {
        let mut supports = vec![BlockSupport::Volume(domain)];
        for binding in &model.boundary_relations {
            if traction.contains(&relation(binding.relation())?) {
                let support = BlockSupport::Boundary(domain_id(binding.boundary())?);
                if !supports.contains(&support) {
                    supports.push(support);
                }
            }
        }
        contributions.push(ContributionBatch::new(
            supports,
            packet_cursor..packet_cursor + traction_count,
            [0],
            traction.iter().copied().map(ResidualOrigin::Relation),
            [],
            [AlgebraicBlock::Field(velocity)],
            [AlgebraicBlock::Field(velocity)],
            [ContributionTerm::Boundary],
        )?);
        packet_cursor += traction_count;
    }

    let mut closures = vec![AlgebraicClosure::EssentialBoundary {
        field: velocity,
        relations: essential,
    }];
    if constraint.is_some() {
        closures.push(AlgebraicClosure::ZeroIntegral { field: pressure });
    } else {
        closures.push(AlgebraicClosure::BoundaryTraction {
            field: pressure,
            relations: traction,
        });
    }
    DiscreteBlockSystem::new(
        DiscreteBlockContext::new(
            program.model(),
            eqiora_realization::SemanticRevision::new(program.revision().0),
            BlockRealizationIdentity::Explicit(resolved.realization_revision()),
            Some(mesh_artifact),
        ),
        fields,
        auxiliaries,
        relations,
        residuals,
        transformations,
        closures,
        contributions,
        packet_cursor,
        1,
        0,
        LinearOperatorProperties::General,
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

fn boundary_parameter_inventory(
    model: &SteadyIncompressibleStokesModel2d,
) -> Vec<Id<kinds::Parameter>> {
    parameter_inventory(
        model
            .normal_pressures()
            .map(|pressure| pressure.expression().parameter_fields())
            .chain(
                model
                    .normal_velocity_expressions()
                    .map(crate::spatial_expression::ScalarSpatialExpression::parameter_fields),
            ),
    )
}

fn steady_boundary_relation_blocks(
    model: &SteadyIncompressibleStokesModel2d,
    bindings: &[BoundaryRelationBinding],
    field: Id<kinds::Field>,
) -> Result<Vec<RelationBlock>, Diagnostic> {
    bindings
        .iter()
        .map(|binding| {
            let disposition = model
                .boundary_entries()
                .find(|(_, entry)| entry.boundary == binding.boundary())
                .map(|(_, entry)| entry.disposition)
                .ok_or_else(|| {
                    invalid_identity(
                        "boundary Relation support in the exact Stokes inventory",
                        binding.boundary(),
                    )
                })?;
            Ok(RelationBlock::new(
                relation(binding.relation())?,
                BlockSupport::Boundary(domain_id(binding.boundary())?),
                RelationDisposition::BoundaryCondition {
                    field,
                    treatment: boundary_treatment_for(disposition)?,
                },
            ))
        })
        .collect()
}

fn boundary_relation_blocks(
    dispositions: &std::collections::BTreeMap<RawId, PhysicalBoundaryDisposition>,
    bindings: &[BoundaryRelationBinding],
    field: Id<kinds::Field>,
) -> Result<Vec<RelationBlock>, Diagnostic> {
    bindings
        .iter()
        .map(|binding| {
            Ok(RelationBlock::new(
                relation(binding.relation())?,
                BlockSupport::Boundary(domain_id(binding.boundary())?),
                RelationDisposition::BoundaryCondition {
                    field,
                    treatment: boundary_treatment_for(
                        *dispositions.get(&binding.boundary()).ok_or_else(|| {
                            invalid_identity(
                                "boundary Relation support in the transient inventory",
                                binding.boundary(),
                            )
                        })?,
                    )?,
                },
            ))
        })
        .collect()
}

fn essential_relations(
    model: &SteadyIncompressibleStokesModel2d,
) -> Result<Vec<Id<kinds::Relation>>, Diagnostic> {
    relations_with_disposition(model, |disposition| {
        disposition == PhysicalBoundaryDisposition::TraceZero
            || matches!(
                disposition,
                PhysicalBoundaryDisposition::Prescribed(law)
                    if law.quantity() == PhysicalBoundaryQuantity::Trace
            )
    })
}

fn traction_relations(
    model: &SteadyIncompressibleStokesModel2d,
) -> Result<Vec<Id<kinds::Relation>>, Diagnostic> {
    relations_with_disposition(model, |disposition| {
        disposition == PhysicalBoundaryDisposition::FluxZero
            || matches!(
                disposition,
                PhysicalBoundaryDisposition::Prescribed(law)
                    if law.quantity() == PhysicalBoundaryQuantity::Flux
            )
    })
}

fn relations_with_disposition(
    model: &SteadyIncompressibleStokesModel2d,
    predicate: impl Fn(PhysicalBoundaryDisposition) -> bool,
) -> Result<Vec<Id<kinds::Relation>>, Diagnostic> {
    model
        .boundary_relations()
        .iter()
        .filter(|binding| {
            model.boundary_entries().any(|(_, entry)| {
                entry.boundary == binding.boundary() && predicate(entry.disposition)
            })
        })
        .map(|binding| relation(binding.relation()))
        .collect()
}

fn transient_boundary_relations(
    model: &TransientIncompressibleNavierStokesModel2d,
    predicate: impl Fn(PhysicalBoundaryDisposition) -> bool,
) -> Result<Vec<Id<kinds::Relation>>, Diagnostic> {
    model
        .boundary_relations
        .iter()
        .filter(|binding| {
            model
                .boundary_dispositions
                .get(&binding.boundary())
                .is_some_and(|disposition| predicate(*disposition))
        })
        .map(|binding| relation(binding.relation()))
        .collect()
}

fn domain_id(id: RawId) -> Result<Id<kinds::Domain>, Diagnostic> {
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
        format!("steady Stokes block inventory expected {expected} identity, received {id}"),
    )
}
