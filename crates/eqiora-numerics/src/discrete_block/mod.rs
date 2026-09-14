//! Closed physics-neutral structure around one deterministic spatial assembly.
//!
//! This module is intentionally crate-private. RFC 0053 uses it to prove the
//! first shared execution vocabulary; RFC 0054 curates the provider facade.
//! Physics-specific reconstruction remains in its existing concrete types.

use eqiora_solver::{AlgebraicBlock, AlgebraicConstraint};
use std::collections::BTreeSet;
use std::fmt;

use eqiora_assembly::{
    AssemblyBackend, AssemblyPacket, AssemblyPacketSetIdentityV1, AssemblyPlan, AssemblyReport,
    AssemblyResult, AssemblyWork,
};
use eqiora_core::ValueFrame;
use eqiora_core::diagnostic::codes;
use eqiora_core::entity::kinds;
use eqiora_core::{
    Diagnostic, DimExponents, DynQuantity, Id, OntologyId, ScalarDomain, ValueShape, ValueType,
};
use eqiora_realization::{
    ConformingTraceQuotient, DefaultPolicyVersion, MeshArtifactReference, RealizationRevision,
    SemanticRevision, Space, SpaceFamily,
};
use eqiora_schema::Model;
use eqiora_solver::{
    CanonicalCsrAgreementFingerprintV1, CanonicalCsrSystemView, LinearOperatorProperties,
};
use sha2::{Digest, Sha256};

use crate::canonical_boundary::BoundaryRelationBinding;
use crate::canonical_boundary::{
    CartesianBoundaryInventory, PhysicalBoundaryDisposition, PhysicalBoundaryQuantity,
};

const BLOCK_SYSTEM_IDENTITY_DOMAIN: &[u8] = b"eqiora.discrete-block-system/v3\0";

/// Exact Realization selection represented by one block system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlockRealizationIdentity {
    Default(DefaultPolicyVersion),
    Explicit(RealizationRevision),
}

/// Model/Realization/mesh identity shared by every block in one system.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct DiscreteBlockContext {
    model: OntologyId<Model>,
    semantic_revision: SemanticRevision,
    realization: BlockRealizationIdentity,
    mesh: Option<MeshArtifactReference>,
}

impl DiscreteBlockContext {
    pub(crate) const fn new(
        model: OntologyId<Model>,
        semantic_revision: SemanticRevision,
        realization: BlockRealizationIdentity,
        mesh: Option<MeshArtifactReference>,
    ) -> Self {
        Self {
            model,
            semantic_revision,
            realization,
            mesh,
        }
    }
}

/// Domain-separated identity of exact block structure, excluding CSR bytes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) struct BlockSystemIdentity([u8; 32]);

/// Algebraic or reconstructed role of one exact Semantic Field.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FieldBlockRole {
    Algebraic,
    EliminatedState,
    CoefficientData,
}

/// One exact Semantic Field and its Realization-owned scalar basis.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct FieldBlock {
    domain: Id<kinds::Domain>,
    field: Id<kinds::Field>,
    space: Option<Space>,
    value_type: ValueType,
    scale: Option<DynQuantity>,
    role: FieldBlockRole,
}

impl FieldBlock {
    pub(crate) fn discrete(
        domain: Id<kinds::Domain>,
        field: Id<kinds::Field>,
        space: Space,
        value_type: ValueType,
        scale: DynQuantity,
        role: FieldBlockRole,
    ) -> Result<Self, Diagnostic> {
        if role == FieldBlockRole::CoefficientData
            || !scale.value().is_finite()
            || scale.value() <= 0.0
            || scale.dim() != value_type.dimension()
        {
            return Err(invalid(
                "an algebraic or eliminated Field block requires a finite positive scale with the Field dimension",
            ));
        }
        Ok(Self {
            domain,
            field,
            space: Some(space),
            value_type,
            scale: Some(scale),
            role,
        })
    }

    pub(crate) const fn coefficient(
        domain: Id<kinds::Domain>,
        field: Id<kinds::Field>,
        value_type: ValueType,
    ) -> Self {
        Self {
            domain,
            field,
            space: None,
            value_type,
            scale: None,
            role: FieldBlockRole::CoefficientData,
        }
    }
}

/// Numerical unknown that is deliberately not a Semantic Field.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct AuxiliaryBlock {
    constraint: AlgebraicConstraint,
    scale: DynQuantity,
}

impl AuxiliaryBlock {
    pub(crate) fn new(
        constraint: AlgebraicConstraint,
        scale: DynQuantity,
    ) -> Result<Self, Diagnostic> {
        if !scale.value().is_finite() || scale.value() <= 0.0 {
            return Err(invalid(
                "an auxiliary block requires a finite strictly positive scale",
            ));
        }
        Ok(Self { constraint, scale })
    }

    const fn block(self) -> AlgebraicBlock {
        match self.constraint {
            AlgebraicConstraint::ZeroIntegral { field } => {
                AlgebraicBlock::ConstraintMultiplier { field }
            }
        }
    }
}

/// Exact support of a semantic Relation or numerical contribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BlockSupport {
    Volume(Id<kinds::Domain>),
    Boundary(Id<kinds::Domain>),
}

/// Why one exact Semantic Relation is present after numerical lowering.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RelationDisposition {
    CoefficientDefinition {
        field: Id<kinds::Field>,
    },
    Residual {
        tested: AlgebraicBlock,
    },
    StateElimination {
        state: Id<kinds::Field>,
        rate: Id<kinds::Field>,
    },
    BoundaryCondition {
        field: Id<kinds::Field>,
        treatment: BoundaryTreatment,
    },
}

/// Exact numerical treatment selected for one normalized boundary Relation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum BoundaryTreatment {
    EssentialElimination,
    Natural { inhomogeneous: bool },
    ConformingInterface { connection: Id<kinds::Connection> },
}

pub(crate) fn boundary_treatment(
    inventory: &CartesianBoundaryInventory<2>,
    binding: BoundaryRelationBinding,
) -> Result<BoundaryTreatment, Diagnostic> {
    let disposition = inventory
        .entries()
        .find(|(_, entry)| entry.boundary() == binding.boundary())
        .map(|(_, entry)| entry.disposition())
        .ok_or_else(|| invalid("a boundary Relation is absent from its exact inventory"))?;
    boundary_treatment_for(disposition)
}

pub(crate) fn boundary_treatment_for(
    disposition: PhysicalBoundaryDisposition,
) -> Result<BoundaryTreatment, Diagnostic> {
    match disposition {
        PhysicalBoundaryDisposition::TraceZero => Ok(BoundaryTreatment::EssentialElimination),
        PhysicalBoundaryDisposition::FluxZero => Ok(BoundaryTreatment::Natural {
            inhomogeneous: false,
        }),
        PhysicalBoundaryDisposition::Prescribed(law) => Ok(match law.quantity() {
            PhysicalBoundaryQuantity::Trace => BoundaryTreatment::EssentialElimination,
            PhysicalBoundaryQuantity::Flux => BoundaryTreatment::Natural {
                inhomogeneous: true,
            },
        }),
        PhysicalBoundaryDisposition::PortBinding { connection, .. } => {
            Ok(BoundaryTreatment::ConformingInterface {
                connection: connection.downcast::<kinds::Connection>().ok_or_else(|| {
                    invalid("a Port-boundary treatment requires a Connection identity")
                })?,
            })
        }
    }
}

pub(crate) fn conforming_interface_relations<'a>(
    inventories: impl IntoIterator<
        Item = (
            &'a CartesianBoundaryInventory<2>,
            &'a [BoundaryRelationBinding],
        ),
    >,
    connection: Id<kinds::Connection>,
) -> Result<Vec<Id<kinds::Relation>>, Diagnostic> {
    let mut result = Vec::new();
    for (inventory, bindings) in inventories {
        for binding in bindings {
            if boundary_treatment(inventory, *binding)?
                == (BoundaryTreatment::ConformingInterface { connection })
            {
                let relation = binding
                    .relation()
                    .downcast::<kinds::Relation>()
                    .ok_or_else(|| {
                        invalid("an interface treatment requires a Relation identity")
                    })?;
                result.push(relation);
            }
        }
    }
    result.sort_by_key(Id::ulid);
    result.dedup();
    if result.is_empty() {
        return Err(invalid(
            "a conforming trace quotient requires exact interface Relations",
        ));
    }
    Ok(result)
}

/// Exact Semantic Relation retained with support and lowered disposition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct RelationBlock {
    relation: Id<kinds::Relation>,
    support: BlockSupport,
    disposition: RelationDisposition,
}

impl RelationBlock {
    pub(crate) const fn new(
        relation: Id<kinds::Relation>,
        support: BlockSupport,
        disposition: RelationDisposition,
    ) -> Self {
        Self {
            relation,
            support,
            disposition,
        }
    }
}

/// Origin of a residual row block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ResidualOrigin {
    Relation(Id<kinds::Relation>),
    AlgebraicConstraint(AlgebraicConstraint),
}

/// Residual rows remain distinct from unknown-coordinate blocks.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ResidualBlock {
    tested: AlgebraicBlock,
    support: BlockSupport,
    origins: Vec<ResidualOrigin>,
}

impl ResidualBlock {
    pub(crate) fn new(
        tested: AlgebraicBlock,
        support: BlockSupport,
        origins: impl IntoIterator<Item = ResidualOrigin>,
    ) -> Result<Self, Diagnostic> {
        let mut origins = origins.into_iter().collect::<Vec<_>>();
        origins.sort_by(residual_origin_order);
        if origins.is_empty() || origins.windows(2).any(|pair| pair[0] == pair[1]) {
            return Err(invalid(
                "a residual block requires a nonempty unique origin inventory",
            ));
        }
        Ok(Self {
            tested,
            support,
            origins,
        })
    }
}

/// Typed coordinate transformation applied before global materialization.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum BlockTransformation {
    EssentialElimination {
        field: Id<kinds::Field>,
        boundary_relations: Vec<Id<kinds::Relation>>,
    },
    BackwardEulerElimination {
        relation: Id<kinds::Relation>,
        state: Id<kinds::Field>,
        rate: Id<kinds::Field>,
        duration: DynQuantity,
    },
    BackwardEulerDerivative {
        relation: Id<kinds::Relation>,
        state: Id<kinds::Field>,
        duration: DynQuantity,
    },
    EnergySkewConvection {
        relation: Id<kinds::Relation>,
        velocity: Id<kinds::Field>,
    },
    ConformingTraceQuotient {
        quotient: ConformingTraceQuotient,
        interface_relations: Vec<Id<kinds::Relation>>,
    },
}

/// Algebraic nullspace or anchoring fact validated by the concrete path.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum AlgebraicClosure {
    EssentialBoundary {
        field: Id<kinds::Field>,
        relations: Vec<Id<kinds::Relation>>,
    },
    ZeroIntegral {
        field: Id<kinds::Field>,
    },
    BoundaryTraction {
        field: Id<kinds::Field>,
        relations: Vec<Id<kinds::Relation>>,
    },
    CompleteOperator {
        field: Id<kinds::Field>,
        relations: Vec<Id<kinds::Relation>>,
    },
}

/// Closed first-slice classification of a fused local contribution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) enum ContributionTerm {
    Mass,
    Stiffness,
    MixedConstraint,
    Load,
    Boundary,
    AlgebraicConstraint,
    Advection,
}

/// One stable batch of existing fused local actions.
///
/// Terms describe the exact contents of the fused evaluator. They do not
/// claim independently materializable term matrices and therefore preserve
/// the established floating-point accumulation order.
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct ContributionBatch {
    supports: Vec<BlockSupport>,
    packet_indices: Vec<usize>,
    target_indices: Vec<usize>,
    origins: Vec<ResidualOrigin>,
    parameters: Vec<Id<kinds::Parameter>>,
    row_blocks: Vec<AlgebraicBlock>,
    column_blocks: Vec<AlgebraicBlock>,
    terms: Vec<ContributionTerm>,
}

impl ContributionBatch {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        supports: impl IntoIterator<Item = BlockSupport>,
        packet_indices: impl IntoIterator<Item = usize>,
        target_indices: impl IntoIterator<Item = usize>,
        origins: impl IntoIterator<Item = ResidualOrigin>,
        parameters: impl IntoIterator<Item = Id<kinds::Parameter>>,
        row_blocks: impl IntoIterator<Item = AlgebraicBlock>,
        column_blocks: impl IntoIterator<Item = AlgebraicBlock>,
        terms: impl IntoIterator<Item = ContributionTerm>,
    ) -> Result<Self, Diagnostic> {
        let mut supports = supports.into_iter().collect::<Vec<_>>();
        supports.sort_by(|left, right| support_order(*left, *right));
        let mut packet_indices = packet_indices.into_iter().collect::<Vec<_>>();
        packet_indices.sort_unstable();
        let mut target_indices = target_indices.into_iter().collect::<Vec<_>>();
        target_indices.sort_unstable();
        let mut origins = origins.into_iter().collect::<Vec<_>>();
        origins.sort_by(residual_origin_order);
        let mut parameters = parameters.into_iter().collect::<Vec<_>>();
        parameters.sort_by_key(Id::ulid);
        let mut row_blocks = row_blocks.into_iter().collect::<Vec<_>>();
        row_blocks.sort_by(block_order);
        let mut column_blocks = column_blocks.into_iter().collect::<Vec<_>>();
        column_blocks.sort_by(block_order);
        let mut terms = terms.into_iter().collect::<Vec<_>>();
        terms.sort_unstable();
        if supports.is_empty()
            || packet_indices.is_empty()
            || origins.is_empty()
            || target_indices.is_empty()
            || row_blocks.is_empty()
            || column_blocks.is_empty()
            || terms.is_empty()
            || has_duplicate(&supports)
            || has_duplicate(&packet_indices)
            || has_duplicate(&target_indices)
            || has_duplicate(&origins)
            || has_duplicate(&parameters)
            || has_duplicate(&row_blocks)
            || has_duplicate(&column_blocks)
            || has_duplicate(&terms)
        {
            return Err(invalid(
                "a contribution batch requires nonempty duplicate-free packets, origins, incidence, and terms",
            ));
        }
        Ok(Self {
            supports,
            packet_indices,
            target_indices,
            origins,
            parameters,
            row_blocks,
            column_blocks,
            terms,
        })
    }
}

/// Complete pre-CSR structure for one closed first-slice spatial operator.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DiscreteBlockSystem {
    context: DiscreteBlockContext,
    fields: Vec<FieldBlock>,
    auxiliaries: Vec<AuxiliaryBlock>,
    relations: Vec<RelationBlock>,
    residuals: Vec<ResidualBlock>,
    transformations: Vec<BlockTransformation>,
    closures: Vec<AlgebraicClosure>,
    contributions: Vec<ContributionBatch>,
    packet_count: usize,
    target_count: usize,
    primary_target: usize,
    required_properties: LinearOperatorProperties,
    identity: BlockSystemIdentity,
}

mod assembly;
mod identity;
mod ordering;
mod validation;

pub(crate) use assembly::{BlockMaterialization, CheckedBlockAssemblyBackend};
use ordering::*;

impl DiscreteBlockSystem {
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        context: DiscreteBlockContext,
        mut fields: Vec<FieldBlock>,
        mut auxiliaries: Vec<AuxiliaryBlock>,
        mut relations: Vec<RelationBlock>,
        mut residuals: Vec<ResidualBlock>,
        mut transformations: Vec<BlockTransformation>,
        mut closures: Vec<AlgebraicClosure>,
        mut contributions: Vec<ContributionBatch>,
        packet_count: usize,
        target_count: usize,
        primary_target: usize,
        required_properties: LinearOperatorProperties,
    ) -> Result<Self, Diagnostic> {
        fields.sort_by_key(|block| block.field.ulid());
        auxiliaries.sort_by(|left, right| block_order(&left.block(), &right.block()));
        relations.sort_by_key(|block| block.relation.ulid());
        residuals.sort_by(residual_block_order);
        for transformation in &mut transformations {
            match transformation {
                BlockTransformation::EssentialElimination {
                    boundary_relations, ..
                } => boundary_relations.sort_by_key(Id::ulid),
                BlockTransformation::ConformingTraceQuotient {
                    interface_relations,
                    ..
                } => interface_relations.sort_by_key(Id::ulid),
                BlockTransformation::BackwardEulerElimination { .. }
                | BlockTransformation::BackwardEulerDerivative { .. }
                | BlockTransformation::EnergySkewConvection { .. } => {}
            }
        }
        transformations.sort_by(transformation_order);
        for closure in &mut closures {
            match closure {
                AlgebraicClosure::EssentialBoundary { relations, .. }
                | AlgebraicClosure::BoundaryTraction { relations, .. }
                | AlgebraicClosure::CompleteOperator { relations, .. } => {
                    relations.sort_by_key(Id::ulid);
                }
                AlgebraicClosure::ZeroIntegral { .. } => {}
            }
        }
        closures.sort_by(closure_order);
        contributions.sort_by(contribution_order);
        let mut value = Self {
            context,
            fields,
            auxiliaries,
            relations,
            residuals,
            transformations,
            closures,
            contributions,
            packet_count,
            target_count,
            primary_target,
            required_properties,
            identity: BlockSystemIdentity([0; 32]),
        };
        value.validate()?;
        value.identity = value.compute_identity();
        Ok(value)
    }
}

#[cfg(test)]
mod tests;
