use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, Id};
use eqiora_realization::CoupledFieldwiseRealizationPlan;
use eqiora_solver::AlgebraicBlock;

use crate::canonical_fsi::FixedReferenceFsiCartesianModel2d;
use crate::discrete_block::{
    BlockSupport, FieldBlock, FieldBlockRole, RelationBlock, RelationDisposition, ResidualBlock,
    ResidualOrigin,
};
use crate::form_compiler::equation_roles::Role;

pub(super) struct VolumeBlocks {
    pub(super) fields: Vec<FieldBlock>,
    pub(super) relations: Vec<RelationBlock>,
    pub(super) residuals: Vec<ResidualBlock>,
}

pub(super) fn volume_blocks(
    model: &FixedReferenceFsiCartesianModel2d,
    plan: &CoupledFieldwiseRealizationPlan,
) -> Result<VolumeBlocks, Diagnostic> {
    let roles = &model.equation_roles;
    let state = plan.time_step().eliminated_state();
    let state_pair = state.pair();
    kinematic_relation(model, state_pair.state(), state_pair.rate())?;
    if roles
        .relations
        .values()
        .filter(|entry| matches!(entry.kind, Role::Kinematic { .. }))
        .count()
        != 1
    {
        return Err(invalid(
            "Plan must account for every derived kinematic relation",
        ));
    }
    let mut fields = Vec::new();
    for (id, (domain, value_type)) in &roles.fields {
        let id = super::field(*id)?;
        let domain = super::domain(*domain)?;
        if roles
            .relations
            .values()
            .any(|entry| entry.kind == (Role::Coefficient { field: id.erase() }))
        {
            fields.push(FieldBlock::coefficient(domain, id, value_type.clone()));
            continue;
        }
        let (space, scale, role) = if id == state_pair.state() {
            (
                state.state_space(),
                state.state_scale().quantity(),
                FieldBlockRole::EliminatedState,
            )
        } else {
            let space = plan
                .spatial()
                .domains()
                .iter()
                .filter(|entry| entry.domain() == domain)
                .flat_map(|entry| entry.field_spaces())
                .find(|binding| binding.field() == id)
                .ok_or_else(|| invalid("derived unknown has no exact Plan field-space binding"))?
                .space();
            let scale = plan
                .scaling()
                .block_scales()
                .iter()
                .find(|scale| scale.block() == AlgebraicBlock::Field(id))
                .ok_or_else(|| invalid("derived unknown has no exact Plan scale"))?
                .scale()
                .quantity();
            (space, scale, FieldBlockRole::Algebraic)
        };
        fields.push(FieldBlock::discrete(
            domain,
            id,
            space,
            value_type.clone(),
            scale,
            role,
        )?);
    }
    let mut relations = Vec::new();
    let mut residuals = Vec::new();
    for (id, entry) in &roles.relations {
        let id = super::relation(*id)?;
        let support = BlockSupport::Volume(super::domain(entry.domain)?);
        let disposition = match entry.kind {
            Role::Coefficient { field } => RelationDisposition::CoefficientDefinition {
                field: super::field(field)?,
            },
            Role::Kinematic { state, rate } => RelationDisposition::StateElimination {
                state: super::field(state)?,
                rate: super::field(rate)?,
            },
            Role::Residual { tested } => {
                let tested = AlgebraicBlock::Field(super::field(tested)?);
                residuals.push(ResidualBlock::new(
                    tested,
                    support,
                    [ResidualOrigin::Relation(id)],
                )?);
                RelationDisposition::Residual { tested }
            }
        };
        relations.push(RelationBlock::new(id, support, disposition));
    }
    Ok(VolumeBlocks {
        fields,
        relations,
        residuals,
    })
}

pub(super) fn coefficient_relation(
    model: &FixedReferenceFsiCartesianModel2d,
    domain: Id<kinds::Domain>,
) -> Result<Id<kinds::Relation>, Diagnostic> {
    unique(model, |entry| {
        entry.domain == domain.erase() && matches!(entry.kind, Role::Coefficient { .. })
    })
}

pub(super) fn residual_relation(
    model: &FixedReferenceFsiCartesianModel2d,
    tested: Id<kinds::Field>,
) -> Result<Id<kinds::Relation>, Diagnostic> {
    unique(model, |entry| {
        entry.kind
            == (Role::Residual {
                tested: tested.erase(),
            })
    })
}

pub(super) fn kinematic_relation(
    model: &FixedReferenceFsiCartesianModel2d,
    state: Id<kinds::Field>,
    rate: Id<kinds::Field>,
) -> Result<Id<kinds::Relation>, Diagnostic> {
    unique(model, |entry| {
        entry.kind
            == (Role::Kinematic {
                state: state.erase(),
                rate: rate.erase(),
            })
    })
}

fn unique(
    model: &FixedReferenceFsiCartesianModel2d,
    matches: impl Fn(&crate::form_compiler::equation_roles::EquationRole) -> bool,
) -> Result<Id<kinds::Relation>, Diagnostic> {
    let mut found = model
        .equation_roles
        .relations
        .iter()
        .filter(|(_, entry)| matches(entry));
    let (id, _) = found
        .next()
        .ok_or_else(|| invalid("Plan role has no matching Model equation"))?;
    if found.next().is_some() {
        return Err(invalid("Plan role matches multiple Model equations"));
    }
    super::relation(*id)
}

fn invalid(message: &str) -> Diagnostic {
    Diagnostic::error(eqiora_core::diagnostic::codes::INVALID_REALIZATION, message)
}
