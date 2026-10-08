//! Global finite weak tests authenticate the original typed residual before action lowering.
use eqiora_compiler::{AuthoredFormExpressionV1 as E, AuthoredFormulationProjection};
use eqiora_core::{Diagnostic, Id, entity::kinds};
use eqiora_sem::KernelProgram;

pub(crate) fn admit_authored_finite_weak_form(
    projection: &AuthoredFormulationProjection,
    program: &KernelProgram,
    relation: Id<kinds::Relation>,
    mode: Id<kinds::Field>,
) -> Result<(), Diagnostic> {
    let reject = || {
        Diagnostic::error(
            eqiora_core::diagnostic::codes::INVALID_DISCRETIZATION,
            "finite weak Formulation must retain the exact source Relation, mode basis and conjugate-test residual",
        )
    };
    let mode_id = mode.ulid().to_string();
    let relation_id = relation.ulid().to_string();
    let [(_, trial, boundaries, _)] = projection.test_restrictions() else {
        return Err(reject());
    };
    let [(equation, _, _)] = projection.equations() else {
        return Err(reject());
    };
    if projection.domain_ulid().is_some()
        || projection.trial_ulids() != [mode_id.clone()]
        || trial != &mode_id
        || !boundaries.is_empty()
        || equation != &relation_id
    {
        return Err(reject());
    }
    super::check_authored_dependence(projection, program)?;
    let typed = program
        .typed_relation_residual(relation)
        .map_err(|_| reject())?;
    let [root] = typed.expression().roots() else {
        return Err(reject());
    };
    let mode_type = program
        .node(mode.erase())
        .and_then(|node| match node {
            eqiora_schema::kernel::KernelNode::Field(field) => Some(field.value_type()),
            _ => None,
        })
        .ok_or_else(reject)?;
    let root_type = typed.node_type(*root).ok_or_else(reject)?;
    if root_type.support.is_some()
        || mode_type.coordinate_basis().is_none()
        || root_type.value_type.coordinate_basis() != mode_type.coordinate_basis()
        || root_type.value_type.scalar_domain() != mode_type.scalar_domain()
    {
        return Err(reject());
    }
    let residual = E::from_expression(typed.expression(), *root)?.ok_or_else(reject)?;
    let expected = E::Inner {
        left: Box::new(E::Test {
            field_ulid: mode_id,
        }),
        right: Box::new(residual),
    };
    if !super::authored_polynomial::matches_weak_residual(
        projection,
        program,
        0,
        &expected,
        &E::Number { value: 0. },
    ) {
        return Err(reject());
    }
    Ok(())
}
