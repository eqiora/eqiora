pub(super) use crate::canonical::{
    continuum_representation, has_edge, is_field, lowering_error, model_lowering_error,
    relation_expression, relations_on,
};
use eqiora_core::entity::kinds;
use eqiora_core::{Diagnostic, RawId};
use eqiora_schema::kernel::typing::TypedResidual;
use eqiora_schema::kernel::{ExprDag, ExprId};
use eqiora_sem::KernelProgram;

pub(super) fn require_continuous_relation(
    program: &KernelProgram,
    relation: RawId,
) -> Result<(), Diagnostic> {
    crate::canonical::require_continuous_relation(
        program,
        relation,
        "canonical fluid Relations require exactly one continuous Activation",
    )
}

pub(super) fn typed_relation(
    program: &KernelProgram,
    relation: RawId,
) -> Result<TypedResidual<RawId>, Diagnostic> {
    let relation_id = relation
        .downcast::<kinds::Relation>()
        .ok_or_else(|| lowering_error(relation, "calculus typing owner is not a Relation"))?;
    program
        .typed_relation_residual(relation_id)
        .map_err(|diagnostics| {
            diagnostics.into_iter().next().unwrap_or_else(|| {
                lowering_error(relation, "calculus typing failed without a diagnostic")
            })
        })
}

pub(super) fn unique_root(expression: &ExprDag, owner: RawId) -> Result<ExprId, Diagnostic> {
    crate::canonical::require_unique_root(
        expression,
        owner,
        "canonical fluid Relation requires exactly one residual root",
    )
}
