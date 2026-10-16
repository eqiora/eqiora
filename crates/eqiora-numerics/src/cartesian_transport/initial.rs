//! Constant-on-support initialization for the bounded scalar transport slice.

use eqiora_core::{Diagnostic, RawId, ScalarDomain};
use eqiora_schema::kernel::{ExprNode, FieldRole, KernelNode, SymbolRef};
use eqiora_sem::KernelProgram;

use crate::canonical::lowering_error;
use crate::canonical_transport::ScalarTransportCartesianModel2d;
use crate::spatial_expression;

pub(super) fn constant_initial_value(
    program: &KernelProgram,
    model: &ScalarTransportCartesianModel2d,
) -> Result<f64, Diagnostic> {
    let state = model.state().erase();
    let Some(KernelNode::Field(field)) = program.node(state) else {
        return Err(invalid(state));
    };
    if field.role() != FieldRole::State
        || field.value_type().scalar_domain() != ScalarDomain::Real
        || !field.value_type().shape().is_scalar()
    {
        return Err(invalid(state));
    }
    let mut initials = program.nodes().filter_map(|node| match node {
        KernelNode::Relation(relation) if relation.is_initial() => Some(relation),
        _ => None,
    });
    let initial = initials.next().ok_or_else(|| invalid(state))?;
    if initials.next().is_some() || initial.equation_sides()?.len() != 1 {
        return Err(lowering_error(
            initial.id().erase(),
            "scalar transport requires exactly one constant-on-support initial equation; additional initial equations are not admitted",
        ));
    }
    let typed = program
        .typed_relation_residual(initial.id())
        .map_err(|errors| {
            errors
                .into_iter()
                .next()
                .expect("failed typing owns diagnostics")
        })?;
    let residuals = program.numerical_residuals(initial.id().erase())?;
    let root = residuals.roots()[0];
    let root_type = typed
        .node_type(root)
        .expect("typed residual owns every root");
    if &root_type.value_type != field.value_type()
        || root_type.support.as_ref().map(|support| *support.domain())
            != Some(model.domain().erase())
    {
        return Err(invalid(initial.id().erase()));
    }
    let expression = &residuals;
    let node = |id: eqiora_schema::kernel::ExprId| &expression.nodes()[id.index() as usize];
    let is_state = |node: &ExprNode| matches!(node, ExprNode::Symbol(SymbolRef::Field(id)) if id.erase() == state);
    let constant = match node(root) {
        // Numerical projection removes only a checked, matching typed zero.
        symbol if is_state(symbol) => return Ok(0.0),
        ExprNode::Sub(left, right) if is_state(node(*left)) => *right,
        ExprNode::Sub(left, right) if is_state(node(*right)) => *left,
        _ => return Err(invalid(initial.id().erase())),
    };
    spatial_expression::lower(program, expression, constant, initial.id().erase(), 2)?
        .constant_value()
        .ok_or_else(|| invalid(initial.id().erase()))
}

fn invalid(owner: RawId) -> Diagnostic {
    lowering_error(
        owner,
        "scalar transport initialization requires one real scalar state = finite constant equation on the exact transported support; absent, shaped, time-dependent, or other-field conditions are not admitted",
    )
}
