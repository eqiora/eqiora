use super::*;
use eqiora_core::{ScalarDomain, ValueFrame};
use eqiora_ir::ScalarOperatorIr;
use eqiora_schema::kernel::typing::{ExpressionType, RootContract, TypedResidual};
use eqiora_schema::kernel::{ActivationKind, ExprDagBuilder, ExprNode, FieldRole, KernelNode};
use std::collections::BTreeSet;

/// Admit one static finite affine Field problem with explicit exact constraint tolerances.
///
/// # Errors
/// Rejects dynamic/spatial/discrete profiles, nonaffine operands, non-square branch systems,
/// missing/foreign/wrong-unit tolerance entries and excessive active-set enumeration.
pub(crate) fn lower_finite_constraints(
    kernel: &KernelProgram,
    enforcement: &FiniteConstraintEnforcement,
) -> Result<FiniteConstraintProblem, Diagnostic> {
    let mut symbols = Vec::new();
    let mut dimensions = Vec::new();
    let mut bindings = Vec::new();
    for node in kernel.nodes() {
        match node {
            KernelNode::Field(field) => {
                let value = field.value_type();
                if field.role() != FieldRole::Variable
                    || value.scalar_domain() != ScalarDomain::Real
                    || !value.shape().is_scalar()
                    || value.frame() != ValueFrame::Invariant
                {
                    return Err(invalid(
                        "finite active-set execution requires invariant real scalar algebraic Fields",
                    ));
                }
                symbols.push(SymbolRef::Field(field.id()));
                dimensions.push(value.dimension());
                if symbols.len() > 256 {
                    return Err(invalid(
                        "finite active-set profile permits at most 256 scalar Fields",
                    ));
                }
            }
            KernelNode::Parameter(parameter) => {
                let value = parameter
                    .value()
                    .real_scalar_value()
                    .ok_or_else(|| invalid("finite active-set Parameters must be real scalars"))?;
                bindings.push((SymbolRef::Parameter(parameter.id()), value.value()));
            }
            KernelNode::Relation(_) | KernelNode::Observable(_) => {}
            KernelNode::Activation(activation)
                if matches!(activation.kind(), ActivationKind::Continuous) => {}
            _ => {
                return Err(invalid(
                    "finite active-set profile rejects clocks, events, spatial support, Ports and non-scalar semantic owners",
                ));
            }
        }
    }
    if symbols.is_empty() {
        return Err(invalid(
            "finite active-set problem has no scalar Field unknowns",
        ));
    }
    let mut relations = Vec::new();
    let mut expected = BTreeSet::new();
    let mut equality_count = 0usize;
    let mut complementarity_count = 0usize;
    let mut expression_nodes = 0usize;
    for node in kernel.nodes() {
        let KernelNode::Relation(relation) = node else {
            continue;
        };
        let conditions = relation.conditions().ok_or_else(|| {
            invalid("finite active-set execution does not reinterpret a conservation Law as a condition Relation")
        })?;
        if relation.is_initial() {
            return Err(invalid(
                "finite static constraints do not admit initialization-only Relations",
            ));
        }
        expression_nodes = expression_nodes
            .checked_add(relation.expression().nodes().len())
            .ok_or_else(|| invalid("finite expression budget overflow"))?;
        if expression_nodes > 65_536 {
            return Err(invalid(
                "finite active-set profile permits at most 65536 expression nodes",
            ));
        }
        if relation.expression().nodes().iter().any(|node| matches!(node, ExprNode::Symbol(symbol) if !matches!(symbol, SymbolRef::Field(_) | SymbolRef::Parameter(_)))) {
            return Err(invalid("finite static constraints reject time, derivatives and activation history"));
        }
        let typed = TypedResidual::<eqiora_core::RawId>::infer(
            relation.expression().clone(),
            None,
            RootContract::RelationOperands,
            |symbol| {
                let value_type = match symbol {
                    SymbolRef::Field(id) => match kernel.node(id.erase()) {
                        Some(KernelNode::Field(value)) => Some(value.value_type()),
                        _ => None,
                    },
                    SymbolRef::Parameter(id) => match kernel.node(id.erase()) {
                        Some(KernelNode::Parameter(value)) => Some(value.value_type()),
                        _ => None,
                    },
                    _ => None,
                };
                value_type
                    .map(|value| ExpressionType::new(value.clone(), None))
                    .ok_or_else(|| {
                        invalid("finite expression symbol is outside the admitted Model")
                    })
            },
        )
        .map_err(|errors| invalid(format!("finite original operand typing failed: {errors:?}")))?;
        let mut operand_dimensions = Vec::new();
        for (ordinal, (kind, (left, right))) in
            conditions.iter().zip(relation.equation_sides()).enumerate()
        {
            let left = typed.node_type(left).expect("admitted root type");
            let right = typed.node_type(right).expect("admitted root type");
            for operand in [left, right] {
                if operand.support.is_some()
                    || operand.value_type.scalar_domain() != ScalarDomain::Real
                    || !operand.shape().is_scalar()
                    || operand.frame() != ValueFrame::Invariant
                {
                    return Err(invalid(
                        "finite active-set conditions require nonspatial invariant real scalar operands",
                    ));
                }
            }
            operand_dimensions.push((left.dimension(), right.dimension()));
            match kind {
                RelationConditionKind::Equality => equality_count += 1,
                RelationConditionKind::Inequality | RelationConditionKind::Complementarity => {
                    let reference = ConstraintRef::new(
                        relation.id(),
                        u32::try_from(ordinal)
                            .map_err(|_| invalid("condition ordinal overflow"))?,
                    );
                    expected.insert(reference);
                    let tolerance = enforcement.tolerance(reference).ok_or_else(|| {
                        invalid("each exact Model constraint requires an explicit tolerance")
                    })?;
                    if tolerance.left().dim() != left.dimension() {
                        return Err(invalid(
                            "constraint left tolerance has the wrong physical dimension",
                        ));
                    }
                    if *kind == RelationConditionKind::Complementarity {
                        complementarity_count += 1;
                        if tolerance
                            .right()
                            .is_none_or(|value| value.dim() != right.dimension())
                        {
                            return Err(invalid(
                                "complementarity requires an independent right-operand tolerance in its own dimension",
                            ));
                        }
                    } else if tolerance.right().is_some() {
                        return Err(invalid(
                            "inequality enforcement must not supply a complementarity tolerance",
                        ));
                    }
                }
            }
        }
        // Prove every original operand affine before considering any active branch.
        // A branch must never hide a nonlinear inactive operand.
        ScalarOperatorIr::lower(relation.expression())?
            .bind_affine(&symbols, &bindings)
            .map_err(|error| {
                invalid(format!(
                    "finite condition operands are not affine: {error:?}"
                ))
            })?;
        relations.push(RelationOperands {
            id: relation.id(),
            expression: relation.expression().clone(),
            conditions: conditions.to_vec(),
            dimensions: operand_dimensions,
        });
    }
    if expected.len() != enforcement.tolerances().len() {
        return Err(invalid(
            "enforcement names a foreign or non-constraint Relation condition",
        ));
    }
    if complementarity_count > 16 || (1u32 << complementarity_count) > enforcement.max_active_sets()
    {
        return Err(invalid(
            "complete active-set enumeration exceeds its explicit bounded budget",
        ));
    }
    if expression_nodes.saturating_mul(1usize << complementarity_count) > 16_777_216 {
        return Err(invalid(
            "finite active-set expression work exceeds 16777216 node evaluations",
        ));
    }
    if equality_count + complementarity_count != symbols.len() {
        return Err(invalid(
            "finite active-set branches must be square: equalities plus complementarity pairs equal scalar Fields",
        ));
    }
    Ok(FiniteConstraintProblem {
        kernel: kernel.clone(),
        symbols,
        dimensions,
        bindings,
        relations,
        enforcement: enforcement.clone(),
        complementarity_count,
    })
}

pub(super) fn branch_expression(
    relation: &RelationOperands,
    mask: u32,
    first_pair: &mut usize,
) -> Result<Option<ExprDag>, Diagnostic> {
    let mut builder = ExprDagBuilder::new();
    for node in relation.expression.nodes() {
        match node {
            ExprNode::PureOperatorApplication(application) => {
                let definition = relation
                    .expression
                    .definition(application.definition())
                    .expect("admitted pure operator definition");
                builder.pure_operator(definition, application.arguments().iter().copied())?;
            }
            _ => {
                builder.push(node.clone())?;
            }
        }
    }
    let mut roots = Vec::new();
    for (kind, pair) in relation
        .conditions
        .iter()
        .zip(relation.expression.roots().as_chunks::<2>().0.iter())
    {
        match kind {
            RelationConditionKind::Equality => roots.push(builder.sub(pair[0], pair[1])?),
            RelationConditionKind::Inequality => {}
            RelationConditionKind::Complementarity => {
                roots.push(if mask & (1 << *first_pair) == 0 {
                    pair[0]
                } else {
                    pair[1]
                });
                *first_pair += 1;
            }
        }
    }
    if roots.is_empty() {
        Ok(None)
    } else {
        builder.finish(roots).map(Some)
    }
}
