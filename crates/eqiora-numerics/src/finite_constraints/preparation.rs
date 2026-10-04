use super::*;
use eqiora_core::{ScalarDomain, ValueFrame};
use eqiora_ir::{ComponentScalarization, ScalarOperatorIr, ScalarSymbolCoordinate};
use eqiora_schema::kernel::typing::{ExpressionType, RootContract, TypedResidual};
use eqiora_schema::kernel::{ActivationKind, ExprDagBuilder, ExprNode, FieldRole, KernelNode};
use std::collections::BTreeSet;

/// Admit one static finite Field problem with exact constraint tolerances or margins.
///
/// # Errors
/// Rejects dynamic/spatial/discrete profiles, nonaffine active-set operands, non-square systems,
/// missing/foreign/wrong-unit tolerance entries and excessive active-set enumeration.
pub(crate) fn lower_finite_constraints(
    kernel: &KernelProgram,
    enforcement: Option<&FiniteConstraintEnforcement>,
) -> Result<FiniteConstraintProblem, Diagnostic> {
    let max_active_sets = enforcement.and_then(FiniteConstraintEnforcement::max_active_sets);
    let strict_interior = enforcement.is_some_and(FiniteConstraintEnforcement::is_strict_interior);
    let mut symbols = Vec::new();
    let mut dimensions = Vec::new();
    let mut coordinates = Vec::new();
    let mut bindings = Vec::new();
    for node in kernel.nodes() {
        match node {
            KernelNode::Field(field) => {
                if kernel.edges().iter().any(|edge| {
                    edge.from() == field.id().erase()
                        && edge.kind() == eqiora_graph::EdgeKind::DefinedOn
                }) {
                    return Err(invalid(
                        "finite execution cannot erase a Field spatial support",
                    ));
                }
                let value = field.value_type();
                if field.role() != FieldRole::Variable
                    || !matches!(
                        value.scalar_domain(),
                        ScalarDomain::Real | ScalarDomain::Complex
                    )
                    || value.frame() != ValueFrame::Invariant
                    || (value.array_rank() != value.shape().rank()
                        && value.finite_bases().next().is_none())
                    || (strict_interior
                        && (value.scalar_domain() != ScalarDomain::Real
                            || !value.shape().is_scalar()))
                {
                    return Err(invalid(
                        "finite affine execution requires invariant numeric Fields; Newton requires real scalars",
                    ));
                }
                symbols.push(SymbolRef::Field(field.id()));
                dimensions.push(value.dimension());
                let count = value
                    .shape()
                    .component_count()
                    .and_then(|count| {
                        count.checked_mul(if value.scalar_domain() == ScalarDomain::Complex {
                            2
                        } else {
                            1
                        })
                    })
                    .ok_or_else(|| invalid("Field coordinate count overflow"))?;
                if coordinates.len().saturating_add(count) > 256 {
                    return Err(invalid(
                        "finite constraint profile permits at most 256 real coordinates",
                    ));
                }
                coordinates.extend(ScalarSymbolCoordinate::for_value(
                    SymbolRef::Field(field.id()),
                    value,
                )?);
            }
            KernelNode::Parameter(parameter) => {
                let value = parameter.value();
                if strict_interior && value.real_scalar_value().is_none() {
                    return Err(invalid("finite Newton Parameters must be real scalars"));
                }
                if value.value_type().frame() != ValueFrame::Invariant
                    || (value.value_type().array_rank() != value.value_type().shape().rank()
                        && value.value_type().finite_bases().next().is_none())
                {
                    return Err(invalid(
                        "finite Parameters require invariant numeric channels",
                    ));
                }
                let count = value
                    .component_count()
                    .checked_mul(
                        if value.value_type().scalar_domain() == ScalarDomain::Complex {
                            2
                        } else {
                            1
                        },
                    )
                    .ok_or_else(|| invalid("Parameter coordinate count overflow"))?;
                if bindings.len().saturating_add(count) > 65_536 {
                    return Err(invalid(
                        "finite Parameter coordinates exceed the 65536-component work bound",
                    ));
                }
                for coordinate in ScalarSymbolCoordinate::for_value(
                    SymbolRef::Parameter(parameter.id()),
                    value.value_type(),
                )? {
                    let scalar = coordinates::component(value, &coordinate)?;
                    bindings.push((coordinate, scalar));
                }
            }
            KernelNode::Domain(domain)
                if matches!(
                    domain.kind(),
                    eqiora_schema::kernel::DomainKind::CoordinateInterval { .. }
                        | eqiora_schema::kernel::DomainKind::CoordinateProduct { .. }
                ) => {}
            // Nominal finite sets retain the identity of already expanded sums and arrays;
            // they add no solve coordinate. Field and operand admission stay independent.
            KernelNode::Relation(_)
            | KernelNode::Observable(_)
            | KernelNode::FiniteSpace(_)
            | KernelNode::IndexSet(_) => {}
            KernelNode::Activation(activation)
                if matches!(activation.kind(), ActivationKind::Continuous) => {}
            _ => {
                return Err(invalid(
                    "finite constraint profile rejects clocks, events, spatial support, Ports and non-scalar semantic owners",
                ));
            }
        }
    }
    if symbols.is_empty() {
        return Err(invalid(
            "finite constraint problem has no scalar Field unknowns",
        ));
    }
    let mut relations = Vec::new();
    let mut expected = BTreeSet::new();
    let mut equality_count = 0usize;
    let mut complementarity_count = 0usize;
    let mut expression_nodes = 0usize;
    let mut affine_node_work = 0usize;
    for node in kernel.nodes() {
        let KernelNode::Relation(relation) = node else {
            continue;
        };
        if kernel.edges().iter().any(|edge| {
            edge.from() == relation.id().erase() && edge.kind() == eqiora_graph::EdgeKind::AppliesOn
        }) {
            return Err(invalid(
                "finite execution cannot erase a Relation spatial support",
            ));
        }
        let conditions = relation.conditions().ok_or_else(|| {
            invalid("finite constraint execution does not reinterpret a conservation Law as a condition Relation")
        })?;
        if relation.is_initial() {
            return Err(invalid(
                "finite static constraints do not admit initialization-only Relations",
            ));
        }
        let expression = super::observables::expand(kernel, relation.expression())?;
        expression_nodes = expression_nodes
            .checked_add(expression.nodes().len())
            .ok_or_else(|| invalid("finite expression budget overflow"))?;
        if expression_nodes > 65_536 {
            return Err(invalid(
                "finite constraint profile permits at most 65536 expression nodes",
            ));
        }
        if expression.nodes().iter().any(|node| matches!(node, ExprNode::Symbol(symbol) if !matches!(symbol, SymbolRef::Field(_) | SymbolRef::Parameter(_)))) {
            return Err(invalid("finite static constraints reject time, derivatives and activation history"));
        }
        let typed = TypedResidual::<eqiora_core::RawId>::infer(
            expression.clone(),
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
        for (ordinal, (kind, (left, right))) in conditions
            .iter()
            .zip(
                expression
                    .roots()
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|sides| (sides[0], sides[1])),
            )
            .enumerate()
        {
            let left = typed.node_type(left).expect("admitted root type");
            let right = typed.node_type(right).expect("admitted root type");
            for operand in [left, right] {
                if operand.support.is_some()
                    || !matches!(
                        operand.value_type.scalar_domain(),
                        ScalarDomain::Real | ScalarDomain::Complex
                    )
                    || operand.frame() != ValueFrame::Invariant
                    || (operand.value_type.array_rank() != operand.shape().rank()
                        && operand.value_type.finite_bases().next().is_none())
                    || ((*kind != RelationConditionKind::Equality || strict_interior)
                        && (operand.value_type.scalar_domain() != ScalarDomain::Real
                            || !operand.shape().is_scalar()))
                {
                    return Err(invalid(
                        "finite equalities require invariant numeric operands; ordered constraints and Newton require real scalars",
                    ));
                }
            }
            operand_dimensions.push((left.dimension(), right.dimension()));
            match kind {
                RelationConditionKind::Equality => {
                    let value_type = eqiora_schema::kernel::typing::additive(left, right)
                        .map_err(|error| invalid(error.to_string()))?
                        .value_type;
                    let count = value_type
                        .shape()
                        .component_count()
                        .and_then(|count| {
                            count.checked_mul(
                                if value_type.scalar_domain() == ScalarDomain::Complex {
                                    2
                                } else {
                                    1
                                },
                            )
                        })
                        .ok_or_else(|| invalid("equality coordinate count overflow"))?;
                    equality_count = equality_count
                        .checked_add(count)
                        .ok_or_else(|| invalid("equality coordinate count overflow"))?;
                }
                RelationConditionKind::Inequality | RelationConditionKind::Complementarity => {
                    let reference = ConstraintRef::new(
                        relation.id(),
                        u32::try_from(ordinal)
                            .map_err(|_| invalid("condition ordinal overflow"))?,
                    );
                    expected.insert(reference);
                    let tolerance = enforcement
                        .and_then(|policy| policy.tolerance(reference))
                        .ok_or_else(|| {
                            invalid("each exact Model constraint requires an explicit tolerance")
                        })?;
                    if tolerance.left().dim() != left.dimension() {
                        return Err(invalid(
                            "constraint left tolerance has the wrong physical dimension",
                        ));
                    }
                    if *kind == RelationConditionKind::Complementarity {
                        if strict_interior {
                            return Err(invalid(
                                "strict-interior execution rejects complementarity",
                            ));
                        }
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
        if equality_count + complementarity_count > coordinates.len() {
            return Err(invalid(
                "finite active-set branches must be square: too many equality coordinates",
            ));
        }
        if !strict_interior {
            // Every scalarized operand row may traverse the original DAG; a
            // complex node may need both parts. Charge that expansion before
            // constructing rows, within the existing finite node-work budget.
            let operand_coordinates = typed
                .expression()
                .roots()
                .iter()
                .try_fold(0usize, |total, root| {
                    let ty = &typed.node_type(*root).expect("admitted operand").value_type;
                    let count = ty.shape().component_count()?.checked_mul(
                        if ty.scalar_domain() == ScalarDomain::Complex {
                            2
                        } else {
                            1
                        },
                    )?;
                    total.checked_add(count)
                })
                .ok_or_else(|| invalid("finite operand coordinate count overflow"))?;
            let parts = if typed
                .node_types()
                .iter()
                .any(|ty| ty.value_type.scalar_domain() == ScalarDomain::Complex)
            {
                2
            } else {
                1
            };
            affine_node_work = affine_node_work.saturating_add(
                expression
                    .nodes()
                    .len()
                    .saturating_mul(operand_coordinates)
                    .saturating_mul(parts),
            );
            if affine_node_work > 16_777_216 {
                return Err(invalid(
                    "finite component expansion exceeds 16777216 node evaluations",
                ));
            }
        }
        // Prove every original operand affine before considering any active branch.
        // A branch must never hide a nonlinear inactive operand.
        if strict_interior {
            ScalarOperatorIr::lower(&expression)?;
        } else {
            for row in ComponentScalarization::lower(&typed)?.rows() {
                row.bind_affine(&coordinates, &bindings).map_err(|error| {
                    invalid(format!(
                        "finite condition operands are not affine: {}",
                        error.message()
                    ))
                })?;
            }
        }
        relations.push(RelationOperands {
            id: relation.id(),
            expression,
            conditions: conditions.to_vec(),
            dimensions: operand_dimensions,
        });
    }
    if expected.len() != enforcement.map_or(0, |policy| policy.tolerances().len()) {
        return Err(invalid(
            "enforcement names a foreign or non-constraint Relation condition",
        ));
    }
    if complementarity_count > 16
        || max_active_sets.is_some_and(|budget| (1u32 << complementarity_count) > budget)
    {
        return Err(invalid(
            "complete active-set enumeration exceeds its explicit bounded budget",
        ));
    }
    let node_work = if strict_interior {
        expression_nodes
    } else {
        affine_node_work
    };
    if node_work.saturating_mul(1usize << complementarity_count) > 16_777_216 {
        return Err(invalid(
            "finite active-set expression work exceeds 16777216 node evaluations",
        ));
    }
    if equality_count + complementarity_count != coordinates.len() {
        return Err(invalid(
            "finite active-set branches must be square: equality coordinates plus complementarity pairs equal unknown coordinates",
        ));
    }
    Ok(FiniteConstraintProblem {
        kernel: kernel.clone(),
        symbols,
        dimensions,
        coordinates,
        bindings,
        parameter_candidates: Vec::new(),
        relations,
        enforcement: enforcement.cloned(),
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
