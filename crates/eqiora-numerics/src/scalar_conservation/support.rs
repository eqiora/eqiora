use super::*;

pub(super) fn typed_relation(
    program: &KernelProgram,
    relation: RawId,
) -> Result<TypedResidual<RawId>, Diagnostic> {
    let relation = relation
        .downcast::<kinds::Relation>()
        .ok_or_else(|| lowering_error(relation, "scalar-conservation owner is not a Relation"))?;
    program
        .typed_relation_residual(relation)
        .map_err(|diagnostics| {
            diagnostics
                .into_iter()
                .next()
                .unwrap_or_else(|| lowering_error(relation.erase(), "typed residual replay failed"))
        })
}

pub(super) fn exactly_one<'a>(
    view: &'a AdditiveResidualView,
    predicate: impl Fn(&SignedOpaqueLeaf) -> bool,
    expectation: &str,
) -> Result<&'a SignedOpaqueLeaf, Diagnostic> {
    let found = view
        .leaves()
        .iter()
        .filter(|leaf| predicate(leaf))
        .collect::<Vec<_>>();
    let [value] = found.as_slice() else {
        return Err(view.mismatch(expectation));
    };
    Ok(*value)
}

pub(super) fn node_dimension(
    typed: &TypedResidual<RawId>,
    value: ExprId,
    owner: RawId,
) -> Result<DimExponents, Diagnostic> {
    typed
        .node_type(value)
        .map(|value| value.dimension())
        .ok_or_else(|| {
            lowering_error(
                owner,
                format!("typed scalar term {} is missing", value.index()),
            )
        })
}

pub(super) fn integrate_dimension(
    dimension: DimExponents,
    spatial_dimensions: usize,
    owner: RawId,
) -> Result<DimExponents, Diagnostic> {
    let increment = i32::try_from(spatial_dimensions)
        .map_err(|_| lowering_error(owner, "integrated source dimension is unrepresentable"))?;
    let measure = DimExponents::from_integers([0, increment, 0, 0, 0, 0, 0])
        .ok_or_else(|| lowering_error(owner, "integration measure dimension is unrepresentable"))?;
    dimension
        .mul(measure)
        .ok_or_else(|| lowering_error(owner, "integrated source dimension overflows"))
}

pub(super) fn contains_exact_derivative(expression: &ExprDag, value: ExprId, field: RawId) -> bool {
    match expression.node(value) {
        Some(ExprNode::Symbol(SymbolRef::Derivative(id, std::num::NonZeroU32::MIN))) => {
            id.erase() == field
        }
        Some(ExprNode::Mul(left, right)) => {
            contains_exact_derivative(expression, *left, field)
                || contains_exact_derivative(expression, *right, field)
        }
        _ => false,
    }
}

pub(super) fn strip_derivative_factor(
    expression: &ExprDag,
    value: ExprId,
    field: RawId,
) -> Option<Option<ExprId>> {
    match expression.node(value) {
        Some(ExprNode::Symbol(SymbolRef::Derivative(id, std::num::NonZeroU32::MIN)))
            if id.erase() == field =>
        {
            Some(None)
        }
        Some(ExprNode::Mul(left, right))
            if contains_exact_derivative(expression, *left, field)
                && !contains_exact_derivative(expression, *right, field) =>
        {
            matches!(
                strip_derivative_factor(expression, *left, field),
                Some(None)
            )
            .then_some(Some(*right))
        }
        Some(ExprNode::Mul(left, right))
            if contains_exact_derivative(expression, *right, field)
                && !contains_exact_derivative(expression, *left, field) =>
        {
            matches!(
                strip_derivative_factor(expression, *right, field),
                Some(None)
            )
            .then_some(Some(*left))
        }
        _ => None,
    }
}

pub(super) fn contains_state_symbol(expression: &ExprDag, root: ExprId) -> bool {
    fn visit(expression: &ExprDag, value: ExprId, seen: &mut BTreeSet<ExprId>) -> bool {
        if !seen.insert(value) {
            return false;
        }
        match expression.node(value) {
            Some(ExprNode::Symbol(
                SymbolRef::Field(_)
                | SymbolRef::Derivative(_, _)
                | SymbolRef::Pre(_)
                | SymbolRef::Next(_),
            )) => true,
            Some(
                ExprNode::Neg(value)
                | ExprNode::PowI(value, _)
                | ExprNode::UnaryMath(_, value)
                | ExprNode::Gradient(value)
                | ExprNode::Divergence(value)
                | ExprNode::SymmetricPart(value)
                | ExprNode::IsotropicLift(value)
                | ExprNode::Trace(value)
                | ExprNode::NormalComponent(value),
            ) => visit(expression, *value, seen),
            Some(
                ExprNode::Add(left, right)
                | ExprNode::Sub(left, right)
                | ExprNode::Mul(left, right)
                | ExprNode::Div(left, right),
            ) => visit(expression, *left, seen) || visit(expression, *right, seen),
            Some(ExprNode::PureOperatorApplication(application)) => application
                .arguments()
                .iter()
                .any(|value| visit(expression, *value, seen)),
            _ => false,
        }
    }
    visit(expression, root, &mut BTreeSet::new())
}

pub(super) fn is_trace(expression: &ExprDag, value: ExprId, field: RawId) -> bool {
    matches!(expression.node(value), Some(ExprNode::Trace(inner)) if is_field(expression, *inner, field))
}

pub(super) fn is_field(expression: &ExprDag, value: ExprId, field: RawId) -> bool {
    matches!(expression.node(value), Some(ExprNode::Symbol(SymbolRef::Field(id))) if id.erase() == field)
}

pub(super) fn robin_coefficient(
    expression: &ExprDag,
    value: ExprId,
    field: RawId,
) -> Option<(ExprId, ExprId)> {
    let ExprNode::Mul(left, right) = expression.node(value)? else {
        return None;
    };
    if is_trace(expression, *left, field) {
        Some((*right, *left))
    } else if is_trace(expression, *right, field) {
        Some((*left, *right))
    } else {
        None
    }
}

pub(super) fn connection_of(program: &KernelProgram, port: RawId) -> Result<RawId, Diagnostic> {
    let connections = program
        .edges()
        .iter()
        .filter(|edge| edge.kind() == EdgeKind::Connects && edge.to() == port)
        .map(|edge| edge.from())
        .collect::<Vec<_>>();
    let [connection] = connections.as_slice() else {
        return Err(lowering_error(
            port,
            format!(
                "scalar interface Port requires exactly one Connection, found {}",
                connections.len()
            ),
        ));
    };
    Ok(*connection)
}

pub(super) fn require_continuous_relation(
    program: &KernelProgram,
    relation: RawId,
) -> Result<(), Diagnostic> {
    let activations = program
        .edges()
        .iter()
        .filter(|edge| edge.kind() == EdgeKind::Activates && edge.to() == relation)
        .filter_map(|edge| match program.node(edge.from()) {
            Some(KernelNode::Activation(activation)) => Some(activation),
            _ => None,
        })
        .collect::<Vec<_>>();
    if activations.len() == 1 && matches!(activations[0].kind(), ActivationKind::Continuous) {
        Ok(())
    } else {
        Err(lowering_error(
            relation,
            "scalar-conservation meaning requires exactly one continuous Activation",
        ))
    }
}
