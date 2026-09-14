use super::*;

pub(super) fn recognize_balance(
    program: &KernelProgram,
    relation: RawId,
    field: RawId,
    dimensions: usize,
    bounds: &[[f64; 2]],
) -> Result<
    (
        DimExponents,
        Option<ScalarStorageMeaning>,
        ScalarFluxMeaning,
        Option<VolumetricSourceMeaning>,
    ),
    Diagnostic,
> {
    if let Some(KernelNode::Relation(definition)) = program.node(relation)
        && let eqiora_schema::kernel::RelationMeaning::Conservation(terms) = definition.meaning()
    {
        return super::retained::retained_balance(
            program, relation, field, dimensions, bounds, *terms,
        );
    }
    require_continuous_relation(program, relation)?;
    let typed = typed_relation(program, relation)?;
    let expression = typed.expression();
    let root = unique_root(expression, relation)?;
    let view = AdditiveResidualView::derive(expression, root, relation)?;
    if !(1..=3).contains(&view.leaves().len()) {
        return Err(view.mismatch(
            "scalar balance requires flux divergence, optional storage, and optional source",
        ));
    }
    let divergence = exactly_one(
        &view,
        |leaf| matches!(expression.node(leaf.value()), Some(ExprNode::Divergence(_))),
        "scalar balance requires exactly one divergence term",
    )?;
    let ExprNode::Divergence(flux_expression) = expression
        .node(divergence.value())
        .expect("classified divergence")
    else {
        unreachable!()
    };
    let coefficient = lower_flux_coefficient(
        program,
        expression,
        *flux_expression,
        field,
        relation,
        dimensions,
    )?;
    validate_positive_affine_coefficient(&coefficient, bounds, relation)?;

    let storage_leaf = view
        .leaves()
        .iter()
        .find(|leaf| contains_exact_derivative(expression, leaf.value(), field));
    let storage = storage_leaf
        .map(|leaf| {
            recognize_storage(
                program,
                expression,
                leaf.value(),
                field,
                relation,
                dimensions,
            )
        })
        .transpose()?;
    if let Some(storage_leaf) = storage_leaf
        && storage_leaf.sign() == divergence.sign()
    {
        return Err(view.mismatch("storage and constitutive divergence must have opposite signs"));
    }
    let remaining = view
        .leaves()
        .iter()
        .filter(|leaf| leaf.value() != divergence.value())
        .filter(|leaf| storage_leaf.is_none_or(|storage| leaf.value() != storage.value()))
        .collect::<Vec<_>>();
    let source = match remaining.as_slice() {
        [] => None,
        [source] if source.sign() == divergence.sign() => {
            if contains_state_symbol(expression, source.value()) {
                return Err(lowering_error(
                    relation,
                    "first scalar-conservation source must be independent of state",
                ));
            }
            let lowered = spatial_expression::lower(
                program,
                expression,
                source.value(),
                relation,
                dimensions,
            )?;
            let dimension = node_dimension(&typed, source.value(), relation)?;
            Some(VolumetricSourceMeaning {
                expression: lowered,
                dimension,
                integrated_dimension: integrate_dimension(dimension, dimensions, relation)?,
                lineage: ScalarTermLineage {
                    relation,
                    expression: source.value(),
                },
            })
        }
        _ => {
            return Err(view.mismatch(
                "scalar source must be the sole non-storage term and share the divergence sign",
            ));
        }
    };
    let balance_dimension = node_dimension(&typed, root, relation)?;
    for leaf in view.leaves() {
        if node_dimension(&typed, leaf.value(), relation)? != balance_dimension {
            return Err(view.mismatch("every scalar balance term must have the root dimension"));
        }
    }
    Ok((
        balance_dimension,
        storage,
        ScalarFluxMeaning {
            coefficient,
            lineage: ScalarTermLineage {
                relation,
                expression: *flux_expression,
            },
        },
        source,
    ))
}

pub(super) fn recognize_storage(
    program: &KernelProgram,
    expression: &ExprDag,
    value: ExprId,
    field: RawId,
    owner: RawId,
    dimensions: usize,
) -> Result<ScalarStorageMeaning, Diagnostic> {
    let coefficient_expression =
        strip_derivative_factor(expression, value, field).ok_or_else(|| {
            lowering_error(
                owner,
                "storage must be one scalar coefficient times the exact Field derivative",
            )
        })?;
    storage_coefficient(
        program,
        expression,
        coefficient_expression,
        value,
        owner,
        dimensions,
    )
}

pub(super) fn storage_coefficient(
    program: &KernelProgram,
    expression: &ExprDag,
    coefficient_expression: Option<ExprId>,
    value: ExprId,
    owner: RawId,
    dimensions: usize,
) -> Result<ScalarStorageMeaning, Diagnostic> {
    let coefficient = match coefficient_expression {
        Some(value) => spatial_expression::lower(program, expression, value, owner, dimensions)?,
        None => ScalarSpatialExpression::constant(dimensions, 1.0),
    };
    let Some(coefficient_value) = coefficient.constant_value() else {
        return Err(lowering_error(
            owner,
            "first scalar-conservation storage coefficient must be constant",
        ));
    };
    if !coefficient_value.is_finite() || coefficient_value <= 0.0 {
        return Err(lowering_error(
            owner,
            "scalar-conservation storage coefficient must be finite and strictly positive",
        ));
    }
    Ok(ScalarStorageMeaning {
        coefficient,
        lineage: ScalarTermLineage {
            relation: owner,
            expression: value,
        },
    })
}
