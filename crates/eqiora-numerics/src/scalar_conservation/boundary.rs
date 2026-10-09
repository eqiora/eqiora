use super::*;

pub(super) fn recognize_exterior_law(
    program: &KernelProgram,
    relation: RawId,
    field: RawId,
    volume_coefficient: &ScalarSpatialExpression<f64>,
    dimensions: usize,
    physical_law: bool,
) -> Result<ScalarExteriorLaw, Diagnostic> {
    if physical_law {
        return recognize_exterior_law_oriented(
            program,
            relation,
            field,
            dimensions,
            |expression, normal| {
                physical_conormal_orientation(
                    program,
                    expression,
                    normal,
                    field,
                    volume_coefficient,
                    relation,
                    dimensions,
                )
            },
        );
    }
    recognize_exterior_law_with_flux(
        program,
        relation,
        field,
        dimensions,
        |expression, normal| {
            validate_normal_flux(
                program,
                expression,
                normal,
                field,
                volume_coefficient,
                relation,
                dimensions,
            )
        },
    )
}

pub(crate) fn recognize_exterior_law_with_flux(
    program: &KernelProgram,
    relation: RawId,
    field: RawId,
    dimensions: usize,
    check_flux: impl FnOnce(&ExprDag, ExprId) -> Result<(), Diagnostic>,
) -> Result<ScalarExteriorLaw, Diagnostic> {
    recognize_exterior_law_oriented(
        program,
        relation,
        field,
        dimensions,
        |expression, normal| {
            check_flux(expression, normal)?;
            Ok(false)
        },
    )
}

fn recognize_exterior_law_oriented(
    program: &KernelProgram,
    relation: RawId,
    field: RawId,
    dimensions: usize,
    check_flux: impl FnOnce(&ExprDag, ExprId) -> Result<bool, Diagnostic>,
) -> Result<ScalarExteriorLaw, Diagnostic> {
    require_continuous_relation(program, relation)?;
    let typed = typed_relation(program, relation)?;
    let expression = typed.expression();
    let root = unique_root(expression, relation)?;
    let view = AdditiveResidualView::derive(expression, root, relation)?;
    let trace = view
        .leaves()
        .iter()
        .find(|leaf| is_trace(expression, leaf.value(), field));
    let normal = view.leaves().iter().find(|leaf| {
        matches!(
            expression.node(leaf.value()),
            Some(ExprNode::NormalComponent { .. })
        )
    });
    let robin = view.leaves().iter().find_map(|leaf| {
        robin_coefficient(expression, leaf.value(), field)
            .map(|(coefficient, trace)| (leaf, coefficient, trace))
    });
    if trace.is_some() && (normal.is_some() || robin.is_some()) {
        return Err(view.mismatch("boundary law cannot prescribe trace and flux simultaneously"));
    }
    let reversed_conormal = normal
        .map(|normal| check_flux(expression, normal.value()))
        .transpose()?
        .unwrap_or(false);
    let normal_sign = normal.map(|normal| {
        if reversed_conormal {
            match normal.sign() {
                crate::additive_residual::AdditiveSign::Positive => {
                    crate::additive_residual::AdditiveSign::Negative
                }
                crate::additive_residual::AdditiveSign::Negative => {
                    crate::additive_residual::AdditiveSign::Positive
                }
            }
        } else {
            normal.sign()
        }
    });
    let operator_sign = trace
        .map(|leaf| leaf.sign())
        .or(normal_sign)
        .or_else(|| robin.map(|(leaf, _, _)| leaf.sign()))
        .ok_or_else(|| view.mismatch("boundary law requires trace, normal flux, or Robin terms"))?;
    let operator_ids = [
        trace.map(|leaf| leaf.value()),
        normal.map(|leaf| leaf.value()),
        robin.map(|(leaf, _, _)| leaf.value()),
    ];
    let values = view
        .leaves()
        .iter()
        .filter(|leaf| !operator_ids.contains(&Some(leaf.value())))
        .collect::<Vec<_>>();
    let (value, datum_expression, datum_negative) = match values.as_slice() {
        [] => (
            ScalarSpatialExpression::constant(dimensions, 0.0),
            None,
            false,
        ),
        [value] if value.sign() != operator_sign || reversed_conormal => {
            if contains_state_symbol(expression, value.value()) {
                return Err(lowering_error(
                    relation,
                    "boundary data must be independent of state",
                ));
            }
            (
                spatial_expression::lower(
                    program,
                    expression,
                    value.value(),
                    relation,
                    dimensions,
                )?
                .multiply(ScalarSpatialExpression::constant(
                    dimensions,
                    if value.sign() == operator_sign {
                        -1.0
                    } else {
                        1.0
                    },
                )),
                Some(value.value()),
                value.sign() == operator_sign,
            )
        }
        _ => return Err(view.mismatch("boundary data must be the sole term opposite its operator")),
    };
    match (trace, normal, robin) {
        (Some(trace), None, None) => Ok(ScalarExteriorLaw::PrescribedTrace {
            value,
            lineage: ScalarExteriorLineage {
                relation,
                operator_expression: trace.value(),
                datum_expression,
                datum_negative,
                robin_coefficient_expression: None,
                robin_trace_expression: None,
            },
        }),
        (None, Some(normal), None) if datum_expression.is_none() => {
            Ok(ScalarExteriorLaw::ZeroOutwardFlux {
                lineage: ScalarExteriorLineage {
                    relation,
                    operator_expression: normal.value(),
                    datum_expression: None,
                    datum_negative: false,
                    robin_coefficient_expression: None,
                    robin_trace_expression: None,
                },
            })
        }
        (None, Some(normal), None) => Ok(ScalarExteriorLaw::PrescribedOutwardFlux {
            value,
            lineage: ScalarExteriorLineage {
                relation,
                operator_expression: normal.value(),
                datum_expression,
                datum_negative,
                robin_coefficient_expression: None,
                robin_trace_expression: None,
            },
        }),
        (None, Some(normal), Some((robin, coefficient_expression, trace_expression)))
            if normal_sign == Some(robin.sign()) =>
        {
            let coefficient = spatial_expression::lower::<f64>(
                program,
                expression,
                coefficient_expression,
                relation,
                dimensions,
            )?;
            let Some(coefficient_value) = coefficient.constant_value() else {
                return Err(lowering_error(
                    relation,
                    "first scalar-conservation Robin trace coefficient must be constant",
                ));
            };
            if !coefficient_value.is_finite() || coefficient_value <= 0.0 {
                return Err(lowering_error(
                    relation,
                    "Robin trace coefficient must be finite and strictly positive",
                ));
            }
            Ok(ScalarExteriorLaw::Robin {
                trace_coefficient: coefficient,
                value,
                lineage: ScalarExteriorLineage {
                    relation,
                    operator_expression: normal.value(),
                    datum_expression,
                    datum_negative,
                    robin_coefficient_expression: Some(coefficient_expression),
                    robin_trace_expression: Some(trace_expression),
                },
            })
        }
        _ => Err(view.mismatch("boundary law has an unsupported operator combination")),
    }
}

pub(super) fn validate_normal_flux(
    program: &KernelProgram,
    expression: &ExprDag,
    normal: ExprId,
    field: RawId,
    volume_coefficient: &ScalarSpatialExpression<f64>,
    relation: RawId,
    dimensions: usize,
) -> Result<(), Diagnostic> {
    let Some(ExprNode::NormalComponent { value: flux, .. }) = expression.node(normal) else {
        return Err(lowering_error(
            relation,
            "expected an outward normal-flux expression",
        ));
    };
    let coefficient =
        lower_flux_coefficient(program, expression, *flux, field, relation, dimensions)?;
    if !coefficient.is_same_coefficient_as(volume_coefficient) {
        return Err(lowering_error(
            relation,
            "boundary/interface flux differs from the exact volume constitutive coefficient",
        ));
    }
    Ok(())
}

pub(super) fn exact_boundaries(
    program: &KernelProgram,
    parent: RawId,
    dimensions: usize,
) -> Result<BTreeMap<(usize, BoundarySide), RawId>, Diagnostic> {
    let mut result = BTreeMap::new();
    for node in program.nodes() {
        let KernelNode::Domain(domain) = node else {
            continue;
        };
        let eqiora_schema::kernel::DomainKind::CartesianBoundary { axis, side } = domain.kind()
        else {
            continue;
        };
        if boundary_parent(program, domain.id().erase()) != Some(parent) {
            continue;
        }
        if *axis >= dimensions || result.insert((*axis, *side), domain.id().erase()).is_some() {
            return Err(lowering_error(
                domain.id().erase(),
                "scalar conservation has an out-of-range or duplicate boundary side",
            ));
        }
    }
    for axis in 0..dimensions {
        for side in [BoundarySide::Lower, BoundarySide::Upper] {
            if !result.contains_key(&(axis, side)) {
                return Err(lowering_error(
                    parent,
                    format!("scalar conservation is missing boundary axis {axis} {side:?}"),
                ));
            }
        }
    }
    Ok(result)
}

/// Convert an admitted physical flux to the diffusion realization's conormal
/// convention without changing the retained boundary equation or its lineage.
fn physical_conormal_orientation(
    program: &KernelProgram,
    expression: &ExprDag,
    normal: ExprId,
    field: RawId,
    volume_coefficient: &ScalarSpatialExpression<f64>,
    relation: RawId,
    dimensions: usize,
) -> Result<bool, Diagnostic> {
    let Some(ExprNode::NormalComponent { value: flux, .. }) = expression.node(normal) else {
        return Err(lowering_error(relation, "expected physical normal flux"));
    };
    let (coefficient, reversed) = super::retained::signed_flux_coefficient(
        program, expression, *flux, field, relation, dimensions,
    )?;
    if coefficient.is_same_coefficient_as(volume_coefficient) {
        return Ok(reversed);
    }
    Err(lowering_error(
        relation,
        "boundary flux differs from the exact physical Law constitutive coefficient",
    ))
}
