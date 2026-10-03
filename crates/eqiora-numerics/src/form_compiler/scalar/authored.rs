mod polynomial;

use eqiora_compiler::{AuthoredFormExpressionV1, AuthoredFormulationProjection};
use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_sem::KernelProgram;

use super::{DerivedScalarGalerkinForm, typed_relation};

pub(crate) fn admit(
    projection: &AuthoredFormulationProjection,
    program: &KernelProgram,
    derived: &DerivedScalarGalerkinForm,
) -> Result<(), Diagnostic> {
    let mut variation_dimensions = Vec::new();
    for (_, left, right) in projection.equations() {
        for expression in [left, right] {
            if let AuthoredFormExpressionV1::Variation {
                functional_ulid, ..
            } = expression
            {
                let id = functional_ulid
                    .parse::<ulid::Ulid>()
                    .map(eqiora_core::Id::<eqiora_core::entity::kinds::Observable>::from_ulid)
                    .map_err(|_| {
                        rejection_with(projection, "variation Observable identity is invalid")
                    })?;
                let Some(eqiora_schema::kernel::KernelNode::Observable(functional)) =
                    program.node(id.erase())
                else {
                    return Err(rejection_with(
                        projection,
                        "variation Observable is outside the live Model",
                    ));
                };
                let typed = program.typed_observable(id).map_err(|_| {
                    rejection_with(projection, "variation energy has invalid live types")
                })?;
                expression.check_functional_variation(functional, &typed)?;
                variation_dimensions.push(functional.value_type().dimension());
            }
        }
    }
    let has_variation = !variation_dimensions.is_empty();
    let test_dimension = if has_variation {
        let Some(eqiora_schema::kernel::KernelNode::Field(field)) = program.node(derived.field)
        else {
            return Err(rejection_with(
                projection,
                "variation trial is not a live Field",
            ));
        };
        field.dimension()
    } else {
        eqiora_core::DimExponents::DIMENSIONLESS
    };
    derived
        .certificate
        .replay_authored_restriction(projection, test_dimension)
        .map_err(|message| rejection_with(projection, message))?;
    let expected_domain = derived.domain.ulid().to_string();
    let expected_trial = derived.field.ulid().to_string();
    let typed = typed_relation(program, derived.volume_relation)?;
    let dag = typed.expression();
    if has_variation {
        let invalid_dimension = || {
            rejection_with(
                projection,
                "variation dimension differs from the strong-law test pairing",
            )
        };
        let spatial_dimension =
            i32::try_from(derived.dimension).map_err(|_| invalid_dimension())?;
        let measure =
            eqiora_core::DimExponents::from_integers([0, spatial_dimension, 0, 0, 0, 0, 0])
                .ok_or_else(invalid_dimension)?;
        let paired_dimension = typed
            .node_type(dag.roots()[0])
            .ok_or_else(invalid_dimension)?
            .dimension()
            .mul(test_dimension)
            .and_then(|dimension| dimension.mul(measure))
            .ok_or_else(invalid_dimension)?;
        if variation_dimensions
            .iter()
            .any(|dimension| *dimension != paired_dimension)
        {
            return Err(invalid_dimension());
        }
    }
    let test = AuthoredFormExpressionV1::Test {
        field_ulid: expected_trial,
    };
    let flux = AuthoredFormExpressionV1::from_expression(dag, derived.volume_nodes.bilinear_flux)?
        .ok_or_else(|| {
            rejection_with(
                projection,
                "source expression exceeds the scalar-primal inventory",
            )
        })?;
    let flux =
        if derived.volume_nodes.divergence_sign == super::super::vocabulary::WeakSign::Negative {
            let (value, negative) = product_sign(flux);
            if negative {
                value
            } else {
                AuthoredFormExpressionV1::Neg {
                    value: Box::new(value),
                }
            }
        } else {
            flux
        };
    let left = AuthoredFormExpressionV1::Integrate {
        domain_ulid: expected_domain.clone(),
        integrand: Box::new(AuthoredFormExpressionV1::Dot {
            left: Box::new(AuthoredFormExpressionV1::Gradient {
                value: Box::new(test.clone()),
            }),
            right: Box::new(flux),
        }),
    };
    let right = AuthoredFormExpressionV1::Integrate {
        domain_ulid: expected_domain,
        integrand: Box::new(AuthoredFormExpressionV1::Mul {
            left: Box::new(test),
            right: Box::new(
                AuthoredFormExpressionV1::from_expression(dag, derived.volume_nodes.source)?
                    .ok_or_else(|| {
                        rejection_with(
                            projection,
                            "source expression exceeds the scalar-primal inventory",
                        )
                    })?,
            ),
        }),
    };
    if has_variation {
        return polynomial::matches_variation(projection, derived.dimension, &left, &right)
            .then_some(())
            .ok_or_else(|| {
                rejection_with(
                    projection,
                    "functional variation differs from the admitted strong-law weak residual",
                )
            });
    }
    if !equivalent(&projection.equations()[0].1, &left) {
        return Err(rejection_with(
            projection,
            "left bilinear term, coefficient, sign, or contraction differs from the admitted primal form",
        ));
    }
    if !equivalent(&projection.equations()[0].2, &right) {
        return Err(rejection_with(
            projection,
            "right source term, sign, or test pairing differs from the admitted primal form",
        ));
    }
    Ok(())
}

// Exact unary-sign movement through multiplication; no coefficient substitution or sampling.
fn product_sign(value: AuthoredFormExpressionV1) -> (AuthoredFormExpressionV1, bool) {
    use AuthoredFormExpressionV1 as E;
    match value {
        E::Neg { value } => {
            let (value, sign) = product_sign(*value);
            (value, !sign)
        }
        E::Mul { left, right } => {
            let (left, left_sign) = product_sign(*left);
            let (right, right_sign) = product_sign(*right);
            (
                E::Mul {
                    left: Box::new(left),
                    right: Box::new(right),
                },
                left_sign ^ right_sign,
            )
        }
        value => (value, false),
    }
}

pub(crate) fn equivalent(
    left: &AuthoredFormExpressionV1,
    right: &AuthoredFormExpressionV1,
) -> bool {
    use AuthoredFormExpressionV1 as Expression;

    match (left, right) {
        (Expression::Add { .. }, Expression::Add { .. }) => {
            multiset(left, true) == multiset(right, true)
        }
        (Expression::Mul { .. }, Expression::Mul { .. }) => {
            multiset(left, false) == multiset(right, false)
        }
        (Expression::Number { value: a }, Expression::Number { value: b }) => {
            a.to_bits() == b.to_bits()
        }
        (Expression::Field { ulid: a }, Expression::Field { ulid: b })
        | (Expression::Parameter { ulid: a }, Expression::Parameter { ulid: b }) => a == b,
        (Expression::Coordinate { axis: a }, Expression::Coordinate { axis: b }) => a == b,
        (Expression::Test { field_ulid: a }, Expression::Test { field_ulid: b }) => a == b,
        (Expression::Neg { value: a }, Expression::Neg { value: b })
        | (Expression::Gradient { value: a }, Expression::Gradient { value: b })
        | (Expression::Sin { value: a }, Expression::Sin { value: b }) => equivalent(a, b),
        (
            Expression::Sub {
                left: al,
                right: ar,
            },
            Expression::Sub {
                left: bl,
                right: br,
            },
        )
        | (
            Expression::Div {
                left: al,
                right: ar,
            },
            Expression::Div {
                left: bl,
                right: br,
            },
        )
        | (
            Expression::Dot {
                left: al,
                right: ar,
            },
            Expression::Dot {
                left: bl,
                right: br,
            },
        ) => equivalent(al, bl) && equivalent(ar, br),
        (
            Expression::Pow {
                base: a,
                exponent: ae,
            },
            Expression::Pow {
                base: b,
                exponent: be,
            },
        ) => ae == be && equivalent(a, b),
        (
            Expression::Integrate {
                domain_ulid: ad,
                integrand: a,
            },
            Expression::Integrate {
                domain_ulid: bd,
                integrand: b,
            },
        ) => ad == bd && equivalent(a, b),
        _ => false,
    }
}

fn multiset(value: &AuthoredFormExpressionV1, addition: bool) -> Vec<String> {
    fn collect<'a>(
        value: &'a AuthoredFormExpressionV1,
        addition: bool,
        values: &mut Vec<&'a AuthoredFormExpressionV1>,
    ) {
        match (addition, value) {
            (true, AuthoredFormExpressionV1::Add { left, right })
            | (false, AuthoredFormExpressionV1::Mul { left, right }) => {
                collect(left, addition, values);
                collect(right, addition, values);
            }
            _ => values.push(value),
        }
    }
    let mut values = Vec::new();
    collect(value, addition, &mut values);
    let mut values = values
        .into_iter()
        .map(|value| serde_json::to_string(value).expect("wire expression serializes"))
        .collect::<Vec<_>>();
    values.sort();
    values
}

fn rejection(message: &str) -> Diagnostic {
    Diagnostic::error(
        codes::INVALID_DISCRETIZATION,
        format!("authored scalar-primal Formulation rejected: {message}"),
    )
}

fn rejection_with(projection: &AuthoredFormulationProjection, message: &str) -> Diagnostic {
    rejection(&format!(
        "{message} (source identity {})",
        projection.source_identity()
    ))
}
