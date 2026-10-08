use crate::form_compiler::authored_polynomial as polynomial;

use eqiora_compiler::{AuthoredFormExpressionV1, AuthoredFormulationProjection};
use eqiora_core::Diagnostic;
use eqiora_core::diagnostic::codes;
use eqiora_sem::KernelProgram;

use super::{DerivedScalarGalerkinForm, typed_relation};

mod dimensions;

pub(crate) fn admit(
    projection: &AuthoredFormulationProjection,
    program: &KernelProgram,
    derived: &DerivedScalarGalerkinForm,
) -> Result<(), Diagnostic> {
    let has_variation =
        derived
            .certificate
            .replay_authored(projection, program, derived.dimension)?;
    let expected_domain = derived.domain.ulid().to_string();
    let expected_trial = derived.field.ulid().to_string();
    let typed = typed_relation(program, derived.volume_relation)?;
    dimensions::check(projection, program, derived, &typed).ok_or_else(|| {
        rejection_with(
            projection,
            "weak residual dimension differs from the strong-law test pairing",
        )
    })?;
    let dag = typed.expression();
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
    let gradient_test = AuthoredFormExpressionV1::Gradient {
        value: Box::new(test.clone()),
    };
    let conjugate = |value| {
        if derived.conjugate_test {
            AuthoredFormExpressionV1::Conjugate {
                value: Box::new(value),
            }
        } else {
            value
        }
    };
    let mut left = AuthoredFormExpressionV1::Integrate {
        domain_ulid: expected_domain.clone(),
        integrand: Box::new(AuthoredFormExpressionV1::Dot {
            left: Box::new(conjugate(gradient_test)),
            right: Box::new(flux),
        }),
    };
    let mut right = None;
    for term in &derived.volume_nodes.values {
        let mut value = AuthoredFormExpressionV1::from_expression(dag, term.source_node)?
            .ok_or_else(|| {
                rejection_with(
                    projection,
                    "source value exceeds the scalar-primal inventory",
                )
            })?;
        if term.sign == super::super::vocabulary::WeakSign::Negative {
            value = AuthoredFormExpressionV1::Neg {
                value: Box::new(value),
            };
        }
        let integral = AuthoredFormExpressionV1::Integrate {
            domain_ulid: expected_domain.clone(),
            integrand: Box::new(AuthoredFormExpressionV1::Mul {
                left: Box::new(conjugate(test.clone())),
                right: Box::new(value),
            }),
        };
        if term.trial_dependent {
            left = AuthoredFormExpressionV1::Add {
                left: Box::new(left),
                right: Box::new(integral),
            };
        } else {
            right = Some(match right {
                None => integral,
                Some(previous) => AuthoredFormExpressionV1::Add {
                    left: Box::new(previous),
                    right: Box::new(integral),
                },
            });
        }
    }
    let mut right = right.unwrap_or(AuthoredFormExpressionV1::Number { value: 0.0 });
    for boundary in &derived.boundary_roles {
        let Some((datum, negative)) = boundary.flux_data else {
            continue;
        };
        let typed = typed_relation(program, boundary.relation)?;
        let mut value = AuthoredFormExpressionV1::from_expression(typed.expression(), datum)?
            .ok_or_else(|| {
                rejection_with(
                    projection,
                    "prescribed flux exceeds the scalar-primal inventory",
                )
            })?;
        if negative {
            value = AuthoredFormExpressionV1::Neg {
                value: Box::new(value),
            };
        }
        right = AuthoredFormExpressionV1::Add {
            left: Box::new(right),
            right: Box::new(AuthoredFormExpressionV1::Integrate {
                domain_ulid: boundary.domain.ulid().to_string(),
                integrand: Box::new(AuthoredFormExpressionV1::Mul {
                    left: Box::new(conjugate(AuthoredFormExpressionV1::Trace {
                        value: Box::new(test.clone()),
                    })),
                    right: Box::new(value),
                }),
            }),
        };
    }
    if polynomial::matches_weak_residual(projection, program, derived.dimension, &left, &right) {
        return Ok(());
    }
    if has_variation || derived.conjugate_test {
        return Err(rejection_with(
            projection,
            "authored weak residual differs from the admitted strong-law weak residual",
        ));
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
pub(super) fn product_sign(value: AuthoredFormExpressionV1) -> (AuthoredFormExpressionV1, bool) {
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
        (
            Expression::Coordinate {
                support_ulid: a_support,
                factor_ulid: a_factor,
                axis: a,
            },
            Expression::Coordinate {
                support_ulid: b_support,
                factor_ulid: b_factor,
                axis: b,
            },
        ) => a == b && a_support == b_support && a_factor == b_factor,
        (Expression::Test { field_ulid: a }, Expression::Test { field_ulid: b }) => a == b,
        (Expression::Neg { value: a }, Expression::Neg { value: b })
        | (Expression::Conjugate { value: a }, Expression::Conjugate { value: b })
        | (Expression::Trace { value: a }, Expression::Trace { value: b })
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
        )
        | (
            Expression::Inner {
                left: al,
                right: ar,
            },
            Expression::Inner {
                left: bl,
                right: br,
            },
        )
        | (
            Expression::Complex { real: al, imag: ar },
            Expression::Complex { real: bl, imag: br },
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

#[cfg(test)]
mod complex_tests {
    use super::*;
    use AuthoredFormExpressionV1 as E;

    #[test]
    fn authored_equivalence_retains_inner_order_and_complex_phase() {
        let test = E::Test {
            field_ulid: "test".into(),
        };
        let field = E::Field {
            ulid: "trial".into(),
        };
        let inner = |left, right| E::Inner {
            left: Box::new(left),
            right: Box::new(right),
        };
        let value = inner(test.clone(), field.clone());
        assert!(equivalent(&value, &value.clone()));
        assert!(!equivalent(&value, &inner(field.clone(), test.clone())));
        assert!(!equivalent(
            &value,
            &E::Dot {
                left: Box::new(test),
                right: Box::new(field.clone())
            }
        ));
        let conjugate = E::Conjugate {
            value: Box::new(field.clone()),
        };
        assert!(equivalent(&conjugate, &conjugate.clone()));
        assert!(!equivalent(&conjugate, &field));
        let complex = |imag| E::Complex {
            real: Box::new(E::Number { value: 1.0 }),
            imag: Box::new(E::Number { value: imag }),
        };
        assert!(equivalent(&complex(2.0), &complex(2.0)));
        assert!(!equivalent(&complex(2.0), &complex(-2.0)));
        assert!(!equivalent(&complex(0.0), &E::Number { value: 1.0 }));
    }
}
