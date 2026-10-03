use super::{AstConstructionError, SourceAstFactory, checked_range, validate_expression};
use crate::{TextRange, ValueTypeSyntax, ValueTypeSyntaxKind};

impl SourceAstFactory {
    /// Construct a bounded mathematical source type without parsing text.
    ///
    /// # Errors
    /// Rejects invalid scalar components, extents, nesting, expressions or ranges.
    pub fn value_type(
        kind: ValueTypeSyntaxKind,
        range: TextRange,
    ) -> Result<ValueTypeSyntax, AstConstructionError> {
        let kind = if let ValueTypeSyntaxKind::Scalar {
            domain: eqiora_core::ScalarDomain::Real,
            dimension,
        } = kind
        {
            *ValueTypeSyntax::real(dimension).kind
        } else {
            kind
        };
        let result = ValueTypeSyntax {
            resolved_nominal: None,
            kind: Box::new(kind),
            range: checked_range(range)?,
        };
        let mut current = &result;
        let mut count = 1_u64;
        for _ in 0..256 {
            let (element, extents) = match current.kind() {
                ValueTypeSyntaxKind::Named(name) => {
                    super::validate_name_path(name)?;
                    return Ok(result);
                }
                ValueTypeSyntaxKind::Counts(name) | ValueTypeSyntaxKind::Index(name) => {
                    super::validate_name_path(name)?;
                    if count != 1 {
                        return Err(AstConstructionError::new(
                            "nominal discrete types do not admit outer arrays or spatial axes",
                        ));
                    }
                    return Ok(result);
                }
                ValueTypeSyntaxKind::Coordinates { scalar, basis } => {
                    super::validate_name_path(&basis.name)?;
                    if !std::ptr::eq(current, &result) {
                        return Err(AstConstructionError::new(
                            "finite coordinates cannot acquire implicit axes",
                        ));
                    }
                    require_scalar(scalar)?;
                    current = scalar;
                    continue;
                }
                ValueTypeSyntaxKind::LinearMap {
                    scalar,
                    source,
                    target,
                } => {
                    super::validate_name_path(&source.name)?;
                    super::validate_name_path(&target.name)?;
                    if !std::ptr::eq(current, &result) {
                        return Err(AstConstructionError::new(
                            "finite maps cannot acquire implicit axes",
                        ));
                    }
                    require_scalar(scalar)?;
                    current = scalar;
                    continue;
                }
                ValueTypeSyntaxKind::Scalar { dimension, domain } => {
                    validate_expression(dimension)?;
                    if matches!(
                        domain,
                        eqiora_core::ScalarDomain::Integer | eqiora_core::ScalarDomain::Boolean
                    ) && !matches!(dimension.kind(), crate::ExprKind::Number(value) if value.to_i64().ok() == Some(1))
                    {
                        return Err(AstConstructionError::new("integer type is dimensionless"));
                    }
                    return Ok(result);
                }
                ValueTypeSyntaxKind::Vector { scalar, extent } => {
                    require_scalar(scalar)?;
                    (scalar.as_ref(), std::slice::from_ref(extent))
                }
                ValueTypeSyntaxKind::Tensor { scalar, extents } => {
                    require_scalar(scalar)?;
                    if !(2..=4).contains(&extents.len()) {
                        return Err(AstConstructionError::new(
                            "tensor requires two to four extents; use vector for one spatial axis",
                        ));
                    }
                    (scalar.as_ref(), extents.as_slice())
                }
                ValueTypeSyntaxKind::Array { element, extent } => {
                    validate_expression(extent)?;
                    if let crate::ExprKind::Number(literal) = extent.kind() {
                        let extent = literal
                            .to_i64()
                            .ok()
                            .and_then(|n| u32::try_from(n).ok())
                            .filter(|n| *n > 0)
                            .ok_or_else(|| {
                                AstConstructionError::new(
                                    "type extent must be a positive u32 integer",
                                )
                            })?;
                        count = count.checked_mul(u64::from(extent)).filter(|n| *n <= 65_536)
                            .ok_or_else(|| AstConstructionError::new("source resource limit exceeded: mathematical type has more than 65536 elements"))?;
                    }
                    current = element;
                    continue;
                }
            };
            for extent in extents {
                if *extent == 0 {
                    return Err(AstConstructionError::new(
                        "type extent must be a positive u32 integer",
                    ));
                }
                count = count.checked_mul(u64::from(*extent)).filter(|n| *n <= 65_536)
                    .ok_or_else(|| AstConstructionError::new(
                        "source resource limit exceeded: mathematical type has more than 65536 elements",
                    ))?;
            }
            current = element;
        }
        Err(AstConstructionError::new(
            "source resource limit exceeded: type nesting exceeds 256",
        ))
    }
}

fn require_scalar(value: &ValueTypeSyntax) -> Result<(), AstConstructionError> {
    if value.is_scalar() || matches!(value.kind(), ValueTypeSyntaxKind::Named(_)) {
        Ok(())
    } else {
        Err(AstConstructionError::new(
            "vector and tensor components require a scalar type",
        ))
    }
}

pub(super) fn validate_syntax(value: &ValueTypeSyntax) -> Result<(), AstConstructionError> {
    SourceAstFactory::value_type(value.kind().clone(), value.range()).map(|_| ())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{ExprKind, Item, format, parse};
    use eqiora_core::ScalarDomain;

    fn extent(n: u32) -> crate::Expr {
        SourceAstFactory::expression(
            ExprKind::Number(crate::DecimalLiteral::parse(&n.to_string()).unwrap()),
            TextRange::default(),
        )
        .unwrap()
    }

    #[test]
    fn component_parameters_share_checked_type_constructors() {
        let checked = eqiora_core::ValueType::scalar(
            ScalarDomain::Complex,
            eqiora_core::DimExponents::DIMENSIONLESS,
        )
        .unwrap()
        .array(3)
        .unwrap();
        let value_type = ValueTypeSyntax::from_checked(&checked, |_| None, |_| None).unwrap();
        let declaration = SourceAstFactory::component_parameter(
            crate::VisibilitySyntax::Public,
            "channels",
            value_type,
            None,
            TextRange::new(0, 0),
        )
        .unwrap();
        assert_eq!(declaration.value_type().to_source(), "array<complex<1>, 3>");
        let document = parse(
            "component.eqi",
            "component C(parameter channels: array<complex<1>, 3>) {}",
        )
        .into_document()
        .unwrap();
        let source = format(&document);
        assert!(source.contains("parameter channels: array<complex<1>, 3>"));
        assert_eq!(
            format(&parse("reparsed.eqi", &source).into_document().unwrap()),
            source
        );
    }

    #[test]
    fn native_array_nesting_stops_at_the_source_depth_limit() {
        let range = TextRange::new(0, 0);
        let dimension = SourceAstFactory::expression(
            ExprKind::Number(crate::DecimalLiteral::parse("1.0").expect("exact literal")),
            range,
        )
        .unwrap();
        let mut value = ValueTypeSyntax::real(dimension);
        for _ in 0..255 {
            value = SourceAstFactory::value_type(
                ValueTypeSyntaxKind::Array {
                    element: Box::new(value),
                    extent: extent(1),
                },
                range,
            )
            .unwrap();
        }
        assert!(
            SourceAstFactory::value_type(
                ValueTypeSyntaxKind::Array {
                    element: Box::new(value),
                    extent: extent(1)
                },
                range,
            )
            .is_err()
        );
    }

    #[test]
    fn native_types_share_source_constructor_rules_and_canonical_emission() {
        let range = TextRange::new(0, 0);
        let scalar = SourceAstFactory::value_type(
            ValueTypeSyntaxKind::Scalar {
                domain: ScalarDomain::Complex,
                dimension: SourceAstFactory::expression(ExprKind::Name("V".into()), range).unwrap(),
            },
            range,
        )
        .unwrap();
        let vector = SourceAstFactory::value_type(
            ValueTypeSyntaxKind::Vector {
                scalar: Box::new(scalar.clone()),
                extent: 2,
            },
            range,
        )
        .unwrap();
        let array = SourceAstFactory::value_type(
            ValueTypeSyntaxKind::Array {
                element: Box::new(vector.clone()),
                extent: extent(3),
            },
            range,
        )
        .unwrap();
        let parameter = SourceAstFactory::parameter(
            "channels",
            array,
            SourceAstFactory::expression(
                ExprKind::Number(crate::DecimalLiteral::parse("0.0").expect("exact literal")),
                range,
            )
            .unwrap(),
            range,
        )
        .unwrap();
        let parsed = parse(
            "types.eqi",
            "model M() { parameter channels: array<vector<complex<V>, 2>, 3> = 0; }",
        )
        .into_document()
        .unwrap();
        let Item::Parameter(source) = &parsed.models()[0].items()[0] else {
            panic!("parameter");
        };
        assert_eq!(
            parameter.value_type().scalar_domain(),
            source.value_type().scalar_domain()
        );
        let model = SourceAstFactory::model(
            parsed.models()[0].visibility(),
            "M",
            Vec::new(),
            vec![Item::Parameter(parameter)],
            range,
        )
        .unwrap();
        let native =
            SourceAstFactory::document(Vec::new(), Vec::new(), Vec::new(), vec![model]).unwrap();
        assert_eq!(format(&native), format(&parsed));

        for kind in [
            ValueTypeSyntaxKind::Vector {
                scalar: Box::new(vector.clone()),
                extent: 2,
            },
            ValueTypeSyntaxKind::Array {
                element: Box::new(scalar.clone()),
                extent: extent(0),
            },
            ValueTypeSyntaxKind::Array {
                element: Box::new(vector),
                extent: extent(32_769),
            },
            ValueTypeSyntaxKind::Tensor {
                scalar: Box::new(scalar.clone()),
                extents: vec![2],
            },
            ValueTypeSyntaxKind::Tensor {
                scalar: Box::new(scalar),
                extents: vec![2; 5],
            },
        ] {
            assert!(SourceAstFactory::value_type(kind, range).is_err());
        }
    }
}
