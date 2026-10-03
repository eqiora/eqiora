use super::*;
use crate::{SourceAstFactory, ValueTypeSyntax, ValueTypeSyntaxKind};

pub(super) fn validate(value: &ValueType) -> Result<(), String> {
    ValueTypeSyntax::validate_checked(value)
        .map(|_| ())
        .map_err(|error| error.to_string())
}

impl ValueTypeSyntax {
    /// Project a checked mathematical type into the bounded source grammar.
    ///
    /// # Errors
    /// Rejects types exceeding source constructor or component-count limits.
    pub fn from_checked(
        value: &ValueType,
        mut resolve: impl FnMut(eqiora_core::RawId) -> Option<crate::NamePath>,
        mut resolve_product: impl FnMut(eqiora_core::FiniteBasis) -> Option<crate::NamePath>,
    ) -> Result<Self, crate::AstConstructionError> {
        Self::validate_checked(value)?;
        let nominal = if let Some(id) = value.enum_definition() {
            Some(ValueTypeSyntaxKind::Named(resolve(id.erase()).ok_or_else(
                || {
                    crate::AstConstructionError::new(
                        "enum is absent from lexical declaration scope",
                    )
                },
            )?))
        } else if let Some((source, target)) = value.map_bases() {
            Some(ValueTypeSyntaxKind::LinearMap {
                scalar: Box::new(component_scalar(value)?),
                source: basis_syntax(source, &mut resolve, &mut resolve_product)?,
                target: basis_syntax(target, &mut resolve, &mut resolve_product)?,
            })
        } else if let Some(basis) = value.coordinate_basis() {
            Some(ValueTypeSyntaxKind::Coordinates {
                scalar: Box::new(component_scalar(value)?),
                basis: basis_syntax(basis, &mut resolve, &mut resolve_product)?,
            })
        } else if let Some(id) = value.finite_space() {
            Some(ValueTypeSyntaxKind::Counts(
                resolve(id.erase()).ok_or_else(|| {
                    crate::AstConstructionError::new(
                        "count space is absent from lexical declaration scope",
                    )
                })?,
            ))
        } else if let Some(id) = value.index_set() {
            let name = resolve(id.erase()).ok_or_else(|| {
                crate::AstConstructionError::new(
                    "index set is absent from the lexical declaration scope",
                )
            })?;
            Some(ValueTypeSyntaxKind::Index(name))
        } else {
            None
        };
        if let Some(kind) = nominal {
            let mut syntax = SourceAstFactory::value_type(kind, crate::TextRange::new(0, 0))?;
            syntax.resolved_nominal = Some(Box::new(value.clone()));
            return Ok(syntax);
        }

        let syntax = project(
            value,
            &GraphPath::new(["value-type"]),
            &mut RangeAllocator::default(),
            &mut HashMap::new(),
            &mut resolve,
            &mut resolve_product,
        );
        SourceAstFactory::value_type(*syntax.kind, syntax.range)
    }
    /// Validate source type resource bounds independently of lexical name projection.
    ///
    /// # Errors
    /// Rejects excessive nesting, rank, or component count.
    pub fn validate_checked(value: &ValueType) -> Result<(), crate::AstConstructionError> {
        // Bound projection work before constructing the source AST. The shared
        // factory then applies the exact source constructor and element-count rules.
        if value.array_rank() >= 256 || value.shape().rank() - value.array_rank() > 4 {
            return Err(crate::AstConstructionError::new(
                "mathematical type exceeds source nesting or spatial-rank limits",
            ));
        }
        if value
            .shape()
            .component_count()
            .is_none_or(|count| count > 65_536)
        {
            return Err(crate::AstConstructionError::new(
                "source type exceeds the 65536-component limit",
            ));
        }
        Ok(())
    }
}

pub(super) fn project(
    value: &ValueType,
    path: &GraphPath,
    ranges: &mut RangeAllocator,
    paths: &mut HashMap<TextRange, GraphPath>,
    resolve: &mut dyn FnMut(eqiora_core::RawId) -> Option<crate::NamePath>,
    resolve_product: &mut dyn FnMut(eqiora_core::FiniteBasis) -> Option<crate::NamePath>,
) -> ValueTypeSyntax {
    if value.enum_definition().is_some()
        || value.finite_bases().next().is_some()
        || value.index_set().is_some()
    {
        let mut syntax = ValueTypeSyntax::from_checked(value, resolve, resolve_product)
            .expect("validated nominal declaration scope");
        syntax.range = ranges.allocate(path, paths);
        return syntax;
    }
    let dimension = dimension_expression(value.dimension(), path, ranges, paths);
    let mut syntax = if value.scalar_domain() == eqiora_core::ScalarDomain::Real {
        ValueTypeSyntax::real(dimension)
    } else {
        ValueTypeSyntax {
            resolved_nominal: None,
            kind: Box::new(ValueTypeSyntaxKind::Scalar {
                domain: value.scalar_domain(),
                dimension,
            }),
            range: ranges.allocate(path, paths),
        }
    };
    syntax.range = ranges.allocate(path, paths);
    let (arrays, spatial) = value.shape().extents().split_at(value.array_rank());
    if !spatial.is_empty() {
        let kind = if spatial.len() == 1 {
            ValueTypeSyntaxKind::Vector {
                scalar: Box::new(syntax),
                extent: spatial[0].get(),
            }
        } else {
            ValueTypeSyntaxKind::Tensor {
                scalar: Box::new(syntax),
                extents: spatial.iter().map(|n| n.get()).collect(),
            }
        };
        syntax = ValueTypeSyntax {
            resolved_nominal: None,
            kind: Box::new(kind),
            range: ranges.allocate(path, paths),
        };
    }
    for extent in arrays.iter().rev() {
        syntax = ValueTypeSyntax {
            resolved_nominal: None,
            kind: Box::new(ValueTypeSyntaxKind::Array {
                element: Box::new(syntax),
                extent: crate::Expr {
                    resolved_enum: None,
                    resolved_nominal: None,
                    kind: crate::ExprKind::Number(
                        crate::DecimalLiteral::parse(&extent.get().to_string())
                            .expect("positive extent"),
                    ),
                    range: crate::TextRange::default(),
                },
            }),
            range: ranges.allocate(path, paths),
        };
    }
    syntax
}

fn component_scalar(value: &ValueType) -> Result<ValueTypeSyntax, crate::AstConstructionError> {
    let scalar = ValueType::scalar(value.scalar_domain(), value.dimension())
        .map_err(|error| crate::AstConstructionError::new(error.to_string()))?;
    ValueTypeSyntax::from_checked(&scalar, |_| None, |_| None)
}

fn basis_syntax(
    basis: eqiora_core::FiniteBasis,
    resolve: &mut impl FnMut(eqiora_core::RawId) -> Option<crate::NamePath>,
    resolve_product: &mut impl FnMut(eqiora_core::FiniteBasis) -> Option<crate::NamePath>,
) -> Result<crate::FiniteBasisSyntax, crate::AstConstructionError> {
    let name = if let Some(space) = basis.space() {
        resolve(space.erase())
    } else {
        resolve_product(if basis.is_dual() { basis.dual() } else { basis })
    }
    .ok_or_else(|| {
        crate::AstConstructionError::new("finite basis is absent from lexical declaration scope")
    })?;
    Ok(crate::FiniteBasisSyntax {
        name,
        dual: basis.is_dual(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Item;
    use eqiora_core::{ScalarDomain, ValueFrame, ValueShape};

    #[test]
    fn native_parameters_preserve_declared_types_in_source_projection() {
        let scalar = ValueType::scalar(ScalarDomain::Complex, DimExponents::DIMENSIONLESS).unwrap();
        for (value_type, expected) in [
            (scalar.clone(), "complex<1>"),
            (scalar.array(3).unwrap(), "array<complex<1>, 3>"),
        ] {
            let parameter = DraftParameter::new(
                "coefficient",
                eqiora_core::ValueLiteral::from_real(value_type.clone(), 0.0).unwrap(),
            );
            assert_eq!(parameter.value_type(), &value_type);
            let draft = Module::new("M", [parameter.into()]).unwrap();
            let native = draft;
            let Item::Parameter(parameter) = &native.model().items()[0] else {
                panic!("Parameter");
            };
            assert_eq!(parameter.value_type().to_source(), expected);
            assert_eq!(
                native
                    .graph_path(parameter.value_type().range())
                    .unwrap()
                    .to_string(),
                "M.coefficient"
            );
            let document = SourceAstFactory::document(
                Vec::new(),
                vec![],
                vec![],
                vec![native.model().clone()],
            )
            .unwrap();
            let source = crate::format(&document);
            let parsed = crate::parse("native.eqi", &source).into_document().unwrap();
            assert_eq!(crate::format(&parsed), source);
        }
        let oversized = ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS)
            .unwrap()
            .array(65_537)
            .unwrap();
        let parameter = DraftParameter::new(
            "large",
            eqiora_core::ValueLiteral::from_real(oversized, 0.0).unwrap(),
        );
        assert!(Module::new("M", [parameter.into()]).is_err());
    }

    #[test]
    fn native_fields_keep_nested_types_and_absent_initial_values() {
        let value = ValueType::shaped(
            ScalarDomain::Complex,
            DimExponents::DIMENSIONLESS,
            ValueShape::new([2]).unwrap(),
            ValueFrame::SpatialCartesian,
        )
        .unwrap()
        .array(3)
        .unwrap();
        let field = DraftField::new("channels", value.clone(), crate::FieldRoleSyntax::Variable);
        assert_eq!(field.value_type(), &value);
        assert_eq!(field.role(), crate::FieldRoleSyntax::Variable);
        let draft = Module::new("M", [field.into()]).unwrap();
        let native = draft;
        let document =
            SourceAstFactory::document(Vec::new(), vec![], vec![], vec![native.model().clone()])
                .unwrap();
        let source = crate::format(&document);
        assert!(source.contains("variable channels: array<vector<complex<1>, 2>, 3>;"));
        let parsed = crate::parse("native.eqi", &source).into_document().unwrap();
        assert_eq!(crate::format(&parsed), source);
        let Item::Field(field) = &native.model().items()[0] else {
            panic!("Field");
        };
        assert_eq!(
            native
                .graph_path(field.value_type().range())
                .unwrap()
                .to_string(),
            "M.channels"
        );
    }

    #[test]
    fn native_fields_obey_source_type_limits() {
        let scalar = ValueType::scalar(ScalarDomain::Real, DimExponents::DIMENSIONLESS).unwrap();
        let oversized = DraftField::new(
            "large",
            scalar.array(65_537).unwrap(),
            crate::FieldRoleSyntax::Variable,
        );
        assert!(Module::new("M", [oversized.into()]).is_err());
    }
}
