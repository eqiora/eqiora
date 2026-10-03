//! Checked construction of finite nominal declarations and family binders.

use std::collections::BTreeSet;

use crate::ast::{Expr, NamePath, TextRange, VisibilitySyntax};
use crate::ast::{FamilyBinderSyntax, NamedDefinitionDecl};

use super::{
    AstConstructionError, SourceAstFactory, checked_identifier, checked_range, validate_expression,
    validate_name_path,
};

impl SourceAstFactory {
    /// Construct a finite space with nonempty, distinct, ordered basis labels.
    ///
    /// # Errors
    /// Rejects invalid identifiers, repeated or absent labels, and malformed ranges.
    pub fn finite_space(
        visibility: VisibilitySyntax,
        name: impl Into<String>,
        labels: Vec<String>,
        range: TextRange,
    ) -> Result<NamedDefinitionDecl, AstConstructionError> {
        if labels.is_empty() {
            return Err(AstConstructionError::new(
                "a finite space requires at least one basis label",
            ));
        }
        let mut distinct = BTreeSet::new();
        for label in &labels {
            checked_identifier(label.clone(), "finite space basis label")?;
            if !distinct.insert(label) {
                return Err(AstConstructionError::new(
                    "finite space basis labels must be distinct",
                ));
            }
        }
        Ok(NamedDefinitionDecl::plain(
            checked_identifier(name, "finite space")?,
            crate::ast::nominal::definition_call(
                "orthonormal",
                labels
                    .into_iter()
                    .map(|name| Expr {
                        resolved_enum: None,
                        resolved_nominal: None,
                        kind: crate::ExprKind::Name(name),
                        range,
                    })
                    .collect(),
                checked_range(range)?,
            ),
            checked_range(range)?,
            visibility,
        ))
    }

    /// Construct an index-set declaration, preserving its exact extent expression.
    /// Integer type, positivity, and structural resource bounds are elaboration checks.
    ///
    /// # Errors
    /// Rejects malformed names, expression trees, and byte ranges.
    pub fn index_set(
        name: impl Into<String>,
        extent: Expr,
        range: TextRange,
    ) -> Result<NamedDefinitionDecl, AstConstructionError> {
        validate_expression(&extent)?;
        Ok(NamedDefinitionDecl::plain(
            checked_identifier(name, "index set")?,
            crate::ast::nominal::definition_call("range", vec![extent], checked_range(range)?),
            checked_range(range)?,
            VisibilitySyntax::Private,
        ))
    }

    /// Construct a local binder over one nominal index-set name.
    ///
    /// # Errors
    /// Rejects malformed identifiers, name paths, and byte ranges.
    pub fn index_family_binder(
        binder: impl Into<String>,
        set: NamePath,
        range: TextRange,
    ) -> Result<FamilyBinderSyntax, AstConstructionError> {
        validate_name_path(&set)?;
        Ok(FamilyBinderSyntax {
            member: checked_identifier(binder, "index family binder")?,
            set,
            range: checked_range(range)?,
        })
    }
}

impl SourceAstFactory {
    /// Add one checked nominal space to an existing compilation unit.
    pub fn with_finite_space(
        mut document: crate::Document,
        declaration: NamedDefinitionDecl,
    ) -> Result<crate::Document, AstConstructionError> {
        validate_definition(&declaration, "orthonormal")?;
        document.finite_spaces.push(declaration);
        Ok(document)
    }

    /// Attach a checked lexical nominal binding while retaining authored syntax.
    ///
    /// # Errors
    /// Rejects a value type with a different nominal role or invalid source bounds.
    pub fn bind_nominal_value_type(
        syntax: &mut crate::ValueTypeSyntax,
        value: eqiora_core::ValueType,
    ) -> Result<(), AstConstructionError> {
        crate::ValueTypeSyntax::validate_checked(&value)?;
        let matches = match syntax.kind() {
            crate::ValueTypeSyntaxKind::Named(_) => value.enum_definition().is_some(),
            crate::ValueTypeSyntaxKind::Coordinates { basis, .. } => value
                .coordinate_basis()
                .is_some_and(|value| value.is_dual() == basis.dual),
            crate::ValueTypeSyntaxKind::LinearMap { source, target, .. } => value
                .map_bases()
                .is_some_and(|(s, t)| s.is_dual() == source.dual && t.is_dual() == target.dual),
            crate::ValueTypeSyntaxKind::Counts(_) => value.is_count(),
            crate::ValueTypeSyntaxKind::Index(_) => value.index_set().is_some(),
            _ => false,
        };
        if !matches {
            return Err(AstConstructionError::new(
                "nominal type binding has a different declaration role",
            ));
        }
        if syntax
            .resolved_nominal
            .as_ref()
            .is_some_and(|previous| previous.as_ref() != &value)
        {
            return Err(AstConstructionError::new(
                "nominal type cannot be rebound to a foreign declaration",
            ));
        }
        syntax.resolved_nominal = Some(Box::new(value));
        Ok(())
    }
}

pub(super) fn validate_definition(
    declaration: &NamedDefinitionDecl,
    constructor: &str,
) -> Result<(), AstConstructionError> {
    if declaration.value_type().is_some()
        || declaration.domain().is_some()
        || declaration.activation().is_some()
    {
        return Err(AstConstructionError::new(
            "nominal definitions cannot carry let type, support or activation assertions",
        ));
    }
    validate_expression(declaration.value())?;
    let crate::ExprKind::Call { callee, arguments } = declaration.value().kind() else {
        return Err(AstConstructionError::new(
            "nominal definition requires its closed constructor",
        ));
    };
    let arguments = arguments.positional().ok_or_else(|| {
        AstConstructionError::new("nominal constructor requires positional arguments")
    })?;
    if callee.as_str() != constructor
        || (constructor == "range" && arguments.len() != 1)
        || arguments.is_empty()
    {
        return Err(AstConstructionError::new(
            "nominal definition has a different constructor or arity",
        ));
    }
    if constructor == "orthonormal" {
        let mut labels = BTreeSet::new();
        for argument in arguments {
            let crate::ExprKind::Name(label) = argument.kind() else {
                return Err(AstConstructionError::new(
                    "finite space basis requires distinct labels",
                ));
            };
            if !labels.insert(label) {
                return Err(AstConstructionError::new(
                    "finite space basis requires distinct labels",
                ));
            }
        }
    } else if declaration.visibility() != VisibilitySyntax::Private {
        return Err(AstConstructionError::new(
            "index sets are private body definitions",
        ));
    }
    Ok(())
}

impl SourceAstFactory {
    /// Bind a closed nominal constructor to its exact lexical declaration type.
    /// The compiler revalidates the declaration identity in its source scope.
    ///
    /// # Errors
    /// Rejects mismatched constructor names, declaration paths, arity, or rebinding.
    #[doc(hidden)]
    pub fn bind_nominal_expression(
        expression: &mut Expr,
        declarations: &[NamePath],
        value: eqiora_core::ValueType,
    ) -> Result<(), AstConstructionError> {
        validate_expression(expression)?;
        crate::ValueTypeSyntax::validate_checked(&value)?;
        let crate::ExprKind::Call { callee, arguments } = expression.kind() else {
            return Err(AstConstructionError::new(
                "nominal binding requires a constructor call",
            ));
        };
        let arguments = arguments.positional().ok_or_else(|| {
            AstConstructionError::new("nominal constructor requires positional arguments")
        })?;
        let duals: Vec<bool> = match callee.as_str() {
            "counts" if value.is_count() => vec![false],
            "coordinates" if value.coordinate_basis().is_some() => vec![
                value
                    .coordinate_basis()
                    .expect("coordinate basis")
                    .is_dual(),
            ],
            "linear_map" if value.map_bases().is_some() => {
                let (source, target) = value.map_bases().expect("map bases");
                vec![source.is_dual(), target.is_dual()]
            }
            "index" if value.index_set().is_some() => vec![false],
            _ => {
                return Err(AstConstructionError::new(
                    "nominal constructor has a different value role",
                ));
            }
        };
        if declarations.len() != duals.len() || arguments.len() != duals.len() + 1 {
            return Err(AstConstructionError::new(
                "nominal constructor has a different basis arity",
            ));
        }
        for ((argument, declaration), dual) in arguments.iter().zip(declarations).zip(duals) {
            let basis = crate::FiniteBasisSyntax::from_expression(argument).ok_or_else(|| {
                AstConstructionError::new(
                    "nominal constructor requires exact declaration arguments",
                )
            })?;
            if basis.name.as_str() != declaration.as_str() || basis.dual != dual {
                return Err(AstConstructionError::new(
                    "nominal constructor binding requires its exact declaration name and duality",
                ));
            }
        }
        if expression
            .resolved_nominal
            .as_ref()
            .is_some_and(|previous| previous.as_ref() != &value)
        {
            return Err(AstConstructionError::new(
                "nominal expression cannot be rebound to a foreign declaration",
            ));
        }
        expression.resolved_nominal = Some(Box::new(value));
        Ok(())
    }
}
