//! Construction of reusable Component declarations.

use crate::ast::formulation::FormulationDecl;
use crate::ast::{
    ComponentDecl, ComponentItem, ConnectionSyntax, Expr, TextRange, VisibilitySyntax,
};

use super::{
    AstConstructionError, SourceAstFactory, checked_identifier, checked_range,
    validate_boundary_connection, validate_boundary_family_binder, validate_expression,
    validate_port_syntax,
};

impl SourceAstFactory {
    /// Construct one borrowed exact-clock requirement without declaring a period.
    ///
    /// # Errors
    /// Rejects malformed identifiers and ranges.
    pub fn clock_requirement(
        name: impl Into<String>,
        range: TextRange,
    ) -> Result<crate::ClockRequirementDecl, AstConstructionError> {
        Ok(crate::ClockRequirementDecl {
            comments: Default::default(),
            name: checked_identifier(name, "required clock")?,
            range: checked_range(range)?,
        })
    }

    /// Construct one reusable Component declaration.
    ///
    /// # Errors
    /// Returns an error for an invalid source identifier, member shape, or byte range.
    pub fn component(
        visibility: VisibilitySyntax,
        name: impl Into<String>,
        signature: Vec<crate::SignatureItem>,
        items: Vec<ComponentItem>,
        range: TextRange,
    ) -> Result<ComponentDecl, AstConstructionError> {
        super::signature::validate_signature(&signature)?;
        for item in &items {
            validate_component_item(item)?;
        }
        Ok(ComponentDecl {
            comments: Default::default(),
            visibility,
            name: checked_identifier(name, "component")?,
            signature,
            items,
            formulations: Vec::new(),
            range: checked_range(range)?,
        })
    }

    /// Construct a Component with one bound mathematical formulation after its members.
    ///
    /// # Errors
    /// Returns an error for an invalid member, identifier, expression, or byte range.
    pub fn component_with_form(
        visibility: VisibilitySyntax,
        name: impl Into<String>,
        signature: Vec<crate::SignatureItem>,
        items: Vec<ComponentItem>,
        form: (String, Vec<String>, crate::FormulationBinding),
        equalities: (Vec<(Expr, Expr)>, TextRange),
        range: TextRange,
    ) -> Result<ComponentDecl, AstConstructionError> {
        super::signature::validate_signature(&signature)?;
        for item in &items {
            validate_component_item(item)?;
        }
        let (equations, formulation_range) = equalities;
        if equations.is_empty() {
            return Err(AstConstructionError::new(
                "Formulation requires an equality",
            ));
        }
        for (left, right) in &equations {
            validate_expression(left)?;
            validate_expression(right)?;
        }
        let (form_name, relations, binding) = form;
        let form_name = checked_identifier(form_name, "Formulation")?;
        if relations.is_empty() {
            return Err(AstConstructionError::new("Formulation requires a Relation"));
        }
        for relation in &relations {
            checked_identifier(relation.clone(), "Formulation Relation")?;
        }
        match &binding {
            crate::FormulationBinding::WeakTests { tests } => {
                if tests.is_empty() {
                    return Err(AstConstructionError::new(
                        "weak Formulation requires a test",
                    ));
                }
                for (name, trial, zero_on) in tests {
                    checked_identifier(name.clone(), "test function")?;
                    checked_identifier(trial.clone(), "trial Field")?;
                    for name in zero_on {
                        checked_identifier(name.clone(), "test boundary")?;
                    }
                }
            }
            crate::FormulationBinding::Interval {
                name,
                lower,
                upper,
                domain,
            } => {
                for name in [name, lower, upper, domain] {
                    checked_identifier(name.clone(), "interval binder")?;
                }
                if name == lower || name == upper || lower == upper {
                    return Err(AstConstructionError::new(
                        "interval binders must be distinct",
                    ));
                }
            }
        }
        let formulation_range = checked_range(formulation_range)?;
        let range = checked_range(range)?;
        Ok(ComponentDecl {
            comments: Default::default(),
            visibility,
            name: checked_identifier(name, "component")?,
            signature,
            items,
            formulations: vec![FormulationDecl {
                comments: Default::default(),
                name: form_name,
                binding,
                relations,
                equations,
                range: formulation_range,
            }],
            range,
        })
    }
}

fn validate_component_item(item: &ComponentItem) -> Result<(), AstConstructionError> {
    let range = match item {
        ComponentItem::Let(declaration) => declaration.range(),
        ComponentItem::Parameter(declaration) => {
            if declaration.visibility() == VisibilitySyntax::Public {
                return Err(AstConstructionError::new(
                    "public parameters belong in the signature",
                ));
            }
            declaration.range()
        }
        ComponentItem::Port(declaration) => {
            if declaration.visibility() == VisibilitySyntax::Public {
                return Err(AstConstructionError::new(
                    "public ports belong in the signature",
                ));
            }
            declaration.range()
        }
        ComponentItem::PortFamily(declaration) => {
            validate_port_syntax(declaration.port().syntax())?;
            validate_boundary_family_binder(declaration.binder())?;
            declaration.range()
        }
        ComponentItem::Observable(declaration) => declaration.range(),
        ComponentItem::Field(declaration) => declaration.range(),
        ComponentItem::Initial(declaration) => declaration.range(),
        ComponentItem::Event(declaration) => declaration.range(),
        ComponentItem::Clock(declaration) => declaration.range(),
        ComponentItem::Relation(declaration) => declaration.range(),
        ComponentItem::RelationFamily(declaration) => {
            validate_boundary_family_binder(declaration.binder())?;
            declaration.range()
        }
        ComponentItem::Connection(declaration) => declaration.range(),
        ComponentItem::BoundaryConnection(declaration) => {
            validate_boundary_connection(declaration)?;
            if declaration.syntax() == ConnectionSyntax::SpatialPeriodic {
                return Err(AstConstructionError::new(
                    "a spatial-periodic Connection belongs only to a closed Model",
                ));
            }
            declaration.range()
        }
        ComponentItem::IndexSet(declaration) => {
            super::nominal::validate_definition(declaration, "range")?;
            declaration.range()
        }
        ComponentItem::Instance(declaration) => declaration.range(),
    };
    checked_range(range).map(|_| ())
}

#[cfg(test)]
mod tests {
    use crate::{format, parse};

    use super::*;

    #[test]
    fn constructs_primal_form_without_model_item_coercion() {
        let parsed = parse(
            "form.eqi",
            "component C() { relation balance { 1 = 0; } form weak for balance { test w: 1 for value zero_on surface; integrate(region, w) = integrate(region, w); } }",
        )
        .into_document()
        .unwrap();
        let source = &parsed.components()[0];
        let (name, relations, equations, range) = source.formulations().next().unwrap();
        let binding = source.formulation_binding(name).unwrap();
        let component = SourceAstFactory::component_with_form(
            VisibilitySyntax::Private,
            "C",
            source.signature().to_vec(),
            source.items().to_vec(),
            (name.into(), relations.to_vec(), binding.clone()),
            (equations.to_vec(), range),
            source.range(),
        )
        .unwrap();
        let document =
            SourceAstFactory::document(Vec::new(), Vec::new(), vec![component], Vec::new())
                .unwrap();

        assert_eq!(document.components()[0].formulations().len(), 1);
        assert!(format(&document).contains("form weak for balance"));
    }
    #[test]
    fn constructs_plural_form_and_rejects_empty_inventories() {
        let parsed = parse("mixed.eqi", "component C() { form weak for momentum,continuity { test v:1 for velocity zero_on surface; test q:1 for pressure; integrate(body,v)=integrate(body,0); integrate(body,q)=integrate(body,0); } }").into_document().unwrap();
        let source = &parsed.components()[0];
        let (name, relations, equations, range) = source.formulations().next().unwrap();
        let binding = source.formulation_binding(name).unwrap();
        let construct = |relations, equations, binding| {
            SourceAstFactory::component_with_form(
                VisibilitySyntax::Private,
                "C",
                vec![],
                vec![],
                (name.into(), relations, binding),
                (equations, range),
                source.range(),
            )
        };
        let component = construct(relations.to_vec(), equations.to_vec(), binding.clone()).unwrap();
        let native = SourceAstFactory::document(vec![], vec![], vec![component], vec![]).unwrap();
        assert_eq!(format(&native), format(&parsed));
        assert!(construct(vec![], equations.to_vec(), binding.clone()).is_err());
        assert!(construct(relations.to_vec(), vec![], binding.clone()).is_err());
        assert!(
            construct(
                relations.to_vec(),
                equations.to_vec(),
                crate::FormulationBinding::WeakTests { tests: vec![] }
            )
            .is_err()
        );
    }
}
