use crate::ast::document::ImportDecl;
use crate::ast::{
    ComponentDecl, ConnectorDecl, Document, Item, ModelDecl, NamePath, PureOperatorDecl, TextRange,
    VisibilitySyntax,
};

use super::{
    AstConstructionError, SourceAstFactory, checked_identifier, checked_range, validate_name_path,
};

impl SourceAstFactory {
    /// Append a checked enum declaration to an owned compilation module.
    #[must_use]
    pub fn with_enumeration(mut document: Document, declaration: crate::EnumDecl) -> Document {
        document.enumerations.push(declaration);
        document
    }

    /// Construct one named flat model from already checked Item values.
    ///
    /// # Errors
    /// Returns an error for an invalid source identifier or byte range.
    pub fn model(
        visibility: VisibilitySyntax,
        name: impl Into<String>,
        signature: Vec<crate::SignatureItem>,
        items: Vec<Item>,
        range: TextRange,
    ) -> Result<ModelDecl, AstConstructionError> {
        super::signature::validate_signature(&signature)?;
        for item in &items {
            if let Item::Coordinate(declaration) = item {
                super::compile_time::validate_coordinate(declaration)?;
            }
            if let Item::IndexSet(declaration) = item {
                super::nominal::validate_definition(declaration, "range")?;
            }
        }
        Ok(ModelDecl {
            comments: Default::default(),
            visibility,
            name: checked_identifier(name, "model")?,
            signature,
            items,
            formulations: Vec::new(),
            range: checked_range(range)?,
        })
    }

    /// Construct a Model with one bound mathematical formulation after its members.
    ///
    /// # Errors
    /// Rejects invalid members, identifiers, expressions, or byte ranges.
    pub fn model_with_form(
        visibility: VisibilitySyntax,
        name: impl Into<String>,
        signature: Vec<crate::SignatureItem>,
        items: Vec<Item>,
        form: (String, Vec<String>, crate::FormulationBinding),
        equalities: (Vec<(crate::Expr, crate::Expr)>, TextRange),
        range: TextRange,
    ) -> Result<ModelDecl, AstConstructionError> {
        let mut model = Self::model(visibility, name, signature, items, range)?;
        model
            .formulations
            .push(super::component::checked_form(form, equalities)?);
        Ok(model)
    }

    /// Replace a Model signature while retaining its declarations and authored forms.
    ///
    /// # Errors
    /// Rejects an invalid signature.
    pub fn with_model_signature(
        mut model: ModelDecl,
        signature: Vec<crate::SignatureItem>,
    ) -> Result<ModelDecl, AstConstructionError> {
        super::signature::validate_signature(&signature)?;
        model.signature = signature;
        Ok(model)
    }

    /// Add one explicit semantic module import to a source document.
    ///
    /// # Errors
    /// Returns an error for an invalid qualified module, alias, or source range.
    pub fn with_import(
        mut document: Document,
        module: NamePath,
        alias: Option<String>,
        range: TextRange,
    ) -> Result<Document, AstConstructionError> {
        validate_name_path(&module)?;
        let alias = alias.unwrap_or_else(|| {
            module
                .segments()
                .last()
                .expect("a validated NamePath is nonempty")
                .to_owned()
        });
        let import = ImportDecl {
            comments: Default::default(),
            module,
            alias: checked_identifier(alias, "module import alias")?,
            range: checked_range(range)?,
        };
        if document
            .imports
            .iter()
            .any(|existing| existing.alias == import.alias)
        {
            return Err(AstConstructionError::new(format!(
                "duplicate module import alias `{}`",
                import.alias
            )));
        }
        document.imports.push(import);
        Ok(document)
    }

    /// Close one compilation unit with an ordered structural-dimension prefix.
    ///
    /// # Errors
    /// Returns an error when the compilation unit contains no declaration.
    pub fn document_with_dimensions(
        enumerations: Vec<crate::EnumDecl>,
        dimensions: Vec<crate::NamedDefinitionDecl>,
        connectors: Vec<ConnectorDecl>,
        components: Vec<ComponentDecl>,
        pure_operators: Vec<PureOperatorDecl>,
        models: Vec<ModelDecl>,
    ) -> Result<Document, AstConstructionError> {
        if enumerations.is_empty()
            && dimensions.is_empty()
            && connectors.is_empty()
            && components.is_empty()
            && pure_operators.is_empty()
            && models.is_empty()
        {
            return Err(AstConstructionError::new(
                "a source document requires at least one top-level declaration",
            ));
        }
        for declaration in &dimensions {
            if declaration.value_type().is_some()
                || declaration.domain().is_some()
                || declaration.activation().is_some()
            {
                return Err(AstConstructionError::new(
                    "dimension aliases accept no type, support, or activation assertion",
                ));
            }
        }
        Ok(Document {
            records: Vec::new(),
            comments: Default::default(),
            imports: Vec::new(),
            enumerations,
            finite_spaces: Vec::new(),
            dimensions,
            property_contracts: Vec::new(),
            property_releases: Vec::new(),
            material_compositions: Vec::new(),
            connectors,
            components,
            pure_operators,
            models,
        })
    }

    /// Close one nonempty compilation unit into a formatter-compatible document.
    ///
    /// A declarations-only document is valid source syntax for a package
    /// library. Executable compilation still requires at least one Model.
    ///
    /// # Errors
    /// Returns an error when the compilation unit is empty.
    pub fn document(
        enumerations: Vec<crate::EnumDecl>,
        connectors: Vec<ConnectorDecl>,
        components: Vec<ComponentDecl>,
        models: Vec<ModelDecl>,
    ) -> Result<Document, AstConstructionError> {
        if enumerations.is_empty()
            && connectors.is_empty()
            && components.is_empty()
            && models.is_empty()
        {
            return Err(AstConstructionError::new(
                "a source document requires at least one top-level declaration",
            ));
        }
        Ok(Document {
            records: Vec::new(),
            comments: Default::default(),
            imports: Vec::new(),
            enumerations,
            finite_spaces: Vec::new(),
            dimensions: Vec::new(),
            property_contracts: Vec::new(),
            property_releases: Vec::new(),
            material_compositions: Vec::new(),
            connectors,
            components,
            pure_operators: Vec::new(),
            models,
        })
    }

    /// Close one nonempty compilation unit including pure operators.
    ///
    /// # Errors
    /// Returns an error when the compilation unit is empty.
    pub fn document_with_pure_operators(
        enumerations: Vec<crate::EnumDecl>,
        connectors: Vec<ConnectorDecl>,
        components: Vec<ComponentDecl>,
        pure_operators: Vec<PureOperatorDecl>,
        models: Vec<ModelDecl>,
    ) -> Result<Document, AstConstructionError> {
        if enumerations.is_empty()
            && connectors.is_empty()
            && components.is_empty()
            && pure_operators.is_empty()
            && models.is_empty()
        {
            return Err(AstConstructionError::new(
                "a source document requires at least one top-level declaration",
            ));
        }
        Ok(Document {
            records: Vec::new(),
            comments: Default::default(),
            imports: Vec::new(),
            enumerations,
            finite_spaces: Vec::new(),
            dimensions: Vec::new(),
            property_contracts: Vec::new(),
            property_releases: Vec::new(),
            material_compositions: Vec::new(),
            connectors,
            components,
            pure_operators,
            models,
        })
    }

    /// Close one or more flat models into a formatter-compatible document.
    ///
    /// # Errors
    /// Returns an error when `models` is empty, matching the source grammar.
    pub fn flat_document(models: Vec<ModelDecl>) -> Result<Document, AstConstructionError> {
        if models.is_empty() {
            return Err(AstConstructionError::new(
                "a source document requires at least one model",
            ));
        }
        Ok(Document {
            records: Vec::new(),
            comments: Default::default(),
            imports: Vec::new(),
            enumerations: Vec::new(),
            finite_spaces: Vec::new(),
            dimensions: Vec::new(),
            property_contracts: Vec::new(),
            property_releases: Vec::new(),
            material_compositions: Vec::new(),
            connectors: Vec::new(),
            components: Vec::new(),
            pure_operators: Vec::new(),
            models,
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::{NamePath, SourceAstFactory, TextRange, format, parse};

    #[test]
    fn checked_factory_adds_one_canonical_module_import_prefix() {
        let range = TextRange::new(0, 0);
        let document = parse("main.eqi", "model Main() {}")
            .into_document()
            .expect("base document");
        let document = SourceAstFactory::with_import(
            document,
            NamePath::from_segments(["library", "parts"], range).unwrap(),
            Some("lib".to_owned()),
            range,
        )
        .expect("checked import");
        assert_eq!(
            format(&document),
            "import library.parts as lib;\n\nmodel Main() {\n}\n"
        );

        assert!(
            SourceAstFactory::with_import(
                document,
                NamePath::from_segments(["other"], range).unwrap(),
                Some("lib".to_owned()),
                range,
            )
            .is_err()
        );

        let implicit = SourceAstFactory::with_import(
            parse("main.eqi", "model Main() {}")
                .into_document()
                .unwrap(),
            NamePath::from_segments(["org", "example", "parts"], range).unwrap(),
            None,
            range,
        )
        .expect("implicit final-segment alias");
        assert_eq!(
            format(&implicit),
            "import org.example.parts;\n\nmodel Main() {\n}\n"
        );
    }
}
