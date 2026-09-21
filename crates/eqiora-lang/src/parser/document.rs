use crate::ast::document::ImportDecl;
use crate::ast::{Document, ModelDecl, NamedDefinitionDecl, TextRange, VisibilitySyntax};
use crate::lexer::TokenKind;

use super::Parser;

impl Parser<'_> {
    pub(super) fn parse_document(&mut self) -> Option<Document> {
        let mut imports = Vec::new();
        let mut enumerations = Vec::new();
        let mut records = Vec::new();
        let mut dimensions = Vec::new();
        let mut finite_spaces = Vec::new();
        let mut property_contracts = Vec::new();
        let mut property_releases = Vec::new();
        let mut material_compositions = Vec::new();
        let mut connectors = Vec::new();
        let mut components = Vec::new();
        let mut pure_operators = Vec::new();
        let mut models = Vec::new();
        let mut models_started = false;
        let mut import_prefix_closed = false;
        while !self.at(TokenKind::Eof) {
            let modifier = if self.at_keyword("public") {
                Some((VisibilitySyntax::Public, self.bump().clone()))
            } else if self.at_keyword("private") {
                Some((VisibilitySyntax::Private, self.bump().clone()))
            } else {
                None
            };
            let visibility = modifier
                .as_ref()
                .map_or(VisibilitySyntax::Private, |(visibility, _)| *visibility);
            let declaration_start = modifier.as_ref().map_or_else(
                || self.current().range().start(),
                |(_, token)| token.range().start(),
            );

            if self.at_keyword("import") {
                if let Some((_, token)) = &modifier {
                    self.error_token(token, "module imports have no visibility modifier");
                }
                if import_prefix_closed {
                    self.error_here("module imports must form a prefix before all declarations");
                }
                if let Some(import) = self.parse_import(declaration_start) {
                    imports.push(import);
                } else {
                    self.recover_top_level();
                }
            } else if self.at_keyword("record") {
                import_prefix_closed = true;
                if let Some(declaration) = self.parse_record(declaration_start, visibility) {
                    records.push(declaration);
                } else {
                    self.recover_top_level();
                }
            } else if self.at_keyword("enum") {
                import_prefix_closed = true;
                if let Some(declaration) = self.parse_enumeration(declaration_start, visibility) {
                    enumerations.push(declaration);
                } else {
                    self.recover_top_level();
                }
            } else if self.at_keyword("dimension") {
                import_prefix_closed = true;
                if let Some(dimension) = self.parse_dimension(declaration_start, visibility) {
                    dimensions.push(dimension);
                } else {
                    self.recover_top_level();
                }
            } else if self.at_keyword("property") {
                import_prefix_closed = true;
                self.parse_top_property(
                    declaration_start,
                    visibility,
                    models_started,
                    &mut property_contracts,
                    &mut property_releases,
                );
            } else if self.at_keyword("material") {
                import_prefix_closed = true;
                if models_started {
                    self.error_here("material compositions must precede model declarations");
                }
                if let Some(composition) =
                    self.parse_material_composition(declaration_start, visibility)
                {
                    material_compositions.push(composition);
                } else {
                    self.recover_top_level();
                }
            } else if self.at_keyword("connector") {
                import_prefix_closed = true;
                if models_started {
                    self.error_here(
                        "compilation-unit Connector declarations must precede model declarations",
                    );
                }
                if let Some(connector) = self.parse_connector(declaration_start, visibility) {
                    connectors.push(connector);
                } else {
                    self.recover_top_level();
                }
            } else if self.at_keyword("component") {
                import_prefix_closed = true;
                if models_started {
                    self.error_here(
                        "compilation-unit component declarations must precede model declarations",
                    );
                }
                if let Some(component) = self.parse_component(declaration_start, visibility) {
                    components.push(component);
                } else {
                    self.recover_top_level();
                }
            } else if self.at_keyword("operator") {
                import_prefix_closed = true;
                if models_started {
                    self.error_here(
                        "compilation-unit pure operator declarations must precede model declarations",
                    );
                }
                if let Some(operator) = self.parse_pure_operator(declaration_start, visibility) {
                    pure_operators.push(operator);
                } else {
                    self.recover_top_level();
                }
            } else if self.at_keyword("space") {
                import_prefix_closed = true;
                if models_started {
                    self.error_here("finite spaces must precede Model declarations");
                }
                if let Some(space) = self.parse_finite_space(declaration_start, visibility) {
                    finite_spaces.push(space);
                } else {
                    self.recover_top_level();
                }
            } else if self.at_keyword("model") {
                import_prefix_closed = true;
                models_started = true;
                if let Some(model) = self.parse_model(declaration_start, visibility) {
                    models.push(model);
                } else {
                    self.recover_top_level();
                }
            } else {
                let expected = if import_prefix_closed {
                    "expected `dimension`, `property`, `connector`, `component`, `operator`, or `model` declaration"
                } else {
                    "expected `import`, `dimension`, `property`, `connector`, `component`, `operator`, or `model` declaration"
                };
                self.error_here(expected);
                self.recover_top_level();
            }
        }
        (!(imports.is_empty()
            && records.is_empty()
            && enumerations.is_empty()
            && finite_spaces.is_empty()
            && dimensions.is_empty()
            && property_contracts.is_empty()
            && property_releases.is_empty()
            && material_compositions.is_empty()
            && connectors.is_empty()
            && components.is_empty()
            && pure_operators.is_empty()
            && models.is_empty()))
        .then_some(Document {
            records,
            comments: Default::default(),
            imports,
            enumerations,
            finite_spaces,
            dimensions,
            property_contracts,
            property_releases,
            material_compositions,
            connectors,
            components,
            pure_operators,
            models,
        })
    }

    pub(super) fn parse_model(
        &mut self,
        declaration_start: u32,
        visibility: VisibilitySyntax,
    ) -> Option<ModelDecl> {
        self.expect_keyword("model")?;
        let name = self.declaration_name("model name")?.text().to_owned();
        let signature = self.parse_signature()?;
        self.expect(TokenKind::LeftBrace, "`{` after model name")?;
        let mut items = Vec::new();
        while !self.at(TokenKind::RightBrace) && !self.at(TokenKind::Eof) {
            match self.parse_item() {
                Some(item) => items.push(item),
                None => self.recover_item(),
            }
        }
        let end = self
            .expect(TokenKind::RightBrace, "`}` to close model")
            .map_or_else(|| self.current().range().end(), |token| token.range().end());
        Some(ModelDecl {
            comments: Default::default(),
            visibility,
            name,
            signature,
            items,
            range: TextRange::new(declaration_start, end),
        })
    }

    fn parse_import(&mut self, start: u32) -> Option<ImportDecl> {
        self.expect_keyword("import")?;
        let module = self.parse_name_path("logical module name")?;
        let alias = if self.at_keyword("as") {
            self.bump();
            self.expect_identifier("module import alias")?
                .text()
                .to_owned()
        } else {
            module
                .segments()
                .last()
                .expect("a parsed NamePath is nonempty")
                .to_owned()
        };
        let end = self
            .expect(TokenKind::Semicolon, "`;` after module import")?
            .range()
            .end();
        Some(ImportDecl {
            comments: Default::default(),
            module,
            alias,
            range: TextRange::new(start, end),
        })
    }

    fn parse_dimension(
        &mut self,
        start: u32,
        visibility: VisibilitySyntax,
    ) -> Option<NamedDefinitionDecl> {
        self.expect_keyword("dimension")?;
        let name = self
            .declaration_name("dimension alias name")?
            .text()
            .to_owned();
        self.expect(TokenKind::Equal, "`=` before dimension expression")?;
        let expression = self.parse_dimension_expression()?;
        let end = self
            .expect(TokenKind::Semicolon, "`;` after dimension alias")?
            .range()
            .end();
        Some(NamedDefinitionDecl::plain(
            name,
            expression,
            TextRange::new(start, end),
            visibility,
        ))
    }
}

#[cfg(test)]
mod tests {
    use crate::{format, parse};

    #[test]
    fn parser_and_formatter_retain_explicit_import_prefix() {
        let source = "import org.example.geometry.channel;\nimport org.example.materials.water as fluid;\nmodel Main() {}";
        let document = parse("main.eqi", source)
            .into_document()
            .expect("module imports parse");
        let imports = document.imports().collect::<Vec<_>>();

        assert_eq!(imports.len(), 2);
        assert_eq!(imports[0].0.as_str(), "org.example.geometry.channel");
        assert_eq!(imports[0].1, "channel");
        assert_eq!(
            &source[imports[0].2.start() as usize..imports[0].2.end() as usize],
            "import org.example.geometry.channel;"
        );

        let formatted = format(&document);
        assert!(formatted.starts_with(
            "import org.example.geometry.channel;\nimport org.example.materials.water as fluid;\n\nmodel Main"
        ));
        let reparsed = parse("main.eqi", &formatted)
            .into_document()
            .expect("formatted imports reparse");
        assert_eq!(format(&reparsed), formatted);

        let misplaced = parse(
            "misplaced.eqi",
            "model Main() {} import org.example.geometry.channel;",
        );
        assert!(
            misplaced
                .diagnostics()
                .iter()
                .any(|diagnostic| diagnostic.message().contains("imports must form a prefix"))
        );
    }

    #[test]
    fn source_module_declarations_are_not_language_syntax() {
        let parsed = parse(
            "primitives.eqi",
            "module library.primitives; public component Resistor() {}",
        );
        assert!(parsed.diagnostics().iter().any(|diagnostic| {
            diagnostic
                .message()
                .contains("expected `import`, `dimension`")
        }));
    }

    #[test]
    fn declaration_diagnostic_offers_header_forms_only_while_their_prefix_is_open() {
        let before = parse("before.eqi", "unexpected");
        assert_eq!(
            before.diagnostics()[0].message(),
            "expected `import`, `dimension`, `property`, `connector`, `component`, `operator`, or `model` declaration"
        );

        let after = parse("after.eqi", "model Main() {} unexpected");
        assert_eq!(
            after.diagnostics()[0].message(),
            "expected `dimension`, `property`, `connector`, `component`, `operator`, or `model` declaration"
        );
    }
}
