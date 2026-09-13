//! Recovering recursive-descent parser for the deliberately small v0 grammar.
use eqiora_core::diagnostic::codes;
use eqiora_core::{Diagnostic, Span};

mod call;
mod clock;
mod comments;
mod compile_time;
mod component;
mod connection;
mod constraint;
mod dimension;
mod document;
mod domain;
mod enumeration;
mod event;
mod expression;
mod formulation;
mod instance;
mod law;
mod nominal;
mod notation;
mod operator;
mod partial;
mod property;
mod record;
mod recovery;
mod relation;
mod select;
mod signature;
mod value_type;

use crate::ast::{
    BinaryOp, BoundaryConnectionDecl, BoundaryPairingSyntax, BoundaryPortReferenceSyntax,
    BoundaryPortSelectorSyntax, BoundarySideSyntax, CallArguments, ComponentItem,
    ComponentParameterDecl, ComponentPortDecl, ComponentPortFamilyDecl, ConnectionDecl,
    ConnectionSyntax, ConnectorDecl, ConnectorQuantitySyntax, ConnectorSyntax, Document,
    DomainDecl, DomainSyntax, ExactIntegerSyntax, Expr, ExprKind, FamilyBinderSyntax, FieldDecl,
    FrameSyntax, InstanceDecl, Item, NamePath, NamedBindingDecl, PortDecl, PortSyntax,
    PureOperatorDecl, PureOperatorFormal, PureValueClassSyntax, SignalDirectionSyntax,
    SupportSlotDecl, SupportSlotSyntax, TextRange, UnaryOp, ValueShapeSyntax, VisibilitySyntax,
};
use crate::lexer::{Token, TokenKind, lex};
use relation::ParsedRelation;

/// Parsed document, lossless tokens, and accumulated source diagnostics.
#[derive(Debug, Clone, PartialEq)]
pub struct ParseResult {
    file: String,
    tokens: Vec<Token>,
    document: Option<Document>,
    diagnostics: Vec<Diagnostic>,
}

impl ParseResult {
    fn missing_declaration_diagnostic(&self, message: &'static str) -> Diagnostic {
        let range = self
            .tokens
            .last()
            .map_or(TextRange::new(0, 0), Token::range);
        Diagnostic::error(codes::SYNTAX_ERROR, message).with_span(Span {
            file: self.file.clone(),
            start: range.start(),
            end: range.end(),
        })
    }

    /// Lossless token stream, including trivia and EOF.
    #[must_use]
    pub fn tokens(&self) -> &[Token] {
        &self.tokens
    }

    /// Recovered syntax tree when at least one top-level declaration was parsed.
    #[must_use]
    pub const fn document(&self) -> Option<&Document> {
        self.document.as_ref()
    }

    /// Lexical and syntactic diagnostics in source order.
    #[must_use]
    pub fn diagnostics(&self) -> &[Diagnostic] {
        &self.diagnostics
    }

    /// Consume the result only when no diagnostic was produced.
    ///
    /// # Errors
    /// Returns all accumulated diagnostics, including recoverable errors.
    pub fn into_document(self) -> Result<Document, Vec<Diagnostic>> {
        let missing =
            self.missing_declaration_diagnostic("source contains no top-level declaration");
        if self.diagnostics.is_empty() {
            self.document.ok_or_else(|| vec![missing])
        } else {
            Err(self.diagnostics)
        }
    }

    /// Consume the result only when it contains an executable Model and no
    /// diagnostic was produced.
    ///
    /// Declaration-only documents are valid library syntax through
    /// [`Self::into_document`], but are not executable compilation entries.
    ///
    /// # Errors
    /// Returns all accumulated diagnostics and reports a missing Model when
    /// the recovered document contains declarations only.
    pub fn into_compilation_document(self) -> Result<Document, Vec<Diagnostic>> {
        let missing =
            self.missing_declaration_diagnostic("source requires at least one `model` declaration");
        let mut diagnostics = self.diagnostics;
        let has_model = self
            .document
            .as_ref()
            .is_some_and(|document| !document.models.is_empty());
        if !has_model {
            diagnostics.push(missing);
        }
        if diagnostics.is_empty() {
            self.document.ok_or_else(|| {
                vec![Diagnostic::error(
                    codes::SYNTAX_ERROR,
                    "parser invariant violated: a Model-bearing document was not retained",
                )]
            })
        } else {
            Err(diagnostics)
        }
    }
}

/// Parse one Eqiora Language source file, including declaration-only package
/// library units.
#[must_use]
pub fn parse(file: impl Into<String>, source: &str) -> ParseResult {
    let file = file.into();
    let (tokens, mut diagnostics) = lex(file.clone(), source);
    let mut parser = Parser {
        file: file.clone(),
        tokens: &tokens,
        cursor: 0,
        expression_recursion: 0,
        exact_numeric: false,
        notations: Vec::new(),
        diagnostics: Vec::new(),
    };
    let mut document = parser.parse_document();
    diagnostics.extend(parser.diagnostics);
    if let Some(document) = &mut document {
        comments::attach(document, &tokens, source, &file, &mut diagnostics);
        notation::attach(document, &tokens, parser.notations);
    }
    ParseResult {
        file,
        tokens,
        document,
        diagnostics,
    }
}

struct Parser<'a> {
    file: String,
    tokens: &'a [Token],
    cursor: usize,
    expression_recursion: usize,
    exact_numeric: bool,
    notations: Vec<(TextRange, crate::Notation)>,
    diagnostics: Vec<Diagnostic>,
}

enum ParsedComponentItem {
    Retained(Box<ComponentItem>),
    Discarded,
}

enum ParsedComponentPort {
    Ordinary(ComponentPortDecl),
    Family(ComponentPortFamilyDecl),
}

enum ParsedConnection {
    Ordinary(ConnectionDecl),
    Boundary(BoundaryConnectionDecl),
}

impl Parser<'_> {
    fn parse_connector(
        &mut self,
        start: u32,
        visibility: VisibilitySyntax,
    ) -> Option<ConnectorDecl> {
        self.expect_keyword("connector")?;
        let name = self.declaration_name("Connector name")?.text().to_owned();
        self.expect(TokenKind::LeftBrace, "`{` before Connector quantities")?;
        let syntax = if self.at_keyword("across") {
            self.bump();
            let across_name = self
                .expect_identifier("across quantity name")?
                .text()
                .to_owned();
            self.expect(TokenKind::Colon, "`:` before across type")?;
            let across_type = self.parse_value_type()?;
            self.expect(TokenKind::Semicolon, "`;` after across quantity")?;
            self.expect_keyword("through")?;
            let through_name = self
                .expect_identifier("through quantity name")?
                .text()
                .to_owned();
            self.expect(TokenKind::Colon, "`:` before through type")?;
            let through_type = self.parse_value_type()?;
            self.expect(TokenKind::Semicolon, "`;` after through quantity")?;
            if across_name == through_name {
                self.error_previous("Connector quantity names must be distinct");
                return None;
            }
            ConnectorSyntax::ScalarPhysical {
                across_name,
                across_type,
                through_name,
                through_type,
            }
        } else {
            self.parse_field_physical_connector()?
        };
        let end = self
            .expect(TokenKind::RightBrace, "`}` after Connector quantities")?
            .range()
            .end();
        Some(ConnectorDecl {
            comments: Default::default(),
            visibility,
            name,
            syntax,
            range: TextRange::new(start, end),
        })
    }

    fn parse_item(&mut self) -> Option<Item> {
        if self.at_keyword("domain") {
            self.parse_domain().map(Item::Domain)
        } else if self.at_keyword("variable") || self.at_keyword("state") {
            self.parse_field(true).map(Item::Field)
        } else if self.at_keyword("initial") {
            self.parse_initial().map(Item::Initial)
        } else if self.at_keyword("parameter") {
            self.parse_parameter().map(Item::Parameter)
        } else if self.at_keyword("observable") {
            self.parse_observable().map(Item::Observable)
        } else if self.at_keyword("indexset") {
            self.parse_index_set().map(Item::IndexSet)
        } else if self.at_keyword("let") {
            self.parse_let().map(Item::Let)
        } else if self.at_keyword("port") {
            self.parse_port().map(Item::Port)
        } else if self.at_keyword("event") {
            self.parse_event().map(Item::Event)
        } else if self.at_keyword("clock") {
            self.parse_clock().map(Item::Clock)
        } else if self.at_keyword("law") {
            self.parse_law().map(Item::Relation)
        } else if self.at_keyword("relation") {
            self.parse_component_relation()
                .map(|relation| match relation {
                    ParsedRelation::Ordinary(value) => Item::Relation(value),
                    ParsedRelation::Family(value) => Item::RelationFamily(value),
                })
        } else if self.at_keyword("connect") {
            self.parse_connection(false)
                .map(|connection| match connection {
                    ParsedConnection::Ordinary(connection) => Item::Connection(connection),
                    ParsedConnection::Boundary(connection) => Item::BoundaryConnection(connection),
                })
        } else if self.at_keyword("instance") {
            self.parse_instance().map(Item::Instance)
        } else {
            self.error_here(
                "expected domain, variable, state, initial, parameter, let, port, clock, relation, law, connect, boundary, or instance",
            );
            None
        }
    }

    fn parse_scalar_physical_types(
        &mut self,
    ) -> Option<(
        String,
        crate::ValueTypeSyntax,
        String,
        crate::ValueTypeSyntax,
    )> {
        self.expect_keyword("scalar_physical")?;
        self.expect(TokenKind::LeftParen, "`(` after `scalar_physical`")?;
        self.expect_keyword("across")?;
        let across_name = self
            .expect_identifier("across quantity name")?
            .text()
            .to_owned();
        self.expect(TokenKind::Colon, "`:` before across type")?;
        let across_type = self.parse_value_type()?;
        self.expect(TokenKind::Comma, "`,` between physical types")?;
        self.expect_keyword("through")?;
        let through_name = self
            .expect_identifier("through quantity name")?
            .text()
            .to_owned();
        self.expect(TokenKind::Colon, "`:` before through type")?;
        let through_type = self.parse_value_type()?;
        self.expect(TokenKind::RightParen, "`)` after physical types")?;
        if across_name == through_name {
            self.error_previous("physical quantity names must be distinct");
            return None;
        }
        Some((across_name, across_type, through_name, through_type))
    }

    fn parse_field_physical_connector(&mut self) -> Option<ConnectorSyntax> {
        let mut trace = None;
        let mut flux = None;
        let mut shape = None;
        let mut frame = None;
        let mut pairing = None;
        let mut orientation = None;

        while !self.at(TokenKind::RightBrace) && !self.at(TokenKind::Eof) {
            let field = self
                .expect_identifier("field-physical Connector field")?
                .text()
                .to_owned();
            match field.as_str() {
                "trace" => {
                    if trace.is_some() {
                        self.error_previous("duplicate `trace` Connector field");
                        return None;
                    }
                    trace = Some(self.parse_connector_quantity("trace")?);
                }
                "flux" => {
                    if flux.is_some() {
                        self.error_previous("duplicate `flux` Connector field");
                        return None;
                    }
                    flux = Some(self.parse_connector_quantity("flux")?);
                }
                "shape" => {
                    if shape.is_some() {
                        self.error_previous("duplicate `shape` Connector field");
                        return None;
                    }
                    shape = Some(self.parse_value_shape()?);
                }
                "frame" => {
                    if frame.is_some() {
                        self.error_previous("duplicate `frame` Connector field");
                        return None;
                    }
                    frame = Some(if self.at_keyword("invariant") {
                        self.bump();
                        FrameSyntax::Invariant
                    } else if self.at_keyword("spatial") {
                        self.bump();
                        FrameSyntax::Spatial
                    } else {
                        self.error_here("expected `invariant` or `spatial` frame");
                        return None;
                    });
                }
                "pairing" => {
                    if pairing.is_some() {
                        self.error_previous("duplicate `pairing` Connector field");
                        return None;
                    }
                    self.expect_keyword("euclidean_boundary_duality")?;
                    pairing = Some(BoundaryPairingSyntax::EuclideanBoundaryDuality);
                }
                "orientation" => {
                    if orientation.is_some() {
                        self.error_previous("duplicate `orientation` Connector field");
                        return None;
                    }
                    self.expect_keyword("parent_outward")?;
                    orientation = Some(());
                }
                _ => {
                    self.error_previous(format!(
                        "unknown field-physical Connector field `{field}`"
                    ));
                    return None;
                }
            }
            self.expect(TokenKind::Semicolon, "`;` after Connector contract entry")?;
        }
        self.require_connector_field(orientation, "orientation parent_outward")?;
        let trace = self.require_connector_field(trace, "trace")?;
        let flux = self.require_connector_field(flux, "flux")?;
        if trace.name == flux.name {
            self.error_previous("Connector quantity names must be distinct");
            return None;
        }
        Some(ConnectorSyntax::FieldPhysical {
            trace,
            flux,
            shape: self.require_connector_field(shape, "shape")?,
            frame: self.require_connector_field(frame, "frame")?,
            pairing: self.require_connector_field(pairing, "pairing")?,
        })
    }

    fn parse_connector_quantity(&mut self, role: &str) -> Option<ConnectorQuantitySyntax> {
        let name = self
            .expect_identifier(&format!("{role} quantity name"))?
            .text()
            .to_owned();
        self.expect(TokenKind::Colon, "`:` before quantity dimension")?;
        Some(ConnectorQuantitySyntax {
            name,
            dimension: Box::new(self.parse_dimension_expression()?),
        })
    }

    fn require_connector_field<T>(&mut self, value: Option<T>, field: &str) -> Option<T> {
        value.or_else(|| {
            self.error_previous(format!(
                "field-physical Connector requires exactly one `{field}` field"
            ));
            None
        })
    }

    fn parse_value_shape(&mut self) -> Option<ValueShapeSyntax> {
        if self.at_keyword("scalar") {
            self.bump();
            return Some(ValueShapeSyntax::Scalar);
        }
        if self.at_keyword("spatial_vector") {
            self.bump();
            return Some(ValueShapeSyntax::SpatialVector);
        }
        self.expect(
            TokenKind::LeftBracket,
            "`scalar`, `spatial_vector`, or `[` for an exact value shape",
        )?;
        let mut extents = Vec::new();
        if !self.at(TokenKind::RightBracket) {
            loop {
                let extent = self.parse_u64("positive value-shape extent")?;
                if extent == 0 {
                    self.error_previous("value-shape extents must be positive");
                    return None;
                }
                let extent = u32::try_from(extent).ok().or_else(|| {
                    self.error_previous("value-shape extent exceeds the portable u32 range");
                    None
                })?;
                extents.push(extent);
                if !self.at(TokenKind::Comma) {
                    break;
                }
                self.bump();
            }
        }
        self.expect(TokenKind::RightBracket, "`]` after exact value shape")?;
        if extents.is_empty() {
            Some(ValueShapeSyntax::Scalar)
        } else {
            Some(ValueShapeSyntax::Exact(extents))
        }
    }

    fn parse_field(&mut self, terminated: bool) -> Option<FieldDecl> {
        let role = if self.at_keyword("variable")
            || (!terminated && (self.at_keyword("input") || self.at_keyword("output")))
        {
            crate::ast::FieldRoleSyntax::Variable
        } else if self.at_keyword("state") {
            crate::ast::FieldRoleSyntax::State
        } else {
            self.error_here("expected `variable` or `state`");
            return None;
        };
        let start = self.bump().range().start();
        let name = self.declaration_name("declaration name")?.text().to_owned();
        self.expect(TokenKind::Colon, "`:` before mathematical type")?;
        let value_type = self.parse_value_type()?;
        let domain = if self.at_keyword("on") {
            self.bump();
            Some(self.expect_identifier("unknown support")?.text().to_owned())
        } else {
            None
        };
        let activation = if self.at_keyword("at") {
            self.bump();
            crate::ActivationSyntax::Named(
                self.expect_identifier("unknown clock")?.text().to_owned(),
            )
        } else {
            crate::ActivationSyntax::Continuous
        };
        let end = if terminated {
            self.expect(TokenKind::Semicolon, "`;` after declaration")?
                .range()
                .end()
        } else {
            self.previous_significant_range().end()
        };
        Some(FieldDecl {
            comments: Default::default(),
            name,
            domain,
            role,
            activation,
            value_type,
            range: TextRange::new(start, end),
        })
    }

    fn parse_component_parameter(
        &mut self,
        start: u32,
        visibility: VisibilitySyntax,
        terminated: bool,
    ) -> Option<ComponentParameterDecl> {
        self.expect_keyword("parameter")?;
        let name = self
            .declaration_name("component Parameter name")?
            .text()
            .to_owned();
        self.expect(TokenKind::Colon, "`:` before component Parameter type")?;
        let value_type = self.parse_value_type()?;
        let default = if self.at(TokenKind::Equal) {
            self.bump();
            Some(self.parse_expression(0)?)
        } else {
            None
        };
        let end = if terminated {
            self.expect(TokenKind::Semicolon, "`;` after component Parameter")?
                .range()
                .end()
        } else {
            self.previous_significant_range().end()
        };
        Some(ComponentParameterDecl {
            comments: Default::default(),
            visibility,
            name,
            value_type,
            default,
            range: TextRange::new(start, end),
        })
    }

    fn parse_support_slot(
        &mut self,
        start: u32,
        visibility: VisibilitySyntax,
    ) -> Option<SupportSlotDecl> {
        self.expect_keyword("support")?;
        let name = self
            .declaration_name("component support-slot name")?
            .text()
            .to_owned();
        self.expect(TokenKind::Colon, "`:` before support-slot contract")?;
        let syntax = if self.at_keyword("volume") {
            self.bump();
            self.expect(TokenKind::LeftParen, "`(` after `volume`")?;
            self.expect_keyword("ambient_dimension")?;
            self.expect(TokenKind::Equal, "`=` after `ambient_dimension`")?;
            let ambient_dimension = self.parse_u64("ambient dimension")?;
            let ambient_dimension = usize::try_from(ambient_dimension).ok().or_else(|| {
                self.error_previous("ambient dimension exceeds this platform's usize range");
                None
            })?;
            self.expect(TokenKind::RightParen, "`)` after volume support contract")?;
            SupportSlotSyntax::Volume { ambient_dimension }
        } else if self.at_keyword("boundary") {
            self.bump();
            self.expect(TokenKind::LeftParen, "`(` after `boundary`")?;
            self.expect_keyword("parent")?;
            self.expect(TokenKind::Equal, "`=` after boundary parent")?;
            let parent = self
                .expect_identifier("parent support-slot name")?
                .text()
                .to_owned();
            self.expect(TokenKind::RightParen, "`)` after boundary support contract")?;
            SupportSlotSyntax::Boundary { parent }
        } else if self.at_keyword("complete_exterior") {
            self.bump();
            self.expect(TokenKind::LeftParen, "`(` after `complete_exterior`")?;
            self.expect_keyword("parent")?;
            self.expect(TokenKind::Equal, "`=` after complete exterior parent")?;
            let parent = self
                .expect_identifier("parent support-slot name")?
                .text()
                .to_owned();
            self.expect(
                TokenKind::RightParen,
                "`)` after complete exterior support contract",
            )?;
            SupportSlotSyntax::CompleteExterior { parent }
        } else {
            self.error_here(
                "expected `volume(...)`, `boundary(...)`, or `complete_exterior(...)` support-slot contract",
            );
            return None;
        };
        let end = self.previous_significant_range().end();
        Some(SupportSlotDecl {
            comments: Default::default(),
            visibility,
            name,
            syntax,
            range: TextRange::new(start, end),
        })
    }

    fn parse_port(&mut self) -> Option<PortDecl> {
        let start = self.expect_keyword("port")?.range().start();
        let name = self.declaration_name("Port name")?.text().to_owned();
        self.expect(TokenKind::Colon, "`:` before Port contract")?;
        let syntax = if self.at_keyword("signal") {
            self.parse_signal_port_syntax()?
        } else {
            let connector = self.parse_name_path("physical Domain or Connector name")?;
            if self.at_keyword("over") {
                self.bump();
                PortSyntax::FieldPhysical {
                    connector,
                    support: self
                        .expect_identifier("boundary support name")?
                        .text()
                        .to_owned(),
                }
            } else if connector.segments().count() == 1 {
                PortSyntax::ScalarPhysical {
                    domain: connector.as_str().to_owned(),
                }
            } else {
                self.error_here("a model-local physical Domain requires a local declaration name");
                return None;
            }
        };
        let end = self
            .expect(TokenKind::Semicolon, "`;` after Port declaration")?
            .range()
            .end();
        Some(PortDecl {
            comments: Default::default(),
            name,
            syntax,
            range: TextRange::new(start, end),
        })
    }

    fn parse_component_port(
        &mut self,
        start: u32,
        visibility: VisibilitySyntax,
        terminated: bool,
    ) -> Option<ParsedComponentPort> {
        self.expect_keyword("port")?;
        let name = self
            .declaration_name("component Port name")?
            .text()
            .to_owned();
        let binder = self
            .at(TokenKind::LeftBracket)
            .then(|| self.parse_boundary_family_binder())
            .flatten();
        self.expect(TokenKind::Colon, "`:` before component Port contract")?;
        let syntax = if self.at_keyword("signal") {
            self.parse_signal_port_syntax()?
        } else {
            let connector = self.parse_name_path("physical Connector name")?;
            if self.at_keyword("over") {
                self.bump();
                PortSyntax::FieldPhysical {
                    connector,
                    support: self
                        .expect_identifier("boundary support-slot name")?
                        .text()
                        .to_owned(),
                }
            } else {
                PortSyntax::ScalarPhysicalConnector { connector }
            }
        };
        let end = if terminated {
            self.expect(TokenKind::Semicolon, "`;` after component Port")?
                .range()
                .end()
        } else {
            self.previous_significant_range().end()
        };
        let port = ComponentPortDecl {
            comments: Default::default(),
            visibility,
            name,
            syntax,
            range: TextRange::new(start, end),
        };
        let Some(binder) = binder else {
            return Some(ParsedComponentPort::Ordinary(port));
        };
        let PortSyntax::FieldPhysical { support, .. } = port.syntax() else {
            self.error_here("a boundary family binder requires a field-physical Port");
            return None;
        };
        if support != binder.member() {
            self.error_here(
                "a field-physical Port family must be declared over its bound boundary member",
            );
            return None;
        }
        Some(ParsedComponentPort::Family(ComponentPortFamilyDecl {
            port,
            binder,
        }))
    }

    fn parse_signal_port_syntax(&mut self) -> Option<PortSyntax> {
        self.expect_keyword("signal")?;
        let direction = if self.at_keyword("input") {
            self.bump();
            SignalDirectionSyntax::Input
        } else if self.at_keyword("output") {
            self.bump();
            SignalDirectionSyntax::Output
        } else {
            self.error_here("expected `input` or `output` after `signal`");
            return None;
        };
        let value_type = self.parse_value_type()?;
        let domain = if self.at_keyword("on") {
            self.bump();
            Some(self.expect_identifier("signal support")?.text().to_owned())
        } else {
            None
        };
        let activation = if self.at_keyword("at") {
            self.bump();
            crate::ActivationSyntax::Named(
                self.expect_identifier("signal clock")?.text().to_owned(),
            )
        } else {
            crate::ActivationSyntax::Continuous
        };
        Some(PortSyntax::Signal {
            direction,
            value_type,
            domain,
            activation,
        })
    }

    fn parse_name_path(&mut self, expected: &str) -> Option<NamePath> {
        let first = self.expect_identifier(expected)?;
        self.parse_name_path_from_first(first, expected)
    }

    fn parse_name_path_from_first(&mut self, first: Token, expected: &str) -> Option<NamePath> {
        let start = first.range().start();
        let mut end = first.range().end();
        let mut segments = vec![first.text().to_owned()];
        while self.at(TokenKind::Dot) {
            self.bump();
            let segment = self.expect_identifier(expected)?;
            end = segment.range().end();
            segments.push(segment.text().to_owned());
        }
        Some(NamePath::from_parsed_segments(
            segments,
            TextRange::new(start, end),
        ))
    }

    fn parse_u64(&mut self, expected: &str) -> Option<u64> {
        let token = self.expect(TokenKind::Number, expected)?;
        token.text().parse::<u64>().ok().or_else(|| {
            self.error_token(&token, format!("{expected} must be an unsigned integer"));
            None
        })
    }

    fn parse_signed_number(&mut self) -> Option<f64> {
        let negative = self.at(TokenKind::Minus);
        if negative {
            self.bump();
        }
        let token = self.expect(TokenKind::Number, "numeric literal")?;
        self.parse_f64(&token)
            .map(|value| if negative { -value } else { value })
    }

    fn parse_f64(&mut self, token: &Token) -> Option<f64> {
        match token.text().parse::<f64>() {
            Ok(value)
                if value == 0.0
                    && token
                        .text()
                        .split(['e', 'E'])
                        .next()
                        .is_some_and(|significand| {
                            significand.bytes().any(|byte| matches!(byte, b'1'..=b'9'))
                        }) =>
            {
                self.error_token(token, "nonzero numeric literal underflows the f64 range");
                None
            }
            Ok(value) if value.is_finite() => Some(value),
            _ => {
                self.error_token(token, "numeric literal must be a finite f64 value");
                None
            }
        }
    }

    fn at_keyword(&mut self, keyword: &str) -> bool {
        self.current().kind() == TokenKind::Identifier && self.current().text() == keyword
    }

    fn expect_keyword(&mut self, keyword: &str) -> Option<Token> {
        if self.at_keyword(keyword) {
            Some(self.bump())
        } else {
            self.error_here(format!("expected `{keyword}`"));
            None
        }
    }

    fn expect_identifier(&mut self, expected: &str) -> Option<Token> {
        self.expect(TokenKind::Identifier, expected)
    }

    fn expect(&mut self, kind: TokenKind, expected: &str) -> Option<Token> {
        if self.at(kind) {
            Some(self.bump())
        } else {
            self.error_here(format!("expected {expected}"));
            None
        }
    }

    fn at(&mut self, kind: TokenKind) -> bool {
        self.current().kind() == kind
    }

    fn current(&mut self) -> &Token {
        self.skip_trivia();
        &self.tokens[self.cursor.min(self.tokens.len() - 1)]
    }

    fn bump(&mut self) -> Token {
        self.skip_trivia();
        let token = self.tokens[self.cursor.min(self.tokens.len() - 1)].clone();
        if token.kind() != TokenKind::Eof {
            self.cursor += 1;
        }
        token
    }

    fn skip_trivia(&mut self) {
        while self
            .tokens
            .get(self.cursor)
            .is_some_and(|token| token.kind().is_trivia())
        {
            self.cursor += 1;
        }
    }

    fn error_here(&mut self, message: impl Into<String>) {
        let token = self.current().clone();
        self.error_token(&token, message);
    }

    fn error_previous(&mut self, message: impl Into<String>) {
        let token = self.tokens[self.cursor.saturating_sub(1)].clone();
        self.error_token(&token, message);
    }

    fn error_token(&mut self, token: &Token, message: impl Into<String>) {
        self.diagnostics.push(
            Diagnostic::error(codes::SYNTAX_ERROR, message).with_span(Span {
                file: self.file.clone(),
                start: token.range().start(),
                end: token.range().end(),
            }),
        );
    }
}

#[cfg(test)]
mod tests;
