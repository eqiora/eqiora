//! Client-neutral, immutable declarations for native model construction.
//!
//! Drafts are an ergonomic control-plane input. They are neither accepted
//! Semantic Models nor a durable wire format. The compiler lowers their
//! synthetic AST through exactly the same path as parsed source.

use std::collections::{HashMap, HashSet};
use std::ops::{Add, Div, Mul, Neg, Sub};

use crate::ast::{BinaryOp, Expr, ExprKind, NamePath, TextRange, UnaryOp};
use crate::draft_spatial::{DraftSpatialDomain, DraftSpatialDomainKind};
use eqiora_core::diagnostic::codes;
use eqiora_core::{Diagnostic, DimExponents, GraphPath, ValueLiteral, ValueType};

/// One immutable native model definition request.
#[derive(Debug, Clone)]
struct ModelDeclarations {
    name: String,
    declarations: Vec<DraftDeclaration>,
}

impl ModelDeclarations {
    /// Close a set of declarations into one atomic model draft.
    ///
    /// # Errors
    /// Returns structured diagnostics for invalid or duplicate names, invalid
    /// scalar values, empty Relations, omitted declaration identities, and
    /// invalid conserving-connection membership.
    pub fn new(
        name: impl Into<String>,
        declarations: impl IntoIterator<Item = DraftDeclaration>,
    ) -> Result<Self, Vec<Diagnostic>> {
        let name = name.into();
        let mut admitted = Vec::new();
        let mut expression_nodes = 0usize;
        for declaration in declarations {
            let error = |message| vec![native_diagnostic(&name, "declarations", message)];
            if admitted.len() == crate::SourceAstFactory::MAX_CONTAINER_MEMBERS {
                return Err(error("native module exceeds the shared declaration limit"));
            }
            if let DraftDeclaration::Parameter(parameter) = &declaration {
                let nodes = crate::SourceAstFactory::value_literal_nodes(
                    parameter.value(),
                    parameter.frame.is_some(),
                )
                .map_err(|failure| {
                    vec![native_diagnostic(
                        &name,
                        "declarations",
                        failure.to_string(),
                    )]
                })?;
                expression_nodes = expression_nodes
                    .checked_add(nodes)
                    .filter(|count| *count <= crate::SourceAstFactory::MAX_EXPRESSION_NODES)
                    .ok_or_else(|| {
                        error("native module exceeds the shared expression node limit")
                    })?;
            }
            if let Some((_, equations)) = declaration.equations() {
                if equations.len() > crate::SourceAstFactory::MAX_CONTAINER_MEMBERS {
                    return Err(error(
                        "native equation group exceeds the shared member limit",
                    ));
                }
                for expression in equations.iter().flat_map(|(left, right)| [left, right]) {
                    expression_nodes = expression_nodes
                        .checked_add(expression.nodes)
                        .filter(|count| *count <= crate::SourceAstFactory::MAX_EXPRESSION_NODES)
                        .ok_or_else(|| {
                            error("native module exceeds the shared expression node limit")
                        })?;
                }
            }
            if let DraftDeclaration::Observable(value) = &declaration {
                expression_nodes = expression_nodes
                    .checked_add(value.expression.nodes)
                    .filter(|count| *count <= crate::SourceAstFactory::MAX_EXPRESSION_NODES)
                    .ok_or_else(|| {
                        error("native module exceeds the shared expression node limit")
                    })?;
            }
            admitted.push(declaration);
        }
        let value = Self {
            name,
            declarations: admitted,
        };
        value.validate()?;
        Ok(value)
    }

    fn validate(&self) -> Result<(), Vec<Diagnostic>> {
        let mut diagnostics = Vec::new();
        let mut names = HashMap::new();
        let mut value_symbols = HashSet::new();
        let mut domain_symbols = HashSet::new();
        let mut spatial_domain_symbols = HashSet::new();
        let mut ports = HashMap::new();

        if !is_language_identifier(&self.name) {
            diagnostics.push(
                Diagnostic::error(
                    codes::LANGUAGE_TYPE_ERROR,
                    format!(
                        "native model name `{}` is not an Eqiora Language identifier",
                        self.name
                    ),
                )
                .with_graph_path(GraphPath::new([self.name.clone()])),
            );
        }

        for declaration in &self.declarations {
            if let Some(name) = declaration.name() {
                if !is_language_identifier(name) {
                    diagnostics.push(native_diagnostic(
                        &self.name,
                        name,
                        format!("declaration name `{name}` is not an Eqiora Language identifier"),
                    ));
                }
                if let Some(previous) = names.insert(name, declaration.kind_name()) {
                    diagnostics.push(native_diagnostic(
                        &self.name,
                        name,
                        format!(
                            "duplicate declaration `{name}` conflicts with an earlier {previous}"
                        ),
                    ));
                }
            }
            match declaration {
                DraftDeclaration::Field(value) => {
                    value_symbols.insert(value.symbol.clone());
                }
                DraftDeclaration::Parameter(value) => {
                    value_symbols.insert(value.symbol.clone());
                }
                DraftDeclaration::PhysicalDomain(value) => {
                    domain_symbols.insert(value.symbol.clone());
                    if !is_language_identifier(&value.across_name)
                        || !is_language_identifier(&value.through_name)
                        || value.across_name == value.through_name
                    {
                        diagnostics.push(native_diagnostic(
                            &self.name,
                            value.name(),
                            "physical quantity names must be distinct valid Eqiora Language identifiers",
                        ));
                    }
                    if !value.across_type.shape().is_scalar()
                        || !value.through_type.shape().is_scalar()
                    {
                        diagnostics.push(native_diagnostic(
                            &self.name,
                            value.name(),
                            "scalar physical quantities require scalar mathematical types",
                        ));
                    }
                }
                DraftDeclaration::SpatialDomain(value) => {
                    spatial_domain_symbols.insert(value.symbol().clone());
                }
                DraftDeclaration::ConservingPort(value) => {
                    ports.insert(value.symbol.clone(), value);
                }
                DraftDeclaration::Enum { name, definition } => {
                    if let Err(error) = nominal::validate_enum(name, definition) {
                        diagnostics.push(native_diagnostic(&self.name, name, error.to_string()));
                    }
                }
                DraftDeclaration::Observable(value) => {
                    value_symbols.insert(value.symbol.clone());
                }
                DraftDeclaration::FiniteSpace { .. }
                | DraftDeclaration::IndexSet { .. }
                | DraftDeclaration::Relation(_)
                | DraftDeclaration::Initial(_)
                | DraftDeclaration::ConservingConnection(_) => {}
            }
            if let Some((path, residuals)) = declaration.equations() {
                if residuals.is_empty() {
                    diagnostics.push(native_diagnostic(
                        &self.name,
                        path,
                        "equation group requires at least one equation",
                    ));
                }
                for error in residuals
                    .iter()
                    .flat_map(|(left, right)| [left, right])
                    .filter_map(DraftExpression::construction_error)
                {
                    diagnostics.push(native_diagnostic(&self.name, path, error.message()));
                }
            }
            match declaration {
                DraftDeclaration::FiniteSpace { name, definition } => {
                    if let Err(error) = self.validate_space(name, definition) {
                        diagnostics.push(native_diagnostic(&self.name, name, error.to_string()));
                    }
                }
                DraftDeclaration::Field(field) => {
                    if let Err(error) = self.validate_enum_type(&field.value_type) {
                        diagnostics.push(native_diagnostic(
                            &self.name,
                            field.name(),
                            error.to_string(),
                        ));
                    }
                    if let Err(message) = crate::ValueTypeSyntax::from_checked(
                        &field.value_type,
                        |id| self.nominal_name(id),
                        |basis| self.product_name(basis),
                    )
                    .map_err(|error| error.to_string())
                    {
                        diagnostics.push(native_diagnostic(&self.name, field.name(), message));
                    }
                }
                DraftDeclaration::Observable(observable) => {
                    if let Some(domain) = &observable.domain
                        && !spatial_domain_symbols.contains(domain.symbol())
                    {
                        diagnostics.push(native_diagnostic(
                            &self.name,
                            observable.name(),
                            "Observable output support references a foreign or omitted Domain",
                        ));
                    }

                    if let Err(error) = self.validate_enum_type(&observable.value_type) {
                        diagnostics.push(native_diagnostic(
                            &self.name,
                            observable.name(),
                            error.to_string(),
                        ));
                    }
                    if let Err(error) = crate::ValueTypeSyntax::from_checked(
                        &observable.value_type,
                        |id| self.nominal_name(id),
                        |basis| self.product_name(basis),
                    ) {
                        diagnostics.push(native_diagnostic(
                            &self.name,
                            observable.name(),
                            error.to_string(),
                        ));
                    }
                    if let Some(error) = observable.expression.construction_error() {
                        diagnostics.push(native_diagnostic(
                            &self.name,
                            observable.name(),
                            error.message(),
                        ));
                    }
                }
                DraftDeclaration::Parameter(parameter) => {
                    if let Err(error) = crate::SourceAstFactory::value_literal(
                        parameter.value(),
                        parameter.frame_name(TextRange::new(0, 1)),
                        TextRange::new(0, 1),
                        |id| self.nominal_name(id),
                        |id| self.enum_definition(id),
                        |basis| self.product_name(basis),
                    ) {
                        diagnostics.push(native_diagnostic(
                            &self.name,
                            parameter.name(),
                            error.to_string(),
                        ));
                    }
                    if let Err(message) = value_type::validate(parameter.value_type()) {
                        diagnostics.push(native_diagnostic(&self.name, parameter.name(), message));
                    }
                }
                _ => {}
            }
        }

        for declaration in &self.declarations {
            let DraftDeclaration::SpatialDomain(domain) = declaration else {
                continue;
            };
            if let DraftSpatialDomainKind::Boundary { parent, .. } = domain.kind()
                && !spatial_domain_symbols.contains(parent.symbol())
            {
                diagnostics.push(native_diagnostic(
                    &self.name,
                    domain.name(),
                    format!(
                        "boundary Domain `{}` references foreign or omitted parent Domain `{}`",
                        domain.name(),
                        parent.name()
                    ),
                ));
            }
        }

        for declaration in &self.declarations {
            let DraftDeclaration::Parameter(parameter) = declaration else {
                continue;
            };
            let Some(frame) = &parameter.frame else {
                continue;
            };
            if !spatial_domain_symbols.contains(frame.symbol()) {
                diagnostics.push(native_diagnostic(
                    &self.name,
                    parameter.name(),
                    format!(
                        "Parameter frame references foreign or omitted Domain `{}`",
                        frame.name()
                    ),
                ));
            }
            let mut volume = frame;
            while let Some(parent) = volume.parent() {
                volume = parent;
            }
            let dimension = volume.bounds().expect("Cartesian frame provider").len();
            if parameter
                .value_type()
                .shape()
                .extents()
                .iter()
                .skip(parameter.value_type().array_rank())
                .any(|extent| extent.get() as usize != dimension)
            {
                diagnostics.push(native_diagnostic(
                    &self.name,
                    parameter.name(),
                    "Parameter spatial axes differ from the frame ambient dimension",
                ));
            }
        }

        for declaration in &self.declarations {
            let DraftDeclaration::Field(field) = declaration else {
                continue;
            };
            let Some(scope) = &field.spatial_scope else {
                continue;
            };
            if !spatial_domain_symbols.contains(scope.domain.symbol()) {
                diagnostics.push(native_diagnostic(
                    &self.name,
                    field.name(),
                    format!(
                        "spatial Field `{}` references foreign or omitted Domain `{}`",
                        field.name(),
                        scope.domain.name()
                    ),
                ));
            }
        }

        for declaration in &self.declarations {
            let DraftDeclaration::Relation(relation) = declaration else {
                continue;
            };
            if let Some(domain) = &relation.domain
                && !spatial_domain_symbols.contains(domain.symbol())
            {
                diagnostics.push(native_diagnostic(
                    &self.name,
                    relation.name(),
                    format!(
                        "spatial Relation `{}` references foreign or omitted Domain `{}`",
                        relation.name(),
                        domain.name()
                    ),
                ));
            }
        }

        for declaration in &self.declarations {
            let DraftDeclaration::ConservingPort(port) = declaration else {
                continue;
            };
            if !domain_symbols.contains(&port.domain.symbol) {
                diagnostics.push(native_diagnostic(
                    &self.name,
                    port.name(),
                    format!(
                        "conserving Port `{}` references foreign or omitted scalar physical Domain `{}`",
                        port.name(),
                        port.domain.name()
                    ),
                ));
            }
        }

        for declaration in &self.declarations {
            let mut referenced = Vec::new();
            let path = if let DraftDeclaration::Observable(value) = declaration {
                value.expression.references(&mut referenced);
                value.name()
            } else if let Some((path, residuals)) = declaration.equations() {
                for (left, right) in residuals {
                    left.references(&mut referenced);
                    right.references(&mut referenced);
                }
                path
            } else {
                continue;
            };
            for reference in referenced {
                match reference {
                    DraftExpressionReference::Value(reference)
                        if !value_symbols.contains(&reference.symbol) =>
                    {
                        diagnostics.push(native_diagnostic(
                            &self.name,
                            path,
                            format!(
                                "declaration `{path}` references foreign or omitted {} `{}`",
                                reference.kind.label(),
                                reference.name
                            ),
                        ));
                    }
                    DraftExpressionReference::Port(reference)
                        if !ports.contains_key(&reference.symbol) =>
                    {
                        diagnostics.push(native_diagnostic(
                            &self.name,
                            path,
                            format!(
                                "declaration `{path}` references foreign or omitted conserving Port `{}`",
                                reference.name
                            ),
                        ));
                    }
                    DraftExpressionReference::Domain(domain)
                        if !spatial_domain_symbols.contains(domain.symbol()) =>
                    {
                        diagnostics.push(native_diagnostic(
                            &self.name,
                            path,
                            format!(
                                "declaration `{path}` references foreign or omitted Domain `{}`",
                                domain.name()
                            ),
                        ));
                    }
                    DraftExpressionReference::Domain(_) => {}
                    DraftExpressionReference::EnumValue(value) => {
                        if let Err(error) = crate::SourceAstFactory::value_literal(
                            value,
                            None,
                            TextRange::new(0, 0),
                            |id| self.nominal_name(id),
                            |id| self.enum_definition(id),
                            |basis| self.product_name(basis),
                        ) {
                            diagnostics.push(native_diagnostic(
                                &self.name,
                                path,
                                error.to_string(),
                            ));
                        }
                    }
                    DraftExpressionReference::Value(_) | DraftExpressionReference::Port(_) => {}
                }
            }
        }

        let mut connected_ports = HashSet::new();
        for declaration in &self.declarations {
            let DraftDeclaration::ConservingConnection(connection) = declaration else {
                continue;
            };
            let path = connection_path(connection);
            if connection.ports.len() < 2 {
                diagnostics.push(native_diagnostic(
                    &self.name,
                    &path,
                    "conserving Connection requires at least two Ports",
                ));
            }
            let mut members = HashSet::new();
            let mut expected_domain = None;
            for port in &connection.ports {
                if !members.insert(port.symbol.clone()) {
                    diagnostics.push(native_diagnostic(
                        &self.name,
                        &path,
                        format!("conserving Connection repeats Port `{}`", port.name()),
                    ));
                    continue;
                }
                let Some(declared) = ports.get(&port.symbol) else {
                    diagnostics.push(native_diagnostic(
                        &self.name,
                        &path,
                        format!(
                            "conserving Connection references foreign or omitted Port `{}`",
                            port.name()
                        ),
                    ));
                    continue;
                };
                if !connected_ports.insert(port.symbol.clone()) {
                    diagnostics.push(native_diagnostic(
                        &self.name,
                        &path,
                        format!(
                            "conserving Port `{}` already belongs to another Connection",
                            port.name()
                        ),
                    ));
                }
                match &expected_domain {
                    None => expected_domain = Some(declared.domain.symbol.clone()),
                    Some(domain) if *domain != declared.domain.symbol => {
                        diagnostics.push(native_diagnostic(
                            &self.name,
                            &path,
                            "conserving Connection requires Ports on the exact same draft-local scalar physical Domain",
                        ));
                    }
                    Some(_) => {}
                }
            }
        }

        if diagnostics.is_empty() {
            Ok(())
        } else {
            Err(diagnostics)
        }
    }
}

/// One declaration admitted by the first native-construction slice.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub enum DraftDeclaration {
    /// One exact module-level enum declaration shared by all native occurrences.
    Enum {
        name: String,
        definition: eqiora_schema::kernel::EnumDef,
    },
    /// An exact registered atomic finite-space definition.
    FiniteSpace {
        name: String,
        definition: eqiora_schema::kernel::FiniteSpaceDef,
    },
    /// An exact registered bounded index-set definition.
    IndexSet {
        name: String,
        definition: eqiora_schema::kernel::IndexSetDef,
    },
    /// Cartesian volume or one oriented boundary Domain.
    SpatialDomain(DraftSpatialDomain),
    /// Nominal scalar physical Domain.
    PhysicalDomain(DraftPhysicalDomain),
    /// Mutable scalar state.
    Field(DraftField),
    /// Revision-local scalar design value.
    Parameter(DraftParameter),
    /// Typed derived output, separate from solve unknowns.
    Observable(DraftObservable),
    /// Scalar conserving Port on one nominal physical Domain.
    ConservingPort(DraftConservingPort),
    /// Continuous implicit equation group.
    Relation(DraftRelation),
    /// Simultaneous fresh-initialization equation sides.
    ///
    /// These are mathematical conditions, separate from numerical guesses.
    /// Empty groups, non-finite constants, and foreign references are rejected
    /// when closing the draft. Types are checked by the common compiler.
    Initial(Vec<(DraftExpression, DraftExpression)>),
    /// Anonymous N-ary conserving connection net.
    ConservingConnection(DraftConservingConnection),
}

impl DraftDeclaration {
    fn name(&self) -> Option<&str> {
        match self {
            Self::Enum { name, .. }
            | Self::FiniteSpace { name, .. }
            | Self::IndexSet { name, .. } => Some(name),
            Self::SpatialDomain(value) => Some(value.name()),
            Self::PhysicalDomain(value) => Some(value.name()),
            Self::Field(value) => Some(value.name()),
            Self::Parameter(value) => Some(value.name()),
            Self::Observable(value) => Some(value.name()),
            Self::ConservingPort(value) => Some(value.name()),
            Self::Relation(value) => Some(value.name()),
            Self::Initial(_) | Self::ConservingConnection(_) => None,
        }
    }

    fn kind_name(&self) -> &'static str {
        match self {
            Self::Enum { .. } => "Enum",
            Self::FiniteSpace { .. } => "FiniteSpace",
            Self::IndexSet { .. } => "IndexSet",
            Self::SpatialDomain(_) => "SpatialDomain",
            Self::PhysicalDomain(_) => "PhysicalDomain",
            Self::Field(_) => "Field",
            Self::Parameter(_) => "Parameter",
            Self::Observable(_) => "Observable",
            Self::ConservingPort(_) => "ConservingPort",
            Self::Relation(_) => "Relation",
            Self::Initial(_) => "Initial",
            Self::ConservingConnection(_) => "ConservingConnection",
        }
    }

    fn equations(&self) -> Option<(&str, &[(DraftExpression, DraftExpression)])> {
        match self {
            Self::Relation(relation) => Some((relation.name(), &relation.equations)),
            Self::Initial(residuals) => Some(("initial", residuals)),
            _ => None,
        }
    }
}

impl From<DraftSpatialDomain> for DraftDeclaration {
    fn from(value: DraftSpatialDomain) -> Self {
        Self::SpatialDomain(value)
    }
}

impl From<DraftPhysicalDomain> for DraftDeclaration {
    fn from(value: DraftPhysicalDomain) -> Self {
        Self::PhysicalDomain(value)
    }
}

impl From<DraftField> for DraftDeclaration {
    fn from(value: DraftField) -> Self {
        Self::Field(value)
    }
}

impl From<DraftParameter> for DraftDeclaration {
    fn from(value: DraftParameter) -> Self {
        Self::Parameter(value)
    }
}

impl From<DraftConservingPort> for DraftDeclaration {
    fn from(value: DraftConservingPort) -> Self {
        Self::ConservingPort(value)
    }
}

impl From<DraftRelation> for DraftDeclaration {
    fn from(value: DraftRelation) -> Self {
        Self::Relation(value)
    }
}

impl From<DraftConservingConnection> for DraftDeclaration {
    fn from(value: DraftConservingConnection) -> Self {
        Self::ConservingConnection(value)
    }
}

/// Immutable nominal Domain for one scalar across/through pair.
///
/// Domain compatibility follows this handle's identity. Equal names and
/// equal dimensions never make separately constructed Domains compatible.
/// Named quantities project through the shared current compiler and Model wire.
#[derive(Debug, Clone)]
pub struct DraftPhysicalDomain {
    symbol: DraftSymbol,
    name: String,
    across_name: String,
    across_type: eqiora_core::ValueType,
    through_name: String,
    through_type: eqiora_core::ValueType,
}

impl DraftPhysicalDomain {
    /// Declare one nominal scalar physical Domain.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        across_name: impl Into<String>,
        across_type: eqiora_core::ValueType,
        through_name: impl Into<String>,
        through_type: eqiora_core::ValueType,
    ) -> Self {
        Self {
            symbol: DraftSymbol::new(),
            name: name.into(),
            across_name: across_name.into(),
            across_type,
            through_name: through_name.into(),
            through_type,
        }
    }

    /// Declaration name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Declared member name of the across quantity.
    #[must_use]
    pub fn across_name(&self) -> &str {
        &self.across_name
    }

    /// Declared member name of the through quantity.
    #[must_use]
    pub fn through_name(&self) -> &str {
        &self.through_name
    }

    /// Complete mathematical type of the across variable.
    #[must_use]
    pub const fn across_type(&self) -> &eqiora_core::ValueType {
        &self.across_type
    }

    /// Complete mathematical type of the through variable.
    #[must_use]
    pub const fn through_type(&self) -> &eqiora_core::ValueType {
        &self.through_type
    }
}

/// Immutable scalar conserving Port bound to one draft-local Domain identity.
#[derive(Debug, Clone)]
pub struct DraftConservingPort {
    symbol: DraftSymbol,
    name: String,
    domain: DraftPhysicalDomain,
}

impl DraftConservingPort {
    /// Declare one scalar conserving Port on `domain`.
    #[must_use]
    pub fn new(name: impl Into<String>, domain: &DraftPhysicalDomain) -> Self {
        Self {
            symbol: DraftSymbol::new(),
            name: name.into(),
            domain: domain.clone(),
        }
    }

    /// Declaration name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Nominal scalar physical Domain handle.
    #[must_use]
    pub const fn domain(&self) -> &DraftPhysicalDomain {
        &self.domain
    }
}

/// Immutable anonymous conserving connection net.
#[derive(Debug, Clone)]
pub struct DraftConservingConnection {
    ports: Vec<DraftConservingPort>,
}

impl DraftConservingConnection {
    /// Request one N-ary conserving connection.
    ///
    /// Membership is checked atomically by [`Module::new`]. In
    /// particular, the closed draft requires at least two distinct declared
    /// Ports on the exact same nominal Domain, and each Port may belong to at
    /// most one Connection.
    #[must_use]
    pub fn new<'a>(ports: impl IntoIterator<Item = &'a DraftConservingPort>) -> Self {
        Self {
            ports: ports.into_iter().cloned().collect(),
        }
    }

    /// Member Ports in request order.
    #[must_use]
    pub fn ports(&self) -> &[DraftConservingPort] {
        &self.ports
    }
}

/// Immutable typed Field declaration, either local or spatially supported.
#[derive(Debug, Clone)]
pub struct DraftField {
    symbol: DraftSymbol,
    name: String,
    value_type: ValueType,
    role: crate::ast::FieldRoleSyntax,
    spatial_scope: Option<DraftSpatialScope>,
}

#[derive(Debug, Clone)]
struct DraftSpatialScope {
    domain: DraftSpatialDomain,
}

impl DraftField {
    /// Declare one unknown with its complete type and explicit evolution role.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        value_type: ValueType,
        role: crate::ast::FieldRoleSyntax,
    ) -> Self {
        Self {
            symbol: DraftSymbol::new(),
            name: name.into(),
            value_type,
            role,
            spatial_scope: None,
        }
    }

    /// Declare one typed unknown over an exact draft-local Domain.
    #[must_use]
    pub fn spatial(
        name: impl Into<String>,
        domain: &DraftSpatialDomain,
        value_type: ValueType,
        role: crate::ast::FieldRoleSyntax,
    ) -> Self {
        Self {
            symbol: DraftSymbol::new(),
            name: name.into(),
            value_type,
            role,
            spatial_scope: Some(DraftSpatialScope {
                domain: domain.clone(),
            }),
        }
    }

    /// Declaration name.
    #[must_use]
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Static SI dimension.
    #[must_use]
    pub const fn dimension(&self) -> DimExponents {
        self.value_type.dimension()
    }

    /// Complete mathematical type, independent of storage and execution.
    #[must_use]
    pub const fn value_type(&self) -> &ValueType {
        &self.value_type
    }

    /// Declared evolution role, independent of numerical initialization guesses.
    #[must_use]
    pub const fn role(&self) -> crate::ast::FieldRoleSyntax {
        self.role
    }

    /// Exact draft-local spatial Domain, when distributed.
    #[must_use]
    pub fn domain(&self) -> Option<&DraftSpatialDomain> {
        self.spatial_scope.as_ref().map(|scope| &scope.domain)
    }

    /// Use this Field as a scalar expression.
    #[must_use]
    pub fn expression(&self) -> DraftExpression {
        DraftExpression::reference(
            self.symbol.clone(),
            self.name.clone(),
            DraftSymbolKind::Field,
        )
    }
}

/// Immutable Parameter declaration owning one complete validated value.
#[derive(Debug, Clone)]
pub struct DraftParameter {
    symbol: DraftSymbol,
    name: String,
    value: ValueLiteral,
    frame: Option<DraftSpatialDomain>,
}

/// Immutable symbolic expression used only while defining a native model.
///
/// Shape and spatial support remain opaque here. The shared semantic
/// validator infers them and checks the compatibility of each equation’s sides.
#[derive(Debug, Clone)]
pub struct DraftExpression {
    syntax: Result<std::sync::Arc<Expr>, crate::AstConstructionError>,
    references: std::sync::Arc<Vec<expression::NativeReference>>,
    depth: usize,
    nodes: usize,
}

impl Neg for DraftExpression {
    type Output = Self;
    fn neg(self) -> Self::Output {
        self.unary(UnaryOp::Neg)
    }
}

macro_rules! impl_binary_expression_operator {
    ($trait:ident, $method:ident, $operator:expr) => {
        impl $trait for DraftExpression {
            type Output = Self;

            fn $method(self, right: Self) -> Self::Output {
                self.binary($operator, right)
            }
        }
    };
}

impl_binary_expression_operator!(Add, add, BinaryOp::Add);
impl_binary_expression_operator!(Sub, sub, BinaryOp::Sub);
impl_binary_expression_operator!(Mul, mul, BinaryOp::Mul);
impl_binary_expression_operator!(Div, div, BinaryOp::Div);

#[derive(Debug, Clone)]
struct DraftReference {
    symbol: DraftSymbol,
    name: String,
    kind: DraftSymbolKind,
}

#[derive(Debug, Clone)]
struct DraftPortReference {
    symbol: DraftSymbol,
    name: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum DraftSymbolKind {
    Observable,
    Field,
    Parameter,
}

impl DraftSymbolKind {
    const fn label(self) -> &'static str {
        match self {
            Self::Observable => "Observable",
            Self::Field => "Field",
            Self::Parameter => "Parameter",
        }
    }
}

mod ast_bridge;
mod dimension;
mod expression;
mod finite;
use expression::DraftExpressionReference;
mod observable;
mod parameter;
pub use observable::DraftObservable;
mod relation;
pub use relation::DraftRelation;
mod nominal;
mod symbol;
mod validation;
mod value_type;
pub use ast_bridge::Module;
use ast_bridge::RangeAllocator;
use dimension::dimension_expression;
pub(crate) use symbol::DraftSymbol;
use validation::{connection_path, is_language_identifier, native_diagnostic};

#[cfg(test)]
mod tests;
