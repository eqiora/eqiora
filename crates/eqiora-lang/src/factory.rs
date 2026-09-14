//! Checked owned-AST construction for compiler-side source transformations.

use core::fmt;

mod compile_time;
use compile_time::validate_named_binding;
mod component;
mod dimension_rewrite;
mod document;
mod domain_validation;
mod expression;
pub(crate) mod expression_visit;
mod signature;
pub(crate) use expression::validate_expression;
mod constraint;
mod enumeration;
mod event;
mod law;
mod nominal;
mod operator;
mod property;
mod property_declaration;
mod record;
mod relation;
mod type_visit;
pub(crate) mod value_literal;
mod value_type;

use crate::ast::{
    ActivationSyntax, BoundaryConnectionDecl, BoundaryPortReferenceSyntax,
    BoundaryPortSelectorSyntax, ClockDecl, ComponentParameterDecl, ComponentPortDecl,
    ComponentPortFamilyDecl, ConnectionDecl, ConnectionSyntax, ConnectorDecl,
    ConnectorQuantitySyntax, ConnectorSyntax, DomainDecl, DomainSyntax, ExactIntegerSyntax, Expr,
    ExprKind, FamilyBinderSyntax, FieldDecl, InstanceDecl, NamePath, NamedBindingDecl,
    NamedDefinitionDecl, ParameterDecl, PortDecl, PortSyntax, PureOperatorDecl, PureOperatorFormal,
    PureValueClassSyntax, RelationCondition, RelationDecl, RelationFamilyDecl, SupportSlotDecl,
    SupportSlotSyntax, TextRange, ValueShapeSyntax, VisibilitySyntax,
};
use domain_validation::validate_domain_syntax;

/// A structural source-AST construction failure.
///
/// This error reports syntax-shape contradictions only. Name resolution,
/// dimensions, component visibility, and other semantic checks remain compiler
/// responsibilities.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AstConstructionError {
    message: String,
}

impl AstConstructionError {
    pub(crate) fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }

    /// Stable human-readable explanation of the structural contradiction.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for AstConstructionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.write_str(&self.message)
    }
}

impl std::error::Error for AstConstructionError {}

/// Checked factory for owned source AST values.
///
/// The factory is intentionally syntax-only. It gives parsers, elaborators,
/// and source transforms one construction boundary without exposing mutable
/// declaration fields or requiring a format-and-reparse cycle.
#[derive(Debug, Clone, Copy, Default)]
pub struct SourceAstFactory;

impl SourceAstFactory {
    /// Maximum depth of one directly constructed expression tree.
    pub const MAX_EXPRESSION_DEPTH: usize = 256;
    /// Maximum expression nodes admitted by one compilation module.
    pub const MAX_EXPRESSION_NODES: usize = 1_000_000;
    /// Maximum declarations or equation members in one source container.
    pub const MAX_CONTAINER_MEMBERS: usize = 65_536;

    /// Construct one nominal Connector declaration.
    ///
    /// # Errors
    /// Returns an error for malformed visibility-independent source shape,
    /// name, expression, or byte range.
    pub fn connector(
        visibility: VisibilitySyntax,
        name: impl Into<String>,
        syntax: ConnectorSyntax,
        range: TextRange,
    ) -> Result<ConnectorDecl, AstConstructionError> {
        validate_connector_syntax(&syntax)?;
        Ok(ConnectorDecl {
            comments: Default::default(),
            visibility,
            name: checked_identifier(name, "Connector")?,
            syntax,
            range: checked_range(range)?,
        })
    }

    /// Construct one named field-physical Connector quantity.
    ///
    /// # Errors
    /// Returns an error for a malformed member name or dimension expression.
    pub fn connector_quantity(
        name: impl Into<String>,
        dimension: Expr,
    ) -> Result<ConnectorQuantitySyntax, AstConstructionError> {
        validate_expression(&dimension)?;
        Ok(ConnectorQuantitySyntax {
            name: checked_identifier(name, "Connector quantity")?,
            dimension: Box::new(dimension),
        })
    }

    /// Construct one component-local scalar Parameter.
    ///
    /// # Errors
    /// Returns an error for a malformed name, expression, or byte range.
    pub fn component_parameter(
        visibility: VisibilitySyntax,
        name: impl Into<String>,
        value_type: crate::ValueTypeSyntax,
        default: Option<Expr>,
        range: TextRange,
    ) -> Result<ComponentParameterDecl, AstConstructionError> {
        Self::value_type(value_type.kind().clone(), value_type.range)?;
        if let Some(default) = &default {
            validate_expression(default)?;
        }
        Ok(ComponentParameterDecl {
            comments: Default::default(),
            visibility,
            name: checked_identifier(name, "component Parameter")?,
            value_type,
            default,
            range: checked_range(range)?,
        })
    }

    /// Construct one component-local Port.
    ///
    /// # Errors
    /// Returns an error for malformed Port syntax, a name, or a byte range.
    pub fn component_port(
        visibility: VisibilitySyntax,
        name: impl Into<String>,
        syntax: PortSyntax,
        range: TextRange,
    ) -> Result<ComponentPortDecl, AstConstructionError> {
        validate_port_syntax(&syntax)?;
        Ok(ComponentPortDecl {
            comments: Default::default(),
            visibility,
            name: checked_identifier(name, "component Port")?,
            syntax,
            range: checked_range(range)?,
        })
    }

    /// Construct one field-physical Port family over a complete exterior.
    ///
    /// # Errors
    /// Returns an error unless the Port is field-physical, its support names
    /// the binder member, and both declarations are structurally valid.
    pub fn component_port_family(
        port: ComponentPortDecl,
        binder: FamilyBinderSyntax,
    ) -> Result<ComponentPortFamilyDecl, AstConstructionError> {
        validate_port_syntax(port.syntax())?;
        checked_range(port.range())?;
        validate_boundary_family_binder(&binder)?;
        match port.syntax() {
            PortSyntax::FieldPhysical { support, .. } if support == binder.member() => {}
            PortSyntax::FieldPhysical { .. } => {
                return Err(AstConstructionError::new(
                    "a Port family support must name its boundary binder member",
                ));
            }
            _ => {
                return Err(AstConstructionError::new(
                    "only a field-physical Port can declare a boundary family",
                ));
            }
        }
        Ok(ComponentPortFamilyDecl { port, binder })
    }

    /// Construct the restricted `[member in complete_exterior]` binder.
    ///
    /// # Errors
    /// Returns an error for malformed identifiers or a reversed range.
    pub fn boundary_family_binder(
        member: impl Into<String>,
        set: impl Into<String>,
        range: TextRange,
    ) -> Result<FamilyBinderSyntax, AstConstructionError> {
        Ok(FamilyBinderSyntax {
            member: checked_identifier(member, "boundary family member")?,
            set: NamePath::single(
                checked_identifier(set, "boundary family support set")?,
                range,
            ),
            range: checked_range(range)?,
        })
    }

    /// Construct one component-local spatial-support slot.
    ///
    /// # Errors
    /// Returns an error for a malformed slot name, parent name, or byte range.
    /// Support-graph and visibility rules remain compiler responsibilities.
    pub fn support_slot(
        visibility: VisibilitySyntax,
        name: impl Into<String>,
        syntax: SupportSlotSyntax,
        range: TextRange,
    ) -> Result<SupportSlotDecl, AstConstructionError> {
        validate_support_slot_syntax(&syntax)?;
        Ok(SupportSlotDecl {
            comments: Default::default(),
            visibility,
            name: checked_identifier(name, "support slot")?,
            syntax,
            range: checked_range(range)?,
        })
    }

    /// Construct a Domain declaration.
    ///
    /// # Errors
    /// Returns an error for malformed source shape, names, expressions, or
    /// ranges. Geometric and dimensional meaning remains a lowering check.
    pub fn domain(
        name: impl Into<String>,
        syntax: DomainSyntax,
        range: TextRange,
    ) -> Result<DomainDecl, AstConstructionError> {
        validate_domain_syntax(&syntax)?;
        Ok(DomainDecl {
            comments: Default::default(),
            name: checked_identifier(name, "Domain")?,
            syntax,
            range: checked_range(range)?,
        })
    }

    /// Construct an owned unknown with independent role, support, and activation.
    ///
    /// # Errors
    /// Rejects malformed identifiers, complete type syntax, or byte ranges.
    pub fn field(
        name: impl Into<String>,
        domain: Option<String>,
        role: crate::ast::FieldRoleSyntax,
        activation: ActivationSyntax,
        value_type: crate::ValueTypeSyntax,
        range: TextRange,
    ) -> Result<FieldDecl, AstConstructionError> {
        if let Some(name) = &domain {
            validate_identifier(name, "unknown support")?;
        }
        if let ActivationSyntax::Named(clock) = &activation {
            validate_identifier(clock, "unknown clock")?;
        }
        Self::value_type(value_type.kind().clone(), value_type.range)?;
        Ok(FieldDecl {
            comments: Default::default(),
            name: checked_identifier(name, "unknown")?,
            domain,
            role,
            activation,
            value_type,
            range: checked_range(range)?,
        })
    }

    /// Construct a model-level Port declaration.
    ///
    /// # Errors
    /// Returns an error for malformed Port syntax, names, or ranges.
    pub fn port(
        name: impl Into<String>,
        syntax: PortSyntax,
        range: TextRange,
    ) -> Result<PortDecl, AstConstructionError> {
        validate_port_syntax(&syntax)?;
        Ok(PortDecl {
            comments: Default::default(),
            name: checked_identifier(name, "Port")?,
            syntax,
            range: checked_range(range)?,
        })
    }

    /// Construct an exact periodic Clock declaration.
    ///
    /// Exact time admission, rational reduction, and nonzero-period checks remain
    /// semantic lowering checks, matching parsed source behavior.
    ///
    /// # Errors
    /// Returns an error for an invalid identifier, range, or expression tree.
    pub fn clock(
        name: impl Into<String>,
        period: Expr,
        phase: Expr,
        range: TextRange,
    ) -> Result<ClockDecl, AstConstructionError> {
        validate_expression(&period)?;
        validate_expression(&phase)?;
        Ok(ClockDecl {
            comments: Default::default(),
            name: checked_identifier(name, "Clock")?,
            period,
            phase,
            range: checked_range(range)?,
        })
    }

    /// Construct a signal or conserving Connection with at least two Ports.
    ///
    /// # Errors
    /// Returns an error for insufficient members or malformed paths/ranges.
    pub fn connection(
        syntax: ConnectionSyntax,
        binder: Option<FamilyBinderSyntax>,
        ports: Vec<Expr>,
        range: TextRange,
    ) -> Result<ConnectionDecl, AstConstructionError> {
        if syntax == ConnectionSyntax::SpatialPeriodic {
            return Err(AstConstructionError::new(
                "a spatial-periodic Connection requires boundary Port references",
            ));
        }
        if ports.len() < 2 {
            return Err(AstConstructionError::new(
                "a Connection requires at least two Port paths",
            ));
        }
        if let Some(binder) = &binder {
            validate_boundary_family_binder(binder)?;
        }
        for endpoint in &ports {
            expression::validate_endpoint(endpoint)?;
        }
        Ok(ConnectionDecl {
            comments: Default::default(),
            syntax,
            binder,
            ports,
            range: checked_range(range)?,
        })
    }

    /// Construct one conserving Connection with boundary-family Port references.
    ///
    /// # Errors
    /// Returns an error for fewer than two Ports, malformed references, or a
    /// declaration containing neither a family binder nor a selector.
    pub fn boundary_connection(
        binder: Option<FamilyBinderSyntax>,
        ports: Vec<BoundaryPortReferenceSyntax>,
        range: TextRange,
    ) -> Result<BoundaryConnectionDecl, AstConstructionError> {
        if ports.len() < 2 {
            return Err(AstConstructionError::new(
                "a Connection requires at least two Port paths",
            ));
        }
        if binder.is_none() && ports.iter().all(|port| port.selector().is_none()) {
            return Err(AstConstructionError::new(
                "a boundary Connection requires a family binder or selector",
            ));
        }
        if let Some(binder) = &binder {
            validate_boundary_family_binder(binder)?;
        }
        for port in &ports {
            validate_boundary_port_reference(port)?;
        }
        Ok(BoundaryConnectionDecl {
            comments: Default::default(),
            syntax: ConnectionSyntax::Conserving,
            binder,
            ports,
            range: checked_range(range)?,
        })
    }

    /// Construct one exact spatial-periodic pair in a closed Model.
    ///
    /// # Errors
    /// Returns an error unless there are exactly two boundary Ports.
    pub fn spatial_periodic_boundary_connection(
        ports: Vec<BoundaryPortReferenceSyntax>,
        range: TextRange,
    ) -> Result<BoundaryConnectionDecl, AstConstructionError> {
        let connection = BoundaryConnectionDecl {
            comments: Default::default(),
            syntax: ConnectionSyntax::SpatialPeriodic,
            binder: None,
            ports,
            range: checked_range(range)?,
        };
        validate_boundary_connection(&connection)?;
        Ok(connection)
    }

    /// Construct one Port path with an optional exact boundary selector.
    ///
    /// # Errors
    /// Returns an error for a malformed path or selector.
    pub fn boundary_port_reference(
        port: NamePath,
        selector: Option<BoundaryPortSelectorSyntax>,
    ) -> Result<BoundaryPortReferenceSyntax, AstConstructionError> {
        validate_name_path(&port)?;
        if let Some(selector) = &selector {
            validate_boundary_port_selector(selector)?;
        }
        Ok(BoundaryPortReferenceSyntax { port, selector })
    }

    /// Construct the closed `[member = target]` Port selector.
    ///
    /// # Errors
    /// Returns an error for malformed identifiers or a reversed range.
    pub fn boundary_port_selector(
        member: impl Into<String>,
        target: impl Into<String>,
        range: TextRange,
    ) -> Result<BoundaryPortSelectorSyntax, AstConstructionError> {
        Ok(BoundaryPortSelectorSyntax {
            member: checked_identifier(member, "boundary selector member")?,
            target: checked_identifier(target, "boundary selector target")?,
            range: checked_range(range)?,
        })
    }

    /// Construct one occurrence with category-free named bindings.
    ///
    /// # Errors
    /// Rejects malformed names, expressions, and ranges.
    pub fn instance(
        name: impl Into<String>,
        definition: NamePath,
        family: Option<crate::FamilyBinderSyntax>,
        bindings: Vec<NamedBindingDecl>,
        range: TextRange,
    ) -> Result<InstanceDecl, AstConstructionError> {
        validate_name_path(&definition)?;
        for binding in &bindings {
            validate_named_binding(binding)?;
        }
        Ok(InstanceDecl {
            comments: Default::default(),
            name: checked_identifier(name, "instance")?,
            definition,
            family,
            bindings,
            range: checked_range(range)?,
        })
    }

    /// Construct one target-directed named binding.
    ///
    /// # Errors
    /// Rejects malformed names, expressions, and ranges.
    pub fn named_binding(
        name: impl Into<String>,
        value: Expr,
        range: TextRange,
    ) -> Result<NamedBindingDecl, AstConstructionError> {
        validate_expression(&value)?;
        Ok(NamedBindingDecl {
            comments: Default::default(),
            name: checked_identifier(name, "binding name")?,
            value,
            range: checked_range(range)?,
        })
    }

    /// Construct one source expression from an owned recursive expression kind.
    ///
    /// # Errors
    /// Returns an error for malformed names, non-finite literals, child
    /// expressions, or byte ranges.
    pub fn expression(kind: ExprKind, range: TextRange) -> Result<Expr, AstConstructionError> {
        let expression = Expr {
            resolved_enum: None,
            resolved_nominal: None,
            kind,
            range: checked_range(range)?,
        };
        validate_expression(&expression)?;
        Ok(expression)
    }
}

fn checked_identifier(
    value: impl Into<String>,
    role: &str,
) -> Result<String, AstConstructionError> {
    let value = value.into();
    validate_identifier(&value, role)?;
    Ok(value)
}

fn validate_identifier(value: &str, role: &str) -> Result<(), AstConstructionError> {
    let mut bytes = value.bytes();
    let valid = matches!(
        bytes.next(),
        Some(first) if first.is_ascii_alphabetic() || first == b'_'
    ) && bytes.all(|byte| byte.is_ascii_alphanumeric() || byte == b'_');
    if valid {
        Ok(())
    } else {
        Err(AstConstructionError::new(format!(
            "{role} `{value}` is not an Eqiora identifier"
        )))
    }
}

fn checked_range(range: TextRange) -> Result<TextRange, AstConstructionError> {
    if range.start() <= range.end() {
        Ok(range)
    } else {
        Err(AstConstructionError::new(format!(
            "source range {}..{} is reversed",
            range.start(),
            range.end()
        )))
    }
}

fn validate_finite(value: f64, role: &str) -> Result<(), AstConstructionError> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(AstConstructionError::new(format!(
            "{role} must be a finite f64"
        )))
    }
}

fn validate_name_path(path: &NamePath) -> Result<(), AstConstructionError> {
    checked_range(path.range())?;
    let mut segments = path.segments();
    let Some(first) = segments.next() else {
        return Err(AstConstructionError::new("a NamePath cannot be empty"));
    };
    validate_identifier(first, "NamePath segment")?;
    for segment in segments {
        validate_identifier(segment, "NamePath segment")?;
    }
    Ok(())
}

fn validate_exact_integer(value: &ExactIntegerSyntax) -> Result<(), AstConstructionError> {
    checked_range(value.range)?;
    let parsed = value.spelling.parse::<u64>().map_err(|_| {
        AstConstructionError::new(format!(
            "exact integer `{}` must be an unsigned decimal integer fitting in u64",
            value.spelling
        ))
    })?;
    if parsed == value.value {
        Ok(())
    } else {
        Err(AstConstructionError::new(format!(
            "exact integer spelling `{}` does not encode stored value {}",
            value.spelling, value.value
        )))
    }
}

fn validate_pure_value_class(
    value_class: &PureValueClassSyntax,
) -> Result<(), AstConstructionError> {
    match value_class {
        PureValueClassSyntax::Typed(value) => value_type::validate_syntax(value),
        PureValueClassSyntax::Scalar => Ok(()),
        PureValueClassSyntax::Spatial { rank } => validate_exact_integer(rank),
    }
}

fn validate_connector_syntax(syntax: &ConnectorSyntax) -> Result<(), AstConstructionError> {
    match syntax {
        ConnectorSyntax::ScalarPhysical {
            across_name,
            across_type,
            through_name,
            through_type,
        } => {
            validate_identifier(across_name, "across quantity")?;
            validate_identifier(through_name, "through quantity")?;
            if across_name == through_name {
                return Err(AstConstructionError::new(
                    "Connector quantity names must be distinct",
                ));
            }
            value_type::validate_syntax(across_type)?;
            value_type::validate_syntax(through_type)
        }
        ConnectorSyntax::FieldPhysical {
            trace, flux, shape, ..
        } => {
            validate_connector_quantity(trace, "trace")?;
            validate_connector_quantity(flux, "flux")?;
            if trace.name == flux.name {
                return Err(AstConstructionError::new(
                    "Connector quantity names must be distinct",
                ));
            }
            validate_value_shape(shape)
        }
    }
}

fn validate_support_slot_syntax(syntax: &SupportSlotSyntax) -> Result<(), AstConstructionError> {
    match syntax {
        SupportSlotSyntax::Volume { .. } => Ok(()),
        SupportSlotSyntax::Boundary { parent } | SupportSlotSyntax::CompleteExterior { parent } => {
            validate_identifier(parent, "boundary parent support slot")
        }
    }
}

fn validate_port_syntax(syntax: &PortSyntax) -> Result<(), AstConstructionError> {
    match syntax {
        PortSyntax::Signal {
            value_type,
            domain,
            activation,
            ..
        } => {
            if let Some(domain) = domain {
                validate_identifier(domain, "signal support")?;
            }
            if let ActivationSyntax::Named(clock) = activation {
                validate_identifier(clock, "signal clock")?;
            }
            SourceAstFactory::value_type(value_type.kind().clone(), value_type.range).map(|_| ())
        }
        PortSyntax::ScalarPhysical { domain } => {
            validate_identifier(domain, "scalar physical Domain")
        }
        PortSyntax::ScalarPhysicalConnector { connector } => validate_name_path(connector),
        PortSyntax::FieldPhysical { connector, support } => {
            validate_name_path(connector)?;
            validate_identifier(support, "field-physical boundary support")
        }
    }
}

fn validate_connector_quantity(
    quantity: &ConnectorQuantitySyntax,
    role: &str,
) -> Result<(), AstConstructionError> {
    validate_identifier(quantity.name(), &format!("{role} quantity"))?;
    validate_expression(quantity.dimension())
}

fn validate_value_shape(shape: &ValueShapeSyntax) -> Result<(), AstConstructionError> {
    match shape {
        ValueShapeSyntax::Scalar | ValueShapeSyntax::SpatialVector => Ok(()),
        ValueShapeSyntax::Exact(extents) if extents.is_empty() => Err(AstConstructionError::new(
            "an empty exact shape must use the canonical Scalar syntax",
        )),
        ValueShapeSyntax::Exact(extents) if extents.contains(&0) => Err(AstConstructionError::new(
            "exact value-shape extents must be positive",
        )),
        ValueShapeSyntax::Exact(_) => Ok(()),
    }
}

fn validate_boundary_family_binder(
    binder: &FamilyBinderSyntax,
) -> Result<(), AstConstructionError> {
    validate_identifier(binder.member(), "boundary family member")?;
    validate_name_path(binder.set())?;
    checked_range(binder.range()).map(|_| ())
}

fn validate_boundary_port_selector(
    selector: &BoundaryPortSelectorSyntax,
) -> Result<(), AstConstructionError> {
    validate_identifier(selector.member(), "boundary selector member")?;
    validate_identifier(selector.target(), "boundary selector target")?;
    checked_range(selector.range()).map(|_| ())
}

fn validate_boundary_port_reference(
    reference: &BoundaryPortReferenceSyntax,
) -> Result<(), AstConstructionError> {
    validate_name_path(reference.port())?;
    if let Some(selector) = reference.selector() {
        validate_boundary_port_selector(selector)?;
    }
    Ok(())
}

fn validate_boundary_connection(
    connection: &BoundaryConnectionDecl,
) -> Result<(), AstConstructionError> {
    if connection.syntax() == ConnectionSyntax::SpatialPeriodic && connection.ports().len() != 2 {
        return Err(AstConstructionError::new(
            "a spatial-periodic Connection requires exactly two Ports",
        ));
    }
    if connection.syntax() != ConnectionSyntax::SpatialPeriodic && connection.ports().len() < 2 {
        return Err(AstConstructionError::new(
            "a Connection requires at least two Port paths",
        ));
    }
    if connection.syntax() == ConnectionSyntax::Signal {
        return Err(AstConstructionError::new(
            "a boundary Connection cannot use signal semantics",
        ));
    }
    if connection.syntax() == ConnectionSyntax::SpatialPeriodic && connection.binder().is_some() {
        return Err(AstConstructionError::new(
            "a spatial-periodic Connection cannot declare a family binder",
        ));
    }
    if connection.syntax() == ConnectionSyntax::Conserving
        && connection.binder().is_none()
        && connection
            .ports()
            .iter()
            .all(|port| port.selector().is_none())
    {
        return Err(AstConstructionError::new(
            "a boundary Connection requires a family binder or selector",
        ));
    }
    if let Some(binder) = connection.binder() {
        validate_boundary_family_binder(binder)?;
    }
    for port in connection.ports() {
        validate_boundary_port_reference(port)?;
    }
    checked_range(connection.range()).map(|_| ())
}

#[cfg(test)]
mod tests;
