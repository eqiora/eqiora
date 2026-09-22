use super::super::scope::{PhysicalNominal, PortContract};
use eqiora_lang::{ActivationSyntax, FieldRoleSyntax};
use eqiora_schema::kernel::typing::{ExpressionType, SpatialSupport};

pub(super) fn bounded_description(mut text: String) -> String {
    let mut characters = text.char_indices();
    if let Some((end, _)) = characters.nth(511)
        && characters.next().is_some()
    {
        text.truncate(end);
        text.push('…');
    }
    text
}

pub(super) fn describe_type(value: &eqiora_core::ValueType) -> String {
    let nominal = value
        .enum_definition()
        .map(|id| format!("; nominal enum {id}"))
        .or_else(|| {
            value
                .finite_space()
                .map(|id| format!("; nominal finite space {id}, counts={}", value.is_count()))
        })
        .or_else(|| {
            value
                .index_set()
                .map(|id| format!("; nominal index set {id}"))
        })
        .unwrap_or_default();
    format!(
        "{:?}; dimension {}; shape {:?}; array rank {}; frame {:?}{nominal}",
        value.scalar_domain(),
        value.dimension(),
        value.shape().extents(),
        value.array_rank(),
        value.frame()
    )
}

pub(super) fn describe_port(port: &PortContract) -> String {
    match port {
        PortContract::Signal {
            direction,
            value_type,
            support,
            activation,
        } => format!(
            "signal {direction:?}; {}; {}; {}",
            describe_type(value_type),
            describe_activation(activation),
            describe_support(support.as_ref()),
        ),
        PortContract::Physical {
            nominal,
            across_type,
            through_type,
            ..
        } => {
            format!(
                "physical; across {}; through {}; nominal {}",
                describe_type(across_type),
                describe_type(through_type),
                describe_nominal(nominal),
            )
        }
        PortContract::BoundaryPhysical {
            nominal,
            connector,
            support,
            ..
        } => format!(
            "field-physical; trace {}; flux {}; {}; nominal {}",
            describe_type(connector.trace_type()),
            describe_type(connector.flux_type()),
            describe_support(Some(support)),
            describe_nominal(nominal),
        ),
    }
}

pub(super) fn describe_field(
    value: &ExpressionType<String>,
    role: FieldRoleSyntax,
    activation: &ActivationSyntax,
) -> String {
    format!(
        "{role:?}; {}; {}; {}",
        describe_type(&value.value_type),
        describe_activation(activation),
        describe_support(value.support.as_ref()),
    )
}

fn describe_nominal(nominal: &PhysicalNominal) -> String {
    match nominal {
        PhysicalNominal::Connector(key) => key.display(),
        PhysicalNominal::ModelDomain(name) => format!("{name} (Model-local)"),
        PhysicalNominal::BoundaryConnector { definition, shape } => {
            format!("{}; shape {:?}", definition.display(), shape.extents())
        }
    }
}

fn describe_activation(activation: &ActivationSyntax) -> String {
    match activation {
        ActivationSyntax::Continuous => "continuous".into(),
        ActivationSyntax::Named(name) => format!("activation {name} (occurrence identity unknown)"),
        _ => "activation unknown".into(),
    }
}

fn describe_support(support: Option<&SpatialSupport<String>>) -> String {
    match support {
        None => "no spatial support".into(),
        Some(SpatialSupport::Volume { domain, dimensions }) => {
            format!("support volume {domain}; axes {dimensions} (definition-local identity)")
        }
        Some(SpatialSupport::Boundary {
            domain,
            parent,
            dimensions,
        }) => {
            format!(
                "support boundary {domain}; parent {parent}; axes {dimensions} (definition-local identity)"
            )
        }
        Some(SpatialSupport::Interface {
            connection,
            dimensions,
        }) => {
            format!("support interface {connection}; axes {dimensions} (definition-local identity)")
        }
    }
}
