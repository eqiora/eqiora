//! Frame context for uniform Parameter types, without attaching spatial support.
use super::*;
use eqiora_lang::{ValueTypeSyntax, ValueTypeSyntaxKind};
use eqiora_schema::kernel::typing::SpatialSupport;

pub(in crate::hierarchy) fn parameter_type(
    file: &str,
    syntax: &ValueTypeSyntax,
    initializer: Option<&Expr>,
    frames: &BTreeMap<String, SpatialSupport<String>>,
) -> Result<ValueType, Diagnostic> {
    fn spatial(syntax: &ValueTypeSyntax) -> bool {
        match syntax.kind() {
            ValueTypeSyntaxKind::Vector { .. } | ValueTypeSyntaxKind::Tensor { .. } => true,
            ValueTypeSyntaxKind::Array { element, .. } => spatial(element),
            _ => false,
        }
    }
    if !spatial(syntax) {
        return crate::value_types::lower_value_type::<String>(file, syntax, None);
    }
    let invalid =
        |message: &str| source_error(codes::LANGUAGE_TYPE_ERROR, file, syntax.range(), message);
    let mut explicit = Vec::new();
    let mut pending = initializer.into_iter().collect::<Vec<_>>();
    while let Some(value) = pending.pop() {
        if value.resolved_enum().is_some() {
            continue;
        }
        match value.kind() {
            ExprKind::Case { value, arms } => {
                pending.push(value.as_ref());
                pending.extend(arms.iter().map(eqiora_lang::CaseArm::value));
            }
            ExprKind::Select {
                condition,
                then_value,
                else_value,
            } => {
                pending.extend([condition.as_ref(), then_value.as_ref(), else_value.as_ref()]);
            }
            ExprKind::Array(values) => pending.extend(values),
            ExprKind::Call { callee, .. } if callee.as_str() == "tensor_value" => {
                let name = super::tensor_values::frame_name(value)
                    .ok_or_else(|| invalid("tensor_value requires an exact named frame support"))?;
                let support = frames.get(name).ok_or_else(|| {
                    invalid("tensor_value frame is not an existing Cartesian support in this scope")
                })?;
                if !explicit.contains(&support) {
                    explicit.push(support);
                }
            }
            _ => {}
        }
    }
    let frame = if let Some(frame) = explicit.first() {
        if explicit
            .iter()
            .any(|other| other.ambient_dimensions() != frame.ambient_dimensions())
        {
            return Err(invalid(
                "explicit tensor frames have incompatible ambient dimensions",
            ));
        }
        *frame
    } else {
        let mut unique = Vec::new();
        for support in frames.values() {
            if support.ambient_dimensions().is_some() && !unique.contains(&support) {
                unique.push(support);
            }
        }
        match unique.as_slice() {
            [frame] => *frame,
            [] => {
                return Err(invalid(
                    "spatial Parameter requires an exact frame support context",
                ));
            }
            _ => {
                return Err(invalid(
                    "spatial Parameter frame context is ambiguous; use tensor_value with an explicit frame",
                ));
            }
        }
    };
    crate::value_types::lower_value_type(file, syntax, Some(frame))
}

pub(super) fn instance_frames(
    file: &str,
    component: &ComponentDecl,
    instance: &InstanceDecl,
    resolve: &mut dyn FnMut(&str) -> Option<SpatialSupport<String>>,
) -> Result<BTreeMap<String, SpatialSupport<String>>, Vec<Diagnostic>> {
    let mut frames = super::super::supports::component_spatial_supports(file, component)?;
    for binding in instance.bindings() {
        if !frames.contains_key(binding.name()) {
            continue;
        }
        let name = match binding.value().kind() {
            ExprKind::Name(name) => name.as_str(),
            ExprKind::Path(name) => name.as_str(),
            _ => continue,
        };
        if let Some(support) = resolve(name) {
            frames.insert(binding.name().to_owned(), support);
        }
    }
    Ok(frames)
}

pub(in crate::hierarchy) fn occurrence(
    support: &SpatialSupport<crate::identity::FullElaborationIdentity>,
) -> SpatialSupport<String> {
    match support {
        SpatialSupport::Coordinates { domain, factors } => SpatialSupport::Coordinates {
            domain: domain.to_string(),
            factors: factors
                .iter()
                .map(|(id, unit, axes)| (id.to_string(), *unit, *axes))
                .collect(),
        },
        SpatialSupport::Volume { domain, dimensions } => SpatialSupport::Volume {
            domain: domain.to_string(),
            dimensions: *dimensions,
        },
        SpatialSupport::Boundary {
            domain,
            parent,
            dimensions,
        } => SpatialSupport::Boundary {
            domain: domain.to_string(),
            parent: parent.to_string(),
            dimensions: *dimensions,
        },
        SpatialSupport::PhysicalInterface {
            domain,
            boundaries,
            parents,
            dimensions,
        } => SpatialSupport::PhysicalInterface {
            domain: domain.to_string(),
            boundaries: Box::new([boundaries[0].to_string(), boundaries[1].to_string()]),
            parents: Box::new([parents[0].to_string(), parents[1].to_string()]),
            dimensions: *dimensions,
        },
        SpatialSupport::Interface {
            connection,
            dimensions,
        } => SpatialSupport::Interface {
            connection: connection.to_string(),
            dimensions: *dimensions,
        },
    }
}

pub(in crate::hierarchy) fn literal(
    file: &str,
    expression: &Expr,
    values: &SymbolicParameterMap,
    resolve_frame: &mut dyn FnMut(&str) -> Option<SpatialSupport<String>>,
) -> Result<ValueLiteral, Diagnostic> {
    let value = super::expression_eval::evaluate_parameter_expression(
        file,
        expression,
        super::ExpressionContext::Let,
        &mut |name, range| {
            values.get(name).cloned().ok_or_else(|| {
                source_error(
                    codes::LANGUAGE_TYPE_ERROR,
                    file,
                    range,
                    "tensor component references an unknown value",
                )
            })
        },
        &mut |_| None,
        resolve_frame,
    )?;
    value.value.ok_or_else(|| {
        source_error(
            codes::LANGUAGE_TYPE_ERROR,
            file,
            expression.range(),
            "tensor_value requires closed component expressions",
        )
    })
}
